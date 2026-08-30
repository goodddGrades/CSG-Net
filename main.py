import math
import numpy as np
import torch
from train import train
from metrics import metrics, calculate_performace
from utils import load_association, build_model, compute_similarity_matrix, run_xgboost_pipeline
import random
import os

def main(args):
    circrna_disease_matrix = load_association(args)
    print('Now Load Dataset ' + str(args.data))
    args.rna_num = circrna_disease_matrix.shape[0]
    args.dis_num = circrna_disease_matrix.shape[1]
    print('rna_num', args.rna_num)
    print('dis_num', args.dis_num)
    n_fold = 5
    if args.data == 8:
        n_fold = 10
    index_tuple = (np.where(circrna_disease_matrix == 1))
    index_tuple_0 = (np.where(circrna_disease_matrix == 0))
    one_list = list(zip(index_tuple[0], index_tuple[1]))
    zero_list = list(zip(index_tuple_0[0], index_tuple_0[1]))
    rnd_state = random.Random(100)
    rnd_state.shuffle(one_list)
    rnd_state.shuffle(zero_list)

    split = math.ceil(len(one_list) / n_fold)
    split_0 = math.ceil(len(zero_list) / n_fold)
    print('split: ', split)
    print('split_0: ', split_0)
    all_tpr = []
    all_fpr = []
    all_recall = []
    all_precision = []
    all_accuracy = []
    all_F1 = []
    for i in range(n_fold):
        test_index = one_list[i * split:(i + 1) * split]
        test_index_0 = zero_list[i * split_0:(i + 1) * split_0]
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        for index in test_index:
            new_circrna_disease_matrix[index[0], index[1]] = 0
        if args.use_svd:
            train_matrix = new_circrna_disease_matrix.copy()
            original_train_matrix = train_matrix.copy()
            if args.use_presmooth:
                print("Applying Similarity Smoothing before SVD...")
                S_lnc = compute_similarity_matrix(train_matrix, method=args.similarity_method)
                if args.dual_side_smooth:
                    S_dis = compute_similarity_matrix(train_matrix.T, method=args.similarity_method)
                    rna_smooth = args.presmooth_alpha * np.dot(S_lnc, train_matrix)
                    dis_smooth = args.presmooth_alpha * np.dot(train_matrix, S_dis)
                    train_matrix_smooth = 0.5 * (rna_smooth + dis_smooth) + (1 - args.presmooth_alpha) * train_matrix
                    print(f"Applied dual-side smoothing with alpha={args.presmooth_alpha}")
                else:
                    train_matrix_smooth = args.presmooth_alpha * np.dot(S_lnc, train_matrix) + (
                                1 - args.presmooth_alpha) * train_matrix
                    print(f"Applied single-side smoothing with alpha={args.presmooth_alpha}")
                train_matrix = train_matrix_smooth
            U, s, Vt = np.linalg.svd(train_matrix, full_matrices=False)

            k = min(args.svd_components, len(s))
            U_k = U[:, :k]
            s_k = s[:k]
            Vt_k = Vt[:k, :]

            train_matrix_svd = (U_k * s_k) @ Vt_k
            mask = original_train_matrix > 0
            train_matrix = np.clip(train_matrix_svd, 0, 1)
            train_matrix[mask] = original_train_matrix[mask]
            rel_matrix = train_matrix
        else:
            rel_matrix = new_circrna_disease_matrix
        if args.neg_sample_ratio > 0:
            train_zero_indices = np.where(rel_matrix < 0.5)
            train_zero_list = list(zip(train_zero_indices[0], train_zero_indices[1]))

            num_positive = np.sum(rel_matrix == 1)
            num_negative_to_keep = int(num_positive * args.neg_sample_ratio)
            rnd_state_neg = random.Random(args.seed + i)
            sampled_negatives = rnd_state_neg.sample(
                train_zero_list,
                min(num_negative_to_keep, len(train_zero_list))
            )
            train_mask = np.zeros_like(rel_matrix)
            train_mask[rel_matrix >= 0.5] = 1
            for idx in sampled_negatives:
                train_mask[idx[0], idx[1]] = 1
        else:
            train_mask = np.ones_like(new_circrna_disease_matrix)

        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
        circnum = rel_matrix.shape[0]
        disnum = rel_matrix.shape[1]
        rel_matrix_tensor = torch.tensor(np.array(rel_matrix).astype(np.float32))
        train_mask_tensor = torch.tensor(np.array(train_mask).astype(np.float32))
        model_init = build_model(args.model_type)
        model = model_init(args)
        if args.cuda:
            model = model.cuda()
            rel_matrix_tensor = rel_matrix_tensor.cuda()
            train_mask_tensor = train_mask_tensor.cuda()
        smooth_factor = args.para
        norm_rel = smooth_factor + (1 - 2 * smooth_factor) * rel_matrix_tensor
        resi, model, zc, zd = train(model, norm_rel, args, args.alpha, i, rel_matrix_tensor, train_mask_tensor)
        if getattr(args, 'use_xgboost', False):
            all_test_indices = test_index + test_index_0
            ymat = run_xgboost_pipeline(
                args, zc, zd, train_mask_tensor,
                circrna_disease_matrix, all_test_indices
            )
        else:
            if isinstance(resi, np.ndarray):
                ymat = resi
            elif args.cuda:
                ymat = resi.cpu().detach().numpy()
            else:
                ymat = resi.detach().numpy()
        S = ymat
        prediction_matrix = S
        zero_matrix = np.zeros((prediction_matrix.shape[0], prediction_matrix.shape[1]))
        score_matrix_temp = prediction_matrix.copy()
        score_matrix = score_matrix_temp + zero_matrix
        minvalue = np.min(score_matrix)
        score_matrix[np.where(roc_circrna_disease_matrix == 2)] = minvalue - 100
        tpr_list, fpr_list, recall_list, precision_list, accuracy_list, F1_list = metrics(score_matrix,
                                                                                          roc_circrna_disease_matrix)
        all_tpr.append(tpr_list)
        all_fpr.append(fpr_list)
        all_recall.append(recall_list)
        all_precision.append(precision_list)
        all_accuracy.append(accuracy_list)
        all_F1.append(F1_list)

    tpr_arr = np.array(all_tpr)
    fpr_arr = np.array(all_fpr)
    recall_arr = np.array(all_recall)
    precision_arr = np.array(all_precision)
    accuracy_arr = np.array(all_accuracy)
    F1_arr = np.array(all_F1)

    mean_cross_tpr = np.mean(tpr_arr, axis=0)
    mean_cross_fpr = np.mean(fpr_arr, axis=0)
    mean_cross_recall = np.mean(recall_arr, axis=0)
    mean_cross_precision = np.mean(precision_arr, axis=0)
    mean_cross_accuracy = np.mean(accuracy_arr, axis=0)

    roc_auc = np.trapz(mean_cross_tpr, mean_cross_fpr)
    AUPR = np.trapz(mean_cross_precision, mean_cross_recall)

    print("AUC:%.4f,AUPR:%.4f" % (roc_auc, AUPR))
    k_idx = 49
    print("Recall=%.4f  Precision=%.4f  F1=%.4f  ACC=%.4f" % (
        np.mean(recall_arr[:, k_idx]), np.mean(precision_arr[:, k_idx]),
        np.mean(F1_arr[:, k_idx]), np.mean(accuracy_arr[:, k_idx])))
    print('-' * 200)

    return roc_auc
