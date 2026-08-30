import argparse
import datetime
import math
import os
import random
from types import SimpleNamespace

import numpy as np

import save_results
from metrics import metrics as orig_metrics
from utils import load_association

SIM_BY_DATA = {1: 'jaccard', 2: 'jaccard', 3: 'jaccard', 4: 'gaussian', 5: 'gaussian'}


def split_folds(M, n_fold=5, seed=100):
    one_list = list(zip(*np.where(M == 1)))
    zero_list = list(zip(*np.where(M == 0)))
    rnd = random.Random(seed)
    rnd.shuffle(one_list)
    rnd.shuffle(zero_list)
    split = math.ceil(len(one_list) / n_fold)
    split_0 = math.ceil(len(zero_list) / n_fold)
    folds = []
    for i in range(n_fold):
        folds.append((one_list[i * split:(i + 1) * split],
                      zero_list[i * split_0:(i + 1) * split_0]))
    return folds


def katz_scores(A):
    beta = 1.0 / (2.0 * np.linalg.norm(A, 2))
    n = A.shape[0]
    inv = np.linalg.inv(np.eye(n) - beta ** 2 * (A @ A.T))
    return beta * (inv @ A)


def rwr_scores(A, restart=0.8):
    n, m = A.shape
    B = np.zeros((n + m, n + m))
    B[:n, n:] = A
    B[n:, :n] = A.T
    outdeg = B.sum(axis=1)
    outdeg[outdeg == 0] = 1.0
    P = B / outdeg[:, None]
    M = np.eye(n + m) - (1 - restart) * P.T
    X0 = np.zeros((n + m, m))
    X0[n + np.arange(m), np.arange(m)] = 1.0
    X = restart * np.linalg.solve(M, X0)
    return X[:n, :]


METHODS = {'KATZ': katz_scores, 'RWR': rwr_scores}


def run_baseline(name, data, M, folds):
    curves, scores = [], []
    for k, (test_pos, test_neg) in enumerate(folds):
        A = M.copy()
        for (i, j) in test_pos:
            A[i, j] = 0
        print(f"  fold{k}: {name} 计算中...", flush=True)
        S = METHODS[name](A)

        roc = np.zeros_like(M, dtype=int)
        roc[M == 1] = 2
        for (i, j) in test_pos:
            roc[i, j] = 1

        score_matrix = S.copy()
        minvalue = np.min(score_matrix)
        score_matrix[roc == 2] = minvalue - 100

        tpr, fpr, rec, prec, acc, F1 = orig_metrics(score_matrix, roc)
        curves.append((list(tpr), list(fpr), list(rec), list(prec), list(acc), list(F1)))
        scores.append((score_matrix.copy(), roc.copy()))
        print(f"  fold{k}: {name} done", flush=True)
    return curves, scores


def main():
    p = argparse.ArgumentParser(description='经典基线 KATZ/NCP, 同协议同口径全格式保存')
    p.add_argument('--data', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                   help='数据集编号 (默认全部 5 个)')
    p.add_argument('--methods', nargs='+', default=['KATZ', 'RWR'])
    args = p.parse_args()

    for data in args.data:
        M = load_association(SimpleNamespace(data=data))
        folds = split_folds(M)
        print(f"\n>>> dataset={data} ({M.shape[0]}x{M.shape[1]}, {int(M.sum())} 关联)", flush=True)
        for name in args.methods:
            curves, scores = run_baseline(name, data, M, folds)
            a = SimpleNamespace(
                data=data, use_svd=False, use_contrastive=False, use_xgboost=False,
                cl_proj_dim=0, presmooth_alpha=0.3, similarity_method=SIM_BY_DATA[data],
                alpha=0.5, beta=0.4, gama=0.05, cl_weight=0.0,
                dual_side_smooth=True, neg_sample_ratio=-1,
                exp_name=f'base_{name}', classifier=name)
            base = save_results.build_tag(a) + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            out_dir = os.path.join('results', base)
            n = 1
            while os.path.exists(out_dir):
                out_dir = os.path.join('results', f'{base}_{n}')
                n += 1
            os.makedirs(out_dir, exist_ok=True)
            means = save_results.save_run(a, curves, scores, out_dir)
            print(f"[Saved] {os.path.abspath(out_dir)}  AUC={means['mean_auc']:.4f} AUPR={means['mean_aupr']:.4f}", flush=True)


if __name__ == '__main__':
    main()
