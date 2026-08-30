import datetime
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import main as main_mod
from parser import parameter_parser
from utils import set_seed

_curves = []
_scores = []
_feats = []
_orig_metrics = None
_orig_train = None

DATASET_NAMES = {
    1: 'CircR2Disease (533 circRNAs x 89 diseases, 613 associations, sparsity 98.71%)',
    2: 'CircR2Cancer (514 circRNAs x 62 diseases, 647 associations, sparsity 97.97%)',
    3: 'circRNADisease (312 circRNAs x 40 diseases, 331 associations, sparsity 97.35%)',
    4: 'Circ2Traits (923 circRNAs x 104 diseases, 37,660 associations, sparsity 60.77%)',
    5: 'Circad (1265 circRNAs x 151 diseases, 1369 associations, sparsity 99.28%)',
}


def _patched_metrics(score_matrix, roc_matrix):
    tpr, fpr, rec, prec, acc, F1 = _orig_metrics(score_matrix, roc_matrix)
    _curves.append((list(tpr), list(fpr), list(rec), list(prec), list(acc), list(F1)))
    _scores.append((score_matrix.copy(), roc_matrix.copy()))
    return tpr, fpr, rec, prec, acc, F1


def _patched_train(model, y0, args, alpha, i, rel, train_mask):
    result = _orig_train(model, y0, args, alpha, i, rel, train_mask)
    zc, zd = result[2], result[3]
    _feats.append((zc.copy(), zd.copy()))
    return result


def install():
    global _orig_metrics, _orig_train
    if _orig_metrics is None:
        _orig_metrics = main_mod.metrics
        main_mod.metrics = _patched_metrics
    if _orig_train is None:
        _orig_train = main_mod.train
        main_mod.train = _patched_train


def _resample_ap(y_true, y_score, ratio, n_repeats=20, seed=42):
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_neg = min(int(len(pos_idx) * ratio), len(neg_idx))
    aps = []
    rng = np.random.default_rng(seed)
    for r in range(n_repeats):
        pick = rng.choice(neg_idx, n_neg, replace=False)
        idx = np.concatenate([pos_idx, pick])
        aps.append(average_precision_score(y_true[idx], y_score[idx]))
    return float(np.mean(aps)), float(np.std(aps))


def build_tag(args):
    exp = getattr(args, 'exp_name', '')
    tag = (f"{exp + '_' if exp else ''}d{args.data}_svd{int(args.use_svd)}_cl{int(args.use_contrastive)}"
           f"_xgb{int(getattr(args, 'use_xgboost', True))}"
           f"_pre{args.presmooth_alpha}_p{getattr(args, 'cl_proj_dim', 0)}"
           f"_sim{getattr(args, 'similarity_method', '?')}"
           f"_abg{args.alpha}_{args.beta}_{args.gama}"
           f"_cw{getattr(args, 'cl_weight', 0)}"
           f"_dss{int(getattr(args, 'dual_side_smooth', True))}"
           f"_nsr{getattr(args, 'neg_sample_ratio', -1)}")
    return tag


def save_run(args, curves, scores, out_dir):
    if not curves:
        return {}
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    fold_aucs, fold_auprs = [], []
    for (tpr, fpr, rec, prec, acc, F1) in curves:
        fold_aucs.append(trapz(np.array(tpr), np.array(fpr)))
        fold_auprs.append(trapz(np.array(prec), np.array(rec)))

    tpr_arr = np.array([c[0] for c in curves])
    fpr_arr = np.array([c[1] for c in curves])
    recall_arr = np.array([c[2] for c in curves])
    precision_arr = np.array([c[3] for c in curves])
    accuracy_arr = np.array([c[4] for c in curves])
    F1_arr = np.array([c[5] for c in curves])

    mean_accuracy = float(np.mean(np.mean(accuracy_arr, axis=1), axis=0))
    mean_recall = float(np.mean(np.mean(recall_arr, axis=1), axis=0))
    mean_precision = float(np.mean(np.mean(precision_arr, axis=1), axis=0))
    mean_F1 = float(np.mean(np.mean(F1_arr, axis=1), axis=0))
    mean_auc = float(np.mean(fold_aucs))
    mean_aupr = float(np.mean(fold_auprs))

    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {args.data}  SVD={args.use_svd}  Contrastive={args.use_contrastive}  "
                f"xgboost={getattr(args, 'use_xgboost', True)}  "
                f"cl_proj_dim={getattr(args, 'cl_proj_dim', 0)}  presmooth_alpha={args.presmooth_alpha}  "
                f"similarity={getattr(args, 'similarity_method', '?')}\n")
        f.write(f"AUC: {mean_auc:.4f} +/- {np.std(fold_aucs):.4f}  "
                f"AUPR: {mean_aupr:.4f} +/- {np.std(fold_auprs):.4f}\n")
        k_idx = 49
        f.write(f"Recall={np.mean(recall_arr[:, k_idx]):.4f}  "
                f"Precision={np.mean(precision_arr[:, k_idx]):.4f}  "
                f"F1={np.mean(F1_arr[:, k_idx]):.4f}  "
                f"ACC={np.mean(accuracy_arr[:, k_idx]):.4f}\n")

    np.savez(os.path.join(out_dir, f'curves_dataset{args.data}.npz'),
             tpr_arr=tpr_arr, fpr_arr=fpr_arr, precision_arr=precision_arr,
             recall_arr=recall_arr, accuracy_arr=accuracy_arr, F1_arr=F1_arr,
             fold_aucs=np.array(fold_aucs), fold_auprs=np.array(fold_auprs))

    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2, ensure_ascii=False)

    return dict(mean_auc=mean_auc, mean_aupr=mean_aupr, mean_accuracy=mean_accuracy,
                mean_recall=mean_recall, mean_precision=mean_precision, mean_F1=mean_F1)


def _save_features(args, out_dir):
    if not getattr(args, 'use_xgboost', True):
        return
    if not _feats or len(_feats) != len(_scores):
        return
    zc_list = np.array([z for z, _ in _feats])
    zd_list = np.array([d for _, d in _feats])
    roc_list = np.array([r.copy() for _, r in _scores])
    test_mask_list = np.array([((r != 2) & (s != 0)).astype(np.uint8) for s, r in _scores])
    np.savez(os.path.join(out_dir, f'features_dataset{args.data}.npz'),
             zc_list=zc_list, zd_list=zd_list, roc_list=roc_list, test_mask_list=test_mask_list)


def _save(args):
    if not _curves:
        return
    base = build_tag(args) + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join('results', base)
    n = 1
    while os.path.exists(out_dir):
        out_dir = os.path.join('results', f'{base}_{n}')
        n += 1
    os.makedirs(out_dir, exist_ok=True)
    means = save_run(args, _curves, _scores, out_dir)
    _save_features(args, out_dir)
    print(f"[Saved] {os.path.abspath(out_dir)}  AUC={means['mean_auc']:.4f} AUPR={means['mean_aupr']:.4f}")


def wrap(main_fn):
    def wrapped(args):
        _curves.clear()
        _scores.clear()
        _feats.clear()
        result = main_fn(args)
        _save(args)
        return result
    return wrapped


def _build_args():
    args = parameter_parser()
    args.cuda = not args.no_cuda and __import__('torch').cuda.is_available()
    args.seed = 1
    args.alpha = 0.5
    args.beta = 0.4
    args.gama = 0.05
    args.use_contrastive = True
    args.cl_weight = 0.1
    return args


def _apply_dataset_config(args, data):
    args.data = data
    args.alpha, args.beta, args.gama = 0.5, 0.4, 0.05
    if data == 8:
        args.alpha, args.beta, args.gama = 0.8, 0.8, 0.8
        args.weight_decay = 1e-8
        args.epochs = 600
        args.neg_sample_ratio = -1
        args.similarity_method = 'jaccard'
    elif data == 9:
        args.alpha, args.beta, args.gama = 0.8, 0.6, 0.8
        args.weight_decay = 1e-10
        args.epochs = 400
        args.neg_sample_ratio = 49
    elif data == 7:
        args.neg_sample_ratio = 31
    elif data == 4:
        args.similarity_method = 'gaussian'
    else:
        args.neg_sample_ratio = -1


def main():
    import argparse
    p = argparse.ArgumentParser(description='原版BiSGTAR(svd+cl)结果保存器, 不改动原版代码')
    p.add_argument('--data', nargs='+', type=int, default=[1, 2, 3, 4, 5], help='数据集编号列表(默认与start.py一致)')
    p.add_argument('--cl', nargs='+', type=int, default=[64], help='cl_proj_dim 列表')
    args = p.parse_args()

    for data in args.data:
        for cl_dim in args.cl:
            a = _build_args()
            _apply_dataset_config(a, data)
            a.cl_proj_dim = cl_dim
            set_seed(a.seed)
            print(f"\n>>> Running dataset={data} cl_proj_dim={cl_dim}")
            wrapped_main = wrap(main_mod.main)
            wrapped_main(a)


install()
if __name__ == '__main__':
    main()
