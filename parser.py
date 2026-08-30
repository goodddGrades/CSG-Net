import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--model-type', type=str, default='BiSGTAR',
                        help='choose the model.')
    parser.add_argument('--rna-num', type=int, default=0, help='circrna number.')
    parser.add_argument('--dis-num', type=int, default=0, help='disease number.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-8,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=256,
                        help='Dimension of representations')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight between lncRNA space and disease space')
    parser.add_argument('--beta', type=float, default=0.4,
                        help='Weight between lncRNA space and disease space')
    parser.add_argument('--gama', type=float, default=0.05,
                        help='Weight between lncRNA space and disease space')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Weight between lncRNA space and disease space')

    parser.add_argument('--data', type=int, default=5,
                        help='Dataset')
    parser.add_argument('--para', type=float, default=1e-2,
                        help='Smooth Factor')
    parser.add_argument('--neg_sample_ratio', type=float, default=-1,
                        help='Negative to positive sample ratio. -1 means use all negatives')
    parser.add_argument('--use_svd', action='store_true', default=True,
                        help='Use SVD preprocessing')
    parser.add_argument('--svd_components', type=int, default=50,
                        help='Number of SVD components to keep')
    parser.add_argument('--svd_denoise', action='store_true', default=False,
                        help='Use SVD for denoising')
    parser.add_argument('--use_presmooth', action='store_true', default=True,
                        help='Use similarity-based pre-smoothing before SVD')
    parser.add_argument('--presmooth_alpha', type=float, default=0.3,
                        help='Smoothing coefficient for pre-smoothing (0-1)')
    parser.add_argument('--dual_side_smooth', action='store_true', default=True,
                        help='Use dual-side smoothing (both RNA and disease similarity)')
    parser.add_argument('--similarity_method', type=str, default='jaccard',
                        choices=['jaccard', 'gaussian', 'cosine'],
                        help='Method to compute similarity matrix')
    parser.add_argument('--use_contrastive', action='store_true', default=True,
                        help='Enable GDCL-NCDA style contrastive loss.')
    parser.add_argument('--cl_proj_dim', type=int, default=256,
                        help='Projection head output dimension (0 to disable projection / contrastive).')
    parser.add_argument('--cl_temperature', type=float, default=0.2,
                        help='Contrastive temperature.')
    parser.add_argument('--cl_weight', type=float, default=0.1,
                        help='Weight of contrastive loss in total loss.')
    parser.add_argument('--cl_warmup_epochs', type=int, default=20,
                        help='Number of epochs to warm up contrastive weight.')
    parser.add_argument('--cl_skip_if_svd', action='store_true', default=False,
                        help='If SVD enabled and this True, skip contrastive.')
    parser.add_argument('--cl_svd_scale', type=float, default=0.1,
                        help='Scale contrastive weight when SVD is used (multiplicative).')
    parser.add_argument('--use_xgboost', action='store_true', default=True,
                        help='Use XGBoost classifier instead of NN matrix multiplication.')
    args = parser.parse_known_args()[0]
    return args
