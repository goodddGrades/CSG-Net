import torch
import numpy as np
from model import BiSGTAR
import xgboost as xgb


def run_xgboost_pipeline(args, zc, zd, train_mask, circrna_disease_matrix, test_indices):
    print(">>> Starting XGBoost Training on GPU..." if args.cuda else ">>> Starting XGBoost Training on CPU...")

    zc_np = zc
    zd_np = zd

    if isinstance(train_mask, torch.Tensor):
        if train_mask.is_cuda:
            train_mask_np = train_mask.cpu().numpy()
        else:
            train_mask_np = train_mask.numpy()
    else:
        train_mask_np = train_mask

    rows0, cols0 = np.where(train_mask_np == 1)
    test_set = {(r, c) for r, c in test_indices}
    keep = np.array([(r, c) not in test_set for r, c in zip(rows0, cols0)])
    train_rows, train_cols = rows0[keep], cols0[keep]
    print(f"XGBoost train pairs: {len(train_rows)}  (排除测试对 {int((~keep).sum())})")

    X_train = np.concatenate([zc_np[train_rows], zd_np[train_cols]], axis=1)
    Y_train = circrna_disease_matrix[train_rows, train_cols]

    test_rows = [x[0] for x in test_indices]
    test_cols = [x[1] for x in test_indices]

    X_test = np.concatenate([zc_np[test_rows], zd_np[test_cols]], axis=1)

    xgb_device = 'cuda' if args.cuda else 'cpu'

    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=500,
        learning_rate=0.05,
        tree_method='hist',
        device=xgb_device,
        eval_metric='auc',
        use_label_encoder=False,
        random_state=args.seed
    )

    clf.fit(X_train, Y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]

    prediction_matrix = np.zeros_like(circrna_disease_matrix, dtype=float)
    prediction_matrix[test_rows, test_cols] = y_prob

    return prediction_matrix

def load_dict(data):
    if data == 1:
        cancer_dict = {'glioma': 7, 'bladder cancer': 9, 'breast cancer': 10, 'cervical cancer': 53,
                       'cervical carcinoma': 64, 'colorectal cancer': 11, 'gastric cancer': 19}
    elif data == 2:
        cancer_dict = {'glioma': 23, 'bladder cancer': 2, 'breast cancer': 4, 'cervical cancer': 6,
                       'colorectal cancer': 12, 'gastric cancer': 20}
    elif data == 3:
        cancer_dict = {'glioma': 20, 'bladder cancer': 19, 'breast cancer': 6, 'cervical cancer': 16,
                       'colorectal cancer': 1, 'gastric cancer': 0}
    elif data == 4:
        cancer_dict = {'bladder cancer': 58, 'breast cancer': 46, 'glioma': 89, 'glioblastoma': 88,
                       'glioblastoma multiforme': 59, 'cervical cancer': 23, 'colorectal cancer': 6,
                       'gastric cancer': 15}
    elif data == 5:
        cancer_dict = {'bladder cancer': 94, 'breast cancer': 53, 'triple-negative breast cancer': 111, 'gliomas': 56,
                       'glioma': 76,
                       'cervical cancer': 65, 'colorectal cancer': 143, 'gastric cancer': 28}
    else:
        cancer_dict = {}
    return cancer_dict


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def build_model(model_type):
    if model_type == 'BiSGTAR':
        return BiSGTAR


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sort_matrix(score_matrix, interact_matrix):
    sort_index = np.argsort(-score_matrix, axis=0)
    score_sorted = np.zeros(score_matrix.shape)
    y_sorted = np.zeros(interact_matrix.shape)
    for i in range(interact_matrix.shape[1]):
        score_sorted[:, i] = score_matrix[:, i][sort_index[:, i]]
        y_sorted[:, i] = interact_matrix[:, i][sort_index[:, i]]
    return y_sorted, score_sorted, sort_index


def compute_similarity_matrix(matrix, method='jaccard', sigma=1.0):
    if method == 'jaccard':
        from sklearn.metrics.pairwise import pairwise_distances
        binary_matrix = (matrix > 0).astype(int)
        dist_matrix = pairwise_distances(binary_matrix, metric='jaccard')
        sim_matrix = 1 - dist_matrix

    elif method == 'gaussian':
        from sklearn.metrics.pairwise import rbf_kernel
        sim_matrix = rbf_kernel(matrix, gamma=1 / (2 * sigma ** 2))

    elif method == 'cosine':
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(matrix)

    row_sums = sim_matrix.sum(axis=1)
    sim_matrix = sim_matrix / row_sums[:, np.newaxis]

    return sim_matrix

def load_association(args):
    if args.data == 5:
        circrna_disease_matrix = np.loadtxt('./data/Dataset5/1265_151_circrna_disease_assoication.csv',
                                            delimiter=',')
    elif args.data == 4:
        circrna_disease_matrix = np.loadtxt('./data/Dataset4/923_104_circrna_disease_assoication.csv',
                                            delimiter=',')
    elif args.data == 3:
        circrna_disease_matrix = np.loadtxt('./data/Dataset3/312_40_circrna_disease_assoication.csv',
                                            delimiter=',')
    elif args.data == 2:
        circrna_disease_matrix = np.loadtxt('./data/Dataset2/514_62_circrna_disease_assoication.csv',
                                            delimiter=',')
    elif args.data == 1:
        circrna_disease_matrix = np.loadtxt('./data/Dataset1/533_89_circrna_disease_assoication.csv',
                                            delimiter=',')
    elif args.data == 6:
        circrna_disease_matrix = np.loadtxt('./data/KGET-Dataset1/330_79_circrna_disease_assoication.csv',
                                            delimiter=',')
    elif args.data == 7:
        circrna_disease_matrix = np.loadtxt('./data/KGET-Dataset2/561_190_circrna_disease_assoication.csv',
                                            delimiter=',')
    elif args.data == 8:
        circrna_disease_matrix = np.loadtxt('./data/Dataset8/l_d2.csv', delimiter=',')
    elif args.data == 9:
        circrna_disease_matrix = np.loadtxt('./data/Dataset9/C_D2.csv', delimiter=',')
    else:
        print('No data available...')
        return ''
    return circrna_disease_matrix
