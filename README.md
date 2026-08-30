# CSG-Net

## Architecture Overview

```
Raw Sparse Matrix
        │
        ▼
┌─────────────────────────────────┐
│  Stage 1: NCP-SVD Preprocessing │  ← main.py (lines 45-73)
│  - Similarity pre-smoothing     │  ← utils.py: compute_similarity_matrix()
│  - SVD decomposition (k=50)     │
│  - Clip [0,1] + recover known   │
└─────────────────────────────────┘
        │
        ▼  Enhanced Matrix (rel_matrix)
┌─────────────────────────────────┐
│  Stage 2: Contra-SGTAR          │  ← model.py: BiSGTAR class
│  - Sparse Quality Control (SQC) │    + train.py: train()
│  - Contrastive Learning (CL)    │
│  - Auxiliary Reconstruction (ARB)│
└─────────────────────────────────┘
        │
        ▼  Latent Features (zc, zd)
┌─────────────────────────────────┐
│  Stage 3: NBC Classifier        │  ← utils.py: run_xgboost_pipeline()
│  - XGBoost on [zc ⊕ zd]        │
└─────────────────────────────────┘
        │
        ▼
   Prediction Scores
```

## Code Mapping (Manuscript → Code)

| Manuscript Section | Code Location | Function/Class |
|---|---|---|
| **2.1 NCP-SVD** | `main.py:45-73` | SVD preprocessing per fold |
| 2.1.3 Network Consistency Projection | `main.py:48-61` + `utils.py:compute_similarity_matrix()` | Pre-smoothing with Jaccard/Gaussian |
| 2.1.4 SVD & Feature Reconstruction | `main.py:62-73` | `np.linalg.svd()` → clip → recover |
| **2.2 Contra-SGTAR** | `model.py:44-82` | `BiSGTAR` class |
| 2.2.1 Sparse Quality Control | `model.py:68-69` | `FeatQC_rna/dis` + `torch.mul()` |
| 2.2.2 Contrastive Learning | `train.py:60-83` | InfoNCE loss computation |
| 2.2.3 Auxiliary Reconstruction | `train.py:28-38` | `rna_mse` / `dis_mse` (MSE loss) |
| 2.2.4 Weight Warm-up | `train.py:86-90` | `cl_warmup_epochs` + `cl_svd_scale` |
| **2.3 NBC Module** | `utils.py:10-76` | `run_xgboost_pipeline()` → XGBoost |

## Key Files

| File | Purpose |
|---|---|
| `start.py` | Entry point. Configures datasets and runs experiments |
| `main.py` | 5-fold CV loop, NCP-SVD preprocessing, evaluation |
| `model.py` | `BiSGTAR` neural network (SQC + TAR autoencoders) |
| `train.py` | Training loop with contrastive + reconstruction losses |
| `parser.py` | All hyperparameters (SVD, CL, XGBoost, smoothing) |
| `utils.py` | Similarity matrices, XGBoost pipeline, data loading |
| `metrics.py` | ROC/PR curve metrics (sweep-based evaluation) |
| `save_results.py` | Auto-saves results to `results/` directory |

## Datasets

| Dataset Name | circRNAs | diseases | args.data |
|---|---|---|---|
| Circ2Traits | 923 | 104 | 4 |
| circad | 1265 | 151 | 5 |
| circR2Disease | 533 | 89 | 1 |

## Running

```bash
python start.py
```
