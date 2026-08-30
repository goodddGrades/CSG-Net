import torch
import torch.nn as nn
import torch.nn.functional as F


def find_key(i, cancer_dict):
    name = list(cancer_dict.keys())[list(cancer_dict.values()).index(i)]
    return name


import torch
import torch.nn as nn
import torch.nn.functional as F

def train(model, y0, args, alpha, i, rel, train_mask):
    loss_list = []
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss(reduction='none')

    for e in range(args.epochs):
        model.train()
        yl, rna_feat, rna_quality, hc, yd, dis_feat, dis_quality, hd, zc, zc_proj, zd, zd_proj = model(y0)
        y = alpha * yl + (1 - alpha) * yd.t()

        rna_confidence = torch.mul(hc, rel)
        dis_confidence = torch.mul(hd, rel.t())

        rna_SPC = torch.mean(rna_quality)
        rna_bce = criterion(hc, rel) * train_mask
        rna_bce = rna_bce.sum() / (train_mask.sum().clamp(min=1.0))
        rna_mse = F.mse_loss(yl, rna_confidence, reduction='none') * train_mask
        rna_mse = rna_mse.sum() / (train_mask.sum().clamp(min=1.0))
        if getattr(args, 'ablate_arb', False):
            rna_mse = rna_mse * 0.0
        rna_TAR = rna_bce + rna_mse
        rna_loss = args.beta * rna_TAR + (1 - args.beta) * rna_SPC

        dis_SPC = torch.mean(dis_quality)
        dis_bce = criterion(hd, rel.t()) * train_mask.t()
        dis_bce = dis_bce.sum() / (train_mask.t().sum().clamp(min=1.0))
        dis_mse = F.mse_loss(yd, dis_confidence, reduction='none') * train_mask.t()
        dis_mse = dis_mse.sum() / (train_mask.t().sum().clamp(min=1.0))
        if getattr(args, 'ablate_arb', False):
            dis_mse = dis_mse * 0.0
        dis_TAR = dis_bce + dis_mse
        dis_loss = args.beta * dis_TAR + (1 - args.beta) * dis_SPC

        loss_inter = alpha * rna_loss + (1 - alpha) * dis_loss

        loss_cls = criterion(y, rel) * train_mask
        loss_cls = loss_cls.sum() / (train_mask.sum().clamp(min=1.0))

        loss_contrast = torch.tensor(0.0, device=rel.device)
        apply_cl = getattr(args, 'use_contrastive', False)
        if apply_cl and not (getattr(args, 'use_svd', False) and getattr(args, 'cl_skip_if_svd', False)):
            if zc_proj is not None and zd_proj is not None:
                zc_n = F.normalize(zc_proj, p=2, dim=1)
                zd_n = F.normalize(zd_proj, p=2, dim=1)
                sim = torch.matmul(zc_n, zd_n.t()) / max(args.cl_temperature, 1e-6)

                pos_mask = (rel > 0.5) & (train_mask > 0.5)
                pos_idx = pos_mask.nonzero(as_tuple=False)
                if pos_idx.size(0) > 0:
                    row_logsumexp = torch.logsumexp(sim, dim=1)
                    pos_rows = pos_idx[:, 0]
                    pos_sims = sim[pos_rows, pos_idx[:, 1]]
                    loss_terms = -pos_sims + row_logsumexp[pos_rows]
                    loss_contrast = loss_terms.mean()
                else:
                    loss_contrast = torch.tensor(0.0, device=rel.device)
            else:
                loss_contrast = torch.tensor(0.0, device=rel.device)

        warmup_epochs = max(getattr(args, 'cl_warmup_epochs', 0), 1)
        warmup_factor = min(1.0, float(e + 1) / warmup_epochs) if getattr(args, 'cl_warmup_epochs', 0) > 0 else 1.0
        current_lambda = getattr(args, 'cl_weight', 0.0) * warmup_factor
        if getattr(args, 'use_svd', False):
            current_lambda = current_lambda * getattr(args, 'cl_svd_scale', 1.0)

        loss = args.gama * loss_cls + (1 - args.gama) * loss_inter + current_lambda * loss_contrast

        loss_list.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            yl, _, _, hc, yd, _, _, hd, _, _, _, _ = model(y0)

    model.eval()
    yl, rna_feat, rna_quality, hc, yd, dis_feat, dis_quality, hd, zc, zc_proj, zd, zd_proj = model(y0)
    y_pred = alpha * yl + (1 - alpha) * yd.t()

    if args.cuda:
        y_pred = y_pred.cpu().detach().numpy()
        zc = zc.cpu().detach().numpy()
        zd = zd.cpu().detach().numpy()
    else:
        y_pred = y_pred.detach().numpy()
        zc = zc.detach().numpy()
        zd = zd.detach().numpy()

    return y_pred, model, zc, zd
