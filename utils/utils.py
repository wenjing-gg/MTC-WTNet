# utils.py  ────────────────────────────────────────────────────────────────
import os, sys, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt, seaborn as sns
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from metrics import assd, hd95   

def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch: int,
    scaler: amp.GradScaler | None = None,
    accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
):
    if not hasattr(criterion, '_opt_checked'):
        opt_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
        miss = [p for p in criterion.parameters() if id(p) not in opt_ids]
        if miss:
            optimizer.add_param_group({
                'params': miss,
                'lr': optimizer.param_groups[0]['lr'],
                'weight_decay': 0.,
            })
            print(f"[Info] Added {len(miss)} criterion params to optimizer.")
        criterion._opt_checked = True

    model.train()
    run_loss = 0.0
    run_correct = 0
    seen = 0

    hist = {k: [] for k in ['total', 'cls', 'seg', 'distill', 'kvloss', 'weights']}

    loader = tqdm(data_loader, file=sys.stdout, ncols=110)
    loader.set_description(f"Epoch {epoch:02d}")
    optimizer.zero_grad(set_to_none=True)

    for step, (imgs, cls_gt, seg_gt) in enumerate(loader, 1):
        imgs   = imgs.to(device, non_blocking=True)
        cls_gt = cls_gt.to(device, non_blocking=True)
        seg_gt = seg_gt.squeeze(1).long().to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', enabled=scaler is not None):
            seg_logits, cls_logits, *mid_logits = model(imgs)
            total_loss, info = criterion(seg_logits, cls_logits, mid_logits, seg_gt, cls_gt)
            
            if hasattr(model, 'kvloss') and model.kvloss is not None:
                kvloss = model.kvloss
                total_loss = total_loss + kvloss
                info['kvloss'] = kvloss
            else:
                info['kvloss'] = torch.tensor(0.0, device=total_loss.device)

        if not torch.isfinite(total_loss):
            loader.write(f"[Warn] non-finite loss at step {step}, skip batch…")
            optimizer.zero_grad(set_to_none=True)
            continue

        (scaler.scale(total_loss / accumulation_steps) if scaler
         else (total_loss / accumulation_steps)).backward()

        if step % accumulation_steps == 0 or step == len(loader):
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            (scaler.step(optimizer) if scaler else optimizer.step())
            if scaler: scaler.update()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            bs = imgs.size(0)
            run_loss += total_loss.item() * bs
            run_correct += (cls_logits.argmax(1) == cls_gt).sum().item()
            seen += bs

            for k in ['total', 'cls', 'seg', 'distill', 'kvloss']:
                if k in info: 
                    hist[k].append(info[k].item())

            w_vec = info.get('weights')
            if w_vec is not None:
                hist['weights'].append(w_vec.detach().cpu().numpy())

            w_disp = ','.join(f"{x:.2f}" for x in w_vec.cpu().numpy()) if w_vec is not None else 'n/a'
            loader.set_postfix({
                'L': f"{run_loss/seen:.4f}",
                'Acc': f"{run_correct/seen:.3f}",
                'W': w_disp
            })

    epoch_loss = run_loss / max(seen, 1)
    epoch_acc  = run_correct / max(seen, 1)
    w_mean = np.mean(hist['weights'], axis=0) if hist['weights'] else np.zeros(3)  # 3个任务的权重
    
    print(f"\nEpoch {epoch:02d} summary")
    print(f"  Total Loss : {epoch_loss:.4f}")
    
    loss_names = {'cls': 'Cls', 'seg': 'Seg', 'distill': 'Distill', 'kvloss': 'KV-MoE'}
    for k, display_name in loss_names.items():
        if hist[k]:
            print(f"  {display_name:<8} Loss : {np.mean(hist[k]):.4f}")
    
    print(f"  Avg Weights: {np.round(w_mean, 3)}")
    print(f"  Acc        : {epoch_acc:.4f}\n")

    return epoch_loss, epoch_acc, criterion

best_auc = best_acc = 0.0

@torch.no_grad()
def evaluate(model,
             data_loader,
             device,
             epoch: int,
             num_classes: int = 2,
             criterion=None,
             writer: Optional[SummaryWriter] = None,
             train_acc: float = None):  
    """返回 loss_ep, youden_acc, auc_val, sens, spec, f1"""
    global best_auc, best_acc
    model.eval()

    ce = nn.CrossEntropyLoss()
    loss_tot = corr_tot = samp_tot = 0
    probs_all, labels_all = [], []

    dscs, iou, asds, hd95s = [], [], [], []

    loader = tqdm(data_loader, file=sys.stdout, ncols=110)
    loader.set_description(f"Evaluating Epoch {epoch}")

    for imgs, cls_gt, seg_gt in loader:
        imgs   = imgs.to(device, non_blocking=True)
        cls_gt = cls_gt.to(device, non_blocking=True)
        seg_gt = seg_gt.squeeze(1).long().to(device, non_blocking=True)

        with torch.autocast('cuda'):
            seg_logits, cls_logits, *_ = model(imgs)

        bs = imgs.size(0)
        loss_tot += ce(cls_logits, cls_gt).item() * bs
        samp_tot += bs
        corr_tot += (cls_logits.argmax(1) == cls_gt).sum().item()

        prob = F.softmax(cls_logits, 1)[:, 1].cpu().numpy()
        probs_all.extend(prob)
        labels_all.extend(cls_gt.cpu().numpy())

        # —— segmentation metrics ——
        pred_seg = torch.argmax(seg_logits, 1).cpu().numpy().astype(np.uint8)
        gt_seg   = seg_gt.cpu().numpy().astype(np.uint8)
        for p, g in zip(pred_seg, gt_seg):
            inter = np.logical_and(p, g); union = np.logical_or(p, g)
            dscs.append((2 * inter.sum()) / (p.sum() + g.sum() + 1e-7))
            iou.append(inter.sum() / (union.sum() + 1e-7))
            a, h = assd(p, g), hd95(p, g)
            if not np.isnan(a): asds.append(a)
            if not np.isnan(h): hd95s.append(h)

        loader.set_postfix({
            'loss': f"{loss_tot / samp_tot:.4f}",
            'acc':  f"{corr_tot / samp_tot:.3f}"
        })

    # —— classification metrics ——
    loss_ep = loss_tot / samp_tot
    acc_ep  = corr_tot / samp_tot
    auc_val = roc_auc_score(labels_all, probs_all)

    if hasattr(criterion, "update_auc"):
        criterion.update_auc(float(auc_val))

    fpr, tpr, thr = roc_curve(labels_all, probs_all)
    idx = (tpr - fpr).argmax(); cutoff = thr[idx]
    y_pred = (np.array(probs_all) >= cutoff).astype(int)

    cm = confusion_matrix(labels_all, y_pred, labels=list(range(num_classes)))
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = tp / (tp + fn + 1e-7); spec = tn / (tn + fp + 1e-7)
    youden_acc = (y_pred == np.array(labels_all)).mean()

    prec = precision_score(labels_all, y_pred, zero_division=0)
    rec  = recall_score(labels_all, y_pred, zero_division=0)
    f1   = f1_score(labels_all, y_pred, zero_division=0)

    # —— segmentation aggregate ——
    dsc_m  = np.mean(dscs) if dscs else 0
    iou_m  = np.mean(iou)  if iou else 0
    asd_m  = np.mean(asds) if asds else 0
    hd95_m = np.mean(hd95s) if hd95s else 0

    # —— console log ——
    print(f"\n=== Eval Epoch {epoch} ===")
    print(f"[Cls] Loss {loss_ep:.4f} | ACC {acc_ep:.4f} | AUC {auc_val:.4f}")
    if train_acc is not None:
        print(f"Train ACC: {train_acc:.4f}")
    if hasattr(criterion, "distillation_enabled"):
        print(f"Distillation : {'Enabled' if criterion.distillation_enabled else 'Disabled'} "
              f"(thr={criterion.distillation_auc_threshold})")
    print(f"Cutoff {cutoff:.3f} | ACC@Cut {youden_acc:.4f} "
          f"| Sens {sens:.4f} | Spec {spec:.4f}")
    print(f"Precision {prec:.4f} | Recall {rec:.4f} | F1 {f1:.4f}")
    print(f"[Seg] DSC {dsc_m:.4f} | IoU {iou_m:.4f} | ASD {asd_m:.4f} | HD95 {hd95_m:.4f}\n")
    print("Confusion Matrix:\n", cm)

    # —— TensorBoard (optional) ——
    if writer is not None:
        writer.add_scalar('Val/AUC',   auc_val,   epoch)
        writer.add_scalar('Val/ACC',   youden_acc, epoch)
        writer.add_scalar('Val/DSC',   dsc_m,     epoch)
        writer.add_scalar('Val/F1',    f1,        epoch)  
        if train_acc is not None:
            writer.add_scalar('Train/ACC', train_acc, epoch)  
        writer.add_figure(
            'Val/ConfMat',
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues').get_figure(),
            global_step=epoch
        )

    # —— save best —— #
    res_dir = "./results"; os.makedirs(res_dir, exist_ok=True)
    if auc_val > best_auc:
        best_auc = auc_val
        sns.heatmap(cm.astype(float) / cm.sum(1, keepdims=True),
                    annot=True, fmt=".3f", cmap='Blues')
        plt.title(f"Epoch {epoch}  ACC={youden_acc:.3f}")
        plt.savefig(os.path.join(res_dir, "best_confusion_matrix.png")); plt.close()

        plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc_val:.4f}")
        plt.scatter(fpr[idx], tpr[idx], c='r', label=f"cut={cutoff:.3f}")
        plt.legend(); plt.savefig(os.path.join(res_dir, "best_roc_curve.png")); plt.close()

        with open(os.path.join(res_dir, "best_metrics.txt"), "w") as f:
            f.write(f"Epoch {epoch}\nAUC {auc_val:.4f}\nACC {youden_acc:.4f}\n"
                    f"Sens {sens:.4f}\nSpec {spec:.4f}\nF1 {f1:.4f}\n"
                    f"DSC {dsc_m:.4f}\nIoU {iou_m:.4f}\nASD {asd_m:.4f}\nHD95 {hd95_m:.4f}\n")
            if train_acc is not None:
                f.write(f"Train ACC {train_acc:.4f}\n")
        print(f"[Save] new best AUC {auc_val:.4f}")

    return loss_ep, youden_acc, auc_val, sens, spec, dsc_m, iou_m, asd_m, hd95_m
