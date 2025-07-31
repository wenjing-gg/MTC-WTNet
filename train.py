#!/usr/bin/env python3
# train.py  ───────────────────────────────────────────────────────────────
import os, sys, time, math, argparse, torch
from datetime import datetime
from typing import List

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import MyNRRDDataSet
from models.mtc_hsdnet import MTC_HSDNet as create_model
from models.loss import MultiTaskLoss
from utils.utils import train_one_epoch, evaluate


# ════════════════════════════════════════════════════════════════
# 0) GPU polling
# ════════════════════════════════════════════════════════════════
def wait_for_available_gpu(threshold: float = .4):
    while True:
        for d in range(2,3):
            free, total = torch.cuda.mem_get_info(d)
            if free / total >= threshold:
                print(f"[Info] cuda:{d} free ({free/1e9:.1f}/{total/1e9:.1f} GB)")
                return d
        print("No GPU free, retry in 60 s…"); time.sleep(60)


# ════════════════════════════════════════════════════════════════
# 1) Warm-up + Cosine LR
# ════════════════════════════════════════════════════════════════
class WarmupCosineLR:
    def __init__(self, opt, initial_lr, max_lr, min_lr, epochs, warm=5):
        self.opt, self.ini, self.max, self.min = opt, initial_lr, max_lr, min_lr
        self.epochs, self.warm = epochs, warm

    def _lr(self, ep):
        if ep < self.warm:
            return self.ini + (self.max - self.ini) * (ep + 1) / self.warm
        cos_ep = self.epochs - self.warm
        return self.min + 0.5 * (self.max - self.min) * (
            1 + math.cos(math.pi * (ep - self.warm) / cos_ep))

    def step(self, ep):
        lr = self._lr(ep)
        for g in self.opt.param_groups:
            g["lr"] = lr


# ════════════════════════════════════════════════════════════════
# 2) Load checkpoint and print detailed differences
# ════════════════════════════════════════════════════════════════
def _strip_known(name: str, prefixes: List[str]):
    keep = True
    while keep:
        keep = False
        for p in prefixes:
            if name.startswith(p):
                name = name[len(p):]; keep = True
    return name

def load_ckpt_verbose(path: str,
                      model: torch.nn.Module,
                      device: torch.device,
                      strip_prefixes: List[str] = None,
                      log_dir: str = "./ckpt_diff_logs"):
    if strip_prefixes is None:
        strip_prefixes = ["module.", "backbone.", "encoder."]

    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    raw = torch.load(path, map_location=device)

    # Extract state_dict
    for key in ("state_dict", "model", "net", "backbone"):
        if key in raw:
            raw = raw[key]
            break

    sd = {_strip_known(k, strip_prefixes): v for k, v in raw.items()}
    info = model.load_state_dict(sd, strict=False)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    miss_file = os.path.join(log_dir, f"missing_{ts}.txt")
    unexp_file= os.path.join(log_dir, f"unexpected_{ts}.txt")
    with open(miss_file, "w") as f:
        for k in info.missing_keys: f.write(k + "\n")
    with open(unexp_file, "w") as f:
        for k in info.unexpected_keys: f.write(k + "\n")

    def _preview(lst, tag, limit=15):
        print(f" {tag:<11}: {len(lst)}")
        if lst:
            show = lst if len(lst) <= limit else lst[:limit] + ["..."]
            for k in show:
                print(f"    {'-' if tag=='missing' else '+'} {k}")

    print(f"[Weights] «{os.path.basename(path)}» loaded")
    _preview(info.missing_keys,   "missing")
    _preview(info.unexpected_keys,"unexpected")
    print(f"► full logs saved to {log_dir}")


# ════════════════════════════════════════════════════════════════
# 3) main
# ════════════════════════════════════════════════════════════════
def main(args):
    device = torch.device(f"cuda:{wait_for_available_gpu()}"
                          if torch.cuda.is_available() else "cpu")

    scaler  = GradScaler('cuda') if args.use_amp else None
    writer  = SummaryWriter()
    os.makedirs("weights", exist_ok=True)

    # ——— Data ———
    train_set = MyNRRDDataSet(args.data_path, split='train')
    val_set   = MyNRRDDataSet(args.data_path, split='test')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True,
                              collate_fn=train_set.collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True,
                              collate_fn=val_set.collate_fn)

    # ——— Model ———
    model = create_model().to(device)
    if args.weights:
        load_ckpt_verbose(args.weights, model, device)

    if args.freeze_layers:
        layers_to_freeze = ["swinViT"]
        frozen_params = 0
        total_params = 0
        
        for n, p in model.named_parameters():
            total_params += 1
            # Check if parameter name contains any layer names that need to be frozen
            should_freeze = any(layer_name in n for layer_name in layers_to_freeze)
            if should_freeze:
                p.requires_grad_(False)
                frozen_params += 1

        print(f"[Freeze] Frozen {frozen_params}/{total_params} parameters in layers: {layers_to_freeze}")

        # Print detailed information of frozen layers
        frozen_layer_names = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                frozen_layer_names.append(n)

        if frozen_layer_names:
            print(f"[Freeze] Frozen layers: {len(frozen_layer_names)} total")
            # Show first few frozen layers as examples
            for i, name in enumerate(frozen_layer_names[:5]):
                print(f"    - {name}")
            if len(frozen_layer_names) > 5:
                print(f"    ... and {len(frozen_layer_names) - 5} more layers")

    # ——— Loss ———
    criterion = MultiTaskLoss().to(device)

    # ——— Optimizer ———
    pg = [{"params": [p for p in model.parameters() if p.requires_grad],
           "lr": args.initial_lr, "weight_decay": 1e-2}]

    # Add all learnable parameters of loss function, using the same learning rate
    criterion_params = [p for p in criterion.parameters() if p.requires_grad]
    if criterion_params:
        pg.append({"params": criterion_params, "lr": args.initial_lr, "weight_decay": 0.})
        print(f"[Info] Added {len(criterion_params)} criterion params with unified LR={args.initial_lr}")

    optimizer = AdamW(pg, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)

    scheduler = WarmupCosineLR(optimizer, args.initial_lr,
                               args.max_lr, args.min_lr, args.epochs, warm=5)

    # ——— Training loop ———
    best_acc = best_auc = best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, _ = train_one_epoch(
            model, criterion, optimizer, train_loader,
            device, epoch, scaler, accumulation_steps=args.accum_steps)

        val_loss, val_acc, val_auc, val_sen, val_spe = evaluate(
            model, val_loader, device, epoch,
            args.num_classes, criterion, writer)

        # TensorBoard
        writer.add_scalar('Loss/train', tr_loss, epoch)
        writer.add_scalar('Loss/val',   val_loss, epoch)
        writer.add_scalar('Acc/train',  tr_acc, epoch)
        writer.add_scalar('Acc/val',    val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(epoch - 1)

        # Save best
        if val_auc > best_auc:
            best_acc, best_auc, best_epoch = val_acc, val_auc, epoch
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "criterion": criterion.state_dict(),
                "epoch": epoch,
                "best_auc": best_auc,
                "best_acc": best_acc
            }, 'weights/best_model.pth')
            print(f"[Save] best @epoch {epoch} | AUC={best_auc:.4f}")

        # Get weight information for display
        if hasattr(criterion, "get_weights"):
            # Use MultiTaskLoss_RT weight information
            weights_info = criterion.get_weights()
            weights_str = ", ".join([f"{k}:{v:.3f}" for k, v in weights_info.items() if 'weight' in k])
            phase_info = ""
            if hasattr(criterion, "get_training_phase"):
                phase_info = f" | Phase: {criterion.get_training_phase()}"
            print(f"Epoch {epoch:03d} | LR={optimizer.param_groups[0]['lr']:.2e} | weights=[{weights_str}]{phase_info}")
        elif hasattr(criterion, "weights"):
            # Compatible with old version weight display
            weights_array = (0.5 * torch.exp(-2 * criterion.log_sigma)).cpu().numpy()
            print(f"Epoch {epoch:03d} | LR={optimizer.param_groups[0]['lr']:.2e} | w={weights_array}")
        else:
            # Simple display when no weight information
            print(f"Epoch {epoch:03d} | LR={optimizer.param_groups[0]['lr']:.2e}")

    # ——— End ———
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "criterion": criterion.state_dict(),
        "epoch": args.epochs},
        'weights/final_model.pth')
    print(f"Finished. Best ACC={best_acc:.4f} (AUC={best_auc:.4f}) at epoch {best_epoch}")


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', default="/home/yuwenjing/data/transfer_VOI_test")
    ap.add_argument('--weights', default='/home/yuwenjing/DeepLearning_ywj/WTMNet/supervised_suprem_swinunetr_2100.pth',
                    help='Path to pre-trained weights')

    ap.add_argument('--num_classes', type=int, default=2)
    ap.add_argument('--epochs',      type=int, default=150)
    ap.add_argument('--batch_size',  type=int, default=2)
    ap.add_argument('--initial_lr',  type=float, default=1e-5)
    ap.add_argument('--max_lr',      type=float, default=1e-4)
    ap.add_argument('--min_lr',      type=float, default=1e-6)
    ap.add_argument('--accum_steps', type=int,   default=4)

    ap.add_argument('--freeze_layers', action='store_false', dest='freeze_layers',
                    default=True, help='Cancel this flag to **not** freeze backbone')
    ap.add_argument('--use_amp', action='store_true', default=False,
                    help='Mixed-precision training')

    args = ap.parse_args()
    main(args)
   