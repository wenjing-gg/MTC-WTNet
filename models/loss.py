import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from monai.losses import DiceCELoss
import math

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.2, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred, target):
        num_classes = pred.size(1)
        target = target.long()
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        target_smoothed = (1 - self.epsilon) * one_hot + self.epsilon / num_classes
        
        log_pred = F.log_softmax(pred, dim=1)
        loss = -torch.sum(target_smoothed * log_pred, dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AdaptiveUncertaintyFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, beta=1.0, epsilon_max=0.1, num_classes=None):
        """
        Args:
            gamma: Focusing parameter for Focal Loss
            beta: Rate at which uncertainty increases with confidence
            epsilon_max: Maximum label smoothing parameter
            num_classes: Number of classes in classification task
        """
        super(AdaptiveUncertaintyFocalLoss, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.epsilon_max = epsilon_max
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits, shape [B, C]
            target: Target class indices, shape [B]
        """
        # Convert targets to one-hot encoding
        if self.num_classes is None:
            self.num_classes = pred.size(1)
        
        # Get predicted probabilities
        pred_softmax = F.softmax(pred, dim=1)
        
        # Create one-hot encoding of targets
        target_one_hot = F.one_hot(target, self.num_classes).float()
        
        # Get prediction probability for the true class
        p_t = torch.sum(pred_softmax * target_one_hot, dim=1)
        
        # Calculate dynamic label smoothing parameter based on confidence
        epsilon_t = self.epsilon_max * (p_t ** self.beta)
        
        # Apply adaptive label smoothing
        smoothed_target = (1 - epsilon_t.view(-1, 1)) * target_one_hot + \
                          epsilon_t.view(-1, 1) / self.num_classes
        
        # Compute the focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute the loss with adaptive smoothing and focal weighting
        log_probs = F.log_softmax(pred, dim=1)
        loss = -torch.sum(smoothed_target * log_probs, dim=1)
        
        # Apply focal weighting
        loss = focal_weight * loss
        
        return loss.mean()

def progressive_supervised_distillation_loss(teacher_logits, student_logits_list, target,
                                           T=2.0, alpha=0.3, layer_weights=None,
                                           num_classes=None):
    """
    Progressive supervised distillation loss - all samples participate in supervised and distillation losses

    Args:
        teacher_logits: Main classifier output (B, num_classes)
        student_logits_list: List of shallow classifier outputs, sorted by depth
        target: True labels (B,)
        T: Distillation temperature
        alpha: Supervised loss weight (between 0-1, 1 means fully supervised, 0 means fully distilled)
        layer_weights: Weight for each layer, automatically calculated if None
        num_classes: Number of classes
    """
    device = teacher_logits.device

    if num_classes is None:
        num_classes = teacher_logits.size(1)

    focal_loss = AdaptiveUncertaintyFocalLoss(num_classes=num_classes)
    ce_loss = LabelSmoothingCrossEntropy()

    # Automatically calculate layer weights (deeper layers have larger weights), ensure on correct device
    if layer_weights is None:
        num_layers = len(student_logits_list)
        layer_weights = torch.linspace(0.5, 1.0, num_layers, device=device)
    elif not isinstance(layer_weights, torch.Tensor):
        layer_weights = torch.tensor(layer_weights, device=device)
    elif layer_weights.device != device:
        layer_weights = layer_weights.to(device)

    # 1. Supervised loss: all student layers calculate loss with true labels
    supervised_loss = 0.0
    total_weight = 0.0
    
    for i, s_logits in enumerate(student_logits_list):
        if s_logits is None:
            continue
            
        weight = layer_weights[i] if i < len(layer_weights) else 1.0
        supervised_loss += weight * focal_loss(s_logits, target)
        
        total_weight += weight
    
    if total_weight == 0:
        return teacher_logits.new_full((), 1e-6)
    
    supervised_loss = supervised_loss / total_weight
    
    # 2. Distillation loss: all samples participate in teacher->student knowledge distillation
    teacher_prob = F.softmax(teacher_logits / T, dim=1)

    distill_loss = 0.0
    total_weight = 0.0

    for i, s_logits in enumerate(student_logits_list):
        if s_logits is None:
            continue

        student_log_prob = F.log_softmax(s_logits / T, dim=1)
        distill_kld = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean')

        weight = layer_weights[i] if i < len(layer_weights) else 1.0
        distill_loss += weight * distill_kld
        total_weight += weight

    if total_weight == 0:
        distill_loss = teacher_logits.new_full((), 1e-6)
    else:
        distill_loss = (distill_loss / total_weight) * (T ** 2)

    # 3. Weighted combination: alpha controls the ratio of supervised vs distillation
    total_loss = alpha * supervised_loss + (1-alpha) * distill_loss
    
    return total_loss

class CEDiceLoss3D(nn.Module):
    """
    二分类也按“多类别”方式计算 Dice + CE：
        • logits  : [B, 2, D, H, W] (background + foreground)
        • targets : [B, 1, D, H, W] or [B, D, H, W] (integer labels)
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        assert num_classes == 2, "This implementation is fixed to num_classes=2 and processed as multi-class"

        # ——Always use softmax=True (multi-class logic)——
        self.dice_ce_loss = DiceCELoss(
            sigmoid=False,          # Turn off sigmoid
            softmax=True,           # Turn on softmax
            squared_pred=True,
            reduction="mean",
            to_onehot_y=True
        )

    def forward(self, seg_logits: torch.Tensor, seg_labels: torch.Tensor):
        # 1. Ensure label type
        assert seg_labels.dtype == torch.long, "seg_labels must be long type"

        # 2. 4D labels → 5D (add channel dimension)
        if seg_labels.dim() == 4:                # [B, D, H, W]
            seg_labels = seg_labels.unsqueeze(1) # [B, 1, D, H, W]

        # 3. Directly calculate multi-class Dice + CE
        return self.dice_ce_loss(seg_logits, seg_labels)


import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        seg_classes: int = 2,
        init_cls_var: float = 1,
        init_seg_var: float = 2,
        init_dst_var: float = 4,
    ):
        super().__init__()
        # Basic single-task losses
        self.cls_loss = AdaptiveUncertaintyFocalLoss(num_classes=num_classes)
        self.seg_loss = CEDiceLoss3D(num_classes=seg_classes)

        # log σ² learnable parameters (s_i = log var)
        self.log_var_cls  = nn.Parameter(torch.log(torch.tensor(init_cls_var)))
        self.log_var_seg  = nn.Parameter(torch.log(torch.tensor(init_seg_var)))
        self.log_var_dst  = nn.Parameter(torch.log(torch.tensor(init_dst_var)))

    # --- For monitoring only ---
    @property
    def cls_weight(self):     # w = 1/σ² = exp(-s)
        return torch.exp(-self.log_var_cls)

    @property
    def seg_weight(self):
        return torch.exp(-self.log_var_seg)

    @property
    def distill_weight(self):
        return torch.exp(-self.log_var_dst)

    def get_weights(self):
        return {
            "cls_weight":     self.cls_weight.item(),
            "seg_weight":     self.seg_weight.item(),
            "distill_weight": self.distill_weight.item(),
        }

    # ---------------------------------------------------------
    def forward(
        self,
        seg_logits: torch.Tensor,
        logits_final: torch.Tensor,
        mid_logits:  List[torch.Tensor],
        seg_gt: torch.Tensor,
        cls_gt: torch.Tensor,
    ):
        # Single-task native losses
        L_cls = self.cls_loss(logits_final, cls_gt)
        L_seg = self.seg_loss(seg_logits, seg_gt)
        L_dst = progressive_supervised_distillation_loss(
            teacher_logits=logits_final,
            student_logits_list=mid_logits,
            target=cls_gt,
        )

        # Uncertainty weighting (Formula 1)
        total = (
            0.5 * torch.exp(-self.log_var_cls) * L_cls + 0.5 * self.log_var_cls +
            0.5 * torch.exp(-self.log_var_seg) * L_seg + 0.5 * self.log_var_seg +
            0.5 * torch.exp(-self.log_var_dst) * L_dst + 0.5 * self.log_var_dst
        )

        info = {
            "total":   total.detach(),
            "cls":     L_cls.detach(),
            "seg":     L_seg.detach(),
            "distill": L_dst.detach(),
            "weights": torch.tensor(
                [self.cls_weight.item(),
                 self.seg_weight.item(),
                 self.distill_weight.item()],
                device=total.device,
            ),
        }
        return total, info