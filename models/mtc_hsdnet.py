from typing import Sequence, Tuple, Type, Union
from kan import KANLinear
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from kornia.filters import sobel
from einops import rearrange
import scipy.ndimage as ndi
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep, optional_import
from attn import SwinTransformer, DualPathInteractionBlock, CrossAttention_flash, SelfAttention_flash
rearrange, _ = optional_import("einops", name="rearrange")

class SCAM(nn.Module):
    def __init__(self, channels, reduction_ratio=4, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_att(x) * x
        x = self.spatial_att(x) * x
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        # 3D adaptive pooling layers (compatible with 3D medical images)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # Output shape [B, C, 1, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # Attention mechanism fully connected layers (with dimension reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),  # Dimension reduction
            nn.ReLU(inplace=True),                       # Non-linear activation
            nn.Linear(channels // reduction, channels)   # Restore original dimension
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Original input shape: [B, C, D, H, W] (typical 3D medical image dimensions)

        # Average pooling path
        avg_out = self.avg_pool(x)          # [B, C, 1, 1, 1]
        avg_out = avg_out.view(avg_out.size(0), -1)  # Flatten to [B, C]
        avg_out = self.fc(avg_out)          # Through fully connected layer [B, C]

        # Max pooling path
        max_out = self.max_pool(x)          # [B, C, 1, 1, 1]
        max_out = max_out.view(max_out.size(0), -1)  # Flatten to [B, C]
        max_out = self.fc(max_out)          # Through fully connected layer [B, C]

        # Feature fusion
        channel_att = self.sigmoid(avg_out + max_out)  # [B, C]

        # Dimension reconstruction for subsequent broadcast multiplication
        return channel_att.view(x.size(0), -1, 1, 1, 1)  # Restore to [B, C, 1, 1, 1]
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class KV_MoE_plus(nn.Module):
    """KAN‑based Vision Mixture‑of‑Experts (enhanced).

    Key changes w.r.t. the original version:
    1.   Two independent LayerNorms (router/head) to avoid parameter coupling.
    2.   Experts consume the *normalized* input for more stable training.
    3.   Added residual connection inside the MoE block.
    """

    def __init__(
        self,
        in_channels: int,                # C of fpn_feat
        seg_channels: int = 2,           # S of seg_logits
        num_classes: int = 2,
        num_experts: int = 8,            # size of shared expert pool
        expansion: float = 2 / 3,
        dropout_rate: float = 0.5,
        temperature: float = 1.0,
        balance_loss_coef: float = 0.01,
        grid: Tuple[int, int, int] = (4, 4, 4),   # gd, gh, gw
        grid_size: int = 5,
        spline_order: int = 3,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        self.Cf = in_channels + seg_channels         # fused channel dim
        self.temperature = temperature
        self.balance_loss_coef = balance_loss_coef
        self.top_k = max(1, top_k)
        self.capacity_factor = capacity_factor
        self.gd, self.gh, self.gw = grid

        # ---------- Expert pool ----------
        hidden = max(4, int(self.Cf * expansion))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    KANLinear(self.Cf, hidden, grid_size, spline_order),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    KANLinear(hidden, self.Cf, grid_size, spline_order),
                )
                for _ in range(num_experts)
            ]
        )

        # Router
        self.router = nn.Linear(self.Cf, num_experts)

        # GAP pooling
        self.gap = nn.AdaptiveAvgPool3d(1)

        # Normalization layers (decoupled)
        self.norm_router = nn.LayerNorm(self.Cf)
        self.norm_head = nn.LayerNorm(self.Cf)

        # Classification head
        self.classifier = KANLinear(
            self.Cf, num_classes, grid_size=grid_size, spline_order=spline_order
        )

    # ---------- Top‑k MoE ----------
    def _topk_moe(self, x: torch.Tensor):
        B, D = x.shape
        E = len(self.experts)
    
        # LayerNorm before routing
        x_norm = self.norm_router(x)
        scores = self.router(x_norm) / self.temperature
        top_val, top_idx = scores.topk(self.top_k, -1)
        top_w = F.softmax(top_val, dim=-1)
    
        capacity = int(self.capacity_factor * B * self.top_k / E) + 1
        out = torch.zeros_like(x_norm)
        load = torch.zeros(E, device=x.device)
        importance = scores.softmax(-1).mean(0)
    
        for e in range(E):
            mask = (top_idx == e).any(-1)
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            if not idx.numel():
                continue
            if idx.numel() > capacity:
                idx = idx[:capacity]

            w = (top_w * (top_idx == e).float()).sum(-1, keepdim=True)[idx]
            y = self.experts[e](x_norm[idx])
            out[idx] += w * y
            load[e] += idx.numel()

        load = load / load.sum().clamp_min(1)
        aux = (importance * load).sum() * E

        # Remove residual connection, let the internal residual structure of expert networks take effect
        return out, aux, scores.max(-1).values

    # ---------- Regular grid splitting ----------
    def _split_blocks(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = feat.shape
        gd, gh, gw = self.gd, self.gh, self.gw
        assert D % gd == 0 and H % gh == 0 and W % gw == 0
        dp, hp, wp = D // gd, H // gh, W // gw
        feat = (
            feat.view(B, C, gd, dp, gh, hp, gw, wp)
            .permute(0, 2, 4, 6, 1, 3, 5, 7)   # B gd gh gw C dp hp wp
            .contiguous()
            .view(-1, C, dp, hp, wp)            # (B*P),C,dp,hp,wp
        )
        return feat  # P = gd * gh * gw

    # ---------- Forward ----------
    def forward(
        self,
        fpn_feat: torch.Tensor,   # [B, C, D, H, W]
        seg_logits: torch.Tensor, # [B, S, D, H, W]
        *,
        return_aux_loss: bool = False,
    ):
        # 1. Channel fusion
        fused_feat = torch.cat([fpn_feat, seg_logits], dim=1)  # [B, Cf, D, H, W]

        # 2. Block splitting + GAP
        blk = self._split_blocks(fused_feat)                   # [B*P, Cf, ...]
        vec = self.gap(blk).flatten(1)                         # [B*P, Cf]

        # 3. Top‑k MoE
        out, aux, conf = self._topk_moe(vec)                   # [B*P, Cf]

        # 4. Block logits (norm‑head)
        logits_blk = self.classifier(self.norm_head(out))      # [B*P, num_cls]

        # 5. Confidence‑weighted mean
        B = fpn_feat.size(0)
        P = self.gd * self.gh * self.gw
        conf = conf.view(B, P)
        weight = F.softmax(conf, dim=-1).unsqueeze(-1)         # [B, P, 1]
        logits = (logits_blk.view(B, P, -1) * weight).sum(1)   # [B, num_cls]

        if return_aux_loss:
            return logits, aux * self.balance_loss_coef
        return logits

class student_FFN(nn.Module):
    """Student classifier network
    """
    def __init__(self, in_channels, num_classes, dropout_rate=0.4):
        """
        Args:
            in_channels: Input feature channels
            num_classes: Number of classes for classification task
            dropout_rate: Dropout probability
        """
        super(student_FFN, self).__init__()

        # Define classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2 // 3),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_channels * 2 // 3, num_classes)
        )

        # Attention layer
        self.eca = eca_layer()

        # Global pooling layer
        self.gap = AvgMaxPooling(in_channels)

    def forward(self, x):
        """Forward propagation
        Args:
            x: Input features [B, C, D, H, W]
        Returns:
            logits: Classification output [B, num_classes]
            feat: Enhanced features [B, C, D, H, W]
        """
        feat = self.eca(x)

        # Global pooling
        pooled = self.gap(feat).view(feat.size(0), -1)

        # Classification
        logits = self.classifier(pooled)

        return logits, feat

class WTFPN(nn.Module):
    """WTM Feature Pyramid Network with added normalization and activation"""
    def __init__(self, in_channels_list, fpn_channels=256):
        """
        Args:
            in_channels_list: List of input feature channels for each layer [C1, C2, C3, C4]
            fpn_channels: FPN output channels
            num_classes: Number of classes for classification task
        """
        super(WTFPN, self).__init__()

        # Lateral 1x1 convolution + BatchNorm + ReLU
        self.lateral_conv1 = nn.Conv3d(in_channels_list[0], fpn_channels, kernel_size=1)
        self.lateral_bn1   = nn.BatchNorm3d(fpn_channels)
        self.lateral_act1  = nn.ReLU(inplace=True)

        self.lateral_conv2 = nn.Conv3d(in_channels_list[1], fpn_channels, kernel_size=1)
        self.lateral_bn2   = nn.BatchNorm3d(fpn_channels)
        self.lateral_act2  = nn.ReLU(inplace=True)

        self.lateral_conv3 = nn.Conv3d(in_channels_list[2], fpn_channels, kernel_size=1)
        self.lateral_bn3   = nn.BatchNorm3d(fpn_channels)
        self.lateral_act3  = nn.ReLU(inplace=True)

        self.lateral_conv4 = nn.Conv3d(in_channels_list[3], fpn_channels, kernel_size=1)
        self.lateral_bn4   = nn.BatchNorm3d(fpn_channels)
        self.lateral_act4  = nn.ReLU(inplace=True)

        # SCAM modules
        self.SCAM_lat1    = SCAM(fpn_channels)
        self.SCAM_lat2    = SCAM(fpn_channels)
        self.SCAM_lat3    = SCAM(fpn_channels)
        self.SCAM_lat4    = SCAM(fpn_channels)

        # 3x3 convolution + BatchNorm + ReLU after fusion
        self.SCAM_fusion3 = SCAM(fpn_channels)
        self.conv3        = nn.Conv3d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.bn3          = nn.BatchNorm3d(fpn_channels)
        self.act3         = nn.ReLU(inplace=True)

        self.SCAM_fusion2 = SCAM(fpn_channels)
        self.conv2        = nn.Conv3d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.bn2          = nn.BatchNorm3d(fpn_channels)
        self.act2         = nn.ReLU(inplace=True)

        self.SCAM_fusion1 = SCAM(fpn_channels)
        self.conv1        = nn.Conv3d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.bn1          = nn.BatchNorm3d(fpn_channels)
        self.act1         = nn.ReLU(inplace=True)

        self.fpn_channels = fpn_channels
        
    def forward(self, feat1, feat2, feat3, feat4):
        """Forward propagation, returns final fused features"""
        # Lateral path
        lat4 = self.lateral_act4(self.lateral_bn4(self.lateral_conv4(feat4)))
        lat4 = self.SCAM_lat4(lat4)

        lat3 = self.lateral_act3(self.lateral_bn3(self.lateral_conv3(feat3)))
        lat3 = self.SCAM_lat3(lat3)

        lat2 = self.lateral_act2(self.lateral_bn2(self.lateral_conv2(feat2)))
        lat2 = self.SCAM_lat2(lat2)

        lat1 = self.lateral_act1(self.lateral_bn1(self.lateral_conv1(feat1)))
        lat1 = self.SCAM_lat1(lat1)

        # Top level
        fpn4 = lat4

        # Middle fusion
        fpn3 = lat3 + F.interpolate(fpn4, scale_factor=2, mode='trilinear', align_corners=False)
        fpn3 = fpn3 + F.avg_pool3d(lat2, kernel_size=2, stride=2)
        fpn3 = self.SCAM_fusion3(fpn3)
        fpn3 = self.act3(self.bn3(self.conv3(fpn3)))

        # Bottom fusion
        fpn2 = lat2 + F.interpolate(fpn3, scale_factor=2, mode='trilinear', align_corners=False)
        fpn2 = fpn2 + F.avg_pool3d(lat1, kernel_size=2, stride=2)
        fpn2 = self.SCAM_fusion2(fpn2)
        fpn2 = self.act2(self.bn2(self.conv2(fpn2)))

        # Final fusion
        fpn1 = lat1 + F.interpolate(fpn2, scale_factor=2, mode='trilinear', align_corners=False)
        fpn1 = self.SCAM_fusion1(fpn1)
        fpn1 = self.act1(self.bn1(self.conv1(fpn1)))

        return fpn1

class MTCNet(nn.Module):
    def __init__(
        self,
        img_size: Union[Sequence[int], int] = (64,64,64),
        in_channels: int = 1,
        num_classes: int = 2,
        out_channels: int = 2,  # Number of classes for segmentation task
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 48,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        norm_name: Union[Tuple, str] = "instance",
        use_depth_pos_embed: bool = True,
    ) -> None:
        super().__init__()

        self.kvloss = None

        # Position encoding
        self.use_depth_pos_embed = use_depth_pos_embed
        if self.use_depth_pos_embed:
            d = img_size[0] if isinstance(img_size, Sequence) else img_size
            self.depth_embed = nn.Parameter(torch.zeros(1, in_channels, d, 1, 1))
            nn.init.trunc_normal_(self.depth_embed, std=0.02)

        # GSA attention layer
        self.gsa_layer = gsa_layer()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )

        # Define encoder blocks (stage1 ~ stage4)
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        # Define decoder blocks (decoder1 ~ decoder5)
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        ) 

        # Segmentation output
        self.seg_out_conv = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels
        )

        self.fpn_channels = 256

        # Student classifiers
        self.classifier1 = student_FFN(2 * feature_size, num_classes)
        self.classifier2 = student_FFN(4 * feature_size, num_classes)
        self.classifier3 = student_FFN(8 * feature_size, num_classes)
        self.classifier4 = student_FFN(16 * feature_size, num_classes)

        self.teacher_moe = KV_MoE_plus(in_channels=self.fpn_channels)

        # WT-FPN Feature Pyramid Network
        self.fpn = WTFPN(
            in_channels_list=[2*feature_size, 4*feature_size, 8*feature_size, 16*feature_size],
            fpn_channels=self.fpn_channels
        )

    def forward(self, x_in):
        """
        Forward function.
        Returns: (seg_logits, logits_final, logits1, logits2, logits3, logits4)
        """

        # ============ 1) Add position encoding ============ #
        if self.use_depth_pos_embed:
            depth_in = x_in.shape[2]
            depth_embed = self.depth_embed[:, :, :depth_in]
            x_in = x_in + depth_embed

        # ============ 2) Through GSA attention layer ============ #
        x_in = self.gsa_layer(x_in)

        # ============ 3) Through Swin Transformer encoder =========== #
        hidden_states_out = self.swinViT(x_in, normalize=True)
        assert len(hidden_states_out) == 5, f"Expected 5 hidden states, got {len(hidden_states_out)}"

        # ============ 4) Through encoder blocks ============ #
        enc1 = self.encoder1(x_in)                          # [B, feature_size, D, H, W]
        enc2 = self.encoder2(hidden_states_out[0])          # [B, feature_size, D/2, H/2, W/2]
        enc3 = self.encoder3(hidden_states_out[1])          # [B, 2*feature_size, D/4, H/4, W/4]
        enc4 = self.encoder4(hidden_states_out[2])          # [B, 4*feature_size, D/8, H/8, W/8]
        enc5 = self.encoder5(hidden_states_out[3])          # [B, 8*feature_size, D/16, H/16, W/16]
        enc10 = self.encoder10(hidden_states_out[4])        # [B, 16*feature_size, D/32, H/32, W/32]

        # ============ 5) Through decoder blocks (segmentation task) ============ #
        dec5 = self.decoder5(enc10, enc5)                   # [B, 8*feature_size, D/16*2, H/16*2, W/16*2]
        dec4 = self.decoder4(dec5, enc4)                    # [B, 4*feature_size, D/8*2, H/8*2, W/8*2]
        dec3 = self.decoder3(dec4, enc3)                    # [B, 2*feature_size, D/4*2, H/4*2, W/4*2]
        dec2 = self.decoder2(dec3, enc2)                    # [B, feature_size, D/2*2, H/2*2, W/2*2]
        dec1 = self.decoder1(dec2, enc1)                    # [B, feature_size, D, H, W]

        # Segmentation output
        seg_logits = self.seg_out_conv(dec1)                # [B, out_channels, D, H, W]

        # ============ 6) Classification task (using student_FFN) ============ #
        logits1, feat1 = self.classifier1(enc3)             # [B, num_classes], [B, 2*feature_size, D/4, H/4, W/4]
        logits2, feat2 = self.classifier2(enc4)             # [B, num_classes], [B, 4*feature_size, D/8, H/8, W/8]
        logits3, feat3 = self.classifier3(enc5)             # [B, num_classes], [B, 8*feature_size, D/16, H/16, W/16]
        logits4, feat4 = self.classifier4(enc10)            # [B, num_classes], [B, 16*feature_size, D/32, H/32, W/32]

        # ============ 7) Feature Pyramid Network ============ #
        fpn_feat = self.fpn(feat1, feat2, feat3, feat4)

        # ============ 8) Replace SegGuidedInteraction ============ #
        # Adjust segmentation logits to FPN feature size
        if seg_logits.shape[2:] != fpn_feat.shape[2:]:
            seg_for_guide = F.interpolate(
                seg_logits,
                size=fpn_feat.shape[2:],
                mode='trilinear',
                align_corners=False
            )
        else:
            seg_for_guide = seg_logits

        # # Directly input adjusted segmentation features and fpn features to Teacher_MoE
        # logits_final = self.teacher_moe(fpn_feat, seg_for_guide)

        # return seg_logits, logits_final, logits1, logits2, logits3, logits4
        # Modified this part: get KV_MoE loss value
        if self.training:
            # Return auxiliary loss during training
            logits_final, aux_loss = self.teacher_moe(fpn_feat, seg_for_guide, return_aux_loss=True)
            self.kvloss = aux_loss
        else:
            # Don't return auxiliary loss during inference
            logits_final = self.teacher_moe(fpn_feat, seg_for_guide, return_aux_loss=False)
            self.kvloss = None

        return seg_logits, logits_final, logits1, logits2, logits3, logits4

class AvgMaxPooling(nn.Module):
    def __init__(self, in_channels):
        """
        Module combining average pooling and max pooling.
        Args:
            in_channels (int): Number of input channels.
        """
        super(AvgMaxPooling, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        # For fusing average pooling and max pooling results
        self.fusion = nn.Conv3d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        avg_out = self.avg_pool(x)  # Average pooling result
        max_out = self.max_pool(x)  # Max pooling result
        combined = torch.cat([avg_out, max_out], dim=1)  # Concatenate by channel
        out = self.fusion(combined)  # Use 1x1 convolution for fusion
        return out

class gsa_layer(nn.Module):
    """GSA module for 5D tensors, treating depth dimension as virtual channels"""
    def __init__(self, k_size=3):
        super(gsa_layer, self).__init__()
        self.hybrid_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Preserve depth dimension
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Assume input shape: [batch_size, channels, depth, height, width]
        b, c, d, h, w = x.size()

        # Merge batch and channel dimensions, use depth as channel
        x_flattened = x.view(b * c, d, h, w)  # [batch_size * channels, depth, height, width]

        # Apply GSA to depth dimension
        y = self.hybrid_pool(x_flattened)  # [batch_size * channels, depth, 1, 1]
        y = y.squeeze(-1).squeeze(-1).unsqueeze(1)  # [batch_size * channels, 1, depth]
        y = self.conv(y)  # 1D convolution, maintain depth slice attention
        y = self.sigmoid(y)  # [batch_size * channels, 1, depth]

        # Restore to 5D tensor shape
        y = y.view(b, c, d, 1, 1)  # [batch_size, channels, depth, 1, 1]

        # Channel-wise attention weighting
        out = x * y.expand_as(x)  # [batch_size, channels, depth, height, width]

        return out
    
class eca_layer(nn.Module):
    """Constructs a 3D ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        # Use 3D average pooling
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # Use 1D convolution to learn inter-channel relationships
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        # Use Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Extract global spatial information from input x
        y = self.avg_pool(x)  # Output shape: [batch_size, channels, 1, 1, 1]

        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2))  # First squeeze last two dimensions, then transpose
        y = y.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)  # Transpose back and restore shape

        y = self.sigmoid(y)

        # Return weighted input
        return x * y.expand_as(x)

