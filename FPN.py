import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

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

class WTFPN(nn.Module):
    """Bidirectional fusion FPN structure using SCAM in lateral connections"""

    def __init__(
        self,
        feature_size: int = 384,
        in_channels: list = [96, 192, 384, 768],
        use_checkpoint: bool = True  # Add gradient checkpointing
    ):
        super().__init__()

        self.feature_size = feature_size
        self.use_checkpoint = use_checkpoint
        # Reduce intermediate channels to lower memory usage
        self.fpn_channels = feature_size
        
        self.lateral_conv1 = nn.Conv3d(in_channels[0], self.fpn_channels, kernel_size=1, bias=False)  # 96 -> 256
        self.lateral_conv2 = nn.Conv3d(in_channels[1], self.fpn_channels, kernel_size=1, bias=False)  # 192 -> 256
        self.lateral_conv3 = nn.Conv3d(in_channels[2], self.fpn_channels, kernel_size=1, bias=False)  # 384 -> 256
        self.lateral_conv4 = nn.Conv3d(in_channels[3], self.fpn_channels, kernel_size=1, bias=False)  # 768 -> 256
        
        # SCAM and convolution after fusion
        self.SCAM_fusion3 = SCAM(self.fpn_channels)
        self.SCAM_fusion2 = SCAM(self.fpn_channels)
        self.SCAM_fusion1 = SCAM(self.fpn_channels)

        # Convolution after fusion - use efficient convolution to reduce computation
        self.conv3 = self._make_efficient_conv(self.fpn_channels)
        self.conv2 = self._make_efficient_conv(self.fpn_channels)
        self.conv1 = self._make_efficient_conv(self.fpn_channels)

        # Final output convolution - directly output to target dimension
        self.final_conv1 = nn.Conv3d(self.fpn_channels, feature_size, kernel_size=1, bias=False)
        self.final_conv2 = nn.Conv3d(self.fpn_channels, feature_size, kernel_size=1, bias=False)
        self.final_conv3 = nn.Conv3d(self.fpn_channels, feature_size, kernel_size=1, bias=False)
        self.final_conv4 = nn.Conv3d(self.fpn_channels, feature_size, kernel_size=1, bias=False)

        self._initialize_weights()

    def _make_efficient_conv(self, channels):
        """Create efficient convolution block"""
        return nn.Sequential(
            # Depthwise separable convolution: first 3x3 depthwise convolution, then 1x1 pointwise convolution
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),  # Depthwise convolution
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),  # Pointwise convolution
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def _safe_interpolate(self, x, target_size, mode='trilinear'):
        """Safe interpolation operation, handling size mismatch issues"""
        if x.shape[2:] == target_size:
            return x
        return F.interpolate(x, size=target_size, mode=mode, align_corners=False)

    def _safe_pool(self, x, target_size):
        """Safe pooling operation, automatically calculate pooling parameters"""
        current_size = x.shape[2:]
        if current_size == target_size:
            return x

        # Calculate pooling kernel_size and stride
        kernel_sizes = []
        strides = []
        for i in range(3):  # D, H, W
            if current_size[i] > target_size[i]:
                ratio = current_size[i] // target_size[i]
                kernel_sizes.append(ratio)
                strides.append(ratio)
            else:
                kernel_sizes.append(1)
                strides.append(1)

        return F.avg_pool3d(x, kernel_size=kernel_sizes, stride=strides)
    
    def forward(self, feat1, feat2, feat3, feat4):
        """
        Bidirectional fusion forward propagation
        Args:
            feat1: [B, 96, D/4, H/4, W/4]   - enc3
            feat2: [B, 192, D/8, H/8, W/8]  - enc4
            feat3: [B, 384, D/16, H/16, W/16] - enc5
            feat4: [B, 768, D/32, H/32, W/32] - enc10
        """
        # ============ Modified FPN feature fusion ============ #
        # Lateral convolution + SCAM
        lat4 = self.lateral_conv4(feat4)  # [B, 256, D/32, H/32, W/32]

        lat3 = self.lateral_conv3(feat3)  # [B, 256, D/16, H/16, W/16]

        lat2 = self.lateral_conv2(feat2)  # [B, 256, D/8, H/8, W/8]

        lat1 = self.lateral_conv1(feat1)  # [B, 256, D/4, H/4, W/4]

        # Bidirectional fusion structure (similar to BiFPN)
        # Top-level path
        fpn4 = lat4

        # Middle layer fusion (upsampling + downsampling)
        # fpn3 = lat3 + upsampled fpn4 + downsampled lat2
        upsampled_fpn4 = self._safe_interpolate(fpn4, lat3.shape[2:])
        downsampled_lat2 = self._safe_pool(lat2, lat3.shape[2:])
        fpn3 = lat3 + upsampled_fpn4 + downsampled_lat2
        fpn3 = self.SCAM_fusion3(fpn3)
        fpn3 = self.conv3(fpn3)

        # Bottom layer path fusion
        # fpn2 = lat2 + upsampled fpn3 + downsampled lat1
        upsampled_fpn3 = self._safe_interpolate(fpn3, lat2.shape[2:])
        downsampled_lat1 = self._safe_pool(lat1, lat2.shape[2:])
        fpn2 = lat2 + upsampled_fpn3 + downsampled_lat1
        fpn2 = self.SCAM_fusion2(fpn2)
        fpn2 = self.conv2(fpn2)

        # Final feature fusion
        # fpn1 = lat1 + upsampled fpn2
        upsampled_fpn2 = self._safe_interpolate(fpn2, lat1.shape[2:])
        fpn1 = lat1 + upsampled_fpn2
        fpn1 = self.SCAM_fusion1(fpn1)
        fpn1 = self.conv1(fpn1)

        # Final output adjusted to target dimension
        output_fpn1 = self.final_conv1(fpn1)  # [B, 256, D/4, H/4, W/4]
        output_fpn2 = self.final_conv2(fpn2)  # [B, 256, D/8, H/8, W/8]
        output_fpn3 = self.final_conv3(fpn3)  # [B, 256, D/16, H/16, W/16]
        output_fpn4 = self.final_conv4(fpn4)  # [B, 256, D/32, H/32, W/32]

        # Clean up intermediate variables to free memory
        del lat1, lat2, lat3, lat4
        del fpn1, fpn2, fpn3, fpn4
        del upsampled_fpn4, downsampled_lat2, upsampled_fpn3, downsampled_lat1, upsampled_fpn2

        return {
            'fpn1': output_fpn1,
            'fpn2': output_fpn2,
            'fpn3': output_fpn3,
            'fpn4': output_fpn4
        }

# Remove unnecessary classes to keep code clean