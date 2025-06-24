import torch
from torch import nn


class SimMIMPreTrainer(nn.Module):
    """基于SimMIM的自监督预训练（文献3.4节）"""

    def __init__(self, base_model, mask_ratio=0.4, patch_size=4):
        super().__init__()
        self.base_model = base_model
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        # 解码器用于图像重建（文献图4）
        self.decoder = nn.Sequential(
            nn.Conv3d(base_model.feature_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=1)
        )

    def _generate_3d_mask(self, shape):
        """生成3D随机掩码（文献3.4节掩码策略）"""
        B, C, Z, Y, X = shape
        num_patches = Z * Y * X
        num_mask = int(self.mask_ratio * num_patches)

        mask = torch.ones((B, 1, Z, Y, X), device=shape[0].device)
        for b in range(B):
            # 随机选择掩码位置（文献图5最佳参数：patch_size=4, ratio=0.4）
            indices = torch.randperm(num_patches)[:num_mask]
            z = indices // (Y * X)
            yx = indices % (Y * X)
            y = yx // X
            x = yx % X
            mask[b, 0, z, y, x] = 0
        return mask

    def forward(self, x):
        """自监督前向传播（文献式9）"""
        # 1. 生成并应用掩码
        mask = self._generate_3d_mask(x.shape)
        masked_x = x * mask

        # 2. 双分支特征提取
        cnn_features = self.base_model.cnn_branch(masked_x)
        dst_features = self.base_model.dst_branch(masked_x)

        # 3. 特征融合与重建
        fused_features = self.base_model.sca_bridge(cnn_features, dst_features)
        reconstructed = self.decoder(fused_features)

        return reconstructed, mask, x


# 损失函数实现（文献3.6.1节）
class SSLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, reconstructed, mask, original):
        """特征级与像素级L1损失（文献式9）"""
        feat_loss = self.l1_loss(reconstructed, original)
        pixel_loss = self.l1_loss(reconstructed * mask, original * mask)
        return feat_loss + pixel_loss