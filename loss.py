import torch
from torch import nn


class MaskedImageModeling(nn.Module):
    """基于掩码图像建模的自监督学习"""

    def __init__(self, base_model, mask_ratio=0.4, patch_size=4):
        super().__init__()
        self.base_model = base_model
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        # 解码器用于图像恢复
        self.decoder = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=1)
        )

    def _generate_mask(self, shape):
        """生成3D掩码（文献图4）"""
        B, C, Z, Y, X = shape
        num_patches = Z * Y * X
        num_mask = int(self.mask_ratio * num_patches)

        mask = torch.ones((B, 1, Z, Y, X), device=shape[0].device)
        for b in range(B):
            indices = torch.randperm(num_patches)[:num_mask]
            z = indices // (Y * X)
            yx = indices % (Y * X)
            y = yx // X
            x = yx % X
            mask[b, 0, z, y, x] = 0
        return mask

    def forward(self, x):
        mask = self._generate_mask(x.shape)
        masked_x = x * mask

        cnn_feat = self.base_model.cnn_branch(masked_x)
        dst_feat = self.base_model.dst_branch(masked_x)
        fused_feat = self.base_model.sca_bridge(cnn_feat, dst_feat)

        reconstructed = self.decoder(fused_feat)
        return reconstructed, mask, x


# 损失函数定义
class SSLLoss(nn.Module):
    """自监督学习损失（文献式9）"""

    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, reconstructed, mask, original):
        # 特征级L1损失
        feat_loss = self.l1_loss(reconstructed, original)
        # 像素级L1损失（仅掩码区域）
        pixel_loss = self.l1_loss(reconstructed * mask, original * mask)
        return feat_loss + pixel_loss


class SegmentationLoss(nn.Module):
    """分割任务损失（文献3.6节）"""

    def __init__(self):
        super().__init__()
        self.dice_loss = self._dice_loss
        self.bce_loss = nn.BCEWithLogitsLoss()

    def _dice_loss(self, pred, target):
        """Dice损失计算"""
        smooth = 1.0
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1.0 - dice

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target.float())
        dice = self._dice_loss(pred, target)
        return 0.5 * bce + 0.5 * dice  # 文献式12