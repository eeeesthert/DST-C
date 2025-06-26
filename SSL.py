# # import torch
# # from torch import nn
# #
# #
# # class SimMIMPreTrainer(nn.Module):
# #     """基于SimMIM的自监督预训练（文献3.4节）"""
# #
# #     def __init__(self, base_model, mask_ratio=0.4, patch_size=4):
# #         super().__init__()
# #         self.base_model = base_model
# #         self.mask_ratio = mask_ratio
# #         self.patch_size = patch_size
# #
# #         # 解码器用于图像重建（文献图4）
# #         self.decoder = nn.Sequential(
# #             nn.Conv3d(base_model.feature_channels, 128, kernel_size=3, padding=1),
# #             nn.BatchNorm3d(128),
# #             nn.ReLU(inplace=True),
# #             nn.Conv3d(128, 64, kernel_size=3, padding=1),
# #             nn.BatchNorm3d(64),
# #             nn.ReLU(inplace=True),
# #             nn.Conv3d(64, 1, kernel_size=1)
# #         )
# #
# #     def _generate_3d_mask(self, shape):
# #         """生成3D随机掩码（文献3.4节掩码策略）"""
# #         B, C, Z, Y, X = shape
# #         num_patches = Z * Y * X
# #         num_mask = int(self.mask_ratio * num_patches)
# #
# #         mask = torch.ones((B, 1, Z, Y, X), device=shape[0].device)
# #         for b in range(B):
# #             # 随机选择掩码位置（文献图5最佳参数：patch_size=4, ratio=0.4）
# #             indices = torch.randperm(num_patches)[:num_mask]
# #             z = indices // (Y * X)
# #             yx = indices % (Y * X)
# #             y = yx // X
# #             x = yx % X
# #             mask[b, 0, z, y, x] = 0
# #         return mask
# #
# #     def forward(self, x):
# #         """自监督前向传播（文献式9）"""
# #         # 1. 生成并应用掩码
# #         mask = self._generate_3d_mask(x.shape)
# #         masked_x = x * mask
# #
# #         # 2. 双分支特征提取
# #         cnn_features = self.base_model.cnn_branch(masked_x)
# #         dst_features = self.base_model.dst_branch(masked_x)
# #
# #         # 3. 特征融合与重建
# #         fused_features = self.base_model.sca_bridge(cnn_features, dst_features)
# #         reconstructed = self.decoder(fused_features)
# #
# #         return reconstructed, mask, x
# #
# #
# # # 损失函数实现（文献3.6.1节）
# # class SSLLoss(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.l1_loss = nn.L1Loss()
# #
# #     def forward(self, reconstructed, mask, original):
# #         """特征级与像素级L1损失（文献式9）"""
# #         feat_loss = self.l1_loss(reconstructed, original)
# #         pixel_loss = self.l1_loss(reconstructed * mask, original * mask)
# #         return feat_loss + pixel_loss
#
# import torch
# from torch import nn
#
# class SimMIMPreTrainer(nn.Module):
#     """基于SimMIM的自监督预训练（文献3.4节）"""
#
#     def __init__(self, base_model, feature_channels, mask_ratio=0.4, patch_size=4):
#         super().__init__()
#         self.base_model = base_model
#         self.mask_ratio = mask_ratio
#         self.patch_size = patch_size
#
#         # 解码器用于图像重建（文献图4）
#         self.decoder = nn.Sequential(
#             nn.Conv3d(feature_channels, 128, kernel_size=3, padding=1),
#             nn.BatchNorm3d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(128, 64, kernel_size=3, padding=1),
#             nn.BatchNorm3d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(64, 1, kernel_size=1)
#         )
#
#     def _generate_3d_mask(self, shape, x):
#         """生成3D随机掩码（文献3.4节掩码策略）"""
#         B, C, Z, Y, X = shape
#         num_patches = Z * Y * X
#         num_mask = int(self.mask_ratio * num_patches)
#
#         mask = torch.ones((B, 1, Z, Y, X), device=x.device)
#         for b in range(B):
#             # 随机选择掩码位置（文献图5最佳参数：patch_size=4, ratio=0.4）
#             indices = torch.randperm(num_patches)[:num_mask]
#             z = indices // (Y * X)
#             yx = indices % (Y * X)
#             y = yx // X
#             x = yx % X
#             mask[b, 0, z, y, x] = 0
#         return mask
#
#     # def forward(self, x):
#     #     """自监督前向传播（文献式9）"""
#     #     # 1. 生成并应用掩码
#     #     mask = self._generate_3d_mask(x.shape, x)
#     #     masked_x = x * mask
#     #     # 在 SSL.py 的 forward 方法中添加调试代码
#     #     print(f"masked_x shape: {masked_x.shape}")
#     #     print(f"masked_x size: {masked_x.numel()}")
#     #
#     #     # 2. 双分支特征提取
#     #     cnn_features = self.base_model.cnn_branch(masked_x)
#     #     dst_features = self.base_model.dst_branch(masked_x)
#     #
#     #     # 3. 特征融合与重建
#     #     fused_features = self.base_model.sca_bridge(cnn_features, dst_features)
#     #     reconstructed = self.decoder(fused_features)
#     #
#     #     return reconstructed, mask, x
#     def forward(self, x):
#         # Masking
#         B, C, D, H, W = x.shape
#         mask = self.generate_mask(x)
#         masked_x = x * mask
#
#         # Get features from both branches
#         cnn_features = self.base_model.cnn_branch(masked_x)  # (B, 512, D', H', W')
#         dst_features = self.base_model.dst_branch(masked_x)  # (B, N, embed_dim)
#
#         # Process CNN features
#         cnn_feat_flat = cnn_features.view(B, self.cnn_feat_dim, -1).mean(dim=2)
#         cnn_feat_proj = self.cnn_proj(cnn_feat_flat)
#
#         # Process DST features (already in (B, N, C) format)
#         dst_feat_flat = dst_features.mean(dim=1)  # Global average pooling over patches
#         dst_feat_proj = self.dst_proj(dst_feat_flat)
#
#         # Combine features
#         combined_features = cnn_feat_proj + dst_feat_proj
#
#         # Reshape for decoder
#         feat_size = int(cnn_features.shape[2])
#         combined_features = combined_features.view(B, 512, 1, 1, 1)
#         combined_features = combined_features.expand(-1, -1, feat_size, feat_size, feat_size)
#
#         # Decode
#         reconstructed = self.decoder(combined_features)
#
#         return reconstructed, mask, x
#
# # 损失函数实现（文献3.6.1节）
# class SSLLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.l1_loss = nn.L1Loss()
#
#     def forward(self, reconstructed, mask, original):
#         """特征级与像素级L1损失（文献式9）"""
#         feat_loss = self.l1_loss(reconstructed, original)
#         pixel_loss = self.l1_loss(reconstructed * mask, original * mask)
#         return feat_loss + pixel_loss
import torch
from torch import nn
import numpy as np


class SimMIMPreTrainer(nn.Module):
    """基于SimMIM的自监督预训练（文献3.4节）"""

    def __init__(self, base_model, feature_channels=64, mask_ratio=0.4, patch_size=4):
        super().__init__()
        self.base_model = base_model
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.feature_channels = feature_channels

        # 解码器用于图像重建（文献图4）
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1)
        )

    def _generate_3d_mask(self, shape, device):
        """生成3D随机掩码（文献3.4节掩码策略）"""
        B, C, Z, Y, X = shape

        # 按照patch_size划分
        patch_z = Z // self.patch_size
        patch_y = Y // self.patch_size
        patch_x = X // self.patch_size

        num_patches = patch_z * patch_y * patch_x
        num_mask = int(self.mask_ratio * num_patches)

        mask = torch.ones((B, C, Z, Y, X), device=device)

        for b in range(B):
            # 随机选择要掩码的patch（文献图5最佳参数：patch_size=4, ratio=0.4）
            mask_indices = torch.randperm(num_patches, device=device)[:num_mask]

            for idx in mask_indices:
                # 将1D索引转换为3D坐标
                pz = idx // (patch_y * patch_x)
                py = (idx % (patch_y * patch_x)) // patch_x
                px = idx % patch_x

                # 计算实际像素坐标
                z_start = pz * self.patch_size
                y_start = py * self.patch_size
                x_start = px * self.patch_size

                z_end = min(z_start + self.patch_size, Z)
                y_end = min(y_start + self.patch_size, Y)
                x_end = min(x_start + self.patch_size, X)

                # 掩码整个patch
                mask[b, :, z_start:z_end, y_start:y_end, x_start:x_end] = 0

        return mask

    def forward(self, x):
        """自监督前向传播（文献式9）"""
        # 1. 生成并应用掩码
        mask = self._generate_3d_mask(x.shape, x.device)
        masked_x = x * mask

        # 2. 双分支特征提取
        cnn_features = self.base_model.cnn_branch(masked_x)
        dst_features = self.base_model.dst_branch(masked_x)

        # 3. 特征融合
        fused_features = self.base_model.sca_bridge(cnn_features, dst_features)

        # 4. 图像重建
        reconstructed = self.decoder(fused_features)

        # 确保输出尺寸与输入一致
        if reconstructed.shape != x.shape:
            reconstructed = nn.functional.interpolate(
                reconstructed, size=x.shape[2:],
                mode='trilinear', align_corners=False
            )

        return reconstructed, mask, x


class SSLLoss(nn.Module):
    """自监督学习损失（文献3.6.1节式9）"""

    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, reconstructed, mask, original):
        """
        计算双路径L1损失（文献式9）
        loss_SSL = l1(F_CNN, F_DST) + l1(I_R, I_INPUT)
        """
        # 像素级L1损失（重建图像与原图像）
        pixel_loss = self.l1_loss(reconstructed, original)

        # 掩码区域损失（重点关注被掩码的区域）
        mask_loss = self.l1_loss(reconstructed * (1 - mask), original * (1 - mask))

        return pixel_loss + mask_loss