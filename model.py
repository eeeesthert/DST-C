# import torch
# from torch import nn
# import torch.nn.functional as F
#
#
# class ResNet3DBranch(nn.Module):
#     """3D ResNet分支（文献3.1节CNN分支）"""
#
#     def __init__(self, in_channels=1, feature_channels=64):
#         super().__init__()
#         # 初始卷积块
#         self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm3d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
#
#         # 残差块（对应文献中的ResNet结构）
#         self.layer1 = self._make_layer(64, 64, blocks=3)
#         self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
#         self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
#         self.layer4 = self._make_layer(256, feature_channels, blocks=3, stride=2)
#
#     def _make_layer(self, in_planes, out_planes, blocks, stride=1):
#         """构建残差层（文献3.1节CNN分支）"""
#         downsample = None
#         if stride != 1 or in_planes != out_planes:
#             downsample = nn.Sequential(
#                 nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride),
#                 nn.BatchNorm3d(out_planes),
#             )
#
#         layers = []
#         layers.append(ResNetBlock(in_planes, out_planes, stride, downsample))
#         for i in range(1, blocks):
#             layers.append(ResNetBlock(out_planes, out_planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         """前向传播（文献3.1节CNN分支流程）"""
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         return x
#
#
# class ResNetBlock(nn.Module):
#     """ResNet基本残差块"""
#
#     def __init__(self, in_planes, out_planes, stride=1, downsample=None):
#         super().__init__()
#         self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)
#         self.bn1 = nn.BatchNorm3d(out_planes)
#         self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm3d(out_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class DSTBranch(nn.Module):
#     def __init__(self, in_channels, embed_dim=768, num_heads=12, num_layers=6, patch_size=16):
#         super().__init__()
#         self.patch_size = patch_size
#         self.embed_dim = embed_dim
#
#         # 3D patch embedding
#         self.patch_embed = nn.Conv3d(
#             in_channels, embed_dim,
#             kernel_size=patch_size,
#             stride=patch_size
#         )
#
#         # Position embedding
#         self.pos_embed = nn.Parameter(torch.zeros(1, 1000, embed_dim))  # Max 1000 patches
#
#         # DST blocks
#         self.blocks = nn.ModuleList([
#            DilatedSamplingTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
#         ])
#
#         self.norm = nn.LayerNorm(embed_dim)
#
#     def forward(self, x):
#         print(f"DSTBranch input shape: {x.shape}")
#         B = x.shape[0]
#
#         # Extract patches
#         x = self.patch_embed(x)
#         print(f"After patch_embed: {x.shape}")
#
#         # Flatten spatial dimensions and transpose
#         x = x.flatten(2).transpose(1, 2)
#         print(f"After flatten and transpose: {x.shape}")
#         # x: (B, C, D, H, W)
#         B = x.shape[0]
#
#         # Extract patches
#         x = self.patch_embed(x)  # (B, embed_dim, D', H', W')
#
#         # Flatten spatial dimensions and transpose
#         # 将5D张量转换为3D张量 (B, N, C)
#         x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
#
#         # Add position embedding
#         N = x.shape[1]
#         x = x + self.pos_embed[:, :N, :]
#
#         # Apply DST blocks
#         for block in self.blocks:
#             x = block(x)
#
#         x = self.norm(x)
#
#         return x
#
# # class DSTBranch(nn.Module):
# #     """扩张采样Transformer分支（文献3.2节DST分支）"""
# #
# #     def __init__(self, in_channels=1, feature_channels=64):
# #         super().__init__()
# #         # 补丁嵌入层（文献3.2节补丁划分）
# #         self.patch_embed = nn.Conv3d(
# #             in_channels, feature_channels,
# #             kernel_size=4, stride=4
# #         )
# #
# #         # 位置编码
# #         # self.pos_embed = nn.Parameter(
# #         #     torch.zeros(1, feature_channels, 32, 32, 32)  # 假设输入尺寸为128x128x128
# #         # )
# #         self.pos_embed = None  # 删除硬编码
# #
# #         # DST块（文献图2扩张采样Transformer）
# #         self.dst_blocks = nn.ModuleList([
# #             DilatedSamplingTransformerBlock(feature_channels, window_size=7, dilation=2),
# #             DilatedSamplingTransformerBlock(feature_channels, window_size=7, dilation=4),
# #             DilatedSamplingTransformerBlock(feature_channels, window_size=7, dilation=6),
# #         ])
# #
# #         # 上采样恢复分辨率（文献3.2节输出处理）
# #         self.upsample = nn.Upsample(
# #             scale_factor=4, mode="trilinear", align_corners=True
# #         )
#
#     # def forward(self, x):
#     #     """前向传播（文献3.2节DST分支流程）"""
#     #     # 补丁嵌入
#     #     x = self.patch_embed(x)
#     #     B, C, Z, Y, X = x.shape
#     #
#     #     # 展平为序列 + 位置编码
#     #     x = x.flatten(2).transpose(1, 2)  # [B, N, C]
#     #     x = x + self.pos_embed.flatten(2).transpose(1, 2)
#     #
#     #     # 通过DST块（文献图2）
#     #     for block in self.dst_blocks:
#     #         x = block(x)
#     #
#     #     # 恢复为3D特征图
#     #     x = x.transpose(1, 2).reshape(B, C, Z, Y, X)
#     #     x = self.upsample(x)
#     #
#     #     return x
#     # def forward(self, x):
#         # print(f"DST input shape: {x.shape}")
#         # # 在每个层后添加形状打印
#         # for i, layer in enumerate(self.layers):
#         #     x = layer(x)
#         #     print(f"After layer {i}: {x.shape}")
#         # return x
#         # # 对输入补零，使其是4的倍数
#         # pad_z = (4 - x.shape[2] % 4) % 4
#         # pad_y = (4 - x.shape[3] % 4) % 4
#         # pad_x = (4 - x.shape[4] % 4) % 4
#         # x = F.pad(x, (0, pad_x, 0, pad_y, 0, pad_z))  # 最后三个维度是 (X, Y, Z)
#         #
#         # # 补丁嵌入
#         # x = self.patch_embed(x)
#         # B, C, Z, Y, X = x.shape
#         #
#         # # 位置编码
#         # if self.pos_embed is None or self.pos_embed.shape != (1, C, Z, Y, X):
#         #     self.pos_embed = nn.Parameter(
#         #         torch.zeros(1, C, Z, Y, X, device=x.device),
#         #         requires_grad=True
#         #     )
#         #
#         # x = x + self.pos_embed
#         #
#         # # 展平为序列
#         # x = x.flatten(2).transpose(1, 2)  # [B, N, C]
#         #
#         # # DST
#         # for block in self.dst_blocks:
#         #     x = block(x)
#         # expected_size = B * C * Z * Y * X
#         # if x.numel() != expected_size:
#         #     print(f"Warning: Size mismatch. Expected {expected_size}, got {x.numel()}")
#         #     # 根据实际情况调整reshape参数
#         # # 在 model.py 第159行之前添加
#         # print(f"Before reshape - x shape: {x.shape}")
#         # print(f"Before reshape - x size: {x.numel()}")
#         # print(f"Target shape: [B={B}, C={C}, Z={Z}, Y={Y}, X={X}]")
#         # # 恢复3D形状
#         # # 获取实际的特征维度
#         # actual_features = x.size(-1)  # 64
#         # actual_tokens = x.size(1)  # 12
#         #
#         # # 计算合适的3D形状
#         # # 如果特征不足，需要调整网络结构或者改变reshape策略
#         # if x.numel() != B * C * Z * Y * X:
#         #     # 方案1：调整目标形状以匹配实际特征数量
#         #     # 例如：reshape为更小的3D体积
#         #     new_size = int(round((actual_features * actual_tokens / C) ** (1 / 3)))
#         #     print(f"Adjusting reshape to: [{B}, {C}, {new_size}, {new_size}, {new_size}]")
#         #     x = x.transpose(1, 2).reshape(B, C, new_size, new_size, new_size)
#         # else:
#         #     x = x.transpose(1, 2).reshape(B, C, Z, Y, X)
#         # # x = x.transpose(1, 2).reshape(B, C, Z, Y, X)
#         #
#         # # 上采样
#         # x = self.upsample(x)
#         # return x
#     # def forward(self, x):
#         #     print(f"DST input shape: {x.shape}")
#         #
#         #     children = list(self.children())
#         #
#         #     for i, child in enumerate(children):
#         #         if isinstance(child, nn.ModuleList):
#         #             # 遍历ModuleList中的每个模块
#         #             for module in child:
#         #                 x = module(x)
#         #                 print(f"After ModuleList module: {x.shape}")
#         #         else:
#         #             x = child(x)
#         #             print(f"After child {i}: {x.shape}")
#         #
#         #     return x
#
# class SpatialChannelAttention(nn.Module):
#     """空间通道注意力交互桥接（文献3.3节SCA模块）"""
#
#     def __init__(self, cnn_channels, dst_channels, lambda1=0.6):
#         super().__init__()
#         self.lambda1 = lambda1  # 文献表2最佳参数λ1=0.6
#         self.lambda2 = 1.0 - lambda1
#
#         # 空间注意力模块（文献3.3节空间注意力）
#         self.spatial_attn = nn.Sequential(
#             nn.Conv3d(2, 1, kernel_size=7, padding=3),
#             nn.Sigmoid()
#         )
#
#         # 通道注意力模块（文献3.3节通道注意力）
#         self.channel_attn = nn.Sequential(
#             nn.AdaptiveAvgPool3d(1),
#             nn.Conv3d(cnn_channels + dst_channels, (cnn_channels + dst_channels) // 16, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv3d((cnn_channels + dst_channels) // 16, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, cnn_feat, dst_feat):
#         """特征融合（文献式7-8）"""
#         # 特征拼接
#         combined = torch.cat([cnn_feat, dst_feat], dim=1)
#
#         # 计算空间注意力图（文献3.3节空间注意力计算）
#         spatial_map = self.spatial_attn(torch.cat([
#             torch.max(combined, dim=1, keepdim=True)[0],
#             torch.mean(combined, dim=1, keepdim=True)
#         ], dim=1))
#
#         # 计算通道注意力图（文献3.3节通道注意力计算）
#         channel_map = self.channel_attn(combined)
#
#         # 加权融合（文献式7）
#         cnn_weighted = cnn_feat * spatial_map
#         dst_weighted = dst_feat * channel_map
#         fused_feat = self.lambda1 * cnn_weighted + self.lambda2 * dst_weighted
#
#         return fused_feat
#
# class DSTC3D(nn.Module):
#     """DST-C双分支3D分割网络（文献图1完整实现）"""
#
#     def __init__(self, in_channels=1, out_channels=1, feature_channels=64, lambda1=0.6):
#         super().__init__()
#
#         # 1. CNN分支（对应文献3.1节ResNet结构）
#         self.cnn_branch = ResNet3DBranch(in_channels, feature_channels)
#
#         # 2. DST分支（对应文献3.2节扩张采样Transformer）
#         self.dst_branch = DSTBranch(in_channels, feature_channels)
#
#         # 3. SCA交互桥接（对应文献3.3节空间通道注意力）
#         self.sca_bridge = SpatialChannelAttention(feature_channels, feature_channels, lambda1)
#
#         # 4. 解码器（对应文献图1绿色解码路径）
#         self.decoder = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
#             nn.Conv3d(feature_channels, feature_channels // 2, kernel_size=3, padding=1),
#             nn.BatchNorm3d(feature_channels // 2),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
#             nn.Conv3d(feature_channels // 2, feature_channels // 4, kernel_size=3, padding=1),
#             nn.BatchNorm3d(feature_channels // 4),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(feature_channels // 4, out_channels, kernel_size=1)
#         )
#
#     def forward(self, x):
#         """前向传播流程（对应文献3.1节网络架构）"""
#         # 分支特征提取
#         cnn_features = self.cnn_branch(x)  # 局部细节特征
#         dst_features = self.dst_branch(x)  # 全局上下文特征
#
#         # SCA桥接融合（文献式7-8）
#         fused_features = self.sca_bridge(cnn_features, dst_features)
#
#         # 解码与分割预测
#         segmentation = self.decoder(fused_features)
#         return segmentation
#
#
# class DilatedSamplingTransformerBlock(nn.Module):
#     """扩张采样自注意力模块（文献图2核心组件）"""
#
#     def __init__(self, dim, window_size=7, dilation=2):
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size
#         self.dilation = dilation
#
#         # 多头自注意力（文献式4-5）
#         self.qkv = nn.Linear(dim, dim * 3)
#         self.proj = nn.Linear(dim, dim)
#
#     def _generate_dilation_mask(self, seq_len):
#         side_len = round(seq_len ** (1 / 3))
#         if side_len ** 3 != seq_len:
#             return torch.arange(0, seq_len, self.dilation)
#
#         coords = torch.meshgrid(
#             torch.arange(side_len),
#             torch.arange(side_len),
#             torch.arange(side_len),
#             indexing='ij'
#         )
#         coords = torch.stack(coords, dim=-1).float()
#         center = side_len // 2
#         distances = torch.sqrt(torch.sum((coords - center) ** 2, dim=-1))
#         mask = (distances % self.dilation == 0) | (distances == 0)
#         return mask.flatten().nonzero(as_tuple=True)[0]
#         print(f"attn.shape = {attn.shape}, N = {N}, dilation_mask.shape = {dilation_mask.shape}")
#
#
#     def forward(self, x):
#
#         """扩张采样自注意力计算（文献式1-3）"""
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # [B, N, C]
#
#         # 计算注意力分数
#         attn = (q @ k.transpose(-2, -1)) / (C ** 0.5)
#
#         # 生成扩张采样掩码
#         dilation_mask = self._generate_dilation_mask(N)
#
#         # 应用扩张采样掩码（文献图2采样机制）
#         attn = attn[:, dilation_mask, :][:, :, dilation_mask]
#
#         # 软最大化与加权求和
#         attn = F.softmax(attn, dim=-1)
#         x = (attn @ v[:, dilation_mask, :])
#
#         # 投影与残差连接
#         x_proj = self.proj(x)
#         x = x_proj + x
#         return x


import torch
from torch import nn
import torch.nn.functional as F
import math


class ResNet3DBranch(nn.Module):
    """3D ResNet分支（文献3.1节CNN分支）"""

    def __init__(self, in_channels=1, feature_channels=64):
        super().__init__()
        # 初始卷积块
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # 残差块（对应文献中的ResNet结构）
        self.layer1 = self._make_layer(64, 64, blocks=3)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256, feature_channels, blocks=3, stride=2)

    def _make_layer(self, in_planes, out_planes, blocks, stride=1):
        """构建残差层（文献3.1节CNN分支）"""
        downsample = None
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_planes),
            )

        layers = []
        layers.append(ResNetBlock(in_planes, out_planes, stride, downsample))
        for i in range(1, blocks):
            layers.append(ResNetBlock(out_planes, out_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播（文献3.1节CNN分支流程）"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNetBlock(nn.Module):
    """ResNet基本残差块"""

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DSTBranch(nn.Module):
    """扩张采样Transformer分支（文献3.2节DST分支）"""

    def __init__(self, in_channels=1, feature_channels=64, patch_size=[4, 4, 4], embed_dim=768, num_heads=12,
                 num_layers=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.feature_channels = feature_channels

        # 3D patch embedding（文献图2补丁嵌入）
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Transformer blocks with dilated sampling（文献图2核心结构）
        self.blocks = nn.ModuleList([
            DilatedSamplingTransformerBlock(
                embed_dim, num_heads,
                window_size=7,
                dilation_rate=2 ** (i + 1)  # 递增膨胀率
            ) for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # 投影到目标特征维度
        self.feature_proj = nn.Linear(embed_dim, feature_channels)

        # 上采样恢复到原始尺寸
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(feature_channels, feature_channels,
                               kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """前向传播（文献3.2节DST分支流程）"""
        B, C, D, H, W = x.shape

        # 1. Patch embedding（文献图2补丁划分）
        x_patches = self.patch_embed(x)  # (B, embed_dim, D', H', W')
        _, _, D_p, H_p, W_p = x_patches.shape

        # 2. 展平为序列格式
        x_seq = x_patches.flatten(2).transpose(1, 2)  # (B, N_patches, embed_dim)

        # 3. 通过DST blocks（文献式1-4）
        for block in self.blocks:
            x_seq = block(x_seq)

        x_seq = self.norm(x_seq)

        # 4. 投影到目标特征维度
        x_seq = self.feature_proj(x_seq)  # (B, N_patches, feature_channels)

        # 5. 重塑回3D形状
        x_3d = x_seq.transpose(1, 2).reshape(B, self.feature_channels, D_p, H_p, W_p)

        # 6. 上采样恢复原始尺寸
        x_out = self.upsample(x_3d)

        return x_out


class DilatedSamplingTransformerBlock(nn.Module):
    """扩张采样自注意力模块（文献图2核心组件）"""

    def __init__(self, dim, num_heads=8, window_size=7, dilation_rate=2, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.dilation_rate = dilation_rate
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 多头自注意力组件
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        # MLP组件
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def window_attention(self, x):
        """窗口自注意力（文献式5）"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x

    def dilated_window_attention(self, x):
        """膨胀窗口自注意力（文献图2膨胀采样）"""
        B, N, C = x.shape

        # 生成膨胀采样索引
        if N >= self.dilation_rate:
            dilated_indices = torch.arange(0, N, self.dilation_rate, device=x.device)
            x_dilated = x[:, dilated_indices, :]  # 膨胀采样

            # 在采样后的序列上应用注意力
            x_dilated = self.window_attention(x_dilated)

            # 恢复到原始序列（简单重复填充）
            x_out = x.clone()
            x_out[:, dilated_indices, :] = x_dilated
        else:
            x_out = self.window_attention(x)

        return x_out

    def forward(self, x):
        """前向传播（文献式1-4）"""
        # 第一个attention：窗口自注意力 + 膨胀采样自注意力（文献式1）
        shortcut = x
        x = self.norm1(x)

        # 标准窗口注意力
        x_wa = self.window_attention(x)

        # 膨胀采样注意力
        x_dwa = self.dilated_window_attention(x)

        # 组合（文献式1）
        x = shortcut + x_wa + x_dwa

        # MLP分支（文献式2）
        x = x + self.mlp(self.norm2(x))

        return x


class SpatialChannelAttention(nn.Module):
    """空间通道注意力交互桥接（文献3.3节SCA模块）"""

    def __init__(self, cnn_channels, dst_channels, lambda1=0.6):
        super().__init__()
        self.lambda1 = lambda1  # 文献表2最佳参数λ1=0.6
        self.lambda2 = 1.0 - lambda1

        # 确保特征维度一致
        if cnn_channels != dst_channels:
            self.cnn_proj = nn.Conv3d(cnn_channels, dst_channels, kernel_size=1)
        else:
            self.cnn_proj = nn.Identity()

        # 空间注意力模块（文献式6）
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 通道注意力模块（文献式7）
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dst_channels, dst_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dst_channels // 16, dst_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, cnn_feat, dst_feat):
        """特征融合（文献式8）"""
        # 确保维度一致
        cnn_feat = self.cnn_proj(cnn_feat)

        # 计算空间注意力图（文献式6）
        dst_max = torch.max(dst_feat, dim=1, keepdim=True)[0]
        dst_mean = torch.mean(dst_feat, dim=1, keepdim=True)
        spatial_map = self.spatial_attn(torch.cat([dst_max, dst_mean], dim=1))

        # 计算通道注意力图（文献式7）
        channel_map = self.channel_attn(cnn_feat)

        # 加权融合（文献式8）
        cnn_weighted = cnn_feat * spatial_map
        dst_weighted = dst_feat * channel_map
        fused_feat = self.lambda1 * cnn_weighted + self.lambda2 * dst_weighted

        return fused_feat


class DSTC3D(nn.Module):
    """DST-C双分支3D分割网络（文献图1完整实现）"""

    def __init__(self, in_channels=1, out_channels=1, feature_channels=64, lambda1=0.6):
        super().__init__()

        # 1. CNN分支（对应文献3.1节ResNet结构）
        self.cnn_branch = ResNet3DBranch(in_channels, feature_channels)

        # 2. DST分支（对应文献3.2节扩张采样Transformer）
        self.dst_branch = DSTBranch(in_channels, feature_channels)

        # 3. SCA交互桥接（对应文献3.3节空间通道注意力）
        self.sca_bridge = SpatialChannelAttention(feature_channels, feature_channels, lambda1)

        # 4. 解码器（对应文献图1绿色解码路径）
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(feature_channels, feature_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(feature_channels // 2, feature_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels // 4, out_channels, kernel_size=1)
        )

    def forward(self, x):
        """前向传播流程（对应文献图1网络架构）"""
        # 分支特征提取
        cnn_features = self.cnn_branch(x)  # 局部细节特征
        dst_features = self.dst_branch(x)  # 全局上下文特征

        # SCA桥接融合（文献式8）
        fused_features = self.sca_bridge(cnn_features, dst_features)

        # 解码与分割预测
        segmentation = self.decoder(fused_features)
        return segmentation