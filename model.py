import torch
from torch import nn
import torch.nn.functional as F


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

    def __init__(self, in_channels=1, feature_channels=64):
        super().__init__()
        # 补丁嵌入层（文献3.2节补丁划分）
        self.patch_embed = nn.Conv3d(
            in_channels, feature_channels,
            kernel_size=4, stride=4
        )

        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, feature_channels, 32, 32, 32)  # 假设输入尺寸为128x128x128
        )

        # DST块（文献图2扩张采样Transformer）
        self.dst_blocks = nn.ModuleList([
            DilatedSamplingTransformerBlock(feature_channels, window_size=7, dilation=2),
            DilatedSamplingTransformerBlock(feature_channels, window_size=7, dilation=4),
            DilatedSamplingTransformerBlock(feature_channels, window_size=7, dilation=6),
        ])

        # 上采样恢复分辨率（文献3.2节输出处理）
        self.upsample = nn.Upsample(
            scale_factor=4, mode="trilinear", align_corners=True
        )

    def forward(self, x):
        """前向传播（文献3.2节DST分支流程）"""
        # 补丁嵌入
        x = self.patch_embed(x)
        B, C, Z, Y, X = x.shape

        # 展平为序列 + 位置编码
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = x + self.pos_embed.flatten(2).transpose(1, 2)

        # 通过DST块（文献图2）
        for block in self.dst_blocks:
            x = block(x)

        # 恢复为3D特征图
        x = x.transpose(1, 2).reshape(B, C, Z, Y, X)
        x = self.upsample(x)

        return x


class SpatialChannelAttention(nn.Module):
    """空间通道注意力交互桥接（文献3.3节SCA模块）"""

    def __init__(self, cnn_channels, dst_channels, lambda1=0.6):
        super().__init__()
        self.lambda1 = lambda1  # 文献表2最佳参数λ1=0.6
        self.lambda2 = 1.0 - lambda1

        # 空间注意力模块（文献3.3节空间注意力）
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 通道注意力模块（文献3.3节通道注意力）
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(cnn_channels + dst_channels, (cnn_channels + dst_channels) // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d((cnn_channels + dst_channels) // 16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, cnn_feat, dst_feat):
        """特征融合（文献式7-8）"""
        # 特征拼接
        combined = torch.cat([cnn_feat, dst_feat], dim=1)

        # 计算空间注意力图（文献3.3节空间注意力计算）
        spatial_map = self.spatial_attn(torch.cat([
            torch.max(combined, dim=1, keepdim=True)[0],
            torch.mean(combined, dim=1, keepdim=True)
        ], dim=1))

        # 计算通道注意力图（文献3.3节通道注意力计算）
        channel_map = self.channel_attn(combined)

        # 加权融合（文献式7）
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
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
            nn.Conv3d(feature_channels, feature_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
            nn.Conv3d(feature_channels // 2, feature_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels // 4, out_channels, kernel_size=1)
        )

    def forward(self, x):
        """前向传播流程（对应文献3.1节网络架构）"""
        # 分支特征提取
        cnn_features = self.cnn_branch(x)  # 局部细节特征
        dst_features = self.dst_branch(x)  # 全局上下文特征

        # SCA桥接融合（文献式7-8）
        fused_features = self.sca_bridge(cnn_features, dst_features)

        # 解码与分割预测
        segmentation = self.decoder(fused_features)
        return segmentation


class DilatedSamplingTransformerBlock(nn.Module):
    """扩张采样自注意力模块（文献图2核心组件）"""

    def __init__(self, dim, window_size=7, dilation=2):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.dilation = dilation

        # 多头自注意力（文献式4-5）
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def _generate_dilation_mask(self, seq_len):
        """生成扩张采样索引（文献图2采样逻辑）"""
        # 假设 seq_len 是一个完全立方数
        side_len = int(seq_len ** (1/3))
        coords = torch.meshgrid(
            torch.arange(side_len),
            torch.arange(side_len),
            torch.arange(side_len),
            indexing='ij'
        )
        coords = torch.stack(coords, dim=-1).float()
        center = side_len // 2

        # 计算到中心的距离
        distances = torch.sqrt(torch.sum((coords - center) ** 2, dim=-1))
        # 选择扩张间隔的点
        mask = (distances % self.dilation == 0) | (distances == 0)
        return mask.flatten()

    def forward(self, x):
        """扩张采样自注意力计算（文献式1-3）"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, N, C]

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) / (C ** 0.5)

        # 生成扩张采样掩码
        dilation_mask = self._generate_dilation_mask(N)

        # 应用扩张采样掩码（文献图2采样机制）
        attn = attn[:, dilation_mask, :][:, :, dilation_mask]

        # 软最大化与加权求和
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v[:, dilation_mask, :])

        # 投影与残差连接
        x_proj = self.proj(x)
        x = x_proj + x
        return x