# import nrrd
# import torch
# import numpy as np
# from scipy import ndimage
# from torch.utils.data import Dataset, DataLoader
# from skimage.transform import resize
# from torchvision import transforms
# import os
# import cv2
#
#
# class ABVSDataset(Dataset):
#     """ABVS数据集处理类，支持.nrrd格式3D医学图像"""
#
#     def __init__(self, data_dir, mode='train', patch_size=128, normalize=True):
#         """
#         初始化数据集
#         data_dir: 数据根目录，需包含train/val/test子目录，每个子目录下包含images和labels文件夹
#         mode: 'train'/'val'/'test'模式
#         patch_size: 训练时裁剪的补丁大小
#         normalize: 是否进行图像标准化
#         """
#         self.data_dir = data_dir
#         self.mode = mode
#         self.patch_size = patch_size
#         self.normalize = normalize
#
#         # 构建数据路径（对应文献4.1.1节数据划分）
#         self.image_dir = os.path.join(data_dir, mode, "images")
#         self.label_dir = os.path.join(data_dir, mode, "labels")
#         self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(".nrrd")]
#
#         # 数据增强转换（仅训练时使用）
#         if mode == 'train':
#             self.transforms = transforms.Compose([
#                 RandomFlip(),
#                 RandomRotation(),
#                 RandomElasticDeformation()
#             ])
#
#     def __len__(self):
#         """返回数据集的样本数量"""
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         """加载并处理单样本数据"""
#         # 1. 读取.nrrd文件（对应文献4.1节数据预处理）
#         img_file = os.path.join(self.image_dir, self.image_files[idx])
#         label_file = os.path.join(self.label_dir, self.image_files[idx].replace("DATA", "MASK"))
#         img_data, img_header = nrrd.read(img_file)
#         label_data, label_header = nrrd.read(label_file)
#
#         # 2. 维度标准化处理（统一为C-Z-Y-X格式）
#         if img_data.ndim == 3:
#             img_data = np.expand_dims(img_data, axis=0)  # 添加通道维度
#             label_data = np.expand_dims(label_data, axis=0)
#         elif img_data.ndim == 4:
#             img_data = img_data.transpose(3, 0, 1, 2)  # 转换为C-Z-Y-X
#             label_data = label_data.transpose(3, 0, 1, 2)
#
#         # 3. 图像标准化（对应文献4.1.1节插值与归一化）
#         if self.normalize:
#             img_data = (img_data - np.mean(img_data)) / (np.std(img_data) + 1e-6)
#             img_data = np.clip(img_data, -3, 3)  # 截断异常值
#
#         # 4. 数据增强（仅训练时）
#         if self.mode == 'train' and self.transforms:
#             img_data, label_data = self.transforms(img_data, label_data)
#
#         # 5. 随机裁剪（对应文献4.2节训练配置）
#         if self.mode == 'train':
#             img_data, label_data = self._random_crop(img_data, label_data)
#
#         return {
#             "image": torch.from_numpy(img_data.astype(np.float32)),
#             "label": torch.from_numpy(label_data.astype(np.int64)),
#             "filename": self.image_files[idx]
#         }
#
#     def _random_crop(self, img, label):
#         """3D随机裁剪，保持标签与图像对齐"""
#         C, Z, Y, X = img.shape
#         z_start = np.random.randint(0, Z - self.patch_size + 1)
#         y_start = np.random.randint(0, Y - self.patch_size + 1)
#         x_start = np.random.randint(0, X - self.patch_size + 1)
#         return (
#             img[:, z_start:z_start + self.patch_size, y_start:y_start + self.patch_size,
#             x_start:x_start + self.patch_size],
#             label[:, z_start:z_start + self.patch_size, y_start:y_start + self.patch_size,
#             x_start:x_start + self.patch_size]
#         )
#
#
# class RandomFlip(object):
#     """3D图像随机翻转增强（文献4.2节数据增强）"""
#
#     def __call__(self, img, label):
#         """
#         img: [C, Z, Y, X] 格式的numpy数组
#         label: [C, Z, Y, X] 格式的numpy数组（C=1）
#         """
#         # 随机决定是否在三个维度上翻转
#         flips = [np.random.randint(0, 2) for _ in range(3)]
#         for i, do_flip in enumerate(flips):
#             if do_flip:
#                 img = np.flip(img, axis=i + 2)  # 沿Z/Y/X轴翻转
#                 label = np.flip(label, axis=i + 2)
#         return img, label
#
#
# class RandomRotation(object):
#     """3D图像随机旋转增强（文献4.2节数据增强）"""
#
#     def __init__(self, max_angle=15):
#         """max_angle: 最大旋转角度（度）"""
#         self.max_angle = max_angle
#
#     def __call__(self, img, label):
#         """
#         img: [C, Z, Y, X] 格式的numpy数组
#         label: [C, Z, Y, X] 格式的numpy数组（C=1）
#         """
#         C, Z, Y, X = img.shape
#         # 随机生成旋转角度（绕三个轴）
#         angle_z = np.random.uniform(-self.max_angle, self.max_angle)
#         angle_y = np.random.uniform(-self.max_angle, self.max_angle)
#         angle_x = np.random.uniform(-self.max_angle, self.max_angle)
#
#         # 对每个切片应用2D旋转（简化3D旋转实现）
#         rotated_img = np.zeros_like(img)
#         rotated_label = np.zeros_like(label)
#
#         for z in range(Z):
#             for y in range(Y):
#                 # 绕Z轴旋转（XY平面）
#                 img_slice = img[0, z, y]
#                 label_slice = label[0, z, y]
#
#                 # 旋转图像切片
#                 img_rot = cv2.warpAffine(
#                     img_slice,
#                     cv2.getRotationMatrix2D((X // 2, Y // 2), angle_z, 1.0),
#                     (X, Y),
#                     flags=cv2.INTER_LINEAR,
#                     borderMode=cv2.BORDER_CONSTANT
#                 )
#
#                 # 旋转标签切片（最近邻插值保持标签离散性）
#                 label_rot = cv2.warpAffine(
#                     label_slice,
#                     cv2.getRotationMatrix2D((X // 2, Y // 2), angle_z, 1.0),
#                     (X, Y),
#                     flags=cv2.INTER_NEAREST,
#                     borderMode=cv2.BORDER_CONSTANT
#                 )
#
#                 rotated_img[0, z, y] = img_rot
#                 rotated_label[0, z, y] = label_rot
#
#         return rotated_img, rotated_label
#
#
# class RandomElasticDeformation(object):
#     """3D图像随机弹性变形增强（文献4.2节数据增强）"""
#
#     def __init__(self, alpha=100, sigma=10):
#         """
#         alpha: 变形强度
#         sigma: 高斯核标准差
#         """
#         self.alpha = alpha
#         self.sigma = sigma
#
#     def __call__(self, img, label):
#         """
#         img: [C, Z, Y, X] 格式的numpy数组
#         label: [C, Z, Y, X] 格式的numpy数组（C=1）
#         """
#         C, Z, Y, X = img.shape
#         # 生成位移场
#         dx = ndimage.gaussian_filter(
#             (np.random.rand(Z, Y, X) * 2 - 1),
#             sigma=self.sigma,
#             mode='constant'
#         ) * self.alpha
#
#         dy = ndimage.gaussian_filter(
#             (np.random.rand(Z, Y, X) * 2 - 1),
#             sigma=self.sigma,
#             mode='constant'
#         ) * self.alpha
#
#         dz = ndimage.gaussian_filter(
#             (np.random.rand(Z, Y, X) * 2 - 1),
#             sigma=self.sigma,
#             mode='constant'
#         ) * self.alpha
#
#         # 生成坐标网格
#         z, y, x = np.meshgrid(
#             np.arange(Z),
#             np.arange(Y),
#             np.arange(X),
#             indexing='ij'
#         )
#
#         # 应用位移场
#         z_deformed = z + dz
#         y_deformed = y + dy
#         x_deformed = x + dx
#
#         # 插值变形图像
#         deformed_img = np.zeros_like(img)
#         deformed_label = np.zeros_like(label)
#
#         for c in range(C):
#             for z_idx in range(Z):
#                 for y_idx in range(Y):
#                     # 对每个2D切片应用变形
#                     img_slice = img[c, z_idx, y_idx]
#                     label_slice = label[c, z_idx, y_idx]
#
#                     # 使用网格插值变形
#                     grid_x = x_deformed[z_idx, y_idx].astype(np.float32)
#                     grid_y = y_deformed[z_idx, y_idx].astype(np.float32)
#
#                     # 图像使用双线性插值
#                     img_deformed = cv2.remap(
#                         img_slice,
#                         grid_x,
#                         grid_y,
#                         cv2.INTER_LINEAR
#                     )
#
#                     # 标签使用最近邻插值
#                     label_deformed = cv2.remap(
#                         label_slice,
#                         grid_x,
#                         grid_y,
#                         cv2.INTER_NEAREST
#                     )
#
#                     deformed_img[c, z_idx, y_idx] = img_deformed
#                     deformed_label[c, z_idx, y_idx] = label_deformed
#
#         return deformed_img, deformed_label


# DST-C/dataset.py
import nrrd
import torch
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from torchvision import transforms
import os
import cv2

class ABVSDataset(Dataset):
    """ABVS数据集处理类，支持.nrrd格式3D医学图像"""

    def __init__(self, data_dir, mode='train', patch_size=128, normalize=True):
        """
        初始化数据集
        data_dir: 数据根目录，需包含train/val/test子目录，每个子目录下包含images和labels文件夹
        mode: 'train'/'val'/'test'模式
        patch_size: 训练时裁剪的补丁大小
        normalize: 是否进行图像标准化
        """
        self.data_dir = data_dir
        self.mode = mode
        self.patch_size = patch_size
        self.normalize = normalize

        # 构建数据路径（对应文献4.1.1节数据划分）
        self.image_dir = os.path.join(data_dir, mode, "images")
        self.label_dir = os.path.join(data_dir, mode, "labels")
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(".nrrd")]

        # 数据增强转换（仅训练时使用）
        if mode == 'train':
            self.transforms = CustomCompose([
                RandomFlip(),
                RandomRotation(),
                RandomElasticDeformation()
            ])

    def __getitem__(self, idx):
        """加载并处理单样本数据"""
        # 1. 读取.nrrd文件（对应文献4.1节数据预处理）
        img_file = os.path.join(self.image_dir, self.image_files[idx])
        label_file = os.path.join(self.label_dir, self.image_files[idx].replace("DATA", "MASK"))
        img_data, img_header = nrrd.read(img_file)
        label_data, label_header = nrrd.read(label_file)

        # 2. 维度标准化处理（统一为C-Z-Y-X格式）
        if img_data.ndim == 3:
            img_data = np.expand_dims(img_data, axis=0)  # 添加通道维度
            label_data = np.expand_dims(label_data, axis=0)
        elif img_data.ndim == 4:
            img_data = img_data.transpose(3, 0, 1, 2)  # 转换为C-Z-Y-X
            label_data = label_data.transpose(3, 0, 1, 2)

        # 3. 图像标准化（对应文献4.1.1节插值与归一化）
        if self.normalize:
            img_data = (img_data - np.mean(img_data)) / (np.std(img_data) + 1e-6)
            img_data = np.clip(img_data, -3, 3)  # 截断异常值

        # 4. 数据增强（仅训练时）
        if self.mode == 'train' and self.transforms:
            img_data, label_data = self.transforms(img_data, label_data)

        # 5. 随机裁剪（对应文献4.2节训练配置）
        if self.mode == 'train':
            img_data, label_data = self._random_crop(img_data, label_data)

        return {
            "image": torch.from_numpy(img_data.astype(np.float32)),
            "label": torch.from_numpy(label_data.astype(np.int64)),
            "filename": self.image_files[idx]
        }

    def __len__(self):
        """返回数据集的样本数量"""
        return len(self.image_files)

    def _random_crop(self, img, label):
        """3D随机裁剪，保持标签与图像对齐"""
        C, Z, Y, X = img.shape
        z_start = np.random.randint(0, Z - self.patch_size + 1)
        y_start = np.random.randint(0, Y - self.patch_size + 1)
        x_start = np.random.randint(0, X - self.patch_size + 1)
        return (
            img[:, z_start:z_start + self.patch_size, y_start:y_start + self.patch_size,
            x_start:x_start + self.patch_size],
            label[:, z_start:z_start + self.patch_size, y_start:y_start + self.patch_size,
            x_start:x_start + self.patch_size]
        )

class CustomCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label

class RandomFlip(object):
    """3D图像随机翻转增强（文献4.2节数据增强）"""

    def __call__(self, img, label):
        """
        img: [C, Z, Y, X] 格式的numpy数组
        label: [C, Z, Y, X] 格式的numpy数组（C=1）
        """
        # 随机决定是否在三个维度上翻转
        flips = [np.random.randint(0, 2) for _ in range(3)]
        for i, do_flip in enumerate(flips):
            if do_flip:
                # 修正轴索引，确保在有效范围内
                axis = i + 1  # 从1开始，对应Z、Y、X轴
                img = np.flip(img, axis=axis)
                label = np.flip(label, axis=axis)
        return img, label

class RandomRotation(object):
    """3D图像随机旋转增强（文献4.2节数据增强）"""

    def __init__(self, max_angle=15):
        """max_angle: 最大旋转角度（度）"""
        self.max_angle = max_angle

    def __call__(self, img, label):
        """
        img: [C, Z, Y, X] 格式的numpy数组
        label: [C, Z, Y, X] 格式的numpy数组（C=1）
        """
        C, Z, Y, X = img.shape
        # 随机生成旋转角度（绕三个轴）
        angle_z = np.random.uniform(-self.max_angle, self.max_angle)
        angle_y = np.random.uniform(-self.max_angle, self.max_angle)
        angle_x = np.random.uniform(-self.max_angle, self.max_angle)

        # 对每个切片应用2D旋转（简化3D旋转实现）
        rotated_img = np.zeros_like(img)
        rotated_label = np.zeros_like(label)

        for c in range(C):
            for z in range(Z):
                for y in range(Y):
                    # 绕Z轴旋转（XY平面）
                    img_slice = img[c, z, y]
                    label_slice = label[c, z, y]

                    # 旋转图像切片
                    img_rot = cv2.warpAffine(
                        img_slice,
                        cv2.getRotationMatrix2D((X // 2, Y // 2), angle_z, 1.0),
                        (X, Y),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT
                    )

                    # 旋转标签切片（最近邻插值保持标签离散性）
                    label_rot = cv2.warpAffine(
                        label_slice,
                        cv2.getRotationMatrix2D((X // 2, Y // 2), angle_z, 1.0),
                        (X, Y),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT
                    )

                    # 确保形状一致
                    if img_rot.shape == (X, Y):
                        rotated_img[c, z, y] = img_rot
                    if label_rot.shape == (X, Y):
                        rotated_label[c, z, y] = label_rot

        return rotated_img, rotated_label

class RandomElasticDeformation(object):
    """3D图像随机弹性变形增强（文献4.2节数据增强）"""

    def __init__(self, alpha=100, sigma=10):
        """
        alpha: 变形强度
        sigma: 高斯核标准差
        """
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img, label):
        """
        img: [C, Z, Y, X] 格式的numpy数组
        label: [C, Z, Y, X] 格式的numpy数组（C=1）
        """
        C, Z, Y, X = img.shape
        # 生成位移场
        dx = ndimage.gaussian_filter(
            (np.random.rand(Z, Y, X) * 2 - 1),
            sigma=self.sigma,
            mode='constant'
        ) * self.alpha

        dy = ndimage.gaussian_filter(
            (np.random.rand(Z, Y, X) * 2 - 1),
            sigma=self.sigma,
            mode='constant'
        ) * self.alpha

        dz = ndimage.gaussian_filter(
            (np.random.rand(Z, Y, X) * 2 - 1),
            sigma=self.sigma,
            mode='constant'
        ) * self.alpha

        # 生成坐标网格
        z, y, x = np.meshgrid(
            np.arange(Z),
            np.arange(Y),
            np.arange(X),
            indexing='ij'
        )

        # 应用位移场
        z_deformed = z + dz
        y_deformed = y + dy
        x_deformed = x + dx

        # 插值变形图像
        deformed_img = np.zeros_like(img)
        deformed_label = np.zeros_like(label)

        for c in range(C):
            for z_idx in range(Z):
                for y_idx in range(Y):
                    # 对每个2D切片应用变形
                    img_slice = img[c, z_idx, y_idx]
                    label_slice = label[c, z_idx, y_idx]

                    # 使用网格插值变形
                    grid_x = x_deformed[z_idx, y_idx].astype(np.float32)
                    grid_y = y_deformed[z_idx, y_idx].astype(np.float32)

                    # 图像使用双线性插值
                    img_deformed = cv2.remap(
                        img_slice,
                        grid_x,
                        grid_y,
                        cv2.INTER_LINEAR
                    )

                    # 标签使用最近邻插值
                    label_deformed = cv2.remap(
                        label_slice,
                        grid_x,
                        grid_y,
                        cv2.INTER_NEAREST
                    )

                    # 确保形状一致
                    if img_deformed.shape == (Y, X):
                        deformed_img[c, z_idx, y_idx] = img_deformed
                    if label_deformed.shape == (Y, X):
                        deformed_label[c, z_idx, y_idx] = label_deformed

        return deformed_img, deformed_label