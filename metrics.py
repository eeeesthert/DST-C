import numpy as np
import torch
# 需要安装: pip install medpy
from medpy import metric



def calculate_dice(preds, labels):
    """
    计算Dice系数（文献4.3节评估指标）
    preds: 预测的二值掩码，形状为[N, Z, Y, X]
    labels: 真实标签的二值掩码，形状为[N, Z, Y, X]
    """
    # 确保输入为numpy数组
    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    # 初始化Dice系数总和
    dice_total = 0.0
    num_samples = preds.shape[0]

    # 遍历每个样本
    for i in range(num_samples):
        pred = preds[i].astype(bool)
        label = labels[i].astype(bool)

        # 确保预测和标签维度一致
        if pred.ndim == 3 and label.ndim == 3:
            # 计算交集
            intersection = np.logical_and(pred, label).sum()
            # 计算并集
            union = np.logical_or(pred, label).sum()
            # 计算Dice系数
            dice = 2.0 * intersection / (union + 1e-6)
            dice_total += dice
        else:
            print(f"样本 {i} 维度不匹配: 预测 {pred.ndim}D, 标签 {label.ndim}D")

    # 计算平均Dice系数
    avg_dice = dice_total / num_samples
    return avg_dice


def calculate_all_metrics(preds, labels):
    """
    计算文献中提到的所有评估指标：Dice、IoU、95% Hausdorff距离、平均表面距离
    preds: 预测的二值掩码，形状为[N, Z, Y, X]
    labels: 真实标签的二值掩码，形状为[N, Z, Y, X]
    """
    metrics = {}
    num_samples = preds.shape[0]

    dice_total = 0.0
    iou_total = 0.0
    hd95_total = 0.0
    asd_total = 0.0

    for i in range(num_samples):
        pred = preds[i].astype(np.uint8)
        label = labels[i].astype(np.uint8)

        # 确保预测和标签为3D
        if pred.ndim != 3 or label.ndim != 3:
            print(f"样本 {i} 不是3D数据，跳过")
            continue

        # 1. 计算Dice系数
        intersection = np.logical_and(pred, label).sum()
        union = np.logical_or(pred, label).sum()
        dice = 2.0 * intersection / (union + 1e-6)
        dice_total += dice

        # 2. 计算IoU
        iou = intersection / (union + 1e-6)
        iou_total += iou

        # 3. 计算95% Hausdorff距离（需要使用scipy或scikit-image）
        try:
            from skimage.metrics import hausdorff_distance
            # 找到边界
            pred_boundary = _get_boundary(pred)
            label_boundary = _get_boundary(label)

            if np.sum(pred_boundary) > 0 and np.sum(label_boundary) > 0:
                hd95 = hausdorff_distance(pred_boundary, label_boundary, distance_metric='euclidean', percentile=95)
                hd95_total += hd95
            else:
                hd95_total += 0  # 边界为空时设为0
        except Exception as e:
            print(f"计算样本 {i} 的HD95时出错: {e}")
            hd95_total += 0

        # 4. 计算平均表面距离
        try:
            from scipy import ndimage
            # 计算表面距离
            pred_surface = ndimage.binary_dilation(pred) ^ pred
            label_surface = ndimage.binary_dilation(label) ^ label

            if np.sum(pred_surface) > 0 and np.sum(label_surface) > 0:
                # 计算每个表面点到另一表面的距离
                pred_dist = ndimage.distance_transform_edt(~label)
                label_dist = ndimage.distance_transform_edt(~pred)

                asd = (np.sum(pred_dist[pred_surface]) + np.sum(label_dist[label_surface])) / (
                            np.sum(pred_surface) + np.sum(label_surface) + 1e-6)
                asd_total += asd
            else:
                asd_total += 0
        except Exception as e:
            print(f"计算样本 {i} 的ASD时出错: {e}")
            asd_total += 0

    # 计算平均值
    metrics["dice"] = dice_total / num_samples
    metrics["iou"] = iou_total / num_samples
    metrics["hd95"] = hd95_total / num_samples
    metrics["asd"] = asd_total / num_samples

    return metrics


def _get_boundary(volume):
    """获取3D体积的边界"""
    from scipy import ndimage
    dilated = ndimage.binary_dilation(volume)
    boundary = dilated ^ volume
    return boundary.astype(np.uint8)