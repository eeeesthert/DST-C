import torch

def dice_coefficient(pred, target, threshold=0.5, eps=1e-6):
    pred = (pred > threshold).float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * intersection + eps) / (union + eps)
    return dice.item()

def iou_score(pred, target, threshold=0.5, eps=1e-6):
    pred = (pred > threshold).float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.item()
