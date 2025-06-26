# # 训练主函数（整合文献4.2节训练配置）
# import nrrd
# import numpy as np
# import torch
# from skimage.restoration._denoise import _gaussian_weight
# from torch.cuda import device
# from tqdm import tqdm
# from SSL import SimMIMPreTrainer, SSLLoss
# from dataloader import get_data_loaders
# from loss import SegmentationLoss
# from metrics import calculate_dice, calculate_all_metrics
# from model import DSTC3D
# from proprocessing import postprocess_segmentation
#
# def set_seed(seed):
#     """设置全局随机种子，确保实验可复现"""
#     import random
#     import numpy as np
#     import torch
#
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#
# def train_ssl_model(ssl_model, ssl_loss, optimizer, train_loader, device, epochs=200):
#     """自监督预训练函数（文献4.5节SSL实验）"""
#     ssl_model.train()
#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         progress_bar = tqdm(train_loader, desc=f"SSL Epoch {epoch + 1}/{epochs}")
#
#         for batch in progress_bar:
#             images = batch["image"].to(device)
#             optimizer.zero_grad()
#
#             # 前向传播（文献3.4节自监督流程）
#             reconstructed, mask, original = ssl_model(images)
#             loss = ssl_loss(reconstructed, mask, original)
#
#             # 反向传播
#             loss.backward()
#             optimizer.step()
#
#             epoch_loss += loss.item()
#             progress_bar.set_postfix(loss=loss.item())
#
#         avg_loss = epoch_loss / len(train_loader)
#         print(f"SSL Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
#
#
# def train_segmentation_model(model, seg_loss, optimizer, train_loader, val_loader, device, epochs=300):
#     """监督训练函数（文献4.6节监督训练）"""
#     best_dice = 0.0
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
#
#     for epoch in range(epochs):
#         # 训练阶段
#         model.train()
#         train_loss = 0.0
#         progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{epochs}")
#
#         for batch in progress_bar:
#             images = batch["image"].to(device)
#             labels = batch["label"].to(device)
#
#             optimizer.zero_grad()
#             outputs = model(images).squeeze()
#             loss = seg_loss(outputs, labels)
#
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item()
#             progress_bar.set_postfix(loss=loss.item())
#
#         avg_train_loss = train_loss / len(train_loader)
#
#         # 验证阶段
#         model.eval()
#         val_loss = 0.0
#         all_preds, all_labels = [], []
#
#         with torch.no_grad():
#             for batch in val_loader:
#                 images = batch["image"].to(device)
#                 labels = batch["label"].to(device)
#
#                 outputs = model(images).squeeze()
#                 loss = seg_loss(outputs, labels)
#
#                 # 转换为概率和二值掩码
#                 preds = torch.sigmoid(outputs) > 0.5
#                 all_preds.append(preds.cpu().numpy())
#                 all_labels.append(labels.cpu().numpy())
#
#                 val_loss += loss.item()
#
#         avg_val_loss = val_loss / len(val_loader)
#         all_preds = np.concatenate(all_preds)
#         all_labels = np.concatenate(all_labels)
#
#         # 计算Dice系数（文献4.3节评估指标）
#         dice = calculate_dice(all_preds, all_labels)
#         scheduler.step()
#
#         print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, "
#               f"Val Loss: {avg_val_loss:.4f}, Dice: {dice:.4f}")
#
#         # 保存最佳模型（文献4.8节模型保存）
#         if dice > best_dice:
#             best_dice = dice
#             torch.save(model.state_dict(), "best_model.pth")
#             print(f"Best model saved, Dice: {best_dice:.4f}")
#
# def validate(model, val_loader, device):
#     model.eval()
#     all_preds = []
#     all_labels = []
#
#     with torch.no_grad():
#         for batch in val_loader:
#             images = batch["image"].to(device)
#             labels = batch["label"].to(device)
#
#             outputs = model(images).squeeze()
#             preds = torch.sigmoid(outputs) > 0.5
#
#             all_preds.append(preds.cpu().numpy())
#             all_labels.append(labels.cpu().numpy())
#
#     # 转换为numpy数组
#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)
#
#     # 计算指标
#     metrics = calculate_all_metrics(all_preds, all_labels)
#     return metrics
#
#
# def train_full_pipeline(config):
#     """端到端训练流程（文献4.3-4.8节实验设置）"""
#     # 1. 数据加载
#     train_loader, val_loader = get_data_loaders(config)
#     print(f"训练集大小: {len(train_loader.dataset)}, 验证集大小: {len(val_loader.dataset)}")
#
#     # 2. 模型初始化
#     model = DSTC3D(
#         in_channels=config["in_channels"],
#         out_channels=config["out_channels"],
#         feature_channels=config["feature_channels"],
#         lambda1=config["lambda1"]
#     ).to(config["device"])
#
#     # 3. 自监督预训练（文献4.5节SSL实验）
#     if config["use_ssl"]:
#         ssl_model = SimMIMPreTrainer(model, config["feature_channels"], config["mask_ratio"], config["patch_size"])
#         ssl_loss = SSLLoss()
#         ssl_optimizer = torch.optim.AdamW(
#             ssl_model.parameters(),
#             lr=config["ssl_lr"],
#             weight_decay=config["weight_decay"]
#         )
#
#         print("开始自监督预训练...")
#         train_ssl_model(
#             ssl_model, ssl_loss, ssl_optimizer, train_loader,
#             config["device"], config["ssl_epochs"]
#         )
#         # 加载预训练权重
#         model.load_state_dict(ssl_model.base_model.state_dict())
#
#     # 4. 监督训练（文献4.6节消融实验）
#     seg_loss = SegmentationLoss()
#     optimizer = torch.optim.SGD(
#         model.parameters(),
#         lr=config["lr"],
#         momentum=0.9,
#         weight_decay=config["weight_decay"]
#     )
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["train_epochs"])
#
#     print("开始监督训练...")
#     train_segmentation_model(
#         model, seg_loss, optimizer, train_loader, val_loader,
#         config["device"], config["train_epochs"]
#     )
#
#     # 在验证阶段计算评估指标
#     metrics = validate(model, val_loader, config["device"])
#     print(f"Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}, "
#           f"HD95: {metrics['hd95']:.4f}, ASD: {metrics['asd']:.4f}")
#
#
# # def train_full_pipeline(config):
# #     """端到端训练流程（文献4.3-4.8节实验设置）"""
# #     # 1. 数据加载
# #     set_seed(config["seed"])
# #     train_loader, val_loader = get_data_loaders(config)
# #     print(f"训练集大小: {len(train_loader.dataset)}, 验证集大小: {len(val_loader.dataset)}")
# #
# #     # 2. 模型初始化
# #     model = DSTC3D(
# #         in_channels=config["in_channels"],
# #         out_channels=config["out_channels"],
# #         feature_channels=config["feature_channels"],
# #         lambda1=config["lambda1"]
# #     ).to(config["device"])
# #
# #     # 3. 自监督预训练（文献4.5节SSL实验）
# #     if config["use_ssl"]:
# #         ssl_model = SimMIMPreTrainer(model, config["mask_ratio"], config["patch_size"])
# #         ssl_loss = SSLLoss()
# #         ssl_optimizer = torch.optim.AdamW(
# #             ssl_model.parameters(),
# #             lr=config["ssl_lr"],
# #             weight_decay=config["weight_decay"]
# #         )
# #
# #         print("开始自监督预训练...")
# #         train_ssl_model(
# #             ssl_model, ssl_loss, ssl_optimizer, train_loader,
# #             config["device"], config["ssl_epochs"]
# #         )
# #         # 加载预训练权重
# #         model.load_state_dict(ssl_model.base_model.state_dict())
# #
# #     # 4. 监督训练（文献4.6节消融实验）
# #     seg_loss = SegmentationLoss()
# #     optimizer = torch.optim.SGD(
# #         model.parameters(),
# #         lr=config["lr"],
# #         momentum=0.9,
# #         weight_decay=config["weight_decay"]
# #     )
# #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["train_epochs"])
# #
# #     print("开始监督训练...")
# #     train_segmentation_model(
# #         model, seg_loss, optimizer, train_loader, val_loader,
# #         config["device"], config["train_epochs"]
# #     )
# #
# #     # 在验证阶段计算评估指标
# #     def validate(model, val_loader, device):
# #         model.eval()
# #         all_preds = []
# #         all_labels = []
# #
# #         with torch.no_grad():
# #             for batch in val_loader:
# #                 images = batch["image"].to(device)
# #                 labels = batch["label"].to(device)
# #
# #                 outputs = model(images).squeeze()
# #                 preds = torch.sigmoid(outputs) > 0.5
# #
# #                 all_preds.append(preds.cpu().numpy())
# #                 all_labels.append(labels.cpu().numpy())
# #
# #         # 转换为numpy数组
# #         all_preds = np.concatenate(all_preds)
# #         all_labels = np.concatenate(all_labels)
# #
# #         # 计算指标
# #         metrics = calculate_all_metrics(all_preds, all_labels)
# #         return metrics
# #
# #     # 在训练循环中调用
# #     metrics = validate(model, val_loader, device)
# #     print(f"Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}, "
# #           f"HD95: {metrics['hd95']:.4f}, ASD: {metrics['asd']:.4f}")
# #
# #

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from SSL import SimMIMPreTrainer, SSLLoss
from dataloader import get_data_loaders
from loss import SegmentationLoss
from metrics import calculate_dice, calculate_all_metrics
from model import DSTC3D
import os
import random


def set_seed(seed):
    """设置全局随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_ssl_model(ssl_model, ssl_loss, optimizer, train_loader, device, epochs=200):
    """自监督预训练函数（文献4.5节SSL实验）"""
    ssl_model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"SSL Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                images = batch["image"].to(device)
                optimizer.zero_grad()

                # 前向传播（文献3.4节自监督流程）
                reconstructed, mask, original = ssl_model(images)
                loss = ssl_loss(reconstructed, mask, original)

                # 反向传播
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            except Exception as e:
                print(f"SSL训练第{epoch + 1}轮第{batch_idx}批次出错: {e}")
                continue

        avg_loss = epoch_loss / len(train_loader)
        print(f"SSL Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        # 每20轮保存一次预训练模型
        if (epoch + 1) % 20 == 0:
            torch.save(ssl_model.state_dict(), f"ssl_model_epoch_{epoch + 1}.pth")


def train_segmentation_model(model, seg_loss, optimizer, scheduler, train_loader, val_loader, device, epochs=300):
    """监督训练函数（文献4.6节监督训练）"""
    best_dice = 0.0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                images = batch["image"].to(device)
                labels = batch["label"].to(device).squeeze(1)  # 移除通道维度

                optimizer.zero_grad()
                outputs = model(images).squeeze(1)  # 移除通道维度
                loss = seg_loss(outputs, labels.float())

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            except Exception as e:
                print(f"训练第{epoch + 1}轮第{batch_idx}批次出错: {e}")
                continue

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        print(f"HD95: {val_metrics['hd95']:.4f}, ASD: {val_metrics['asd']:.4f}")

        # 保存最佳模型（文献4.8节模型保存）
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Best model saved, Dice: {best_dice:.4f}")

        # 每50轮保存一次检查点
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice
            }, f"checkpoint_epoch_{epoch + 1}.pth")


def validate(model, val_loader, device):
    """验证函数"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            try:
                images = batch["image"].to(device)
                labels = batch["label"].to(device).squeeze(1)

                outputs = model(images).squeeze(1)
                preds = torch.sigmoid(outputs) > 0.5

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            except Exception as e:
                print(f"验证时出错: {e}")
                continue

    if len(all_preds) == 0:
        return {'dice': 0.0, 'iou': 0.0, 'hd95': 0.0, 'asd': 0.0}

    # 转换为numpy数组
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 计算指标
    metrics = calculate_all_metrics(all_preds, all_labels)
    return metrics


def train_full_pipeline(config):
    """端到端训练流程（文献4.3-4.8节实验设置）"""
    # 设置随机种子
    set_seed(config["seed"])

    # 创建保存目录
    os.makedirs("checkpoints", exist_ok=True)

    # 1. 数据加载
    train_loader, val_loader = get_data_loaders(config)
    print(f"训练集大小: {len(train_loader.dataset)}, 验证集大小: {len(val_loader.dataset)}")

    # 2. 模型初始化（文献图1架构）
    model = DSTC3D(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        feature_channels=config["feature_channels"],
        lambda1=config["lambda1"]
    ).to(config["device"])

    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 3. 自监督预训练（文献4.5节SSL实验）
    if config["use_ssl"]:
        print("开始自监督预训练...")
        ssl_model = SimMIMPreTrainer(
            model,
            feature_channels=config["feature_channels"],
            mask_ratio=config["mask_ratio"],
            patch_size=config["patch_size"]
        ).to(config["device"])

        ssl_loss = SSLLoss()
        ssl_optimizer = torch.optim.AdamW(
            ssl_model.parameters(),
            lr=config["ssl_lr"],
            weight_decay=config["weight_decay"]
        )

        train_ssl_model(
            ssl_model, ssl_loss, ssl_optimizer, train_loader,
            config["device"], config["ssl_epochs"]
        )

        # 加载预训练权重到主模型
        model.load_state_dict(ssl_model.base_model.state_dict(), strict=False)
        print("自监督预训练完成，权重已加载到主模型")

    # 4. 监督训练（文献4.6节微调）
    print("开始监督训练...")
    seg_loss = SegmentationLoss()

    # 使用SGD优化器（文献4.2节训练配置）
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=0.9,
        weight_decay=config["weight_decay"]
    )

    # 余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["train_epochs"]
    )

    train_segmentation_model(
        model, seg_loss, optimizer, scheduler, train_loader, val_loader,
        config["device"], config["train_epochs"]
    )

    # 最终验证
    print("训练完成，进行最终验证...")
    final_metrics = validate(model, val_loader, config["device"])
    print("最终验证结果:")
    print(f"Dice: {final_metrics['dice']:.4f}")
    print(f"IoU: {final_metrics['iou']:.4f}")
    print(f"HD95: {final_metrics['hd95']:.4f}")
    print(f"ASD: {final_metrics['asd']:.4f}")

    return model, final_metrics
