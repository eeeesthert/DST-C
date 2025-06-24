# 训练主函数（整合文献4.2节训练配置）
import nrrd
import numpy as np
import torch
from skimage.restoration._denoise import _gaussian_weight
from torch.cuda import device
from tqdm import tqdm
from SSL import SimMIMPreTrainer
from dataloader import get_data_loaders
from loss import SegmentationLoss, SSLLoss
from metrics import calculate_dice, calculate_all_metrics
from model import DSTC3D
from proprocessing import postprocess_segmentation


def train_ssl_model(ssl_model, ssl_loss, optimizer, train_loader, device, epochs=200):
    """自监督预训练函数（文献4.5节SSL实验）"""
    ssl_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"SSL Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
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

        avg_loss = epoch_loss / len(train_loader)
        print(f"SSL Epoch {epoch + 1}, Loss: {avg_loss:.4f}")


def train_segmentation_model(model, seg_loss, optimizer, train_loader, val_loader, device, epochs=300):
    """监督训练函数（文献4.6节监督训练）"""
    best_dice = 0.0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = seg_loss(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images).squeeze()
                loss = seg_loss(outputs, labels)

                # 转换为概率和二值掩码
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # 计算Dice系数（文献4.3节评估指标）
        dice = calculate_dice(all_preds, all_labels)
        scheduler.step()

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Dice: {dice:.4f}")

        # 保存最佳模型（文献4.8节模型保存）
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Best model saved, Dice: {best_dice:.4f}")
def train_full_pipeline(config):
    """端到端训练流程（文献4.3-4.8节实验设置）"""
    # 1. 数据加载
    train_loader, val_loader = get_data_loaders(config)
    print(f"训练集大小: {len(train_loader.dataset)}, 验证集大小: {len(val_loader.dataset)}")

    # 2. 模型初始化
    model = DSTC3D(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        feature_channels=config["feature_channels"],
        lambda1=config["lambda1"]
    ).to(config["device"])

    # 3. 自监督预训练（文献4.5节SSL实验）
    if config["use_ssl"]:
        ssl_model = SimMIMPreTrainer(model, config["mask_ratio"], config["patch_size"])
        ssl_loss = SSLLoss()
        ssl_optimizer = torch.optim.AdamW(
            ssl_model.parameters(),
            lr=config["ssl_lr"],
            weight_decay=config["weight_decay"]
        )

        print("开始自监督预训练...")
        train_ssl_model(
            ssl_model, ssl_loss, ssl_optimizer, train_loader,
            config["device"], config["ssl_epochs"]
        )
        # 加载预训练权重
        model.load_state_dict(ssl_model.base_model.state_dict())

    # 4. 监督训练（文献4.6节消融实验）
    seg_loss = SegmentationLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=0.9,
        weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["train_epochs"])

    print("开始监督训练...")
    train_segmentation_model(
        model, seg_loss, optimizer, train_loader, val_loader,
        config["device"], config["train_epochs"]
    )

    # 在验证阶段计算评估指标
    def validate(model, val_loader, device):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images).squeeze()
                preds = torch.sigmoid(outputs) > 0.5

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # 转换为numpy数组
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # 计算指标
        metrics = calculate_all_metrics(all_preds, all_labels)
        return metrics

    # 在训练循环中调用
    metrics = validate(model, val_loader, device)
    print(f"Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}, "
          f"HD95: {metrics['hd95']:.4f}, ASD: {metrics['asd']:.4f}")


# 推理函数（支持大体积3D图像）
def inference(model, input_path, output_path, config):
    """
    模型推理流程（文献4.8节测试设置）
    input_path: 输入.nrrd图像路径
    output_path: 输出分割结果路径
    """
    model.eval()
    model.to(config["device"])

    # 1. 加载.nrrd图像
    img_data, header = nrrd.read(input_path)
    if img_data.ndim == 3:
        img_data = np.expand_dims(img_data, axis=0)  # 增加通道维度
    img_tensor = torch.from_numpy(img_data.astype(np.float32)).unsqueeze(0).to(config["device"])

    # 2. 滑动窗口推理（处理大体积图像，文献4.2节推理策略）
    z_size, y_size, x_size = img_tensor.shape[2], img_tensor.shape[3], img_tensor.shape[4]
    output = np.zeros((1, 1, z_size, y_size, x_size), dtype=np.float32)
    window_size = config["patch_size"]
    step_size = window_size // 2  # 重叠率50%

    with torch.no_grad():
        for z in range(0, z_size, step_size):
            for y in range(0, y_size, step_size):
                for x in range(0, x_size, step_size):
                    z_end, y_end, x_end = min(z + window_size, z_size), min(y + window_size, y_size), min(
                        x + window_size, x_size)
                    window = img_tensor[:, :, z:z_end, y:y_end, x:x_end]

                    # 模型推理
                    pred = model(window).cpu().numpy()

                    # 高斯权重融合（文献4.8节测试策略）
                    weight = _gaussian_weight(pred.shape[2:], sigma=window_size / 4)
                    output[:, :, z:z_end, y:y_end, x:x_end] += pred * weight

    # 3. 后处理
    pred_map = np.squeeze(output, axis=(0, 1))
    global_max = np.max(pred_map)
    processed = postprocess_segmentation(
        pred_map, global_max,
        config["min_volume"], config["max_volume"],
        config["alpha1"], config["alpha2"]
    )

    # 4. 保存结果
    nrrd.write(output_path, processed, header)
    print(f"分割结果已保存至: {output_path}")