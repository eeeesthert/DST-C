
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config
from dataloader import get_data_loaders
from dataset import ABVSDataset
from UNet3D import UNet3D
from metrics_1 import dice_coefficient, iou_score
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


torch.backends.cudnn.enabled = False  # 防止CUDNN大卷积爆显存

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config = config.get_config()
train_loader, val_loader = get_data_loaders(config)
print(f"训练集大小: {len(train_loader.dataset)}, 验证集大小: {len(val_loader.dataset)}")

model = UNet3D(in_channels=config["in_channels"], out_channels=config["out_channels"]).to(config['device'])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
best_dice = 0.0

def sliding_window_inference(model, image, window_size=(64,64,64), overlap=0.5):
    model.eval()
    _, D, H, W = image.shape
    stride = [int(w * (1 - overlap)) for w in window_size]
    output = torch.zeros((1, 1, D, H, W), device=image.device)
    count_map = torch.zeros_like(output)

    for z in range(0, D, stride[0]):
        for y in range(0, H, stride[1]):
            for x in range(0, W, stride[2]):
                zs, ys, xs = z, y, x
                ze, ye, xe = min(z + window_size[0], D), min(y + window_size[1], H), min(x + window_size[2], W)
                patch = image[:, :, zs:ze, ys:ye, xs:xe]

                pad = [
                    0, window_size[2] - patch.shape[4],
                    0, window_size[1] - patch.shape[3],
                    0, window_size[0] - patch.shape[2]
                ]
                patch = nn.functional.pad(patch, pad)

                with torch.no_grad():
                    pred = model(patch)
                    pred = pred[:, :, :ze - zs, :ye - ys, :xe - xs]
                    output[:, :, zs:ze, ys:ye, xs:xe] += pred
                    count_map[:, :, zs:ze, ys:ye, xs:xe] += 1
    output = output / torch.clamp(count_map, min=1e-5)
    return output

for epoch in range(1, 101):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{100} - Training"):
        images = batch["image"].to(config['device'])
        masks = batch["label"].to(config['device']).float()

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch [{epoch}/100], Train Loss: {train_loss/len(train_loader):.4f}")

    # ---------- 验证 ----------
    model.eval()
    dice_total = 0.0
    iou_total = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{100} - Validation"):
            images = batch["image"].to(config['device'])
            masks = batch["label"].to(config['device']).float()

            outputs = sliding_window_inference(model, images)

            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
            dice_total += dice
            iou_total += iou

    avg_dice = dice_total / len(val_loader)
    avg_iou = iou_total / len(val_loader)
    print(f"Validation Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")

    if avg_dice > best_dice:
        best_dice = avg_dice
        torch.save(model.state_dict(), 'best_model.pth')
        print("✅ Saved best model!")
    torch.cuda.empty_cache()