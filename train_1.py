import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader
import config
from dataloader import get_data_loaders
from dataset import ABVSDataset
from UNet3D import UNet3D
from metrics_1 import dice_coefficient, iou_score
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


torch.backends.cudnn.enabled = False  # 防止CUDNN大卷积爆显存

# train_dataset = ABVSDataset(split='train')  # DST-C 风格，默认读取 train 集
# val_dataset = ABVSDataset(split='val')

# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
config = config.get_config()
train_loader, val_loader = get_data_loaders(config)
print(f"训练集大小: {len(train_loader.dataset)}, 验证集大小: {len(val_loader.dataset)}")

model = UNet3D(in_channels=config["in_channels"],out_channels=config["out_channels"]).to(config['device'])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
best_dice = 0.0

for epoch in range(1, 201):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{200} - Training"):
        images = batch["image"].to(config['device'])
        masks = batch["label"].to(config['device']).float()  # ✅ 这里加 .float()

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()


    print(f"Epoch [{epoch}/200], Train Loss: {train_loss/len(train_loader):.4f}")

    # ---------- 验证 ----------
    model.eval()
    dice_total = 0.0
    iou_total = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{200} - Validation"):
            images = batch["image"].to(config['device'])
            masks = batch["label"].to(config['device']).float()  # ✅ 这里加 .float()

            outputs = model(images)
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


