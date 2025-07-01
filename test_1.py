import torch
# from torch.utils.data import DataLoader
from dataloader import get_data_loaders
from dataset import ABVSDataset
from UNet3D import UNet3D
from metrics_1 import dice_coefficient, iou_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = ABVSDataset(split='test')
test_loader = get_data_loaders(test_dataset, batch_size=1, shuffle=False)

model = UNet3D(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

dice_total = 0.0
iou_total = 0.0

with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)

        dice = dice_coefficient(outputs, masks)
        iou = iou_score(outputs, masks)

        dice_total += dice
        iou_total += iou

avg_dice = dice_total / len(test_loader)
avg_iou = iou_total / len(test_loader)

print(f"Test Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")
