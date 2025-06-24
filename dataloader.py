from torch.utils.data.dataloader import DataLoader

from dataset import ABVSDataset


def get_data_loaders(config):
    """获取数据加载器（文献4.1节数据处理）"""
    # 训练集
    train_dataset = ABVSDataset(
        data_dir=config["data_dir"],
        mode="train",
        patch_size=config["patch_size"],
        normalize=config["normalize"]
    )

    # 验证集
    val_dataset = ABVSDataset(
        data_dir=config["data_dir"],
        mode="val",
        patch_size=config["patch_size"],
        normalize=config["normalize"]
    )

    # 数据加载器（文献4.2节训练配置）
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    return train_loader, val_loader