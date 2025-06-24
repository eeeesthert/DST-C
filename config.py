import torch


def get_config():
    """实验配置参数（对应文献表1-3与4.2节）"""
    return {
        "data_path": "./data",  # 改为可配置路径
        "num_classes": 2,
        "seed": 42,  # 添加随机种子
        "augmentation": {
            "rotate_range": 15,  # 可配置的数据增强参数
            "elastic_deformation": True
        },
        "postprocessing": {
            "threshold": 0.5,  # 可配置的后处理参数
            "min_volume": 50
        },
        "data_dir": "./data",  # 数据目录结构
        "in_channels": 1,  # 输入通道数
        "out_channels": 1,  # 输出通道数（二分类）
        "feature_channels": 64,  # 特征通道数（文献图1）
        "lambda1": 0.6,  # SCA权重（文献表2最佳参数）
        "patch_size": 128,  # 训练补丁大小
        "batch_size": 1,  # 批量大小（受GPU内存限制）
        "num_workers": 4,  # 数据加载线程数
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

        # 自监督参数（文献4.5节表3）
        "use_ssl": True,
        "ssl_epochs": 200,
        "ssl_lr": 5e-4,
        "mask_ratio": 0.4,  # 最佳掩码比例（文献图5）

        # 监督训练参数（文献4.2节）
        "train_epochs": 300,
        "lr": 2e-4,
        "weight_decay": 1e-5,

        # 后处理参数（文献3.5节）
        "min_volume": 50,  # 最小区域体积（体素数）
        "max_volume": 5000,  # 最大区域体积
        "alpha1": 0.3,  # 区域生长参数1
        "alpha2": 0.5 , # 区域生长参数2
        "normalize": True  # 添加 normalize 键并设置默认值
    }
