# 推理函数（支持大体积3D图像）
import nrrd
import numpy as np
import torch
from skimage.restoration._denoise import _gaussian_weight

from proprocessing import postprocess_segmentation


def test(model, input_path, output_path, config):
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