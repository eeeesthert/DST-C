import numpy as np
from scipy import ndimage


def postprocess_segmentation(pred_map, global_max, min_volume=50, max_volume=5000,
                             alpha1=0.3, alpha2=0.5):
    """
    完整后处理流程（文献3.5节全流程）
    pred_map: 模型输出概率图 [Z, Y, X]
    global_max: 全局最大概率值
    """
    # 1. 自适应阈值区域生长（文献算法1）
    processed = adaptive_threshold_region_growth(pred_map, global_max, alpha1, alpha2)

    # 2. 连通区域体积过滤（文献图10假阳性移除）
    labels, counts = np.unique(processed, return_counts=True)
    valid_mask = np.zeros_like(processed, dtype=bool)

    for label, count in zip(labels[1:], counts[1:]):  # 跳过背景标签0
        if min_volume <= count <= max_volume:
            valid_mask |= (processed == label)

    # 3. 形态学操作优化边界（补充文献未明确的常见后处理）
    valid_mask = ndimage.binary_closing(valid_mask, structure=np.ones((3, 3, 3)))
    valid_mask = ndimage.binary_opening(valid_mask, structure=np.ones((3, 3, 3)))

    return valid_mask.astype(np.uint8)


def adaptive_threshold_region_growth(seg_map, global_max, alpha1, alpha2):
    """
    自适应阈值区域生长（文献算法1详细实现）
    seg_map: 分割概率图 [Z, Y, X]
    global_max: 全局最大概率值
    alpha1: 局部最大值与全局最大值的比例系数
    alpha2: 邻域像素阈值系数
    """
    z_size, y_size, x_size = seg_map.shape
    processed = np.zeros_like(seg_map, dtype=np.uint8)
    block_size = 64  # 分块大小，平衡内存与效率

    # 分块处理大体积数据（文献4.2节滑动窗口思想）
    for z_start in range(0, z_size, block_size):
        for y_start in range(0, y_size, block_size):
            for x_start in range(0, x_size, block_size):
                z_end = min(z_start + block_size, z_size)
                y_end = min(y_start + block_size, y_size)
                x_end = min(x_start + block_size, x_size)
                block = seg_map[z_start:z_end, y_start:y_end, x_start:x_end]

                local_max = np.max(block)
                if local_max > alpha1 * global_max:
                    # 找到局部最大值作为种子点
                    seed_idx = np.unravel_index(np.argmax(block), block.shape)
                    g_z, g_y, g_x = z_start + seed_idx[0], y_start + seed_idx[1], x_start + seed_idx[2]

                    # 执行3D区域生长（文献算法1核心步骤）
                    region = _region_growth(seg_map, (g_z, g_y, g_x), alpha2 * local_max)
                    processed[region] = 1

    return processed


def _region_growth(volume, seed, threshold):
    """3D区域生长核心算法（文献算法1内部逻辑）"""
    z, y, x = seed
    queue = [(z, y, x)]
    visited = np.zeros_like(volume, dtype=bool)
    visited[z, y, x] = True

    # 6邻域系统（上下左右前后）
    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    while queue:
        cz, cy, cx = queue.pop(0)
        for dz, dy, dx in directions:
            nz, ny, nx = cz + dz, cy + dy, cx + dx
            if (0 <= nz < volume.shape[0] and 0 <= ny < volume.shape[1] and 0 <= nx < volume.shape[2]
                    and not visited[nz, ny, nx] and volume[nz, ny, nx] >= threshold):
                visited[nz, ny, nx] = True
                queue.append((nz, ny, nx))

    return visited