"""
utils/patches.py
纯 numpy 的 patch 提取工具，无 torch 依赖。
"""

import numpy as np


def extract_patches(x, label_cls, label_reg, patch_size=128, stride=128):
    """从大区域提取多个 patch。返回 list of (x_patch, cls_patch, reg_patch)。"""
    C, H, W = x.shape
    patches = []
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patches.append((
                x[:, i:i+patch_size, j:j+patch_size],
                label_cls[i:i+patch_size, j:j+patch_size],
                label_reg[i:i+patch_size, j:j+patch_size],
            ))
    return patches
