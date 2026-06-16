# FY4A/FY4B AGRI 降水反演 — U-Net 双头模型

基于 FY4A/FY4B AGRI 红外亮温数据，使用轻量 U-Net 双头模型（分类 + 回归）进行像素级降水反演，以 GPM IMERG 作为标签。

## 项目结构

```
precip_demo/
├── config.py              # 全局配置（路径、区域、通道、日期）
├── build_cache_v3.py      # 构建 npz 缓存（多进程，KDTree 加速）
├── model.py               # U-Net 模型定义（双头：分类 + 回归）
├── train_unet.py          # U-Net 训练 + 评估
├── analyze_stratified.py  # 分层分析评估
├── compare_fy4a_fy4b.py   # FY4A/FY4B 对比
├── extrapolate_figure.py  # 外推可视化
├── diagnose_*.py          # 诊断脚本
├── test_*.py              # 通道组合测试
└── utils/
    ├── data_io.py         # AGRI/GPM HDF5 读取、重采样、归一化
    └── patches.py         # patch 提取（纯 numpy）
```

## 数据流

```
FY4A/FY4B AGRI HDF5 (FDI + GEO)      GPM IMERG HDF5
        |                                    |
  read_agri_disk()                    read_gpm_imerg()
  _derive_latlon() (GEOS 投影)       resample_gpm_to_region()
        |                                    |
  KDTree 重采样到 0.1° 格点           裁剪到目标区域
  compute_bt_diffs() → 4 IR + 3 BTD = 7 通道
  normalize() (z-score)
        |                                    |
        └──────── extract_patches() ─────────┘
                   (128×128 patch)
                         |
                   npz 缓存 (x, cls, reg)
                         |
                    ┌────┴────┐
                    ▼         ▼
              U-Net Encoder (4层)
              + Skip Connections
              + Bottleneck
                    │
              ┌─────┴─────┐
              ▼           ▼
         分类头 (cls)   回归头 (reg)
         有雨/无雨      log1p(mm/h)
              │           │
              └─────┬─────┘
                    ▼
         sigmoid(cls) > 阈值 → expm1(reg) → mm/h
         sigmoid(cls) ≤ 阈值 → 0
```

## 模型架构

轻量 U-Net，7 通道 IR+BTD 输入，双头输出：

| 组件 | 结构 | 输出尺寸 |
|------|------|----------|
| Encoder | 4 层 ConvBlock + MaxPool | 128→64→32→16→8 |
| Bottleneck | ConvBlock (256ch) | 8×8 |
| Decoder | 4 层 UpConv + Skip + ConvBlock | 16→32→64→128 |
| 分类头 | 1×1 Conv → sigmoid | (B, 1, 128, 128) 有雨概率 |
| 回归头 | 1×1 Conv → expm1 | (B, 1, 128, 128) mm/h |

## 目标区域

| 参数 | 值 |
|------|-----|
| 经度 | 80°E ~ 131.2°E |
| 纬度 | 10°S ~ 41.2°N |
| 格点 | 512×512 (0.1° 分辨率) |
| Patch | 128×128, 无重叠 |

## 通道配置

| 类型 | 通道 | 说明 |
|------|------|------|
| IR | CH09 (6.25μm) | 高层水汽 |
| IR | CH10 (7.10μm) | 中层水汽 |
| IR | CH12 (10.8μm) | 红外窗区 |
| IR | CH13 (12.0μm) | 红外窗区 |
| BTD | CH09-CH10 | 水汽吸收差 |
| BTD | CH09-CH13 | 深对流指标 |
| BTD | CH12-CH13 | split-window |

## 快速开始

### 1. 构建缓存

```bash
python build_cache_v3.py
```

读取 FY4A/FY4B AGRI + GPM IMERG HDF5 文件，提取 128×128 patch，保存为 npz 缓存。

### 2. 训练

```bash
python train_unet.py
```

U-Net 双头训练：
- **分类头**：BCEWithLogitsLoss，处理类别不平衡
- **回归头**：仅在雨像素上训练，log1p 目标

### 3. 评估

```bash
python analyze_stratified.py
```

## 评估指标

| 指标 | 目标 |
|------|------|
| CSI | ≥ 0.35 |
| POD | ≥ 0.60 |
| FAR | < 0.60 |
| R | ≥ 0.50 |

## 依赖

```bash
pip install numpy scipy h5py torch scikit-learn matplotlib
```
