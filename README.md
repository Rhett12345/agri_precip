# FY4A AGRI 降水反演 — XGBoost 双头模型

基于 FY4A AGRI 红外亮温数据，使用 XGBoost 双头模型（分类 + 回归）进行像素级降水反演，以 GPM IMERG 作为标签。

## 项目结构

```
precip_demo/
├── config.py              # 全局配置（路径、区域、通道、日期）
├── build_cache_v3.py      # 构建 npz 缓存（多进程，KDTree 加速）
├── train_xgb.py           # XGB 双头训练 + 评估
└── utils/
    ├── data_io.py         # AGRI/GPM HDF5 读取、重采样、归一化
    └── patches.py         # patch 提取（纯 numpy）
```

## 数据流

```
FY4A AGRI HDF5 (FDI + GEO)          GPM IMERG HDF5
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
                  ┌──────┴──────┐
                  ▼              ▼
           XGBClassifier    XGBRegressor
           (有雨/无雨)      (降水量, 仅雨像素)
                  │              │
                  └──────┬──────┘
                         ▼
              prob > 阈值 → 回归预测 → expm1 → mm/h
              prob ≤ 阈值 → 预测 = 0
```

## 目标区域

| 参数 | 值 |
|------|-----|
| 经度 | 80°E ~ 131.2°E |
| 纬度 | 10°S ~ 41.2°N |
| 格点 | 512×512 (0.1° 分辨率) |
| Patch | 128×128, 无重叠, 每场景 16 个 |

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

读取 FY4A AGRI + GPM IMERG HDF5 文件，提取 128×128 patch，保存为 npz 缓存到 `workdir/cache/`。

### 2. 训练

```bash
python train_xgb.py
```

双头训练：
- **分类头**：XGBClassifier，AUCPR 优化，处理类别不平衡
- **回归头**：XGBRegressor，仅在雨像素上训练，log1p 目标，加权 (1 + 6×y_true)

输出模型到 `workdir/models/xgb_cls.json` 和 `xgb_reg.json`。

## 模型架构

| 组件 | 算法 | 输入 | 输出 |
|------|------|------|------|
| 分类头 | XGBClassifier | 7 维特征/像素 | 有雨概率 [0,1] |
| 回归头 | XGBRegressor | 7 维特征/像素 (仅雨) | log1p(mm/h) |

## 评估指标

| 指标 | 目标 |
|------|------|
| CSI | ≥ 0.35 |
| POD | ≥ 0.60 |
| FAR | < 0.60 |
| R | ≥ 0.50 |

## 依赖

```bash
pip install numpy scipy h5py xgboost scikit-learn
```
