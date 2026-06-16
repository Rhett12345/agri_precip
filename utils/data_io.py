"""
utils/data_io.py
FY4A AGRI disk HDF5 数据读取 + GPM IMERG 读取 + 区域裁剪 + 重采样
v4: KDTree 加速重采样，移除慢速 griddata，修复经纬度缓存
"""

import os
import re
import glob
import h5py
import numpy as np
from datetime import datetime
from scipy.spatial import cKDTree

from config import (
    REGION, GPM_RES, AGRI_CHANNELS, BT_DIFF_PAIRS,
    REGION_H, REGION_W, PRECIP_THRESHOLD,
)

# ============================================================
# 全局缓存
# ============================================================
_AGRI_LATLON_CACHE = {}   # geo_dir -> (lats, lons)
_GEO_TREE_CACHE = {}      # geo_file -> mapping array
_TARGET_PTS = None         # 重采样目标网格点


def _get_agri_latlon(geo_file: str) -> tuple[np.ndarray, np.ndarray]:
    """缓存 AGRI 经纬度（同一目录下所有文件共享同一 GEO 网格）。"""
    geo_dir = os.path.dirname(geo_file)
    if geo_dir not in _AGRI_LATLON_CACHE:
        _AGRI_LATLON_CACHE[geo_dir] = _derive_latlon(geo_file)
    return _AGRI_LATLON_CACHE[geo_dir]


def get_tree_and_mapping(geo_file: str) -> np.ndarray:
    """获取 KDTree 映射索引（同目录 GEO 文件共享同一棵树）。"""
    # 同一天的 GEO 文件共享相同的经纬度和投影参数，因此映射相同
    cache_key = os.path.dirname(geo_file)
    if cache_key in _GEO_TREE_CACHE:
        return _GEO_TREE_CACHE[cache_key]

    src_lat, src_lon = _get_agri_latlon(geo_file)
    valid = np.isfinite(src_lat) & np.isfinite(src_lon)
    src_pts = np.stack([src_lat[valid], src_lon[valid]], axis=1)
    valid_indices = np.where(valid.ravel())[0]

    tree = cKDTree(src_pts)

    global _TARGET_PTS
    if _TARGET_PTS is None:
        target_lats = np.arange(REGION['lat_min'] + GPM_RES / 2,
                                REGION['lat_max'], GPM_RES)
        target_lons = np.arange(REGION['lon_min'] + GPM_RES / 2,
                                REGION['lon_max'], GPM_RES)
        tgt_lat2d, tgt_lon2d = np.meshgrid(target_lats, target_lons, indexing='ij')
        _TARGET_PTS = np.stack([tgt_lat2d.ravel(), tgt_lon2d.ravel()], axis=1)

    _, idxs = tree.query(_TARGET_PTS)
    mapping = valid_indices[idxs]
    _GEO_TREE_CACHE[cache_key] = mapping
    return mapping


def fast_resample_from_mapping(agri_data: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """用预计算 KDTree 映射快速重采样。agri_data: (C, H, W) -> (C, REGION_H, REGION_W)。"""
    C, H, W = agri_data.shape
    src_flat = agri_data.reshape(C, -1)
    return src_flat[:, mapping].reshape(C, REGION_H, REGION_W)


# ============================================================
# 通用工具
# ============================================================

def parse_fy4_datetime(filepath: str) -> datetime | None:
    """从FY4 AGRI文件名中解析时间。"""
    basename = os.path.basename(filepath)
    m = re.search(r'(\d{14})_\d{14}', basename)
    if m:
        return datetime.strptime(m.group(1), '%Y%m%d%H%M%S')
    return None


def parse_gpm_datetime(filepath: str) -> datetime | None:
    """从GPM IMERG V07B文件名解析中心时间。"""
    basename = os.path.basename(filepath)
    m = re.search(r'(\d{8})-S(\d{6})-E(\d{6})', basename)
    if m:
        date_str = m.group(1)
        dt_start = datetime.strptime(date_str + m.group(2), '%Y%m%d%H%M%S')
        dt_end   = datetime.strptime(date_str + m.group(3), '%Y%m%d%H%M%S')
        return dt_start + (dt_end - dt_start) / 2
    return None


def _attr_scalar(obj, key: str, default=None):
    """从 h5py 对象属性中读取标量值（处理 array/bytes）。"""
    v = obj.attrs.get(key, default)
    if v is None:
        return default
    if isinstance(v, np.ndarray):
        v = v.reshape(-1)[0] if v.size else default
    if isinstance(v, (bytes, bytearray)):
        try:
            v = v.decode()
        except Exception:
            return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _dataset_scaled(ds: h5py.Dataset) -> np.ndarray:
    """读取数据集，处理 FillValue + Scale/Offset。"""
    arr = ds[()].astype(np.float64)
    fv = _attr_scalar(ds, "FillValue")
    if fv is not None:
        arr[arr == fv] = np.nan
    slope  = _attr_scalar(ds, "Slope", 1.0)
    inter  = _attr_scalar(ds, "Intercept", 0.0)
    return arr * slope + inter


def _wrap_lon(lon: np.ndarray) -> np.ndarray:
    """将经度包裹到 [-180, 180]。"""
    return (((lon + 180.0) % 360.0) - 180.0).astype(np.float32)


def _find_geo_file(fdi_path: str) -> str | None:
    """根据 FDI 文件路径找到对应的 GEO 文件。"""
    fdi_dir = os.path.dirname(fdi_path)
    fdi_name = os.path.basename(fdi_path)
    geo_name = fdi_name.replace("_FDI-_", "_GEO-_")
    geo_path = os.path.join(fdi_dir, geo_name)
    if os.path.exists(geo_path):
        return geo_path
    return None


# ============================================================
# AGRI 经纬度计算（GEOS 投影）
# ============================================================

def _derive_latlon(geo_file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    从 GEO 文件的 ColumnNumber/LineNumber + 轨道参数反算经纬度。
    使用严格的 GEOS 投影公式（参考 FY4A AGRI 官方文档）。
    """
    with h5py.File(geo_file, 'r') as gf:
        line_ds = gf.get("Navigation/LineNumber") or gf.get("LineNumber")
        col_ds  = gf.get("Navigation/ColumnNumber") or gf.get("ColumnNumber")
        if line_ds is None or col_ds is None:
            raise KeyError("GEO 文件缺少 LineNumber/ColumnNumber")
        line = _dataset_scaled(line_ds)
        col  = _dataset_scaled(col_ds)

        lon0_deg = _attr_scalar(gf, "NOMCenterLon")
        sat_h    = _attr_scalar(gf, "NOMSatHeight")
        ea_attr  = _attr_scalar(gf, "dEA")
        flat_inv = _attr_scalar(gf, "dObRecFlat")
        samp_ang = _attr_scalar(gf, "dSamplingAngle")
        step_ang = _attr_scalar(gf, "dSteppingAngle")

        if None in [lon0_deg, sat_h, ea_attr, flat_inv, samp_ang, step_ang]:
            raise KeyError("GEO 文件缺少轨道参数")

        ea_km    = ea_attr / 1000.0 if ea_attr > 1e5 else ea_attr
        sat_h_km = sat_h / 1000.0 if sat_h > 1e5 else sat_h
        H_sat    = sat_h_km + ea_km if sat_h_km < 40000 else sat_h_km
        eb_km    = ea_km * (1.0 - 1.0 / flat_inv)

        line[~np.isfinite(line) | (line < 0)] = np.nan
        col[~np.isfinite(col) | (col < 0)]   = np.nan

        begin_pixel = _attr_scalar(gf, "Begin Pixel Number", 0.0)
        end_pixel   = _attr_scalar(gf, "End Pixel Number", 2747.0)
        begin_line  = _attr_scalar(gf, "Begin Line Number", 0.0)
        end_line    = _attr_scalar(gf, "End Line Number", 2747.0)
        coff = (begin_pixel + end_pixel) / 2.0
        loff = (begin_line + end_line) / 2.0

        H, W = line.shape
        mid_row, mid_col = H // 2, W // 2
        col_vals  = col[mid_row, :][np.isfinite(col[mid_row, :])]
        line_vals = line[:, mid_col][np.isfinite(line[:, mid_col])]
        col_step  = float(np.nanmedian(np.diff(np.unique(col_vals)))) if len(col_vals) > 1 else -1.0
        line_step = float(np.nanmedian(np.diff(np.unique(line_vals)))) if len(line_vals) > 1 else -1.0

        x_pix = samp_ang * 1e-6 / (col_step if abs(col_step) > 1.5 else 1.0)
        y_pix = step_ang * 1e-6 / (line_step if abs(line_step) > 1.5 else 1.0)

        x = (col - coff) * x_pix
        y = (line - loff) * y_pix
        lon0 = np.deg2rad(lon0_deg)

        ea2, eb2 = ea_km**2, eb_km**2
        cosx, sinx = np.cos(x), np.sin(x)
        cosy, siny = np.cos(y), np.sin(y)
        a = sinx**2 + cosx**2 * (cosy**2 + (ea2 / eb2) * siny**2)
        b = -2.0 * H_sat * cosx * cosy
        c = H_sat**2 - ea2
        disc = b**2 - 4.0 * a * c

        lat = np.full(line.shape, np.nan, np.float64)
        lon = np.full(line.shape, np.nan, np.float64)
        valid = np.isfinite(disc) & (disc >= 0.0)
        if valid.any():
            sd  = np.sqrt(disc[valid])
            sn  = (-b[valid] - sd) / (2.0 * a[valid])
            s1  = H_sat - sn * cosx[valid] * cosy[valid]
            s2  = sn * sinx[valid] * cosy[valid]
            s3  = -sn * siny[valid]
            sxy = np.sqrt(s1**2 + s2**2)
            lat[valid] = np.rad2deg(np.arctan((ea2 / eb2) * s3 / sxy))
            lon[valid] = np.rad2deg(np.arctan2(s2, s1) + lon0)

        bad = (lat < -90) | (lat > 90)
        lat[bad] = np.nan
        lon[bad] = np.nan

    return lat.astype(np.float32), _wrap_lon(lon)


# ============================================================
# FY4 AGRI 读取
# ============================================================

def _lut_calibrate(raw: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """使用 CAL LUT 将 DN 值转为物理量。"""
    raw_i = raw.astype(np.int64)
    out = np.full(raw.shape, np.nan, dtype=np.float32)
    ok = (raw_i >= 0) & (raw_i < len(lut))
    out[ok] = lut[raw_i[ok]].astype(np.float32)
    out[(raw_i >= 65534) | (raw_i < 0)] = np.nan
    return out


def read_agri_disk(filepath: str, channels: list = None,
                   rdcc_nbytes: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 FY4A AGRI disk FDI + GEO 文件对。
    返回:
        data  : (C, H, W) float32，LUT 标定后的亮温/反射率
        lons  : (H, W) 经度
        lats  : (H, W) 纬度
    rdcc_nbytes: HDF5 chunk cache 大小（字节），0 表示使用默认值。
    """
    if channels is None:
        channels = AGRI_CHANNELS

    geo_file = _find_geo_file(filepath)
    if geo_file is None:
        raise FileNotFoundError(f"找不到对应的 GEO 文件: {filepath}")

    # 使用缓存的经纬度（修复：之前直接调用 _derive_latlon 导致重复计算）
    lats, lons = _get_agri_latlon(geo_file)

    open_kwargs = {'rdcc_nbytes': rdcc_nbytes} if rdcc_nbytes > 0 else {}
    with h5py.File(filepath, 'r', **open_kwargs) as f:
        bands = []
        for ch_idx in channels:
            ch_num = ch_idx + 1
            raw_key = f"NOMChannel{ch_num:02d}"
            cal_key = f"CALChannel{ch_num:02d}"

            raw = f[raw_key][()].astype(np.float32)
            if cal_key in f:
                lut = f[cal_key][()].astype(np.float32)
                bt = _lut_calibrate(raw, lut)
            else:
                bt = raw
                bt[bt > 60000] = np.nan
            bands.append(bt)

    data = np.stack(bands, axis=0).astype(np.float32)
    return data, lons, lats


# ============================================================
# GPM 读取
# ============================================================

def read_gpm_imerg(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 GPM IMERG V07B HDF5 文件。
    返回:
        precip : (H, W) float32，mm/h
        lons   : (W,) float32
        lats   : (H,) float32
    """
    with h5py.File(filepath, 'r') as f:
        grid = f["Grid"]
        precip = grid["precipitation"][()].astype(np.float32)
        lats   = grid["lat"][()].astype(np.float32)
        lons   = grid["lon"][()].astype(np.float32)

    if precip.ndim == 3:
        precip = precip[0]
    if precip.shape == (3600, 1800):
        precip = precip.T

    precip = precip.copy()
    precip[precip < -9000] = np.nan
    return precip, lons, lats


# ============================================================
# 归一化
# ============================================================

_IR_MEAN = np.array([245.14, 257.90, 279.05, 278.06], dtype=np.float32)
_IR_STD  = np.array([  6.03,   7.16,  10.82,  10.91], dtype=np.float32)
_BTD_MEAN = np.array([-12.76, -32.92,   0.99], dtype=np.float32)
_BTD_STD  = np.array([  3.50,   5.50,   1.50], dtype=np.float32)
CHANNEL_MEAN = np.concatenate([_IR_MEAN, _BTD_MEAN])
CHANNEL_STD  = np.concatenate([_IR_STD,  _BTD_STD])


# ============================================================
# 区域裁剪 + 重采样
# ============================================================

def resample_gpm_to_region(
    gpm_data: np.ndarray,
    gpm_lons: np.ndarray,
    gpm_lats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    裁剪 GPM 数据到目标区域（快速切片）。
    返回:
        label_cls  : (REGION_H, REGION_W) 0/1 二分类标签
        label_reg  : (REGION_H, REGION_W) float32 原始降水量 (mm/h)
    """
    target_lons = np.arange(REGION['lon_min'] + GPM_RES / 2,
                            REGION['lon_max'], GPM_RES)
    target_lats = np.arange(REGION['lat_min'] + GPM_RES / 2,
                            REGION['lat_max'], GPM_RES)

    if gpm_lats[0] > gpm_lats[-1]:
        gpm_lats = gpm_lats[::-1]
        gpm_data = gpm_data[::-1, :]

    lat_idx = np.searchsorted(gpm_lats, target_lats).clip(0, len(gpm_lats) - 1)
    lon_idx = np.searchsorted(gpm_lons, target_lons).clip(0, len(gpm_lons) - 1)

    precip_region = gpm_data[np.ix_(lat_idx, lon_idx)].astype(np.float32)
    precip_region = np.where(np.isfinite(precip_region), precip_region, 0.0)
    label_cls = (precip_region >= PRECIP_THRESHOLD).astype(np.int64)
    label_reg = precip_region.astype(np.float32)
    return label_cls, label_reg


def compute_bt_diffs(ir_data: np.ndarray) -> np.ndarray:
    """从 IR 通道数据计算亮温差通道。ir_data: (N_IR, H, W) -> (N_BTD, H, W)。"""
    btd_channels = []
    for ch_a, ch_b in BT_DIFF_PAIRS:
        idx_a = AGRI_CHANNELS.index(ch_a)
        idx_b = AGRI_CHANNELS.index(ch_b)
        btd_channels.append(ir_data[idx_a] - ir_data[idx_b])
    return np.stack(btd_channels, axis=0).astype(np.float32)


def normalize(data: np.ndarray) -> np.ndarray:
    """(C, H, W) -> z-score 标准化，NaN 用通道均值填充。"""
    data = data.copy()
    n_ch = data.shape[0]
    n = min(n_ch, len(CHANNEL_MEAN))
    for c in range(n):
        data[c] = np.where(np.isfinite(data[c]), data[c], CHANNEL_MEAN[c])
    data[:n] = (data[:n] - CHANNEL_MEAN[:n, None, None]) / (CHANNEL_STD[:n, None, None] + 1e-6)
    return data


# ============================================================
# 文件列表工具
# ============================================================

def list_agri_files(base_dir: str) -> list[str]:
    """列出指定目录下的 AGRI FDI HDF 文件。"""
    pattern = os.path.join(base_dir, '**', '*_FDI-_*.HDF')
    return sorted(glob.glob(pattern, recursive=True))


def find_matching_gpm(agri_time: datetime, gpm_dir: str,
                      max_minutes: int = 18) -> str | None:
    """根据 AGRI 时间找最近的 GPM IMERG 半小时文件。"""
    date_str = agri_time.strftime('%Y%m%d')
    day_dir = os.path.join(gpm_dir, date_str)
    if not os.path.isdir(day_dir):
        return None

    all_gpm = sorted(glob.glob(os.path.join(day_dir, '*.HDF5')))
    if not all_gpm:
        return None

    best, best_diff = None, float('inf')
    for f in all_gpm:
        t = parse_gpm_datetime(f)
        if t is None:
            continue
        diff = abs((t - agri_time).total_seconds() / 60)
        if diff < best_diff:
            best_diff = diff
            best = f

    return best if best_diff <= max_minutes else None
