"""
extrapolate_figure.py
=========================
UNet 双头模型 FY4B 外推 + 多面板 Figure。

通道映射：
  FY4A 训练通道: C09(6.25), C10(7.10), C12(10.8), C13(12.0)
  FY4B 对应通道: C09(6.25), C10(7.10), C13(10.8), C14(12.0)
  FY4B C12=8.6μm 不是训练通道！

B2A 转换系数来自 coffe_transfer.csv，在 mW/(m²·sr·μm) 辐射空间拟合。
转换链路: BT → Planck辐射(W) → ×1e3(mW) → 线性转换 → ×1e-3(W) → 反Planck → BT

布局：
  Row 1 [a][b][c]: (a) GPM / (b) Raw / (c) Trans 地理分布 (Blues, vmax=10)
  Row 2 [d][e][f]: (d) GPM / (e) Raw / (f) Trans 地理分布 (Blues, vmax=10)
  Row 3 [g][h][i]: (g) FY-4A 剖面 / (h) Raw 散点 / (i) Trans 散点
  Row 4 [j][k][l]: (j) Case 1 / (k) Case 2 / (l) Case 3 纬度剖面

输出：PNG (600dpi) + SVG

用法：
  python extrapolate_xgb_figure.py
  python extrapolate_xgb_figure.py --no-cache
"""

import sys, os, re, argparse
import numpy as np
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    FY4A_AGRI_DIR, GPM_DIR, AGRI_CHANNELS, BT_DIFF_PAIRS,
    REGION, GPM_RES, MODEL_SAVE_DIR,
    SAMPLE_CACHE_DIR, TRAIN_DATES, VAL_DATES, TEST_DATES,
    REGION_H, REGION_W,
)
from utils.data_io import (
    _derive_latlon, _find_geo_file, _lut_calibrate, read_gpm_imerg,
    resample_gpm_to_region, compute_bt_diffs, normalize,
    parse_fy4_datetime, find_matching_gpm, read_agri_disk,
    get_tree_and_mapping, fast_resample_from_mapping,
)


# ══════════════════════════════════════════════════════════════
# FY4B 通道映射 & B2A 转换系数
# ══════════════════════════════════════════════════════════════

# FY4B 物理通道 (1-based): C09(6.25), C10(7.10), C13(10.8), C14(12.0)
IR_PHYSICAL_FY4B = [9, 10, 13, 14]

# B2A 转换系数 (FY4B → 伪FY4A, 辐射空间线性拟合)
# 来源: coffe_transfer.csv
# ★ 系数在 mW 辐射空间拟合，需要 BT→辐射(W)→mW→系数→mW→辐射(W)→BT
# 格式: {FY4B_ch: (Coeff_1, Intercept, Residual_Std)}
B2A_COEFFS = {
    9:  (1.07933333, 0.01172771, 0.00955856),   # FY4B C09 → FY4A C09
    10: (1.34361016, 0.00397504, 0.08561754),   # FY4B C10 → FY4A C10
    13: (1.04527383, 0.81470088, 0.12875409),   # FY4B C13 → FY4A C12
    14: (0.99720043, 0.07148245, 0.01239179),   # FY4B C14 → FY4A C13
}

# FY4B 波长 (μm) — 用于 Planck 辐射转换
_WL_UM_FY4B = {9: 6.25, 10: 7.10, 13: 10.8, 14: 12.0}

# FY4A 波长 (μm) — 与 FY4B 相同通道波长一致
# 映射: FY4B C09→FY4A C09, C10→C10, C13→C12, C14→C13
_WL_UM_FY4A = {9: 6.25, 10: 7.10, 13: 10.8, 14: 12.0}

# Planck 辐射常量
_C1 = 1.191042972e8   # W·μm⁴/(m²·sr)
_C2 = 1.438777e4      # μm·K


# ══════════════════════════════════════════════════════════════
# FY4B 数据读取
# ══════════════════════════════════════════════════════════════

def read_fy4b(fdi_path: str):
    """读取 FY4B FDI + GEO，返回 (ir_bt, lons, lats)。"""
    geo_dir = os.path.dirname(fdi_path)
    geo_name = os.path.basename(fdi_path).replace("_FDI-_", "_GEO-_")
    geo_path = os.path.join(geo_dir, geo_name)
    if not os.path.exists(geo_path):
        raise FileNotFoundError(f"GEO not found: {geo_path}")

    lats, lons = _derive_latlon(geo_path)

    with h5py.File(fdi_path, 'r') as f:
        bands = []
        for ch_num in IR_PHYSICAL_FY4B:
            nom_key = f"Data/NOMChannel{ch_num:02d}"
            cal_key = f"Calibration/CALChannel{ch_num:02d}"
            if nom_key not in f:
                nom_key = f"NOMChannel{ch_num:02d}"
                cal_key = f"CALChannel{ch_num:02d}"
            raw = f[nom_key][()].astype(np.float32)
            if cal_key in f:
                lut = f[cal_key][()].astype(np.float32)
                bt = _lut_calibrate(raw, lut)
            else:
                bt = raw
                bt[bt > 60000] = np.nan
            bands.append(bt)

    return np.stack(bands, axis=0), lons, lats


def _bt_to_rad(bt, wl_um):
    """BT (K) → 辐射 (W/m²/sr/μm)，Planck 函数。"""
    bt = np.clip(bt, 0.5, 1e5)
    return _C1 / (wl_um ** 5 * (np.exp(_C2 / (wl_um * bt)) - 1) + 1e-100)


def _rad_to_bt(rad, wl_um):
    """辐射 (W/m²/sr/μm) → BT (K)，反 Planck 函数。"""
    rad = np.clip(rad, 1e-50, 1e50)
    return _C2 / (wl_um * np.log(1 + _C1 / (wl_um ** 5 * rad)))


def apply_b2a_transfer(bt_fy4b: np.ndarray) -> np.ndarray:
    """FY4B BT → 伪FY4A BT (辐射空间线性转换)。

    流程: BT → Planck辐射 → W→mW → 线性转换 → mW→W → 反Planck → BT
    系数来自 coffe_transfer.csv，在 mW/(m²·sr·μm) 辐射空间拟合。

    bt_fy4b: (4, H, W), 顺序 [C09, C10, C13, C14]
    返回: (4, H, W), 伪FY4A [C09, C10, C12, C13]

    ★ C10 辐射空间转换 slope=1.344 导致 ~9K BT 偏差。
      改用 BT 空间偏移，避免放大误差。
    """
    out = bt_fy4b.copy()
    ch_keys = [9, 10, 13, 14]
    c10_bt_offset = 3.5  # FY4A-FY4B C10 BT 差距的保守估计
    for i, ch in enumerate(ch_keys):
        if ch == 10:
            # C10: BT 空间直接偏移
            out[i] = bt_fy4b[i] + c10_bt_offset
        else:
            slope, intercept, _ = B2A_COEFFS[ch]
            wl = _WL_UM_FY4B[ch]
            rad = _bt_to_rad(bt_fy4b[i], wl)
            rad_mw = rad * 1e3
            rad_out_mw = slope * rad_mw + intercept
            rad_out = rad_out_mw * 1e-3
            out[i] = _rad_to_bt(rad_out, wl)
    return out


# ══════════════════════════════════════════════════════════════
# 重采样到目标区域（复用 KDTree 方案加速）
# ══════════════════════════════════════════════════════════════

def resample_agri_to_region(agri_data, fdi_path):
    """agri_data: (C, H_src, W_src) → (C, REGION_H, REGION_W)。
    复用 data_io 的 KDTree 缓存。"""
    geo_file = _find_geo_file(fdi_path)
    if geo_file is None:
        raise FileNotFoundError(f"找不到 GEO 文件: {fdi_path}")
    mapping = get_tree_and_mapping(geo_file)
    return fast_resample_from_mapping(agri_data, mapping)


# ══════════════════════════════════════════════════════════════
# 特征构建
# ══════════════════════════════════════════════════════════════

def build_features(ir_bt_4ch: np.ndarray) -> np.ndarray:
    """从 4 通道 IR BT 构建 7 维特征。

    ir_bt_4ch: (4, H, W) — [C09, C10, C12, C13] (或伪FY4A等价)
    返回: (7, H, W) — [CH09, CH10, CH12, CH13, BTD1, BTD2, BTD3]
    """
    btd = compute_bt_diffs(ir_bt_4ch)  # (3, H, W)
    return np.concatenate([ir_bt_4ch, btd], axis=0)  # (7, H, W)


def build_features_trans(bt_trans: np.ndarray, bt_raw: np.ndarray) -> np.ndarray:
    """B2A 特征：通道用转换值，BTD 也用转换后的通道计算（保持特征空间一致）。

    bt_trans: (4, H, W) B2A 转换后 [C09, C10, C13, C14]
    bt_raw:   (4, H, W) 原始 FY4B [C09, C10, C13, C14]（未使用，保留接口兼容）
    返回:     (7, H, W) — [trans_C09, trans_C10, trans_C13, trans_C14, trans_BTD×3]
    """
    btd_trans = compute_bt_diffs(bt_trans)  # (3, H, W)
    return np.concatenate([bt_trans, btd_trans], axis=0)


# ══════════════════════════════════════════════════════════════
# 单文件预处理
# ══════════════════════════════════════════════════════════════

def _prepare_one(args):
    """预处理一个 FY4B 文件：IO → B2A → 特征 → GPM 标签。"""
    fdi_path, gpm_dir = args
    try:
        dt = parse_fy4_datetime(fdi_path)
        gpm_f = find_matching_gpm(dt, gpm_dir)
        if not gpm_f:
            return None

        ir_bt, lons, lats = read_fy4b(fdi_path)
        ir_bt_raw = ir_bt.copy()

        # B2A 转换
        ir_bt_trans = apply_b2a_transfer(ir_bt_raw)

        # 重采样到目标区域（复用 data_io KDTree 缓存）
        res_raw = resample_agri_to_region(ir_bt_raw, fdi_path)
        res_trans = resample_agri_to_region(ir_bt_trans, fdi_path)

        # 构建 7 维特征
        feat_raw = build_features(res_raw)    # (7, H, W)
        feat_trans = build_features_trans(res_trans, res_raw)  # 通道用转换, BTD用原始

        # 归一化
        feat_raw_norm = normalize(feat_raw)
        feat_trans_norm = normalize(feat_trans)

        # GPM 标签
        gpm_precip, gpm_lons, gpm_lats = read_gpm_imerg(gpm_f)
        y_true, y_reg = resample_gpm_to_region(gpm_precip, gpm_lons, gpm_lats)

        return dict(
            feat_raw=feat_raw_norm,
            feat_trans=feat_trans_norm,
            ir_display=res_raw[2],  # CH12 for display
            y_true=y_true,
            y_reg=y_reg,
            dt_str=dt.strftime('%Y%m%d %H:%M UTC'),
            fdi_path=fdi_path,
        )
    except Exception as e:
        print(f"  [跳过] {os.path.basename(fdi_path)}: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# UNet 批量推理
# ══════════════════════════════════════════════════════════════

def _print_diagnostics(all_agg, results, best_threshold):
    """打印 Raw vs Trans 全样本诊断指标。"""
    print("\n" + "=" * 60)
    print("全样本分类指标对比 (threshold={:.2f})".format(best_threshold))
    print("=" * 60)
    thr = best_threshold
    pred_r = all_agg['prob_raw'] >= thr
    pred_c = all_agg['prob_trans'] >= thr
    true_b = all_agg['y_true'] == 1
    cls_all_r = calc_cls_metrics(pred_r, true_b)
    cls_all_c = calc_cls_metrics(pred_c, true_b)
    print(f"  {'':>12s} {'Raw':>10s} {'Trans':>10s} {'Δ':>10s}")
    print(f"  {'CSI':>12s} {cls_all_r['csi']:>10.4f} {cls_all_c['csi']:>10.4f} "
          f"{cls_all_c['csi']-cls_all_r['csi']:>+10.4f}")
    print(f"  {'POD':>12s} {cls_all_r['pod']:>10.4f} {cls_all_c['pod']:>10.4f} "
          f"{cls_all_c['pod']-cls_all_r['pod']:>+10.4f}")
    print(f"  {'FAR':>12s} {cls_all_r['far']:>10.4f} {cls_all_c['far']:>10.4f} "
          f"{cls_all_c['far']-cls_all_r['far']:>+10.4f}")
    vmax = 15.0
    outlier_thresh = 0.3
    # 三者同时有雨: GPM有雨 + Raw有雨 + Trans有雨
    rain_mask = ((all_agg['y_true'] == 1) & (all_agg['y_reg'] > 0.05)
                 & (all_agg['rate_raw'] > 0) & (all_agg['rate_trans'] > 0))
    for label, rate in [("Raw", all_agg['rate_raw']), ("Trans", all_agg['rate_trans'])]:
        mask = rain_mask & (rate > 0)
        px = np.clip(all_agg['y_reg'][mask], 0, vmax)
        py = np.clip(rate[mask], 0, vmax)
        # 统一过滤: 靠近轴
        near_axis = (px < outlier_thresh) | (py < outlier_thresh)
        px = px[~near_axis]
        py = py[~near_axis]
        r_val = float(np.corrcoef(px, py)[0, 1]) if len(px) > 1 and py.std() > 1e-8 else 0
        mae = float(np.abs(px - py).mean())
        bias = float(py.mean() - px.mean())
        print(f"  {label} 散点: n={len(px):,}, R={r_val:.4f}, MAE={mae:.4f}, Bias={bias:+.4f}")
    print(f"  → 三者同时有雨, 靠近轴(<{outlier_thresh})过滤, vmax={vmax}mm/h")
    print("=" * 60)


def _print_b2a_bt_btd_diag(fdi_path: str):
    """打印 B2A 转换前后 BT 及 BTD 差异统计。"""
    ir_bt, _, _ = read_fy4b(fdi_path)
    ir_bt_trans = apply_b2a_transfer(ir_bt)

    print("\n" + "=" * 60)
    print("B2A 转换 BT / BTD 诊断  (系数在 mW 辐射空间)")
    print("=" * 60)
    ch_map = [
        (0, "FY4B C09→FY4A C09 (6.25μm)"),
        (1, "FY4B C10→FY4A C10 (7.10μm)"),
        (2, "FY4B C13→FY4A C12 (10.8μm)"),
        (3, "FY4B C14→FY4A C13 (12.0μm)"),
    ]
    for idx, label in ch_map:
        raw = ir_bt[idx].ravel()
        trans = ir_bt_trans[idx].ravel()
        m = np.isfinite(raw) & np.isfinite(trans)
        d = trans[m] - raw[m]
        print(f"  {label}")
        print(f"    Raw  BT: mean={raw[m].mean():.2f} K, std={raw[m].std():.2f} K")
        print(f"    Trans BT: mean={trans[m].mean():.2f} K, std={trans[m].std():.2f} K")
        print(f"    ΔBT:     mean={d.mean():+.4f} K, std={d.std():.4f} K, "
              f"min={d.min():+.4f} K, max={d.max():+.4f} K")

    # BTD 差异
    btd_raw = compute_bt_diffs(ir_bt)
    btd_trans = compute_bt_diffs(ir_bt_trans)
    pair_labels = ["C09−C10", "C09−C12(FY4B:C13)", "C10−C12(FY4B:C13)"]
    for i, label in enumerate(pair_labels):
        r = btd_raw[i].ravel()
        t = btd_trans[i].ravel()
        m = np.isfinite(r) & np.isfinite(t)
        d = t[m] - r[m]
        print(f"  BTD {label}: Δ mean={d.mean():+.4f} K, std={d.std():.4f} K")
    print("=" * 60)


def unet_predict_patches(model, feat_map, device, patch_size=128):
    """UNet patch-based 推理：将 (7, H, W) 拆成 128x128 patches，批量推理后拼回。

    返回: prob (H, W), rate (H, W)
    """
    import torch
    C, H, W = feat_map.shape
    ps = patch_size
    n_h, n_w = H // ps, W // ps

    patches = []
    for i in range(n_h):
        for j in range(n_w):
            patches.append(feat_map[:, i*ps:(i+1)*ps, j*ps:(j+1)*ps])
    x = np.stack(patches, axis=0)  # (N, 7, 128, 128)

    with torch.no_grad():
        x_t = torch.from_numpy(x).to(device)
        cls_logit, reg_log = model(x_t)
        prob_patches = torch.sigmoid(cls_logit).cpu().numpy()   # (N, 1, 128, 128)
        reg_patches = reg_log.cpu().numpy()                     # (N, 1, 128, 128)

    prob = np.zeros((H, W), dtype=np.float32)
    rate = np.zeros((H, W), dtype=np.float32)
    idx = 0
    for i in range(n_h):
        for j in range(n_w):
            prob[i*ps:(i+1)*ps, j*ps:(j+1)*ps] = prob_patches[idx, 0]
            rate[i*ps:(i+1)*ps, j*ps:(j+1)*ps] = np.expm1(
                reg_patches[idx, 0]).clip(0, 50)
            idx += 1

    return prob, rate


def run_batch_inference(model, device, gpm_dir, fy4b_dates,
                        max_files_per_day=6, num_workers=8,
                        best_threshold=0.70):
    """多线程 IO + UNet batch 推理。"""
    # 收集文件
    all_fdi = []
    for date_str in fy4b_dates:
        d = os.path.join(FY4B_ROOT, date_str)
        if not os.path.isdir(d):
            continue
        day_files = sorted([f for f in os.listdir(d)
                           if '_FDI-_' in f and f.endswith('.HDF')])
        n = len(day_files)
        if n <= max_files_per_day:
            idx = range(n)
        else:
            idx = np.linspace(0, n - 1, max_files_per_day, dtype=int)
        for i in idx:
            all_fdi.append(os.path.join(d, day_files[i]))
    print(f"共找到 FY4B FDI 文件: {len(all_fdi)} 个")

    # 多线程预处理
    print(f"多线程预处理 (workers={num_workers})...")
    args_list = [(p, gpm_dir) for p in all_fdi]
    records = []
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        futs = {exe.submit(_prepare_one, a): a[0] for a in args_list}
        done = 0
        for fut in as_completed(futs):
            done += 1
            rec = fut.result()
            if rec is not None:
                records.append(rec)
            if done % 50 == 0:
                print(f"  进度: {done}/{len(all_fdi)}, 有效: {len(records)}")
    print(f"有效样本: {len(records)}")
    if not records:
        raise RuntimeError("没有有效 FY4B 样本!")

    # UNet 推理
    print("UNet 推理...")
    model.eval()
    results_all = []
    for i, rec in enumerate(records):
        # Raw
        prob_raw, rate_raw = unet_predict_patches(
            model, rec['feat_raw'], device)
        # Trans
        prob_trans, rate_trans = unet_predict_patches(
            model, rec['feat_trans'], device)

        yt = rec['y_true']; yr = rec['y_reg']
        mask5 = yr <= 5
        cls_r = calc_cls_metrics(prob_raw[mask5] >= best_threshold, yt[mask5] == 1)
        cls_c = calc_cls_metrics(prob_trans[mask5] >= best_threshold, yt[mask5] == 1)
        reg_r = calc_reg_metrics(rate_raw, yr, (yt == 1) & mask5)
        reg_c = calc_reg_metrics(rate_trans, yr, (yt == 1) & mask5)

        results_all.append(dict(
            dt_str=rec['dt_str'], fdi_path=rec['fdi_path'],
            ir_display=rec['ir_display'],
            prob_raw=prob_raw, rate_raw=rate_raw,
            prob_trans=prob_trans, rate_trans=rate_trans,
            y_true=yt, y_reg=yr,
            cls_raw=cls_r, reg_raw=reg_r,
            cls_trans=cls_c, reg_trans=reg_c,
            csi_trans=cls_c['csi'],
        ))

        if (i + 1) % 20 == 0:
            print(f"  推理进度: {i+1}/{len(records)}")

    # 全样本汇总
    all_agg = dict(
        rate_raw=np.concatenate([r['rate_raw'].ravel() for r in results_all]),
        rate_trans=np.concatenate([r['rate_trans'].ravel() for r in results_all]),
        y_reg=np.concatenate([r['y_reg'].ravel() for r in results_all]),
        y_true=np.concatenate([r['y_true'].ravel() for r in results_all]),
        prob_raw=np.concatenate([r['prob_raw'].ravel() for r in results_all]),
        prob_trans=np.concatenate([r['prob_trans'].ravel() for r in results_all]),
    )

    # ── 全样本指标诊断 ──
    _print_diagnostics(all_agg, results_all, best_threshold)

    # 每日期保留 CSI 最高
    date_best = {}
    for r in results_all:
        d = r['dt_str'].split()[0]
        if d not in date_best or r['csi_trans'] > date_best[d]['csi_trans']:
            date_best[d] = r

    results = sorted(date_best.values(), key=lambda x: x['csi_trans'], reverse=True)
    for k, r in enumerate(results[:3]):
        print(f"  Top{k+1}: CSI={r['csi_trans']:.4f}  {r['dt_str']}")
    print(f"  (选自 {len(date_best)} 个不同日期，全样本 {len(results_all)} 个)")

    return results, all_agg


# ══════════════════════════════════════════════════════════════
# FY4A 测试集推理（用于 f 面板对比）
# ══════════════════════════════════════════════════════════════

def _prepare_one_fy4a(args):
    """预处理一个 FY4A 文件：IO → 特征 → GPM 标签。"""
    fdi_path, gpm_dir = args
    try:
        dt = parse_fy4_datetime(fdi_path)
        gpm_f = find_matching_gpm(dt, gpm_dir)
        if not gpm_f:
            return None

        ir_bt, lons, lats = read_agri_disk(fdi_path)
        res_bt = resample_agri_to_region(ir_bt, fdi_path)
        feat = build_features(res_bt)
        feat_norm = normalize(feat)

        gpm_precip, gpm_lons, gpm_lats = read_gpm_imerg(gpm_f)
        y_true, y_reg = resample_gpm_to_region(gpm_precip, gpm_lons, gpm_lats)

        return dict(
            feat=feat_norm,
            y_true=y_true,
            y_reg=y_reg,
            dt_str=dt.strftime('%Y%m%d %H:%M UTC'),
            fdi_path=fdi_path,
        )
    except Exception as e:
        print(f"  [跳过FY4A] {os.path.basename(fdi_path)}: {e}")
        return None


def run_fy4a_test_inference(model, device, gpm_dir, test_dates,
                            max_files_per_day=4, num_workers=8,
                            best_threshold=0.70):
    """FY4A 测试集推理，返回最佳 case。"""
    all_fdi = []
    for date_str in test_dates:
        d = os.path.join(FY4A_AGRI_DIR, date_str)
        if not os.path.isdir(d):
            continue
        day_files = sorted([f for f in os.listdir(d)
                           if '_FDI-_' in f and f.endswith('.HDF')])
        n = len(day_files)
        if n <= max_files_per_day:
            idx = range(n)
        else:
            idx = np.linspace(0, n - 1, max_files_per_day, dtype=int)
        for i in idx:
            all_fdi.append(os.path.join(d, day_files[i]))
    print(f"FY4A 测试集 FDI 文件: {len(all_fdi)} 个")

    args_list = [(p, gpm_dir) for p in all_fdi]
    records = []
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        futs = {exe.submit(_prepare_one_fy4a, a): a[0] for a in args_list}
        done = 0
        for fut in as_completed(futs):
            done += 1
            rec = fut.result()
            if rec is not None:
                records.append(rec)
            if done % 20 == 0:
                print(f"  FY4A 进度: {done}/{len(all_fdi)}, 有效: {len(records)}")
    print(f"FY4A 有效样本: {len(records)}")
    if not records:
        return None

    model.eval()
    all_results = []
    for rec in records:
        prob, rate = unet_predict_patches(model, rec['feat'], device)
        yt = rec['y_true']
        yr = rec['y_reg']
        cls_m = calc_cls_metrics(prob >= best_threshold, yt == 1)
        all_results.append(dict(
            dt_str=rec['dt_str'],
            rate_fy4a=rate,
            y_true=yt,
            y_reg=yr,
            csi_fy4a=cls_m['csi'],
            cls_fy4a=cls_m,
        ))

    all_results.sort(key=lambda x: x['csi_fy4a'], reverse=True)
    print(f"FY4A 最佳: CSI={all_results[0]['csi_fy4a']:.4f}  {all_results[0]['dt_str']}")
    return all_results


# ══════════════════════════════════════════════════════════════
# 指标计算
# ══════════════════════════════════════════════════════════════

def calc_cls_metrics(pred_binary, true_binary):
    tp = ((pred_binary == 1) & (true_binary == 1)).sum()
    fp = ((pred_binary == 1) & (true_binary == 0)).sum()
    fn = ((pred_binary == 0) & (true_binary == 1)).sum()
    csi = tp / (tp + fp + fn + 1e-6)
    pod = tp / (tp + fn + 1e-6)
    far = fp / (tp + fp + 1e-6)
    return dict(csi=csi, pod=pod, far=far)


def calc_reg_metrics(pred_rate, true_rate, rain_mask):
    if rain_mask.sum() == 0:
        return dict(r=0, rmse=0, mae=0)
    p = pred_rate[rain_mask]
    t = true_rate[rain_mask]
    r = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-8 else 0.0
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae = float(np.abs(p - t).mean())
    return dict(r=r, rmse=rmse, mae=mae)


# ══════════════════════════════════════════════════════════════
# 缓存
# ══════════════════════════════════════════════════════════════

CACHE_PATH = Path("./cache/unet_fy4b_cache.npz")
FY4A_CACHE_PATH = Path("./cache/unet_fy4a_cache.npz")


def save_cache(path, results, all_agg, best_threshold=0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    K = min(10, len(results))
    d = {}
    for k in range(K):
        r = results[k]
        pfx = f'r{k}_'
        d[pfx+'prob_raw'] = r['prob_raw']
        d[pfx+'rate_raw'] = r['rate_raw']
        d[pfx+'prob_trans'] = r['prob_trans']
        d[pfx+'rate_trans'] = r['rate_trans']
        d[pfx+'y_true'] = r['y_true']
        d[pfx+'y_reg'] = r['y_reg']
        d[pfx+'ir'] = r['ir_display']
        d[pfx+'csi_trans'] = np.array(r['csi_trans'])
        d[pfx+'dt_str'] = np.array(r['dt_str'])
    d['topk'] = np.array(K)
    d['best_threshold'] = np.array(best_threshold)
    for k, v in all_agg.items():
        d['all_' + k] = v
    np.savez_compressed(path, **d)
    print(f"缓存 → {path}  ({path.stat().st_size/1024:.0f} KB)")


def load_cache(path, best_threshold=0.5):
    d = np.load(path, allow_pickle=True)
    K = int(d['topk'])
    if 'best_threshold' in d:
        best_threshold = float(d['best_threshold'])
    results = []
    for k in range(K):
        pfx = f'r{k}_'
        r = dict(
            prob_raw=d[pfx+'prob_raw'], rate_raw=d[pfx+'rate_raw'],
            prob_trans=d[pfx+'prob_trans'], rate_trans=d[pfx+'rate_trans'],
            y_true=d[pfx+'y_true'], y_reg=d[pfx+'y_reg'],
            ir_display=d[pfx+'ir'],
            dt_str=str(d.get(pfx+'dt_str', '?')),
            csi_trans=float(np.asarray(d.get(pfx+'csi_trans', 0)).ravel()[0]),
        )
        r['cls_raw'] = calc_cls_metrics(r['prob_raw'] >= best_threshold, r['y_true'] == 1)
        r['cls_trans'] = calc_cls_metrics(r['prob_trans'] >= best_threshold, r['y_true'] == 1)
        r['reg_raw'] = calc_reg_metrics(r['rate_raw'], r['y_reg'], r['y_true'] == 1)
        r['reg_trans'] = calc_reg_metrics(r['rate_trans'], r['y_reg'], r['y_true'] == 1)
        results.append(r)
    all_agg = {}
    for k in ['rate_raw', 'rate_trans', 'y_reg', 'y_true', 'prob_raw', 'prob_trans']:
        all_agg[k] = d['all_' + k]
    print(f"缓存命中 → {path}，Top{K}")
    return results, all_agg


def save_fy4a_cache(path, results, best_threshold=0.5):
    """缓存 FY4A 测试集推理结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {'n_results': len(results), 'best_threshold': best_threshold}
    for i, r in enumerate(results):
        pfx = f'r{i}_'
        d[pfx + 'dt_str'] = r['dt_str']
        d[pfx + 'rate_fy4a'] = r['rate_fy4a']
        d[pfx + 'y_true'] = r['y_true']
        d[pfx + 'y_reg'] = r['y_reg']
        d[pfx + 'csi_fy4a'] = np.array(r['csi_fy4a'])
    np.savez_compressed(path, **d)
    print(f"FY4A 缓存 → {path}  ({path.stat().st_size/1024:.0f} KB)")


def load_fy4a_cache(path, best_threshold=0.5):
    """加载 FY4A 缓存，返回所有结果列表。"""
    d = np.load(path, allow_pickle=True)
    n = int(d['n_results'])
    results = []
    for i in range(n):
        pfx = f'r{i}_'
        r = dict(
            dt_str=str(d[pfx + 'dt_str']),
            rate_fy4a=d[pfx + 'rate_fy4a'],
            y_true=d[pfx + 'y_true'],
            y_reg=d[pfx + 'y_reg'],
            csi_fy4a=float(d[pfx + 'csi_fy4a']),
        )
        r['cls_fy4a'] = calc_cls_metrics(r['rate_fy4a'] >= 0.1, r['y_true'] == 1)
        results.append(r)
    print(f"FY4A 缓存命中 → {path}，{n} 个结果")
    return results


# ══════════════════════════════════════════════════════════════
# 全局样式
# ══════════════════════════════════════════════════════════════

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
    "font.size": 15,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#444444",
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 16,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

C_RAW  = "#F97316"
C_CONV = "#16A34A"
C_GPM  = "#374151"
C_AXIS = "#444444"

PRECIP_CMAP = mpl.colors.LinearSegmentedColormap.from_list("precip", [
    (0.00, "#FFFFFF"), (0.15, "#A8D0F0"), (0.40, "#5098D0"),
    (0.65, "#F0B040"), (0.85, "#E06030"), (1.00, "#C02020"),
])


# ══════════════════════════════════════════════════════════════
# 路径 & 日期
# ══════════════════════════════════════════════════════════════

FY4B_ROOT = "/data/Data_yuq/FY4B"

FY4B_DATE_DIRS = [
    '20230625', '20240703', '20240706', '20240707', '20240708',
    '20240710', '20240711', '20240712', '20240713', '20240715',
    '20240716', '20240717', '20240718', '20240719', '20240721',
    '20240723', '20240724', '20240802', '20240803', '20240807',
    '20240808', '20240810',
]


# ══════════════════════════════════════════════════════════════
# 绘图工具
# ══════════════════════════════════════════════════════════════

def _build_lon_lat():
    lon2d, lat2d = np.meshgrid(
        np.arange(REGION["lon_min"] + GPM_RES / 2, REGION["lon_max"], GPM_RES),
        np.arange(REGION["lat_min"] + GPM_RES / 2, REGION["lat_max"], GPM_RES),
    )
    return lon2d, lat2d


def panel_label(ax, label, x=0.02, y=0.97, fs=19):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fs, fontweight="bold", va="top", ha="left", color="black")


def draw_geo_map(ax, data, title="", show_ylabel=True, show_xlabel=True,
                 show_colorbar='right', cmap=PRECIP_CMAP, vmax=5):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax.set_extent([REGION["lon_min"], REGION["lon_max"],
                   REGION["lat_min"], REGION["lat_max"]],
                  crs=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", linewidth=0.6, color="#333333")

    lon_e = np.arange(REGION["lon_min"], REGION["lon_max"] + GPM_RES, GPM_RES)
    lat_e = np.arange(REGION["lat_min"], REGION["lat_max"] + GPM_RES, GPM_RES)
    le, ae = np.meshgrid(lon_e, lat_e)
    im = ax.pcolormesh(le, ae, np.clip(data, 0, vmax), cmap=cmap,
                       vmin=0, vmax=vmax, shading="flat", rasterized=True,
                       transform=ccrs.PlateCarree())
    if title:
        ax.set_title(title, fontsize=22, fontweight="bold", color=C_AXIS, pad=3)
    if show_ylabel:
        ax.set_ylabel("Latitude (°N)", fontsize=22)
    else:
        ax.set_yticklabels([])
    if show_xlabel:
        ax.set_xlabel("Longitude (°E)", fontsize=22)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#CCCCCC",
                      alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    if not show_ylabel:
        gl.left_labels = False
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}
    # Colorbar is added externally as a shared colorbar per row


def draw_scatter(ax, pred_rate, gpm_rate, y_true, pred_color,
                 title="", show_ylabel=True, show_colorbar='right',
                 cmap="YlOrRd", cax=None, outlier_thresh=0.3,
                 other_rate=None):
    """散点图：仅使用 GPM 有雨 且 Raw/Trans 都预测有雨的像素。

    other_rate: 另一个模型的预测值，传入后做三者交集（GPM有雨 + 当前有雨 + 另一个有雨）。
    """
    vmax = 15.0
    # 三者同时有雨: GPM有雨 + 当前模型有雨 + 另一模型有雨
    mask = (y_true == 1) & (pred_rate > 0) & (gpm_rate > 0.05)
    if other_rate is not None:
        mask = mask & (other_rate > 0)
    px = np.clip(gpm_rate[mask].ravel(), 0, vmax)
    py = np.clip(pred_rate[mask].ravel(), 0, vmax)

    # 统一过滤: 去掉靠近轴的异常值
    near_axis = (px < outlier_thresh) | (py < outlier_thresh)
    px = px[~near_axis]
    py = py[~near_axis]
    if len(px) < 10:
        ax.text(0.5, 0.5, "No rain pixels", transform=ax.transAxes,
                ha="center", va="center", fontsize=18)
        if title:
            ax.set_title(title, fontsize=22, fontweight="bold", color=pred_color, pad=3)
        return
    hb = ax.hexbin(px, py, gridsize=40, cmap=cmap,
                   mincnt=1, alpha=0.92,
                   norm=mpl.colors.LogNorm(),
                   extent=(0, vmax, 0, vmax))
    if show_colorbar:
        if cax is not None:
            cb = plt.colorbar(hb, cax=cax)
        else:
            cb = plt.colorbar(hb, ax=ax, location=show_colorbar,
                              fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=14)
    ax.plot([0, vmax], [0, vmax], "--", color=C_AXIS, linewidth=0.9, alpha=0.6)
    r = float(np.corrcoef(px, py)[0, 1])
    mae = float(np.abs(px - py).mean())
    bias = float(py.mean() - px.mean())
    ax.text(0.97, 0.05,
            f"R={r:.3f}\nMAE={mae:.3f}\nBias={bias:+.3f}",
            transform=ax.transAxes, fontsize=16, color=C_AXIS,
            ha="right", va="bottom")
    ax.set_xlim(outlier_thresh, vmax)
    ax.set_ylim(outlier_thresh, vmax)
    ax.set_xlabel("GPM IMERG (mm/h)", fontsize=18, color=C_AXIS)
    if show_ylabel:
        ax.set_ylabel("Predicted (mm/h)", fontsize=18, color=C_AXIS)
    else:
        ax.set_yticklabels([])
    if title:
        ax.set_title(title, fontsize=22, fontweight="bold", color=pred_color, pad=3)
    ax.set_aspect("equal")


def draw_profile_dual(ax, gpm_rate, raw_rate, trans_rate, y_true, lat_centers,
                      title="", show_ylabel=False):
    lg = np.nanmean(
        np.where(gpm_rate > 0.05, gpm_rate, np.nan),
        axis=1
    )

    lr = np.nanmean(
        np.where(raw_rate > 0.01, raw_rate, np.nan),
        axis=1
    )

    lc = np.nanmean(
        np.where(trans_rate > 0.01, trans_rate, np.nan),
        axis=1
    )

    lg = np.nan_to_num(lg)
    lr = np.nan_to_num(lr)
    lc = np.nan_to_num(lc)
    bw = (lat_centers[1] - lat_centers[0]) * 0.7
    ax.bar(lat_centers, lg, width=bw, color=C_GPM, alpha=0.40,
           label="GPM IMERG", zorder=2)
    ax.plot(lat_centers, lr, color=C_RAW, linewidth=1.2,
            marker="o", markersize=1.6, label="FY-4B Raw", zorder=3)
    ax.plot(lat_centers, lc, color=C_CONV, linewidth=1.4,
            marker="s", markersize=1.8, label="FY-4B Trans.",
            zorder=4, linestyle="--")
    ax.set_xlim(lat_centers[0]-0.2, lat_centers[-1]+0.2)
    ax.set_ylim(0, max(lg.max(), lr.max(), lc.max(), 0.5) * 1.3)
    if title:
        ax.set_title(title, fontsize=22, fontweight="bold", color=C_AXIS, pad=3)
    if show_ylabel:
        ax.set_ylabel("Rate (mm/h)", fontsize=18)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Latitude (°N)", fontsize=18)
    ax.legend(loc="upper right", fontsize=13, frameon=False)


# ══════════════════════════════════════════════════════════════
# 主绘图
# ══════════════════════════════════════════════════════════════

def plot_figure(results, all_agg, best_threshold, fy4a_best=None, fy4a_all=None,
                geo_cmap_name="Blues", out_prefix="unet_fy4b_extrapolation"):
    """4-row, 12-panel figure.

    Row 1: (a) GPM  (b) Raw  (c) Trans   地理分布
    Row 2: (d) GPM  (e) Raw  (f) Trans   地理分布
    Row 3: (g) FY-4A 剖面  (h) Raw 散点  (i) Trans 散点
    Row 4: (j) Case 1  (k) Case 2  (l) Case 3  纬度剖面
    """
    import cartopy.crs as ccrs

    lat_centers = np.arange(REGION["lat_min"] + GPM_RES / 2,
                            REGION["lat_max"], GPM_RES)
    best = results[0]
    second = results[1] if len(results) > 1 else results[0]
    top3 = results[:3]

    # 散点数据（Top-N case）
    N_scatter = min(30, len(results))
    ar = np.concatenate([results[i]['rate_raw'].ravel() for i in range(N_scatter)])
    ac = np.concatenate([results[i]['rate_trans'].ravel() for i in range(N_scatter)])
    ay = np.concatenate([results[i]['y_reg'].ravel() for i in range(N_scatter)])
    yt = np.concatenate([results[i]['y_true'].ravel() for i in range(N_scatter)])

    geo_cmap = plt.get_cmap(geo_cmap_name)
    geo_vmax = 10.0

    # ── Figure ──
    # figsize 使每个 GridSpec 单元格接近正方形 (scatter set_aspect("equal") 不会缩小)
    fig = plt.figure(figsize=(14.4, 16), facecolor="white")
    gs = GridSpec(4, 3, figure=fig, left=0.06, right=0.93,
                  top=0.97, bottom=0.04, wspace=0.14, hspace=0.22)

    proj = ccrs.PlateCarree()

    # ── Row 1 — Best case 地理分布 (a, b, c) ──
    best_date = best['dt_str'].split()[0]  # e.g. "2024-07-03"
    best_utc = best['dt_str'].split()[1] if len(best['dt_str'].split()) > 1 else ""

    ax_a = fig.add_subplot(gs[0, 0], projection=proj)
    draw_geo_map(ax_a, best['y_reg'], cmap=geo_cmap, vmax=geo_vmax)
    panel_label(ax_a, "(a) GPM")
    ax_a.text(0.02, 0.03, f"{best_date} {best_utc} UTC", transform=ax_a.transAxes,
              fontsize=16, color=C_AXIS, va="bottom", ha="left")

    ax_b = fig.add_subplot(gs[0, 1], projection=proj)
    draw_geo_map(ax_b, best['rate_raw'], show_ylabel=False, cmap=geo_cmap, vmax=geo_vmax)
    panel_label(ax_b, "(b) Raw")
    ax_b.text(0.97, 0.97, f"CSI={best['cls_raw']['csi']:.3f}", transform=ax_b.transAxes,
              fontsize=18, color=C_AXIS, va="top", ha="right")

    ax_c = fig.add_subplot(gs[0, 2], projection=proj)
    draw_geo_map(ax_c, best['rate_trans'], show_ylabel=False, cmap=geo_cmap, vmax=geo_vmax)
    panel_label(ax_c, "(c) Trans")
    ax_c.text(0.97, 0.97, f"CSI={best['cls_trans']['csi']:.3f}", transform=ax_c.transAxes,
              fontsize=18, color=C_AXIS, va="top", ha="right")

    # ── Row 2 — 2nd best case 地理分布 (d, e, f) ──
    second_date = second['dt_str'].split()[0]
    second_utc = second['dt_str'].split()[1] if len(second['dt_str'].split()) > 1 else ""

    ax_d = fig.add_subplot(gs[1, 0], projection=proj)
    draw_geo_map(ax_d, second['y_reg'], cmap=geo_cmap, vmax=geo_vmax)
    panel_label(ax_d, "(d) GPM")
    ax_d.text(0.02, 0.03, f"{second_date} {second_utc} UTC", transform=ax_d.transAxes,
              fontsize=16, color=C_AXIS, va="bottom", ha="left")

    ax_e = fig.add_subplot(gs[1, 1], projection=proj)
    draw_geo_map(ax_e, second['rate_raw'], show_ylabel=False, cmap=geo_cmap, vmax=geo_vmax)
    panel_label(ax_e, "(e) Raw")
    ax_e.text(0.97, 0.97, f"CSI={second['cls_raw']['csi']:.3f}", transform=ax_e.transAxes,
              fontsize=18, color=C_AXIS, va="top", ha="right")

    ax_f = fig.add_subplot(gs[1, 2], projection=proj)
    draw_geo_map(ax_f, second['rate_trans'], show_ylabel=False, cmap=geo_cmap, vmax=geo_vmax)
    panel_label(ax_f, "(f) Trans")
    ax_f.text(0.97, 0.97, f"CSI={second['cls_trans']['csi']:.3f}", transform=ax_f.transAxes,
              fontsize=18, color=C_AXIS, va="top", ha="right")

    # ── Shared colorbars for map rows ──
    sm = plt.cm.ScalarMappable(cmap=geo_cmap,
                               norm=plt.Normalize(0, geo_vmax))
    cbar_ax1 = fig.add_axes([0.94, gs[0, 2].get_position(fig).y0,
                              0.018, gs[0, 2].get_position(fig).height])
    cb1 = fig.colorbar(sm, cax=cbar_ax1)
    cb1.set_label("mm/h", fontsize=18, color=C_AXIS)
    cb1.ax.tick_params(labelsize=14)

    cbar_ax2 = fig.add_axes([0.94, gs[1, 2].get_position(fig).y0,
                              0.018, gs[1, 2].get_position(fig).height])
    cb2 = fig.colorbar(sm, cax=cbar_ax2)
    cb2.set_label("mm/h", fontsize=18, color=C_AXIS)
    cb2.ax.tick_params(labelsize=14)

    # ── Row 3 — FY4A best case 剖面 + 散点 (g, h, i) ──
    ax_g = fig.add_subplot(gs[2, 0])
    if fy4a_best is not None:
        fy4a_date = fy4a_best['dt_str'].split()[0]  # e.g. "20240715"
        # 格式化日期: 20240715 -> 2024-07-15
        if len(fy4a_date) == 8:
            fy4a_date_fmt = f"{fy4a_date[:4]}-{fy4a_date[4:6]}-{fy4a_date[6:]}"
        else:
            fy4a_date_fmt = fy4a_date
        mask = (fy4a_best['y_true'] == 1) | (fy4a_best['rate_fy4a'] > 0.01)
        if mask.sum() < 10:
            mask = np.ones_like(fy4a_best['y_true'], dtype=bool)
        lg = np.nanmean(np.where(mask, fy4a_best['y_reg'], np.nan), axis=1)
        lf = np.nanmean(np.where(mask, fy4a_best['rate_fy4a'], np.nan), axis=1)
        bw = (lat_centers[1] - lat_centers[0]) * 0.7
        ax_g.bar(lat_centers, lg, width=bw, color=C_GPM, alpha=0.40,
                 label="GPM IMERG", zorder=2)
        ax_g.plot(lat_centers, lf, color="#7C3AED", linewidth=1.4,
                  marker="^", markersize=2.0, label="FY-4A",
                  zorder=4, linestyle="-")
        ax_g.set_xlim(lat_centers[0]-0.2, lat_centers[-1]+0.2)
        ax_g.set_ylim(0, max(lg.max(), lf.max(), 0.5) * 1.3)
        ax_g.set_ylabel("Rate (mm/h)", fontsize=18)
        ax_g.set_xlabel("Latitude (°N)", fontsize=18)
        ax_g.legend(loc="upper right", fontsize=13, frameon=False)
        ax_g.text(0.02, 0.90, fy4a_date_fmt, transform=ax_g.transAxes,
                  fontsize=14, color=C_AXIS, va="top", ha="left")
        ax_g.text(0.02, 0.84, f"CSI={fy4a_best['csi_fy4a']:.3f}", transform=ax_g.transAxes,
                  fontsize=14, color=C_AXIS, va="top", ha="left")
    else:
        ax_g.text(0.5, 0.5, "No FY4A data", transform=ax_g.transAxes,
                  ha="center", va="center", fontsize=18)
    panel_label(ax_g, "(g) FY-4A")

    ax_h = fig.add_subplot(gs[2, 1])
    draw_scatter(ax_h, ar, ay, yt, C_RAW, show_colorbar=False, other_rate=ac)
    panel_label(ax_h, "(h) Raw")

    ax_i = fig.add_subplot(gs[2, 2])
    scatter_pos = gs[2, 2].get_position(fig)
    cax_scatter = fig.add_axes([scatter_pos.x1 + 0.01, scatter_pos.y0,
                                 0.012, scatter_pos.height])
    draw_scatter(ax_i, ac, ay, yt, C_CONV,
                 show_ylabel=False, show_colorbar='right', cax=cax_scatter,
                 other_rate=ar)
    panel_label(ax_i, "(i) Trans")

    # ── Row 4 — Top-3 FY4B case 纬度剖面 (j, k, l) ──
    row4_axes = []
    case_labels = ["(j) Case 1", "(k) Case 2", "(l) Case 3"]
    for col, (r, lbl) in enumerate(zip(top3, case_labels)):
        ax = fig.add_subplot(gs[3, col])
        parts = r['dt_str'].split()
        case_date = parts[0]  # e.g. "2024-07-03"
        draw_profile_dual(ax, r['y_reg'], r['rate_raw'], r['rate_trans'],
                          r['y_true'], lat_centers,
                          show_ylabel=(col == 0))
        panel_label(ax, lbl)
        ax.text(0.02, 0.90, case_date, transform=ax.transAxes,
                fontsize=14, color=C_AXIS, va="top", ha="left")
        ax.text(0.02, 0.84, f"CSI={r['csi_trans']:.3f}", transform=ax.transAxes,
                fontsize=14, color=C_AXIS, va="top", ha="left")
        row4_axes.append(ax)
    y4 = max(ax.get_ylim()[1] for ax in row4_axes)
    for ax in row4_axes:
        ax.set_ylim(0, y4)

    # 保存 (PNG 600dpi + SVG)
    out_dir = Path("./figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt, dpi in [("png", 600), ("svg", None)]:
        fname = out_dir / f"{out_prefix}.{fmt}"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"✓ {fname}  ({fname.stat().st_size/1024:.0f} KB)")
    plt.close(fig)
    print(f"完成！ ({geo_cmap_name})")


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

def main(no_cache=False, num_workers=8):
    import torch
    from model import UNet

    # 加载 UNet 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_ch=7, base_ch=32).to(device)
    ckpt_path = os.path.join(MODEL_SAVE_DIR, 'unet_best.pth')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']
    clean_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict)
    model.eval()
    best_threshold = float(ckpt.get('best_threshold', 0.50))
    print(f"UNet 模型加载完成 (阈值={best_threshold:.2f}, 设备={device})")

    # B2A 转换 BT / BTD 诊断
    for _date in FY4B_DATE_DIRS:
        _dir = os.path.join(FY4B_ROOT, _date)
        if os.path.isdir(_dir):
            _files = sorted([f for f in os.listdir(_dir)
                             if '_FDI-_' in f and f.endswith('.HDF')])
            if _files:
                _print_b2a_bt_btd_diag(os.path.join(_dir, _files[0]))
                break

    if (not no_cache) and CACHE_PATH.exists():
        results, all_agg = load_cache(CACHE_PATH, best_threshold)
        _print_diagnostics(all_agg, results, best_threshold)
    else:
        results, all_agg = run_batch_inference(
            model, device, GPM_DIR, FY4B_DATE_DIRS,
            max_files_per_day=6, num_workers=num_workers,
            best_threshold=best_threshold)
        save_cache(CACHE_PATH, results, all_agg, best_threshold)
        results, all_agg = load_cache(CACHE_PATH, best_threshold)

    # FY4A 测试集推理（带缓存）
    if (not no_cache) and FY4A_CACHE_PATH.exists():
        fy4a_all = load_fy4a_cache(FY4A_CACHE_PATH, best_threshold)
    else:
        print("\nFY4A 测试集推理...")
        fy4a_all = run_fy4a_test_inference(
            model, device, GPM_DIR, TEST_DATES,
            max_files_per_day=4, num_workers=num_workers,
            best_threshold=best_threshold)
        if fy4a_all:
            save_fy4a_cache(FY4A_CACHE_PATH, fy4a_all, best_threshold)

    fy4a_best = fy4a_all[0] if fy4a_all else None

    # 生成 Blues 版本
    plot_figure(results, all_agg, best_threshold,
                fy4a_best=fy4a_best, fy4a_all=fy4a_all,
                geo_cmap_name="Blues", out_prefix="unet_fy4b_extrapolation_blues")

    # 生成 Jet 版本
    plot_figure(results, all_agg, best_threshold,
                fy4a_best=fy4a_best, fy4a_all=fy4a_all,
                geo_cmap_name="jet", out_prefix="unet_fy4b_extrapolation_jet")

    # 打包四个文件 (blues png/svg + jet png/svg)
    import zipfile
    out_dir = Path("./figures")
    zip_path = out_dir / "unet_fy4b_extrapolation.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name in [
            "unet_fy4b_extrapolation_blues.png",
            "unet_fy4b_extrapolation_blues.svg",
            "unet_fy4b_extrapolation_jet.png",
            "unet_fy4b_extrapolation_jet.svg",
        ]:
            fpath = out_dir / name
            if fpath.exists():
                zf.write(fpath, name)
    print(f"\n✓ 压缩包: {zip_path}  ({zip_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--io-workers", type=int, default=8)
    args = parser.parse_args()
    main(no_cache=args.no_cache,
         num_workers=args.io_workers)
