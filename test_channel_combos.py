"""测试不同通道转换组合对 R 的影响。"""
import os, sys, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from extrapolate_xgb_figure import (
    read_fy4b, apply_b2a_transfer, build_features, build_features_trans,
    compute_bt_diffs, B2A_COEFFS, _WL_UM_FY4B, _bt_to_rad, _rad_to_bt,
    normalize, FY4B_ROOT, FY4B_DATE_DIRS, FY4B_GPM_DIR, parse_fy4_datetime,
    find_matching_gpm, read_gpm_rate, resample_agri_to_region, read_gpm_label,
    _GEO_LUT_CACHE, UNet, unet_predict_patches, PATCH_SIZE
)
import torch

def apply_b2a_selective(bt_fy4b, convert_chs):
    """选择性转换指定通道。convert_chs: set of channel indices (0-3) to convert."""
    out = bt_fy4b.copy()
    ch_keys = [9, 10, 13, 14]
    for i, ch in enumerate(ch_keys):
        if i not in convert_chs:
            continue
        slope, intercept, _ = B2A_COEFFS[ch]
        wl = _WL_UM_FY4B[ch]
        rad = _bt_to_rad(bt_fy4b[i], wl)
        rad_mw = rad * 1e3
        rad_out_mw = slope * rad_mw + intercept
        rad_out = rad_out_mw * 1e-3
        out[i] = _rad_to_bt(rad_out, wl)
    return out


def process_file(fdi_path, gpm_dir, model, device, convert_chs, use_trans_btd):
    """处理单个文件，返回 (rate_raw, rate_trans, y_true, y_reg)。"""
    dt = parse_fy4_datetime(fdi_path)
    gpm_f = find_matching_gpm(dt, gpm_dir)
    if not gpm_f:
        return None

    ir_bt, lons, lats = read_fy4b(fdi_path)
    ir_bt_raw = ir_bt.copy()
    ir_bt_trans = apply_b2a_selective(ir_bt, convert_chs)

    res_raw = resample_agri_to_region(ir_bt_raw, fdi_path)
    res_trans = resample_agri_to_region(ir_bt_trans, fdi_path)

    feat_raw = build_features(res_raw)
    if use_trans_btd:
        feat_trans = build_features_trans(res_trans, res_raw)
    else:
        # BTD 用原始值
        btd_raw = compute_bt_diffs(res_raw)
        feat_trans = np.concatenate([res_trans, btd_raw], axis=0)

    feat_raw_n = normalize(feat_raw)
    feat_trans_n = normalize(feat_trans)

    prob_raw, rate_raw = unet_predict_patches(model, feat_raw_n, device, PATCH_SIZE)
    prob_trans, rate_trans = unet_predict_patches(model, feat_trans_n, device, PATCH_SIZE)

    gpm_rate = read_gpm_rate(gpm_f)
    y_reg = resample_gpm_to_match(gpm_rate, ...)
    y_true = read_gpm_label(gpm_f, ...)
    # This won't work directly, need proper resampling

    return rate_raw, rate_trans, y_true, y_reg


def main():
    """Test different channel combinations."""
    # For now, just print what we'd test
    print("Channel indices: 0=C09, 1=C10, 2=C13, 3=C14")
    print()

    combos = [
        ("None (Raw)", set(), False),
        ("All + trans BTD", {0,1,2,3}, True),
        ("All + raw BTD", {0,1,2,3}, False),
        ("Skip C10 + trans BTD", {0,2,3}, True),
        ("Skip C10 + raw BTD", {0,2,3}, False),
        ("Only C14", {3}, False),
        ("Only C13+C14", {2,3}, False),
        ("Only C09+C14", {0,3}, False),
        ("C09+C13+C14 + raw BTD", {0,2,3}, False),
    ]

    for name, chs, trans_btd in combos:
        ch_names = ['C09', 'C10', 'C13', 'C14']
        ch_str = '+'.join(ch_names[i] for i in sorted(chs)) if chs else 'None'
        btd_str = 'trans_BTD' if trans_btd else 'raw_BTD'
        print(f"  {name}: convert=[{ch_str}], BTD={btd_str}")


if __name__ == "__main__":
    main()
