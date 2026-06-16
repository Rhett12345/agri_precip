"""诊断每个通道/特征对 UNet 预测的影响。

思路：逐个扰动特征，看预测变化大小。
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from extrapolate_xgb_figure import (
    read_fy4b, apply_b2a_transfer, build_features, build_features_trans,
    compute_bt_diffs, normalize, FY4B_ROOT, FY4B_DATE_DIRS, GPM_DIR,
    parse_fy4_datetime, find_matching_gpm,
    resample_gpm_to_region, resample_agri_to_region,
    unet_predict_patches, B2A_COEFFS, _WL_UM_FY4B, _bt_to_rad, _rad_to_bt,
)
from utils.data_io import read_gpm_imerg
from model import UNet
from config import MODEL_SAVE_DIR


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_ch=7, base_ch=32).to(device)
    ckpt_path = os.path.join(MODEL_SAVE_DIR, 'unet_best.pth')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']
    clean_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict)
    model.eval()
    print(f"模型加载完成 ({device})")

    # 找一个文件
    fdi_path = None
    for _date in FY4B_DATE_DIRS:
        _dir = os.path.join(FY4B_ROOT, _date)
        if os.path.isdir(_dir):
            _files = sorted([f for f in os.listdir(_dir) if '_FDI-_' in f and f.endswith('.HDF')])
            if _files:
                fdi_path = os.path.join(_dir, _files[0])
                break

    ir_bt, lons, lats = read_fy4b(fdi_path)
    ir_bt_trans = apply_b2a_transfer(ir_bt)

    res_raw = resample_agri_to_region(ir_bt, fdi_path)
    res_trans = resample_agri_to_region(ir_bt_trans, fdi_path)

    feat_raw = build_features(res_raw)
    feat_trans = build_features_trans(res_trans, res_raw)
    feat_raw_n = normalize(feat_raw)
    feat_trans_n = normalize(feat_trans)

    prob_raw, rate_raw = unet_predict_patches(model, feat_raw_n, device)
    prob_trans, rate_trans = unet_predict_patches(model, feat_trans_n, device)

    feat_names = ['C09', 'C10', 'C13', 'C14', 'BTD1(C09-C10)', 'BTD2(C09-C13)', 'BTD3(C10-C13)']

    print("\n" + "=" * 70)
    print("特征差异分析 (Raw vs Trans, 归一化后)")
    print("=" * 70)

    for k, name in enumerate(feat_names):
        r = feat_raw_n[k].ravel()
        t = feat_trans_n[k].ravel()
        m = np.isfinite(r) & np.isfinite(t)
        d = t[m] - r[m]
        corr = np.corrcoef(r[m], t[m])[0, 1]
        print(f"  {name:20s}: Δmean={d.mean():+.6f}, Δstd={d.std():.6f}, "
              f"corr={corr:.6f}, raw_mean={r[m].mean():.4f}")

    print("\n" + "=" * 70)
    print("预测差异分析")
    print("=" * 70)
    d_prob = prob_trans - prob_raw
    d_rate = rate_trans - rate_raw
    print(f"  Prob: Δmean={d_prob.mean():+.6f}, Δstd={d_prob.std():.6f}")
    print(f"  Rate: Δmean={d_rate.mean():+.6f} mm/h, Δstd={d_rate.std():.6f} mm/h")

    # 逐通道扰动测试
    print("\n" + "=" * 70)
    print("逐通道扰动测试 (±1σ 扰动对预测的影响)")
    print("=" * 70)

    base_prob, base_rate = unet_predict_patches(model, feat_raw_n, device)

    for k, name in enumerate(feat_names):
        feat_perturbed = feat_raw_n.copy()
        sigma = feat_raw_n[k].std()
        feat_perturbed[k] += sigma  # +1σ

        prob_p, rate_p = unet_predict_patches(model, feat_perturbed, device)

        d_prob = prob_p - base_prob
        d_rate = rate_p - base_rate
        print(f"  {name:20s}: σ={sigma:.4f}, "
              f"Δprob_mean={d_prob.mean():+.6f}, Δrate_mean={d_rate.mean():+.6f} mm/h")

    # B2A 转换的 BT 偏移量 vs 特征 σ
    print("\n" + "=" * 70)
    print("B2A 转换偏移 vs 特征标准差")
    print("=" * 70)
    ch_keys = [9, 10, 13, 14]
    for i, (ch, name) in enumerate(zip(ch_keys, ['C09', 'C10', 'C13', 'C14'])):
        raw_bt = res_raw[i].ravel()
        trans_bt = res_trans[i].ravel()
        m = np.isfinite(raw_bt) & np.isfinite(trans_bt)
        delta_bt = (trans_bt[m] - raw_bt[m]).mean()
        sigma_bt = raw_bt[m].std()
        ratio = abs(delta_bt) / sigma_bt * 100
        print(f"  {name}: ΔBT={delta_bt:+.3f}K, σ_BT={sigma_bt:.3f}K, "
              f"|Δ|/σ={ratio:.2f}%")

    # BTD 偏移
    btd_raw = compute_bt_diffs(res_raw)
    btd_trans = compute_bt_diffs(res_trans)
    btd_names = ['C09-C10', 'C09-C13', 'C10-C13']
    for j, name in enumerate(btd_names):
        r = btd_raw[j].ravel()
        t = btd_trans[j].ravel()
        m = np.isfinite(r) & np.isfinite(t)
        delta = (t[m] - r[m]).mean()
        sigma = r[m].std()
        ratio = abs(delta) / sigma * 100 if sigma > 0 else 0
        print(f"  BTD {name}: Δ={delta:+.3f}K, σ={sigma:.3f}K, |Δ|/σ={ratio:.2f}%")


if __name__ == "__main__":
    main()
