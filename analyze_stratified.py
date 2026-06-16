"""
分层分析：按降水量级对比 UNet 预测 vs GPM 真值
检查低值偏高、高值偏低问题
"""

import os
import re
import sys
import glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import SAMPLE_CACHE_DIR, MODEL_SAVE_DIR, VAL_DATES
from model import UNet


def load_patches(cache_dir, split_dates, max_files=None):
    all_npz = sorted(glob.glob(os.path.join(cache_dir, '*.npz')))
    if split_dates:
        date_set = set(split_dates)
        filtered = []
        for f in all_npz:
            basename = os.path.basename(f)
            dates_in_name = re.findall(r'\d{8}', basename)
            if any(d in date_set for d in dates_in_name):
                filtered.append(f)
        all_npz = filtered
    if max_files:
        all_npz = all_npz[:max_files]

    x_list, cls_list, reg_list = [], [], []
    for f in all_npz:
        try:
            npz = np.load(f)
            x_list.append(npz['x'])
            cls_list.append(npz['cls'])
            reg_list.append(npz['reg'])
        except Exception:
            pass

    x = np.concatenate(x_list, axis=0).astype(np.float32)
    cls = np.concatenate(cls_list, axis=0).astype(np.float32)
    reg = np.concatenate(reg_list, axis=0).astype(np.float32)
    return x, cls, reg


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = UNet(in_ch=7, base_ch=32).to(device)
    ckpt = torch.load(os.path.join(MODEL_SAVE_DIR, 'unet_best.pth'),
                       map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']
    # torch.compile 保存的模型带 _orig_mod. 前缀
    clean_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict)
    model.eval()
    best_th = ckpt['best_threshold']
    print(f"模型阈值: {best_th}")

    # 加载数据，后半部分作为测试集（测试日期无缓存）
    print("加载数据...")
    x_all, cls_all, reg_all = load_patches(SAMPLE_CACHE_DIR, VAL_DATES, max_files=500)
    n_half = len(x_all) // 2
    x_test, cls_test, reg_test = x_all[n_half:], cls_all[n_half:], reg_all[n_half:]
    print(f"测试集（验证集后半）: {len(x_test)} patches, 有雨像素比例: {cls_test.mean()*100:.1f}%")

    # 推理
    print("推理中...")
    prob_list, reg_list = [], []
    with torch.no_grad():
        for i in range(0, len(x_test), 32):
            x_batch = torch.from_numpy(x_test[i:i+32]).to(device)
            cls_logit, reg_log = model(x_batch)
            prob_list.append(torch.sigmoid(cls_logit).cpu().numpy())
            reg_list.append(reg_log.cpu().numpy())

    prob = np.concatenate(prob_list, axis=0).reshape(-1)
    pred_reg_log = np.concatenate(reg_list, axis=0).reshape(-1)
    cls_flat = cls_test.reshape(-1)
    reg_flat = reg_test.reshape(-1)

    # 还原到 mm/h
    pred_reg = np.expm1(pred_reg_log).clip(0, 50)
    true_reg = np.expm1(reg_flat).clip(0, 50)

    # 推理时校准：修正 per-bin 系统偏差
    # 校准因子基于验证集 per-bin BiasRatio 优化
    pred_calibrated = pred_reg.copy()
    pred_calibrated = np.where(pred_reg < 0.5, pred_reg * 0.3, pred_calibrated)
    pred_calibrated = np.where((pred_reg >= 0.5) & (pred_reg < 1.0), pred_reg * 0.5, pred_calibrated)
    pred_calibrated = np.where((pred_reg >= 1.0) & (pred_reg < 2.0), pred_reg * 0.85, pred_calibrated)
    pred_calibrated = np.where((pred_reg >= 2.0) & (pred_reg < 5.0), pred_reg * 1.0, pred_calibrated)
    pred_calibrated = np.where((pred_reg >= 5.0) & (pred_reg < 10.0), pred_reg * 1.5, pred_calibrated)
    pred_calibrated = np.where(pred_reg >= 10.0, pred_reg * 1.8, pred_calibrated)

    # 只看有雨像素（GPM 有雨）
    rain_mask = cls_flat > 0.5
    pred_rain = pred_reg[rain_mask]
    pred_cal_rain = pred_calibrated[rain_mask]
    true_rain = true_reg[rain_mask]
    prob_rain = prob[rain_mask]

    print(f"\n有雨像素数: {rain_mask.sum():,}")
    print(f"预测有雨像素数 (th={best_th}): {(prob >= best_th).sum():,}")

    # ============================================================
    # 分层分析：按 GPM 真值降水量级
    # ============================================================
    bins = [
        (0.1, 0.5, "0.1-0.5 mm/h (小雨)"),
        (0.5, 1.0, "0.5-1.0 mm/h (小-中雨)"),
        (1.0, 2.0, "1.0-2.0 mm/h (中雨)"),
        (2.0, 5.0, "2.0-5.0 mm/h (中-大雨)"),
        (5.0, 10.0, "5.0-10.0 mm/h (大雨)"),
        (10.0, 20.0, "10.0-20.0 mm/h (暴雨)"),
        (20.0, 50.0, "20.0+ mm/h (大暴雨)"),
    ]

    # --- 校准前 ---
    print("\n" + "=" * 80)
    print("分层分析：校准前（仅分析 GPM 有雨像素）")
    print("=" * 80)
    print(f"\n{'量级':<28} {'像素数':>8} {'GPM均值':>8} {'Pred均值':>8} "
          f"{'BiasRatio':>10} {'MAE':>8} {'R':>6} {'POD':>6}")
    print("-" * 90)

    for lo, hi, label in bins:
        mask = (true_rain >= lo) & (true_rain < hi)
        n = mask.sum()
        if n < 10:
            continue
        t = true_rain[mask]
        p = pred_rain[mask]
        prob_bin = prob_rain[mask]
        print(f"{label:<28} {n:>8,} {t.mean():>8.3f} {p.mean():>8.3f} "
              f"{p.mean()/(t.mean()+1e-8):>10.3f} {np.abs(p-t).mean():>8.3f} "
              f"{np.corrcoef(p,t)[0,1] if p.std()>1e-8 else 0:>6.3f} "
              f"{(prob_bin>=best_th).mean():>6.3f}")

    # --- 校准后 ---
    print("\n" + "=" * 80)
    print("分层分析：校准后（仅分析 GPM 有雨像素）")
    print("=" * 80)
    print(f"\n{'量级':<28} {'像素数':>8} {'GPM均值':>8} {'Cal均值':>8} "
          f"{'BiasRatio':>10} {'MAE':>8} {'R':>6} {'POD':>6}")
    print("-" * 90)

    for lo, hi, label in bins:
        mask = (true_rain >= lo) & (true_rain < hi)
        n = mask.sum()
        if n < 10:
            continue
        t = true_rain[mask]
        p = pred_cal_rain[mask]
        prob_bin = prob_rain[mask]
        print(f"{label:<28} {n:>8,} {t.mean():>8.3f} {p.mean():>8.3f} "
              f"{p.mean()/(t.mean()+1e-8):>10.3f} {np.abs(p-t).mean():>8.3f} "
              f"{np.corrcoef(p,t)[0,1] if p.std()>1e-8 else 0:>6.3f} "
              f"{(prob_bin>=best_th).mean():>6.3f}")

    # ============================================================
    # 整体统计（校准后）
    # ============================================================
    print("\n" + "=" * 80)
    print("整体统计：校准后（仅 GPM 有雨像素）")
    print("=" * 80)
    print(f"  GPM 均值:    {true_rain.mean():.3f} mm/h")
    print(f"  Cal 均值:    {pred_cal_rain.mean():.3f} mm/h")
    print(f"  BiasRatio:   {pred_cal_rain.mean() / (true_rain.mean() + 1e-8):.3f}")
    print(f"  MAE:         {np.abs(pred_cal_rain - true_rain).mean():.3f} mm/h")
    print(f"  RMSE:        {np.sqrt(np.mean((pred_cal_rain - true_rain)**2)):.3f} mm/h")
    r_cal = np.corrcoef(pred_cal_rain, true_rain)[0, 1]
    print(f"  R:           {r_cal:.4f}")

    # ============================================================
    # 分层分析：按预测概率分组（看分类置信度）
    # ============================================================
    print("\n" + "=" * 80)
    print("分类置信度分层：按预测概率分组")
    print("=" * 80)

    prob_bins = [
        (0.0, 0.1, "0.0-0.1"),
        (0.1, 0.2, "0.1-0.2"),
        (0.2, 0.3, "0.2-0.3"),
        (0.3, 0.4, "0.3-0.4"),
        (0.4, 0.5, "0.4-0.5"),
        (0.5, 0.6, "0.5-0.6"),
        (0.6, 0.7, "0.6-0.7"),
        (0.7, 0.8, "0.7-0.8"),
        (0.8, 0.9, "0.8-0.9"),
        (0.9, 1.0, "0.9-1.0"),
    ]

    print(f"\n{'概率区间':<12} {'像素数':>10} {'实际有雨比例':>12} {'GPM均值':>8} {'Pred均值':>8}")
    print("-" * 55)

    for lo, hi, label in prob_bins:
        mask = (prob >= lo) & (prob < hi)
        n = mask.sum()
        if n == 0:
            continue

        actual_rain_rate = cls_flat[mask].mean()
        gpm_mean = true_reg[mask].mean()
        pred_mean = pred_reg[mask].mean()

        print(f"{label:<12} {n:>10,} {actual_rain_rate:>12.3f} {gpm_mean:>8.3f} {pred_mean:>8.3f}")

    # ============================================================
    # 无雨像素误报分析
    # ============================================================
    print("\n" + "=" * 80)
    print("无雨像素误报分析（GPM 无雨但预测有雨）")
    print("=" * 80)

    no_rain_mask = cls_flat <= 0.5
    false_rain_mask = no_rain_mask & (prob >= best_th)
    print(f"  GPM 无雨像素数: {no_rain_mask.sum():,}")
    print(f"  误报像素数:     {false_rain_mask.sum():,}")
    print(f"  误报率 (FAR):   {false_rain_mask.sum() / (no_rain_mask.sum() + 1e-6):.4f}")
    if false_rain_mask.sum() > 0:
        print(f"  误报平均概率:   {prob[false_rain_mask].mean():.3f}")
        print(f"  误报平均预测量: {pred_reg[false_rain_mask].mean():.3f} mm/h")


if __name__ == '__main__':
    main()
