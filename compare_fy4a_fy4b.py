"""对比 FY4A 真实 BT 与 FY4B 转换后 BT，验证转换系数准确性。

通道映射（来自 CSV 系数文件）:
  FY4B C09(6.25μm)  → FY4A C09(6.25μm)    idx 0
  FY4B C10(7.10μm)  → FY4A C10(7.10μm)    idx 1
  FY4B C13(10.8μm)  → FY4A C12(10.8μm)    idx 2
  FY4B C14(12.0μm)  → FY4A C13(12.0μm)    idx 3
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import FY4A_AGRI_DIR, TEST_DATES, AGRI_CHANNELS
from utils.data_io import read_agri_disk
from extrapolate_xgb_figure import (
    read_fy4b, apply_b2a_transfer,
    FY4B_ROOT, FY4B_DATE_DIRS, B2A_COEFFS, _WL_UM_FY4B,
)


def load_fy4a_bt(fy4a_dir, dates, max_per_date=1):
    """加载 FY4A BT 数据。"""
    all_bt = []
    for date_str in dates:
        d = os.path.join(fy4a_dir, date_str)
        if not os.path.isdir(d):
            continue
        files = sorted([f for f in os.listdir(d) if f.endswith('.HDF')])
        for f in files[:max_per_date]:
            try:
                data, _, _ = read_agri_disk(os.path.join(d, f))
                all_bt.append(data)
            except Exception:
                pass
    return all_bt


def load_fy4b_bt(fy4b_root, fy4b_dates, max_per_date=1):
    """加载 FY4B BT 数据。"""
    all_bt = []
    for date_str in fy4b_dates:
        d = os.path.join(fy4b_root, date_str)
        if not os.path.isdir(d):
            continue
        files = sorted([f for f in os.listdir(d) if '_FDI-_' in f and f.endswith('.HDF')])
        for f in files[:max_per_date]:
            try:
                bt, _, _ = read_fy4b(os.path.join(d, f))
                all_bt.append(bt)
            except Exception:
                pass
    return all_bt


def sample_flat(bt_list, max_pixels=300000):
    """从 BT 列表中采样像素，返回 (4, N)。"""
    all_samples = []
    per_file = max_pixels // max(len(bt_list), 1)
    for bt in bt_list:
        C, H, W = bt.shape
        flat = bt.reshape(C, -1)
        n = flat.shape[1]
        if n > per_file:
            idx = np.random.choice(n, per_file, replace=False)
            all_samples.append(flat[:, idx])
        else:
            all_samples.append(flat)
    return np.concatenate(all_samples, axis=1)


def safe_stats(vals):
    """安全计算统计量。"""
    if len(vals) == 0:
        return None
    return {
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'p5': float(np.percentile(vals, 5)),
        'p95': float(np.percentile(vals, 95)),
    }


def main():
    np.random.seed(42)
    print("=" * 70)
    print("FY4A vs FY4B BT 分布对比 + 转换系数验证")
    print("=" * 70)

    # 加载 FY4A 数据
    print("\n加载 FY4A 数据...")
    fy4a_bts = load_fy4a_bt(FY4A_AGRI_DIR, TEST_DATES, max_per_date=1)
    print(f"  加载了 {len(fy4a_bts)} 个 FY4A 文件")

    # 加载 FY4B 数据（只取前5个日期）
    fy4b_dates_sub = FY4B_DATE_DIRS[:5]
    print("加载 FY4B 数据...")
    fy4b_bts = load_fy4b_bt(FY4B_ROOT, fy4b_dates_sub, max_per_date=1)
    print(f"  加载了 {len(fy4b_bts)} 个 FY4B 文件")

    if not fy4a_bts or not fy4b_bts:
        print("ERROR: 数据加载失败")
        return

    # 采样像素
    print("\n采样像素...")
    fy4a_sample = sample_flat(fy4a_bts, 300000)   # (4, N)
    fy4b_sample = sample_flat(fy4b_bts, 300000)    # (4, N)
    print(f"  FY4A: {fy4a_sample.shape[1]} 像素")
    print(f"  FY4B: {fy4b_sample.shape[1]} 像素")

    # 预计算 FY4B 转换后的 BT（逐像素）
    print("\n计算 B2A 转换...")
    N = fy4b_sample.shape[1]
    fy4b_trans_sample = np.zeros_like(fy4b_sample)
    ch_keys = [9, 10, 13, 14]
    for i, ch in enumerate(ch_keys):
        if ch == 10:
            # C10 使用 BT 偏移
            fy4b_trans_sample[i] = fy4b_sample[i] + 3.5
        else:
            slope, intercept, _ = B2A_COEFFS[ch]
            wl = _WL_UM_FY4B[ch]
            from extrapolate_xgb_figure import _bt_to_rad, _rad_to_bt
            rad = _bt_to_rad(fy4b_sample[i], wl)
            rad_mw = rad * 1e3
            rad_out_mw = slope * rad_mw + intercept
            rad_out = rad_out_mw * 1e-3
            fy4b_trans_sample[i] = _rad_to_bt(rad_out, wl)
    print("  B2A 转换完成")

    # 通道名称
    ch_names = ['C09(6.25μm)', 'C10(7.10μm)', 'C12/C13(10.8μm)', 'C13/C14(12.0μm)']

    print("\n" + "=" * 70)
    print("BT 分布对比（按波长对齐）")
    print("=" * 70)

    for k, name in enumerate(ch_names):
        fa = fy4a_sample[k]
        fb = fy4b_sample[k]
        ft = fy4b_trans_sample[k]

        ma = np.isfinite(fa) & (fa > 100) & (fa < 350)
        mb = np.isfinite(fb) & (fb > 100) & (fb < 350)
        mt = np.isfinite(ft) & (ft > 100) & (ft < 350)

        sa = safe_stats(fa[ma])
        sb = safe_stats(fb[mb])
        st = safe_stats(ft[mt])

        if sa and sb and st:
            print(f"\n  {name}:")
            print(f"    FY4A:      mean={sa['mean']:.2f}K, std={sa['std']:.2f}K, P5={sa['p5']:.1f}K, P95={sa['p95']:.1f}K")
            print(f"    FY4B raw:  mean={sb['mean']:.2f}K, std={sb['std']:.2f}K, P5={sb['p5']:.1f}K, P95={sb['p95']:.1f}K")
            print(f"    FY4B trans: mean={st['mean']:.2f}K, std={st['std']:.2f}K, P5={st['p5']:.1f}K, P95={st['p95']:.1f}K")
            print(f"    FY4A - FY4B raw:  Δmean={sa['mean']-sb['mean']:+.2f}K")
            print(f"    FY4A - FY4B trans: Δmean={sa['mean']-st['mean']:+.2f}K")

            # 提供的 B2A 系数
            slope, intercept, _ = B2A_COEFFS[ch_keys[k]]
            print(f"    提供的 B2A 系数: slope={slope:.6f}, intercept={intercept:.6f} (辐射空间 mW)")

    # BTD 对比
    print("\n" + "=" * 70)
    print("BTD 分布对比")
    print("=" * 70)

    btd_pairs = [(0, 1), (0, 2), (1, 2)]
    btd_names = ['C09-C10', 'C09-C12/C13', 'C10-C12/C13']

    for j, (a, b) in enumerate(btd_pairs):
        btd_a = fy4a_sample[a] - fy4a_sample[b]
        btd_b = fy4b_sample[a] - fy4b_sample[b]
        btd_t = fy4b_trans_sample[a] - fy4b_trans_sample[b]

        ma = np.isfinite(btd_a)
        mb = np.isfinite(btd_b)
        mt = np.isfinite(btd_t)

        sa = safe_stats(btd_a[ma])
        sb = safe_stats(btd_b[mb])
        st = safe_stats(btd_t[mt])

        if sa and sb and st:
            print(f"\n  BTD {btd_names[j]}:")
            print(f"    FY4A:      mean={sa['mean']:.2f}K, std={sa['std']:.2f}K")
            print(f"    FY4B raw:  mean={sb['mean']:.2f}K, std={sb['std']:.2f}K")
            print(f"    FY4B trans: mean={st['mean']:.2f}K, std={st['std']:.2f}K")
            print(f"    FY4A - FY4B raw:  Δmean={sa['mean']-sb['mean']:+.2f}K, Δstd={sa['std']-sb['std']:+.2f}K")
            print(f"    FY4A - FY4B trans: Δmean={sa['mean']-st['mean']:+.2f}K, Δstd={sa['std']-st['std']:+.2f}K")


if __name__ == "__main__":
    main()
