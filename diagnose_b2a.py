"""诊断 B2A 转换系数问题：分析转换是否真的让 FY4B 更接近 FY4A。"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from extrapolate_xgb_figure import (
    read_fy4b, apply_b2a_transfer, build_features, build_features_trans,
    compute_bt_diffs, B2A_COEFFS, _WL_UM_FY4B, _bt_to_rad, _rad_to_bt,
    FY4B_ROOT, FY4B_DATE_DIRS
)

def main():
    # 找一个 FY4B 文件
    fdi_path = None
    for _date in FY4B_DATE_DIRS:
        _dir = os.path.join(FY4B_ROOT, _date)
        if os.path.isdir(_dir):
            _files = sorted([f for f in os.listdir(_dir) if '_FDI-_' in f and f.endswith('.HDF')])
            if _files:
                fdi_path = os.path.join(_dir, _files[0])
                break
    if not fdi_path:
        print("No FY4B file found!")
        return

    ir_bt, _, _ = read_fy4b(fdi_path)
    ir_bt_trans = apply_b2a_transfer(ir_bt)

    ch_names = ['C09(6.25μm)', 'C10(7.10μm)', 'C13(10.8μm)', 'C14(12.0μm)']
    ch_keys = [9, 10, 13, 14]

    print("=" * 70)
    print("B2A 转换系数诊断")
    print("=" * 70)

    # 1. 系数分析
    print("\n[1] 转换系数:")
    for ch, name in zip(ch_keys, ch_names):
        slope, intercept, resid_std = B2A_COEFFS[ch]
        wl = _WL_UM_FY4B[ch]
        # 计算典型辐射值 (300K 对应的辐射)
        rad_300k = _bt_to_rad(np.float32(300.0), wl)
        rad_300k_mw = rad_300k * 1e3
        intercept_ratio = intercept / rad_300k_mw * 100  # 截距占比
        print(f"  {name}: slope={slope:.4f}, intercept={intercept:.4f} mW "
              f"(占300K辐射{intercept_ratio:.2f}%), R²={0.9811 if ch==10 else 0.9986:.4f}")

    # 2. BT 分布对比
    print("\n[2] BT 分布 (原始 vs 转换):")
    for i, (ch, name) in enumerate(zip(ch_keys, ch_names)):
        raw = ir_bt[i].ravel()
        trans = ir_bt_trans[i].ravel()
        m = np.isfinite(raw) & np.isfinite(trans)
        raw_v, trans_v = raw[m], trans[m]
        delta = trans_v - raw_v
        print(f"  {name}:")
        print(f"    Raw:  mean={raw_v.mean():.2f}K, std={raw_v.std():.2f}K, "
              f"P5={np.percentile(raw_v,5):.1f}K, P95={np.percentile(raw_v,95):.1f}K")
        print(f"    Trans: mean={trans_v.mean():.2f}K, std={trans_v.std():.2f}K")
        print(f"    ΔBT:  mean={delta.mean():+.3f}K, std={delta.std():.3f}K")

    # 3. 辐射空间分析
    print("\n[3] 辐射空间分析 (mW):")
    for i, (ch, name) in enumerate(zip(ch_keys, ch_names)):
        slope, intercept, _ = B2A_COEFFS[ch]
        wl = _WL_UM_FY4B[ch]
        raw_bt = ir_bt[i].ravel()
        m = np.isfinite(raw_bt)
        rad_mw = _bt_to_rad(raw_bt[m], wl) * 1e3
        rad_out_mw = slope * rad_mw + intercept
        print(f"  {name}:")
        print(f"    输入辐射: mean={rad_mw.mean():.4f} mW, std={rad_mw.std():.4f} mW")
        print(f"    intercept={intercept:.4f} mW, 占输出辐射比={intercept/rad_out_mw.mean()*100:.2f}%")
        print(f"    斜率效应: {slope*rad_mw.mean():.4f} → 输出mean={rad_out_mw.mean():.4f} mW")

    # 4. BTD 影响分析
    print("\n[4] BTD 影响:")
    btd_raw = compute_bt_diffs(ir_bt)
    btd_trans = compute_bt_diffs(ir_bt_trans)
    btd_names = ['C09-C10', 'C09-C13', 'C10-C13']
    for j, name in enumerate(btd_names):
        r = btd_raw[j].ravel()
        t = btd_trans[j].ravel()
        m = np.isfinite(r) & np.isfinite(t)
        d = t[m] - r[m]
        print(f"  {name}: Δmean={d.mean():+.3f}K, Δstd={d.std():.3f}K, "
              f"原始std={r[m].std():.3f}K")

    # 5. 特征空间一致性检查
    print("\n[5] 特征空间分析:")
    feat_raw = build_features(ir_bt)
    feat_trans = build_features_trans(ir_bt_trans, ir_bt)
    feat_names = ['C09', 'C10', 'C13', 'C14', 'BTD1', 'BTD2', 'BTD3']
    for k, name in enumerate(feat_names):
        r = feat_raw[k].ravel()
        t = feat_trans[k].ravel()
        m = np.isfinite(r) & np.isfinite(t)
        d = t[m] - r[m]
        corr = np.corrcoef(r[m], t[m])[0, 1] if m.sum() > 1 else 0
        print(f"  {name}: Δmean={d.mean():+.4f}, corr={corr:.6f}, "
              f"raw_std={r[m].std():.4f}, trans_std={t[m].std():.4f}")

    # 6. 关键问题: 截距单位验证
    print("\n[6] 截距单位验证:")
    print("  如果截距是 mW 空间拟合的，应该很小。")
    print("  C13 intercept=0.815 mW 异常大！")
    print("  C10 slope=1.344 异常大！")
    print("  这两个通道的转换可能引入了系统偏差。")

    # 7. 跳过 C13 截距的效果测试
    print("\n[7] 测试: 去掉 C13 截距 (设为0):")
    for i, ch in enumerate(ch_keys):
        if ch == 13:
            slope, _, resid_std = B2A_COEFFS[ch]
            wl = _WL_UM_FY4B[ch]
            raw_bt = ir_bt[i].ravel()
            m = np.isfinite(raw_bt)
            # 有截距
            rad_mw = _bt_to_rad(raw_bt[m], wl) * 1e3
            out_with = _rad_to_bt((slope * rad_mw + 0.8147) * 1e-3, wl)
            # 无截距
            out_without = _rad_to_bt((slope * rad_mw) * 1e-3, wl)
            d_with = out_with - raw_bt[m]
            d_without = out_without - raw_bt[m]
            print(f"  C13 有截距: Δmean={d_with.mean():+.3f}K, std={d_with.std():.3f}K")
            print(f"  C13 无截距: Δmean={d_without.mean():+.3f}K, std={d_without.std():.3f}K")


if __name__ == "__main__":
    main()
