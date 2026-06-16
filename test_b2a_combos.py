"""测试不同 B2A 通道转换组合，找出最优方案。

只改外推部分，不改训练代码。
"""
import os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))

# 复用主脚本的基础设施
from extrapolate_xgb_figure import (
    read_fy4b, compute_bt_diffs, B2A_COEFFS, _WL_UM_FY4B,
    _bt_to_rad, _rad_to_bt, normalize, FY4B_ROOT, FY4B_DATE_DIRS,
    FY4B_GPM_DIR, FY4A_ROOT, FY4A_DATE_DIRS, FY4A_GPM_DIR,
    parse_fy4_datetime, find_matching_gpm, read_gpm_rate,
    resample_agri_to_region, read_gpm_label, _GEO_LUT_CACHE,
    PATCH_SIZE, IR_PHYSICAL_FY4B, UNet
)
import torch


def apply_b2a_selective(bt_fy4b, convert_mask):
    """选择性 B2A 转换。convert_mask: 4-element bool list [C09,C10,C13,C14]。"""
    out = bt_fy4b.copy()
    ch_keys = [9, 10, 13, 14]
    for i, ch in enumerate(ch_keys):
        if not convert_mask[i]:
            continue
        slope, intercept, _ = B2A_COEFFS[ch]
        wl = _WL_UM_FY4B[ch]
        rad = _bt_to_rad(bt_fy4b[i], wl)
        rad_mw = rad * 1e3
        rad_out_mw = slope * rad_mw + intercept
        rad_out = rad_out_mw * 1e-3
        out[i] = _rad_to_bt(rad_out, wl)
    return out


def build_features_mixed(bt_mixed):
    """从混合 BT 构建 7 维特征（BTD 从混合通道计算）。"""
    btd = compute_bt_diffs(bt_mixed)
    return np.concatenate([bt_mixed, btd], axis=0)


def unet_predict_patches(model, feat_map, device, patch_size=PATCH_SIZE):
    """UNet patch-based 推理。"""
    C, H, W = feat_map.shape
    ps = patch_size
    n_h, n_w = H // ps, W // ps

    patches = []
    for i in range(n_h):
        for j in range(n_w):
            patches.append(feat_map[:, i*ps:(i+1)*ps, j*ps:(j+1)*ps])

    batch = np.stack(patches, axis=0)
    batch_t = torch.from_numpy(batch).to(device)

    with torch.no_grad():
        out = model(batch_t)
        if isinstance(out, tuple):
            logit, reg = out
        else:
            logit = out[:, 0:1]
            reg = out[:, 1:2]
        prob = torch.sigmoid(logit).cpu().numpy()[:, 0]
        rate = reg.cpu().numpy()[:, 0]

    prob_map = np.zeros((H, W), dtype=np.float32)
    rate_map = np.zeros((H, W), dtype=np.float32)
    idx = 0
    for i in range(n_h):
        for j in range(n_w):
            prob_map[i*ps:(i+1)*ps, j*ps:(j+1)*ps] = prob[idx]
            rate_map[i*ps:(i+1)*ps, j*ps:(j+1)*ps] = rate[idx]
            idx += 1

    return prob_map, rate_map


def _process_one(args):
    """处理单个 FY4B 文件。"""
    fdi_path, gpm_dir, convert_mask, device_str = args
    device = torch.device(device_str)

    dt = parse_fy4_datetime(fdi_path)
    gpm_f = find_matching_gpm(dt, gpm_dir)
    if not gpm_f:
        return None

    ir_bt, lons, lats = read_fy4b(fdi_path)

    # 选择性转换
    ir_bt_trans = apply_b2a_selective(ir_bt, convert_mask)

    # 重采样
    res_trans = resample_agri_to_region(ir_bt_trans, fdi_path)

    # 构建特征（BTD 从混合通道计算）
    feat = build_features_mixed(res_trans)
    feat_n = normalize(feat)

    # UNet 推理
    model = UNet(in_ch=7).to(device)
    ckpt = os.path.join(os.path.dirname(__file__), 'unet_rain_pred', 'checkpoints', 'unet_mixed_best.pth')
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    prob, rate = unet_predict_patches(model, feat_n, device)

    # GPM
    gpm_rate = read_gpm_rate(gpm_f)
    from data_io import make_gpm_lut
    gpm_lut = make_gpm_lut(gpm_f, lats, lons)
    y_reg = gpm_lut['rate']
    y_true = gpm_lut['label']

    return {
        'prob': prob,
        'rate': rate,
        'y_true': y_true,
        'y_reg': y_reg,
    }


def calc_r(px, py):
    """计算相关系数。"""
    if len(px) < 10 or py.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(px, py)[0, 1])


def main():
    # 收集 FY4B 文件
    fdi_files = []
    for _date in FY4B_DATE_DIRS:
        _dir = os.path.join(FY4B_ROOT, _date)
        if os.path.isdir(_dir):
            _files = sorted([f for f in os.listdir(_dir)
                            if '_FDI-_' in f and f.endswith('.HDF')])
            for f in _files[:5]:  # 每天只取 5 个文件加速测试
                fdi_files.append(os.path.join(_dir, f))

    print(f"找到 {len(fdi_files)} 个 FY4B 文件")

    # 测试组合
    combos = [
        ("Raw (不转换)",      [False, False, False, False]),
        ("全转+trans BTD",    [True,  True,  True,  True ]),
        ("跳C10+trans BTD",   [True,  False, True,  True ]),
        ("跳C10+raw BTD",     [True,  False, True,  True ]),  # 需要不同BTD逻辑
        ("只转C14",           [False, False, False, True ]),
        ("只转C13+C14",       [False, False, True,  True ]),
        ("只转C09+C14",       [True,  False, False, True ]),
        ("C09+C13+C14",       [True,  False, True,  True ]),
    ]

    # 加载模型一次
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_ch=7).to(device)
    ckpt = os.path.join(os.path.dirname(__file__), 'unet_rain_pred', 'checkpoints', 'unet_mixed_best.pth')
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"模型加载完成 ({device})")

    for name, mask in combos:
        print(f"\n{'='*50}")
        print(f"方案: {name}  转换通道: C09={mask[0]}, C10={mask[1]}, C13={mask[2]}, C14={mask[3]}")
        print(f"{'='*50}")

        all_px, all_py = [], []

        for fdi_path in fdi_files:
            dt = parse_fy4_datetime(fdi_path)
            gpm_f = find_matching_gpm(dt, FY4B_GPM_DIR)
            if not gpm_f:
                continue

            ir_bt, lons, lats = read_fy4b(fdi_path)
            ir_bt_trans = apply_b2a_selective(ir_bt, mask)

            res_trans = resample_agri_to_region(ir_bt_trans, fdi_path)
            feat = build_features_mixed(res_trans)
            feat_n = normalize(feat)

            prob, rate = unet_predict_patches(model, feat_n, device)

            gpm_rate = read_gpm_rate(gpm_f)
            from data_io import make_gpm_lut
            gpm_lut = make_gpm_lut(gpm_f, lats, lons)
            y_reg = gpm_lut['rate']
            y_true = gpm_lut['label']

            # 三者同时有雨
            m = (y_true == 1) & (rate > 0) & (y_reg > 0.05)
            px = np.clip(y_reg[m].ravel(), 0, 15)
            py = np.clip(rate[m].ravel(), 0, 15)
            # 靠近轴过滤
            keep = (px >= 0.3) & (py >= 0.3)
            all_px.extend(px[keep])
            all_py.extend(py[keep])

        all_px = np.array(all_px)
        all_py = np.array(all_py)

        if len(all_px) > 10:
            r = calc_r(all_px, all_py)
            mae = float(np.abs(all_px - all_py).mean())
            bias = float(all_py.mean() - all_px.mean())
            print(f"  n={len(all_px):,}, R={r:.4f}, MAE={mae:.4f}, Bias={bias:+.4f}")
        else:
            print(f"  样本不足: {len(all_px)}")


if __name__ == "__main__":
    main()
