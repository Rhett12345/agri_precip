"""
train_unet.py
轻量 UNet 双头训练：分类（有雨/无雨）+ 回归（log1p 降水量）
直接读取 npz 缓存，patch-based 空间训练
"""

import os
import re
import sys
import time
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from config import SAMPLE_CACHE_DIR, MODEL_SAVE_DIR, FIGURE_DIR
from model import UNet

torch.backends.cudnn.benchmark = True

# ============================================================
# 目标
# ============================================================
GOAL_CSI = 0.35
GOAL_R = 0.50
GOAL_POD = 0.60
GOAL_FAR = 0.60


# ============================================================
# 数据加载（patch-based，保留空间结构）
# ============================================================

def load_patches(cache_dir, split_dates, max_files=None):
    """
    从 npz 缓存加载 patch 数据，保留 (C, H, W) 空间结构。
    返回: x (N, 7, 128, 128), cls (N, 128, 128), reg (N, 128, 128)
    """
    all_npz = sorted(glob.glob(os.path.join(cache_dir, '*.npz')))
    if not all_npz:
        raise FileNotFoundError(f"无缓存文件: {cache_dir}")

    if split_dates:
        date_set = set(split_dates)
        filtered = []
        for f in all_npz:
            basename = os.path.basename(f)
            # 从文件名提取 8 位日期做 set 查找，O(1) 替代逐日期子串匹配
            dates_in_name = re.findall(r'\d{8}', basename)
            if any(d in date_set for d in dates_in_name):
                filtered.append(f)
        all_npz = filtered

    if max_files:
        all_npz = all_npz[:max_files]

    print(f"加载 {len(all_npz)} 个文件...")

    x_list, cls_list, reg_list = [], [], []
    for i, f in enumerate(all_npz):
        try:
            npz = np.load(f)
            x_list.append(npz['x'])      # (16, 7, 128, 128)
            cls_list.append(npz['cls'])   # (16, 128, 128)
            reg_list.append(npz['reg'])   # (16, 128, 128)
        except Exception as e:
            if i < 3:
                print(f"  跳过 {os.path.basename(f)}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  已加载 {i+1}/{len(all_npz)} 文件")

    if not x_list:
        raise RuntimeError(f"所有文件加载失败: {cache_dir}")

    x = np.concatenate(x_list, axis=0).astype(np.float32)
    cls = np.concatenate(cls_list, axis=0).astype(np.float32)
    reg = np.concatenate(reg_list, axis=0).astype(np.float32)

    print(f"总 patch: {len(x)}, 有雨像素比例: {cls.mean()*100:.1f}%")
    return x, cls, reg


# ============================================================
# 指标计算
# ============================================================

def calc_metrics(y_true_cls, prob, threshold, y_true_reg=None, y_pred_reg=None):
    pred_cls = (prob >= threshold).astype(int)
    tp = ((pred_cls == 1) & (y_true_cls == 1)).sum()
    fp = ((pred_cls == 1) & (y_true_cls == 0)).sum()
    fn = ((pred_cls == 0) & (y_true_cls == 1)).sum()
    tn = ((pred_cls == 0) & (y_true_cls == 0)).sum()

    csi = tp / (tp + fp + fn + 1e-6)
    pod = tp / (tp + fn + 1e-6)
    far = fp / (tp + fp + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    f1 = 2 * prec * pod / (prec + pod + 1e-6)

    result = dict(csi=csi, pod=pod, far=far, prec=prec, f1=f1,
                  tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

    if y_pred_reg is not None:
        rain_mask = y_true_cls == 1
        if rain_mask.sum() > 0:
            p = y_pred_reg[rain_mask]
            t = y_true_reg[rain_mask]
            r = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-8 else 0.0
            mae = float(np.abs(p - t).mean())
            rmse = float(np.sqrt(np.mean((p - t) ** 2)))
            bias_ratio = float(p.mean() / (t.mean() + 1e-8))
            result.update(r=r, mae=mae, rmse=rmse, bias_ratio=bias_ratio)
        else:
            result.update(r=0, mae=0, rmse=0, bias_ratio=1.0)

    return result


def threshold_scan(prob, y_true_cls, y_true_reg, y_pred_reg,
                   thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.10, 0.71, 0.05)

    best_csi = 0
    best_th = 0.5
    results = []

    for th in thresholds:
        m = calc_metrics(y_true_cls, prob, th, y_true_reg, y_pred_reg)
        results.append((th, m))
        if m['csi'] > best_csi:
            best_csi = m['csi']
            best_th = th

    return best_th, best_csi, results


def print_goal_check(m, label=""):
    """打印 Goal Check 结果，返回是否全部通过。"""
    title = f"{label} Goal Check" if label else "Goal Check"
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    goals = [
        ('CSI', m['csi'], GOAL_CSI, '≥'),
        ('POD', m['pod'], GOAL_POD, '≥'),
        ('FAR', m['far'], GOAL_FAR, '<'),
    ]
    if 'r' in m:
        goals.append(('R', m['r'], GOAL_R, '≥'))
    if 'bias_ratio' in m:
        br = m['bias_ratio']
        br_ok = 0.7 <= br <= 1.3
        print(f"  BiasRatio: {br:.3f} {'✓' if br_ok else '✗'} (目标 0.7~1.3)")

    all_pass = True
    for name, val, target, op in goals:
        if op == '≥':
            ok = val >= target
        else:
            ok = val < target
        print(f"  {name}: {val:.4f} {op} {target} {'✓' if ok else '✗'}")
        if not ok:
            all_pass = False

    if 'bias_ratio' in m:
        br = m['bias_ratio']
        if not (0.7 <= br <= 1.3):
            all_pass = False

    if all_pass:
        print(f"\n{label + ' ' if label else ''}所有目标达成！")
    else:
        print(f"\n{label + ' ' if label else ''}部分目标未达成")
    return all_pass


# ============================================================
# 训练循环
# ============================================================

def _intensity_weights(reg_log_vals: torch.Tensor) -> torch.Tensor:
    """
    按降水量级给回归损失加权。reg_log_vals 在 log1p 空间。
    mm/h 阈值: 2, 5, 10, 15  →  log1p: 1.099, 1.792, 2.398, 2.773
    """
    w = torch.ones_like(reg_log_vals)
    w = w + 1.0 * (reg_log_vals > 1.099)    # 2-5 mm/h  → w=2
    w = w + 6.0 * (reg_log_vals > 1.792)    # 5-10 mm/h → w=8
    w = w + 7.0 * (reg_log_vals > 2.398)    # 10-15 mm/h → w=15
    w = w + 0.0 * (reg_log_vals > 2.773)    # >15 mm/h 保持 w=15
    return w


def train_one_epoch(model, loader, optimizer, device, reg_weight=1.0,
                    scaler=None, accum_steps=1, intensity_weight=False):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    n_batches = 0

    optimizer.zero_grad(set_to_none=True)
    for step, (x, cls, reg) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        cls = cls.to(device, non_blocking=True)
        reg = reg.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            cls_logit, reg_log = model(x)
            cls_loss = nn.functional.binary_cross_entropy_with_logits(
                cls_logit.squeeze(1), cls)
            rain_mask = cls > 0.5
            if rain_mask.sum() > 0:
                reg_pred = reg_log.squeeze(1)[rain_mask]
                reg_true = reg[rain_mask]
                if intensity_weight:
                    w = _intensity_weights(reg_true)
                    diff = reg_pred - reg_true
                    abs_diff = diff.abs()
                    huber = torch.where(abs_diff < 1.0,
                                        0.5 * abs_diff ** 2,
                                        abs_diff - 0.5)
                    reg_loss = (w * huber).sum() / w.sum()
                else:
                    reg_loss = nn.functional.smooth_l1_loss(reg_pred, reg_true)
            else:
                reg_loss = torch.tensor(0.0, device=device)
            loss = (cls_loss + reg_weight * reg_loss) / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps
        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()
        n_batches += 1

    return (total_loss / n_batches,
            total_cls_loss / n_batches,
            total_reg_loss / n_batches)


@torch.no_grad()
def validate(model, x_np, cls_np, reg_np, device, batch_size=16):
    """
    在整个验证集上跑推理，返回 (prob, pred_reg) 摊平为像素级。
    x_np: (N, 7, 128, 128), cls_np/reg_np: (N, 128, 128)
    """
    model.eval()
    prob_list = []
    reg_list = []

    for i in range(0, len(x_np), batch_size):
        x_batch = torch.from_numpy(x_np[i:i+batch_size]).to(device, non_blocking=True)
        cls_logit, reg_log = model(x_batch)
        prob_list.append(torch.sigmoid(cls_logit).cpu().numpy())
        reg_list.append(reg_log.cpu().numpy())

    prob = np.concatenate(prob_list, axis=0)   # (N, 1, 128, 128)
    pred_reg_log = np.concatenate(reg_list, axis=0)

    # 摊平为像素级
    prob_flat = prob.reshape(-1)
    pred_reg_log_flat = pred_reg_log.reshape(-1)
    cls_flat = cls_np.reshape(-1)
    reg_flat = reg_np.reshape(-1)

    # expm1 还原到 mm/h
    pred_reg_flat = np.expm1(pred_reg_log_flat).clip(0, 50)
    reg_flat = np.expm1(reg_flat).clip(0, 50)

    return prob_flat, pred_reg_flat, cls_flat, reg_flat


# ============================================================
# 主函数
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-train-files', type=int, default=2000)
    parser.add_argument('--max-val-files', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--reg-weight', type=float, default=1.0)
    parser.add_argument('--accum-steps', type=int, default=1,
                        help='梯度累积步数，等效 batch=batch_size*accum_steps')
    parser.add_argument('--no-amp', action='store_true',
                        help='禁用混合精度训练')
    parser.add_argument('--base-ch', type=int, default=32,
                        help='UNet 基础通道数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='AdamW 权重衰减')
    parser.add_argument('--intensity-weight', action='store_true',
                        help='回归损失按降水量级加权（改善高值低估）')
    args = parser.parse_args()

    # 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    from config import TRAIN_DATES, VAL_DATES, TEST_DATES

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    print("=" * 60)
    print("UNet 双头训练 (log1p/expm1)")
    print("=" * 60)

    # 加载数据
    print("\n[1/5] 加载训练数据...")
    t0 = time.time()
    x_train, cls_train, reg_train = load_patches(
        SAMPLE_CACHE_DIR, TRAIN_DATES, max_files=args.max_train_files)
    print(f"  耗时 {time.time()-t0:.0f}s")

    print("\n[2/5] 加载验证数据...")
    t0 = time.time()
    x_val, cls_val, reg_val = load_patches(
        SAMPLE_CACHE_DIR, VAL_DATES, max_files=args.max_val_files)
    print(f"  耗时 {time.time()-t0:.0f}s")

    # DataLoader
    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(cls_train),
        torch.from_numpy(reg_train),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)

    # 模型
    model = UNet(in_ch=7, base_ch=args.base_ch).to(device)
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {n_params:,}")

    use_amp = not args.no_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print("AMP 混合精度: 开启")

    if args.weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)

    # 训练
    print(f"\n[3/5] 训练 ({args.epochs} epochs, batch={args.batch_size}, "
          f"accum={args.accum_steps}, effective_batch={args.batch_size * args.accum_steps})...")
    best_csi = 0
    best_epoch = 0
    best_th_global = 0.5

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, cls_loss, reg_loss = train_one_epoch(
            model, train_loader, optimizer, device, args.reg_weight,
            scaler=scaler, accum_steps=args.accum_steps,
            intensity_weight=args.intensity_weight)
        elapsed = time.time() - t0

        # 每 5 个 epoch 或最后一个 epoch 做验证
        if epoch % 5 == 0 or epoch == args.epochs:
            prob_val, pred_reg_val, cls_val_flat, reg_val_flat = validate(
                model, x_val, cls_val, reg_val, device, batch_size=args.batch_size)
            best_th, val_csi, _ = threshold_scan(
                prob_val, cls_val_flat, reg_val_flat, pred_reg_val)
            m = calc_metrics(cls_val_flat, prob_val, best_th,
                             reg_val_flat, pred_reg_val)
            scheduler.step(val_csi)

            marker = ""
            if val_csi > best_csi:
                best_csi = val_csi
                best_epoch = epoch
                best_th_global = best_th
                # 保存最佳模型
                model_path = os.path.join(MODEL_SAVE_DIR, 'unet_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_csi': best_csi,
                    'best_threshold': best_th,
                }, model_path)
                marker = " ★"

            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"loss={loss:.4f} cls={cls_loss:.4f} reg={reg_loss:.4f}  "
                  f"CSI={m['csi']:.4f} POD={m['pod']:.4f} FAR={m['far']:.4f}  "
                  f"th={best_th:.2f}  {elapsed:.0f}s{marker}")
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"loss={loss:.4f} cls={cls_loss:.4f} reg={reg_loss:.4f}  "
                  f"{elapsed:.0f}s")

    print(f"\n最佳 epoch: {best_epoch}, CSI={best_csi:.4f}, 阈值={best_th_global:.2f}")

    # 加载最佳模型做最终评估
    print("\n[4/5] 验证集最终评估...")
    ckpt = torch.load(os.path.join(MODEL_SAVE_DIR, 'unet_best.pth'),
                       map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']
    # 统一键名：去掉 _orig_mod. 前缀，用原始 UNet 加载
    raw_model = UNet(in_ch=7, base_ch=args.base_ch).to(device)
    clean_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    raw_model.load_state_dict(clean_dict)
    model = raw_model
    best_th_global = ckpt['best_threshold']

    prob_val, pred_reg_val, cls_val_flat, reg_val_flat = validate(
        model, x_val, cls_val, reg_val, device)
    m_best = calc_metrics(cls_val_flat, prob_val, best_th_global,
                          reg_val_flat, pred_reg_val)

    print(f"\n最优阈值 {best_th_global:.2f} 详细指标:")
    print(f"  CSI={m_best['csi']:.4f}  (目标≥{GOAL_CSI})")
    print(f"  POD={m_best['pod']:.4f}  (目标≥{GOAL_POD})")
    print(f"  FAR={m_best['far']:.4f}  (目标<{GOAL_FAR})")
    if 'r' in m_best:
        print(f"  R={m_best['r']:.4f}    (目标≥{GOAL_R})")
    if 'bias_ratio' in m_best:
        print(f"  BiasRatio={m_best['bias_ratio']:.3f} (目标0.7~1.3)")

    print_goal_check(m_best)

    # 测试集评估
    print("\n[5/5] 测试集评估...")
    all_npz = sorted(glob.glob(os.path.join(SAMPLE_CACHE_DIR, '*.npz')))
    test_date_set = set(TEST_DATES)
    has_test = any(
        any(d in test_date_set for d in re.findall(r'\d{8}', os.path.basename(f)))
        for f in all_npz
    )

    if has_test:
        x_test, cls_test, reg_test = load_patches(
            SAMPLE_CACHE_DIR, TEST_DATES, max_files=500)
    else:
        print("  测试日期无缓存，使用验证集后半部分作为泛化检查...")
        n_half = len(x_val) // 2
        x_test, cls_test, reg_test = x_val[n_half:], cls_val[n_half:], reg_val[n_half:]
        x_val, cls_val, reg_val = x_val[:n_half], cls_val[:n_half], reg_val[:n_half]
        prob_val, pred_reg_val, cls_val_flat, reg_val_flat = validate(
            model, x_val, cls_val, reg_val, device)
        best_th_global, _, _ = threshold_scan(
            prob_val, cls_val_flat, reg_val_flat, pred_reg_val)
        # 用新的验证子集重新计算 m_best，保证泛化检查可比
        m_best = calc_metrics(cls_val_flat, prob_val, best_th_global,
                              reg_val_flat, pred_reg_val)

    prob_test, pred_reg_test, cls_test_flat, reg_test_flat = validate(
        model, x_test, cls_test, reg_test, device)
    m_test = calc_metrics(cls_test_flat, prob_test, best_th_global,
                          reg_test_flat, pred_reg_test)

    print(f"\n测试集指标 (阈值={best_th_global:.2f}):")
    print(f"  CSI={m_test['csi']:.4f}  (目标≥{GOAL_CSI})")
    print(f"  POD={m_test['pod']:.4f}  (目标≥{GOAL_POD})")
    print(f"  FAR={m_test['far']:.4f}  (目标<{GOAL_FAR})")
    if 'r' in m_test:
        print(f"  R={m_test['r']:.4f}    (目标≥{GOAL_R})")
    if 'bias_ratio' in m_test:
        print(f"  BiasRatio={m_test['bias_ratio']:.3f} (目标0.7~1.3)")

    print_goal_check(m_test, label="测试集")

    # 泛化检查
    csi_gap = abs(m_best['csi'] - m_test['csi'])
    print(f"\n泛化检查: |val_CSI - test_CSI| = {csi_gap:.4f} "
          f"{'✓' if csi_gap < 0.15 else '✗'} (阈值<0.15)")


if __name__ == '__main__':
    main()
