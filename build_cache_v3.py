"""
build_cache_v3.py
patch-based 缓存构建（v5）— KDTree 加速，堆叠缓存格式
优化：
  - 按 GEO 文件分组，避免 worker 间重复构建 KDTree
  - GPM 文件 worker 内缓存（半小时一帧，~30 个 AGRI 共享）
  - savez 替代 savez_compressed（省掉 LZMA 压缩，快 5-10x）
  - HDF5 chunk cache (rdcc_nbytes) 加速重复读取
"""

import os
import sys
import time
import numpy as np
from multiprocessing import Pool
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    FY4A_AGRI_DIR, GPM_DIR, SAMPLE_CACHE_DIR,
    AGRI_CHANNELS, PATCH_SIZE, PATCH_STRIDE,
    MAX_SAMPLES_PER_DAY, TRAIN_DATES, VAL_DATES,
)
from utils.data_io import (
    parse_fy4_datetime, find_matching_gpm,
    read_agri_disk, read_gpm_imerg,
    resample_gpm_to_region,
    list_agri_files, normalize,
    _find_geo_file, get_tree_and_mapping,
    fast_resample_from_mapping, compute_bt_diffs,
)
from utils.patches import extract_patches

# HDF5 chunk cache: 256 MB
_RDCC_NBYTES = 256 * 1024 * 1024

# ── worker 内 GPM 缓存 ──
_gpm_cache: dict[str, tuple] = {}


def _read_gpm_cached(gpm_f: str) -> tuple:
    """worker 内按路径缓存 GPM 读取，避免同一半小时文件重复 IO。"""
    if gpm_f not in _gpm_cache:
        _gpm_cache[gpm_f] = read_gpm_imerg(gpm_f)
    return _gpm_cache[gpm_f]


def process_single_file(agri_f, gpm_f, cache_dir, geo_file=None):
    """处理单个文件，返回 patch 数或 -1。"""
    basename = os.path.splitext(os.path.basename(agri_f))[0]
    cache_path = os.path.join(cache_dir, f"{basename}_patches.npz")

    if os.path.exists(cache_path):
        try:
            npz = np.load(cache_path)
            return int(npz['x'].shape[0])
        except Exception:
            pass

    try:
        if geo_file is None:
            geo_file = _find_geo_file(agri_f)
        if geo_file is None:
            return -1

        mapping = get_tree_and_mapping(geo_file)

        agri_data, _, _ = read_agri_disk(agri_f, AGRI_CHANNELS,
                                         rdcc_nbytes=_RDCC_NBYTES)
        ir_resampled = fast_resample_from_mapping(agri_data, mapping)

        btd = compute_bt_diffs(ir_resampled)
        x = np.concatenate([ir_resampled, btd], axis=0).astype(np.float32)
        x = normalize(x)

        gpm_precip, gpm_lons, gpm_lats = _read_gpm_cached(gpm_f)
        label_cls, label_reg = resample_gpm_to_region(gpm_precip, gpm_lons, gpm_lats)
        label_reg = np.log1p(label_reg).astype(np.float32)

        C, H, W = x.shape
        if H < PATCH_SIZE or W < PATCH_SIZE:
            return -1

        patches = extract_patches(x, label_cls, label_reg, PATCH_SIZE, PATCH_STRIDE)
        if not patches:
            return -1

        x_all = np.stack([p[0] for p in patches])
        cls_all = np.stack([p[1] for p in patches])
        reg_all = np.stack([p[2] for p in patches])

        # 不压缩：省掉 LZMA，写入快 5-10x
        tmp = cache_path + f'.tmp.{os.getpid()}'
        np.savez(tmp, x=x_all, cls=cls_all, reg=reg_all)
        tmp_actual = tmp + '.npz'
        if os.path.exists(tmp_actual):
            os.replace(tmp_actual, cache_path)
        else:
            raise FileNotFoundError(f"savez 未生成预期文件: {tmp_actual}")

        return len(patches)
    except Exception as e:
        print(f"[WARN] {os.path.basename(agri_f)}: {type(e).__name__}: {e}", flush=True)
        return -1


def collect_file_pairs(dates):
    """收集所有 AGRI-GPM 文件对。"""
    pairs = []
    for date_str in dates:
        date_dir = os.path.join(FY4A_AGRI_DIR, date_str)
        if not os.path.isdir(date_dir):
            continue
        files = list_agri_files(date_dir)
        if not files:
            continue
        files = files[:MAX_SAMPLES_PER_DAY]
        for agri_f in files:
            agri_time = parse_fy4_datetime(agri_f)
            if agri_time is None:
                continue
            gpm_f = find_matching_gpm(agri_time, GPM_DIR)
            if gpm_f is None:
                continue
            pairs.append((agri_f, gpm_f))
    return pairs


def geo_worker(args):
    """
    每个 worker 处理一组 GEO 文件对应的所有 AGRI 文件。
    worker 内部共享 KDTree 缓存 + GPM 缓存。
    """
    geo_group, cache_dir = args
    results = []
    for agri_f, gpm_f, geo_f in geo_group:
        results.append(process_single_file(agri_f, gpm_f, cache_dir, geo_file=geo_f))
    return results


def main():
    os.makedirs(SAMPLE_CACHE_DIR, exist_ok=True)

    all_dates = TRAIN_DATES + VAL_DATES
    print(f"收集文件对（{len(all_dates)} 天）...")
    pairs = collect_file_pairs(all_dates)
    print(f"共 {len(pairs)} 对文件")

    # 按 GEO 文件分组（geo_file 一起传入避免 worker 内重复查找）
    geo_to_pairs = defaultdict(list)
    for agri_f, gpm_f in pairs:
        geo_f = _find_geo_file(agri_f)
        if geo_f:
            geo_to_pairs[geo_f].append((agri_f, gpm_f, geo_f))

    unique_geos = list(geo_to_pairs.keys())
    unique_gpm = len(set(gpm_f for _, gpm_f in pairs))
    print(f"唯一 GEO 文件数: {len(unique_geos)}")
    print(f"唯一 GPM 文件数: {unique_gpm}")

    # 将 GEO 文件均匀分配给 workers
    n_workers = 10
    worker_geo_lists = [[] for _ in range(n_workers)]
    for i, geo_f in enumerate(unique_geos):
        worker_geo_lists[i % n_workers].extend(geo_to_pairs[geo_f])

    args_list = []
    for w in range(n_workers):
        if worker_geo_lists[w]:
            args_list.append((worker_geo_lists[w], SAMPLE_CACHE_DIR))

    print(f"分配给 {len(args_list)} 个 worker，每个处理 ~{len(pairs)//len(args_list)} 文件")
    print(f"开始处理（{PATCH_SIZE}x{PATCH_SIZE} patch）...")
    t0 = time.time()

    total_success = 0
    total_failed = 0
    total_patches = 0

    with Pool(len(args_list)) as pool:
        for worker_results in pool.imap_unordered(geo_worker, args_list):
            for result in worker_results:
                if result > 0:
                    total_success += 1
                    total_patches += result
                else:
                    total_failed += 1

            processed = total_success + total_failed
            if processed % 200 == 0:
                elapsed = time.time() - t0
                speed = processed / elapsed
                eta = (len(pairs) - processed) / speed if speed > 0 else 0
                print(f"  [{processed}/{len(pairs)}] 成功={total_success} 失败={total_failed} "
                      f"patches={total_patches} 速度={speed:.1f}/s ETA={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n完成！耗时 {elapsed:.0f}s")
    print(f"成功: {total_success} 文件, 失败: {total_failed} 文件")
    print(f"总 patch 数: {total_patches}")
    print(f"缓存目录: {SAMPLE_CACHE_DIR}")


if __name__ == '__main__':
    main()
