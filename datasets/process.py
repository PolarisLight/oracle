#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================== 配置 ==================
# 原始数据根目录
SRC_ROOT = Path("H:\DatasetD\HUST-OBC\deciphered")

# 搬运后的目标根目录
DROOT = Path("H:\DatasetD\HUST-OBC\processed")

# 支持的图片扩展名
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

# 并行线程数
NUM_WORKERS = 8
# ========================================


def is_combo_folder(name: str) -> bool:
    """名字里有下划线且每段都是数字"""
    if "_" not in name:
        return False
    return all(p.isdigit() for p in name.split("_"))


def iter_image_files(root: Path):
    """递归遍历 root 下的所有图片文件"""
    for dirpath, _, filenames in os.walk(root):
        dp = Path(dirpath)
        for fn in filenames:
            if Path(fn).suffix.lower() in EXTS:
                yield dp / fn


def unique_destination(dst_dir: Path, filename: str) -> Path:
    """避免覆盖，加序号"""
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = dst_dir / filename
    idx = 1
    while candidate.exists():
        candidate = dst_dir / f"{base}_{idx}{ext}"
        idx += 1
    return candidate


def copy_one(src: Path, dst_dir: Path):
    """单文件复制"""
    dst = unique_destination(dst_dir, src.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return src, dst


def process_folder(folder: Path):
    """处理一个类文件夹（不论单类还是组合类）"""
    rel = folder.relative_to(SRC_ROOT)
    dst_dir = DROOT / rel
    dst_dir.mkdir(parents=True, exist_ok=True)

    futures = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        if is_combo_folder(folder.name):
            # 组合类 -> 从子文件夹取图
            parts = folder.name.split("_")
            for p in parts:
                sub = folder / p
                if not sub.exists() or not sub.is_dir():
                    continue
                for img in iter_image_files(sub):
                    futures.append(ex.submit(copy_one, img, dst_dir))
        else:
            # 普通类 -> 直接取图
            for img in iter_image_files(folder):
                futures.append(ex.submit(copy_one, img, dst_dir))

        results = []
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {folder.name}"):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"[ERROR] {e}")
        return results


def main():
    all_folders = [d for d in SRC_ROOT.iterdir() if d.is_dir()]

    print(f"[INFO] Found {len(all_folders)} class folders to process")
    for folder in all_folders:
        process_folder(folder)

    print("[DONE]")




if __name__ == "__main__":
    # main()
    pass
