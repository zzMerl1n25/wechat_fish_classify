# find_bad_images.py
import os
from pathlib import Path
import time
from PIL import Image

DATA_ROOT = os.getenv("FIND_BAD_DATA_ROOT", "CHANGE_ME_FIND_BAD_DATA_ROOT")

# 每处理多少张打印一次进度
PRINT_EVERY = 200

def format_seconds(sec: float) -> str:
    if sec is None or sec < 0:
        return "--:--:--"
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def collect_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    return files

def scan_split(split: str):
    root = Path(DATA_ROOT) / split
    files = collect_images(root)
    total = len(files)

    bad = []
    t0 = time.time()

    print(f"\n[SCAN] {split} | total={total}")

    for i, p in enumerate(files, start=1):
        try:
            # verify() 检查文件结构（不完全解码）
            with Image.open(p) as im:
                im.verify()

            # 重新 open + convert 模拟训练时的读图流程
            with Image.open(p) as im:
                im.convert("RGB")

        except Exception as e:
            bad.append((str(p), repr(e)))

        # 进度打印
        if (i == 1) or (i % PRINT_EVERY == 0) or (i == total):
            elapsed = time.time() - t0
            speed = i / max(elapsed, 1e-6)
            remain = total - i
            eta = remain / max(speed, 1e-6)

            percent = (i / total * 100.0) if total > 0 else 100.0
            print(
                f"[{split}] {i}/{total} ({percent:.2f}%) | "
                f"bad={len(bad)} | {speed:.1f} img/s | ETA={format_seconds(eta)}"
            )

    return bad

if __name__ == "__main__":
    if "CHANGE_ME" in str(DATA_ROOT):
        raise ValueError("请设置 FIND_BAD_DATA_ROOT 环境变量，或直接修改脚本路径。")
    all_bad = []

    for split in ["train", "val", "test"]:
        if not (Path(DATA_ROOT) / split).exists():
            print(f"[WARN] split not found: {split} (skip)")
            continue
        bad = scan_split(split)
        print(f"[DONE] {split} bad={len(bad)}")
        all_bad.extend(bad)

    out = Path("bad_images.txt")
    out.write_text("\n".join([f"{p}\t{err}" for p, err in all_bad]), encoding="utf-8")

    print(f"\n[ALL DONE] total_bad={len(all_bad)}")
    print("saved -> bad_images.txt")
