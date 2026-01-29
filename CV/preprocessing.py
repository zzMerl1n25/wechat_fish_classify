import time
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import cv2
import numpy as np
from PIL import Image


# ===================== 你只需要改这里 =====================
IN_ROOT  = Path(r"D:\wechat_fish_classify\CV\Image_data\WildFish++_Release")
OUT_ROOT = Path(r"D:\wechat_fish_classify\CV\Image_data\WildFish++_preprocessed")

OUT_SIZE = 384

INCLUDE_BASE = True       # 是否输出 base 图（仅拉伸到正方形+resize） -> _000
NUM_AUGS = 5              # 增强图数量（旋转后裁切） -> _001.._00x
ROTATE_DEG = 180           # 增强时旋转角度：[-20, +20]

JPG_QUALITY = 95
WORKERS = 8               # NVMe可8；普通SSD 4~6；机械2~4
PRINT_EVERY_DONE = 200    # 完成多少张“原图任务”打印一次进度

# 保险规则：增强图裁掉>30%就丢弃重做
MIN_KEEP_RATIO = 0.70     # 保留面积比例 >= 0.70 才合格（裁掉<=30%）
MAX_RETRY_PER_OUT = 20    # 每张增强输出最多重试次数
# =========================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# -------------------- Base：直接拉伸成正方形 --------------------
def resize_stretch_square(pil_img: Image.Image) -> Image.Image:
    """
    base 图：不裁剪、不padding，直接拉伸/压缩到 OUT_SIZE×OUT_SIZE
    """
    return pil_img.resize((OUT_SIZE, OUT_SIZE), Image.BICUBIC)


# -------------------- 增强：旋转(黑边)→定位→裁切 --------------------
def rotate_with_black(img_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        img_bgr, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return rotated


def estimate_bbox_by_edges_ignore_black(img_bgr: np.ndarray, min_area=150):
    h, w = img_bgr.shape[:2]

    max_side = 720
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img_small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    else:
        img_small = img_bgr

    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    valid_mask = (gray > 10).astype(np.uint8) * 255

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.bitwise_and(edges, edges, mask=valid_mask)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None

    x, y, bw, bh = cv2.boundingRect(c)

    if scale != 1.0:
        x = int(x / scale)
        y = int(y / scale)
        bw = int(bw / scale)
        bh = int(bh / scale)

    x1, y1, x2, y2 = x, y, x + bw, y + bh
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    if (x2 - x1) * (y2 - y1) < 0.01 * w * h:
        return None

    return (x1, y1, x2, y2)


def make_square_crop_containing_bbox(bbox, img_w, img_h, rng: random.Random,
                                    margin=0.18, scale_range=(1.4, 2.2), shift_range=0.12):
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    mx = int(bw * margin)
    my = int(bh * margin)
    x1e = max(0, x1 - mx)
    y1e = max(0, y1 - my)
    x2e = min(img_w, x2 + mx)
    y2e = min(img_h, y2 + my)

    bw2 = x2e - x1e
    bh2 = y2e - y1e
    base = max(bw2, bh2)

    side = int(base * rng.uniform(*scale_range))
    side = max(side, base)
    side = min(side, max(img_w, img_h))

    cx = (x1 + x2) // 2 + int(rng.uniform(-shift_range, shift_range) * bw)
    cy = (y1 + y2) // 2 + int(rng.uniform(-shift_range, shift_range) * bh)

    rx1 = cx - side // 2
    ry1 = cy - side // 2
    rx2 = rx1 + side
    ry2 = ry1 + side

    if rx1 < 0:
        rx2 -= rx1; rx1 = 0
    if ry1 < 0:
        ry2 -= ry1; ry1 = 0
    if rx2 > img_w:
        dx = rx2 - img_w
        rx1 -= dx; rx2 = img_w
        rx1 = max(0, rx1)
    if ry2 > img_h:
        dy = ry2 - img_h
        ry1 -= dy; ry2 = img_h
        ry1 = max(0, ry1)

    if not (rx1 <= x1 and ry1 <= y1 and rx2 >= x2 and ry2 >= y2):
        cx = (x1e + x2e) // 2
        cy = (y1e + y2e) // 2
        side = max(bw2, bh2)
        rx1 = max(0, cx - side // 2)
        ry1 = max(0, cy - side // 2)
        rx2 = min(img_w, rx1 + side)
        ry2 = min(img_h, ry1 + side)

    return (int(rx1), int(ry1), int(rx2), int(ry2))


def aug_one_once(pil_img: Image.Image, rng: random.Random):
    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    angle = rng.uniform(-ROTATE_DEG, ROTATE_DEG) if ROTATE_DEG > 0 else 0.0
    rot_bgr = rotate_with_black(img_bgr, angle) if ROTATE_DEG > 0 else img_bgr
    H, W = rot_bgr.shape[:2]

    bbox = estimate_bbox_by_edges_ignore_black(rot_bgr)

    if bbox is None:
        side = int(min(H, W) * rng.uniform(0.75, 1.0))
        cx = W // 2 + int(rng.uniform(-0.08, 0.08) * W)
        cy = H // 2 + int(rng.uniform(-0.08, 0.08) * H)
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(W, x1 + side)
        y2 = min(H, y1 + side)
    else:
        x1, y1, x2, y2 = make_square_crop_containing_bbox(bbox, W, H, rng)

    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)
    keep_ratio = (crop_w * crop_h) / float(W * H)

    crop_bgr = rot_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    out = Image.fromarray(crop_rgb).resize((OUT_SIZE, OUT_SIZE), Image.BICUBIC)
    return out, keep_ratio


def aug_one_with_retry(pil_img: Image.Image, rng: random.Random):
    best_img = None
    best_ratio = -1.0
    rejected = 0

    for _ in range(MAX_RETRY_PER_OUT):
        out, ratio = aug_one_once(pil_img, rng)
        if ratio > best_ratio:
            best_ratio = ratio
            best_img = out
        if ratio >= MIN_KEEP_RATIO:
            return out, rejected, ratio
        rejected += 1

    return best_img, rejected, best_ratio


def worker_process_one_image(task):
    img_path, out_class_dir = task
    img_path = Path(img_path)
    out_class_dir = Path(out_class_dir)

    try:
        pil = Image.open(img_path).convert("RGB")
    except Exception:
        return (False, 0, 1, 0)

    stem = img_path.stem
    saved = 0
    rejected_total = 0

    base_seed = (hash(str(img_path)) & 0xFFFFFFFF)
    idx = 0

    # base：拉伸到正方形
    if INCLUDE_BASE:
        base_img = resize_stretch_square(pil)
        base_name = f"{stem}_{idx:03d}.jpg"  # _000
        base_img.save(out_class_dir / base_name, quality=JPG_QUALITY)
        saved += 1
        idx += 1

    # aug：旋转后裁切（带保险重试）
    for k in range(NUM_AUGS):
        rng = random.Random(base_seed + 1000 + k)
        out_img, rejected, keep_ratio = aug_one_with_retry(pil, rng)
        rejected_total += rejected

        out_name = f"{stem}_{idx:03d}.jpg"   # _001..（如果 INCLUDE_BASE=True）
        out_img.save(out_class_dir / out_name, quality=JPG_QUALITY)
        saved += 1
        idx += 1

    return (True, saved, 0, rejected_total)


def main():
    assert IN_ROOT.exists(), f"输入目录不存在：{IN_ROOT}"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([p for p in IN_ROOT.iterdir() if p.is_dir()])
    per_img_out = (1 if INCLUDE_BASE else 0) + NUM_AUGS

    print(f"[INFO] Classes: {len(class_dirs)}")
    print(f"[INFO] WORKERS={WORKERS}, OUT_SIZE={OUT_SIZE}")
    print(f"[INFO] INCLUDE_BASE={INCLUDE_BASE}, NUM_AUGS={NUM_AUGS} => per image outputs={per_img_out}")
    print(f"[INFO] AUG ROTATE_DEG=±{ROTATE_DEG}")
    print(f"[INFO] MIN_KEEP_RATIO={MIN_KEEP_RATIO}, MAX_RETRY_PER_OUT={MAX_RETRY_PER_OUT}")
    print(f"[INFO] Input : {IN_ROOT}")
    print(f"[INFO] Output: {OUT_ROOT}\n")

    tasks = []
    for cd in class_dirs:
        out_cd = OUT_ROOT / cd.name
        out_cd.mkdir(parents=True, exist_ok=True)
        for p in cd.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                tasks.append((str(p), str(out_cd)))

    total = len(tasks)
    print(f"[INFO] Total input images: {total}")
    print(f"[INFO] Estimated output images: {total} * {per_img_out} = {total * per_img_out}\n")

    start = time.time()
    done = 0
    saved_total = 0
    failed_total = 0
    rejected_total = 0

    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(worker_process_one_image, t) for t in tasks]
        for fut in as_completed(futures):
            ok, saved, failed, rejected = fut.result()
            done += 1
            saved_total += saved
            failed_total += failed
            rejected_total += rejected

            if done % PRINT_EVERY_DONE == 0 or done == total:
                elapsed = time.time() - start
                speed = done / elapsed if elapsed > 0 else 0.0
                remain = total - done
                eta = (remain / speed) if speed > 0 else 0.0
                pct = (done / total * 100.0) if total > 0 else 0.0

                print(
                    f"[PROGRESS] {done}/{total} ({pct:.2f}%) | saved={saved_total} | failed={failed_total} | "
                    f"rejected={rejected_total} | {speed:.2f} img/s | ETA {fmt_time(eta)}"
                )

    print("\n[DONE]")
    print(f"Input images processed: {done}")
    print(f"Output images saved:   {saved_total}")
    print(f"Failed reads:          {failed_total}")
    print(f"Rejected (retries):    {rejected_total}")
    print(f"Total time:            {fmt_time(time.time() - start)}")
    print(f"Output root:           {OUT_ROOT}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
