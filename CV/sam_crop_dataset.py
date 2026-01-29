# sam_crop_dataset.py
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ====== 你需要改的 3 个路径 ======
SRC_ROOT = r"D:\wechat_fish_classify\CV\Image_data\WildFish++_Release_split"         # 原始分类数据集根目录（含 train/val/test）
DST_ROOT = r"D:\wechat_fish_classify\CV\Image_data\WildFish++_Release_split_CROP"   # 输出裁剪版数据集根目录
SAM_CKPT  = r"D:\wechat_fish_classify\CV\sam\sam_vit_b_01ec64.pth"                                # SAM 权重文件
MODEL_TYPE = "vit_b"  # vit_b / vit_l / vit_h

# ====== 一些稳健性参数（你这种“一鱼很大”通常默认就够） ======
MIN_AREA_RATIO = 0.30     # mask 面积占比太小 -> 可能不是鱼
MAX_AREA_RATIO = 0.95     # 太大 -> 可能把整张背景都当成前景
PAD = 0.12                # bbox 外扩一点，避免裁掉鱼鳍
COPY_IF_FAIL = True       # 没找到合适 mask 时是否复制原图到输出

REQUIRE_LANDSCAPE = True   # 竖向裁剪视为错误
MIN_WH_RATIO = 1.05        # 宽高比阈值：w/h >= 1.05 才算“横向”（可改 1.1 更严格）

def imread_cn(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def imwrite_cn(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", img)[1].tofile(str(path))

def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2

def pad_bbox(x1, y1, x2, y2, w, h, pad=0.1):
    bw, bh = (x2 - x1), (y2 - y1)
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(w - 1, int(x2 + bw * pad))
    y2 = min(h - 1, int(y2 + bh * pad))
    return x1, y1, x2, y2

def choose_best_mask(masks, img_w, img_h):
    """
    选择最可能是“鱼主体”的 mask：
    - 面积比例在阈值内
    - （可选）要求 bbox 为横向：w/h >= MIN_WH_RATIO
    - 优先面积大，其次 stability_score / predicted_iou
    返回 best_mask（或 None）
    """
    img_area = img_w * img_h
    candidates = []

    for m in masks:
        area = float(m.get("area", 0.0))
        ratio = area / max(img_area, 1)
        if ratio < MIN_AREA_RATIO or ratio > MAX_AREA_RATIO:
            continue

        seg = m.get("segmentation", None)
        if seg is None:
            continue

        box = bbox_from_mask(seg)
        if box is None:
            continue

        x1, y1, x2, y2 = box
        bw = (x2 - x1 + 1)
        bh = (y2 - y1 + 1)
        if bw <= 0 or bh <= 0:
            continue

        wh_ratio = bw / bh  # 宽高比

        # ✅ 保险：竖向长方形认为剪错了，直接排除
        if REQUIRE_LANDSCAPE and wh_ratio < MIN_WH_RATIO:
            continue

        stability = float(m.get("stability_score", 0.0))
        piou = float(m.get("predicted_iou", 0.0))

        # 排序关键字：面积优先，其次稳定性/自评iou
        key = (area, stability, piou)
        candidates.append((key, m))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def main():
    src = Path(SRC_ROOT)
    dst = Path(DST_ROOT)
    dst.mkdir(parents=True, exist_ok=True)

    # 初始化 SAM 自动分割器（官方就支持 “generate masks for an entire image” 的用法）:contentReference[oaicite:3]{index=3}
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CKPT)

    # 如果你本机有 CUDA，取消注释会更快：
    import torch
    if torch.cuda.is_available():
         sam = sam.to("cuda")

    import torch
    print("[INFO] torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("[INFO] cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[INFO] gpu:", torch.cuda.get_device_name(0))

    # 你如果有把 sam.to("cuda") 打开，下面会显示 cuda:0
    try:
        print("[INFO] sam device:", next(sam.parameters()).device)
    except Exception as e:
        print("[WARN] cannot read sam device:", repr(e))

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=8,  # 先 8，速度会明显提升；不够再调 16
        crop_n_layers=0,  # 关掉多层裁剪（很耗时）
        pred_iou_thresh=0.90,
        stability_score_thresh=0.95,
        box_nms_thresh=0.7,
        min_mask_region_area=0,  # 可选：不做小区域后处理，省时间
    )
    # 收集所有图片
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_imgs = []
    for split in ["train", "val", "test"]:
        split_dir = src / split
        if not split_dir.exists():
            continue
        for p in split_dir.rglob("*"):
            if p.suffix.lower() in exts:
                all_imgs.append(p)

    report_path = dst / "bbox_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["src_path", "dst_path", "ok", "x1", "y1", "x2", "y2", "mask_area_ratio"])

        for p in tqdm(all_imgs, desc="SAM -> BBox -> Crop"):
            rel = p.relative_to(src)
            out_path = (dst / rel).with_suffix(".jpg")

            img = imread_cn(p)
            if img is None:
                writer.writerow([str(p), str(out_path), 0, "", "", "", "", ""])
                continue

            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            masks = mask_generator.generate(img_rgb)
            best = choose_best_mask(masks, w, h)

            if best is None:
                # 失败：复制原图 or 跳过
                if COPY_IF_FAIL:
                    imwrite_cn(out_path, img)
                writer.writerow([str(p), str(out_path), 0, "", "", "", "", ""])
                continue

            box = bbox_from_mask(best["segmentation"])
            if box is None:
                if COPY_IF_FAIL:
                    imwrite_cn(out_path, img)
                writer.writerow([str(p), str(out_path), 0, "", "", "", "", ""])
                continue

            x1, y1, x2, y2 = pad_bbox(*box, w, h, pad=PAD)
            bw = x2 - x1 + 1
            bh = y2 - y1 + 1
            if REQUIRE_LANDSCAPE and (bw / max(bh, 1)) < MIN_WH_RATIO:
                # 认为剪错：按失败处理
                if COPY_IF_FAIL:
                    imwrite_cn(out_path, img)
                writer.writerow([str(p), str(out_path), 0, "", "", "", "", ""])
                continue

            crop = img[y1:y2 + 1, x1:x2 + 1]
            if crop.size == 0:
                if COPY_IF_FAIL:
                    imwrite_cn(out_path, img)
                writer.writerow([str(p), str(out_path), 0, "", "", "", "", ""])
                continue

            imwrite_cn(out_path, crop)

            area_ratio = float(best.get("area", 0.0)) / max(w * h, 1)
            writer.writerow([str(p), str(out_path), 1, x1, y1, x2, y2, f"{area_ratio:.6f}"])

    print("Done.")
    print("Cropped dataset:", dst)
    print("Report saved:", report_path)

if __name__ == "__main__":
    main()
