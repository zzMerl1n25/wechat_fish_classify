# -*- coding: utf-8 -*-
"""
鱼类识别推理服务（FastAPI）— 输入/输出分离版本
- 输入接口：POST /input   （上传图片，返回 request_id）
- 输出接口：GET  /output/{request_id} （按 request_id 获取预测结果）
- 列表接口：GET  /outputs?limit=50    （获取最近 N 条结果）

依赖：
pip install fastapi uvicorn pillow torch torchvision

运行：
python infer_api.py
或：
uvicorn infer_api:app --host <YOUR_HOST> --port <YOUR_PORT>
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 你环境有 OpenMP 冲突就保留

import io
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


# =========================
# 你需要改的路径（指向训练输出目录）
# =========================
RUN_DIR = os.getenv("INFER_RUN_DIR", "").strip()
MODEL_FILE = os.getenv("INFER_MODEL_FILE", "best_model.pt")
CLASS_MAP_FILE = os.getenv("INFER_CLASS_MAP_FILE", "class_to_idx.json")

IMG_SIZE = int(os.getenv("INFER_IMG_SIZE", "224"))
TOPK = int(os.getenv("INFER_TOPK", "5"))

# 端口交给环境变量（或用 uvicorn 命令传入）
HOST = os.getenv("INFER_HOST", "").strip()
PORT = os.getenv("INFER_PORT", "").strip()

# 是否保存上传图片（方便你排查/做历史）
SAVE_INPUT_IMAGE = True
SAVE_DIRNAME = "inference_records"    # 会创建在 RUN_DIR 下面


# =========================
# 与训练一致的 ResizeWithPad
# =========================
class ResizeWithPad:
    def __init__(self, size: int, fill=(123, 116, 103)):
        self.size = int(size)
        self.fill = fill

    def __call__(self, img: Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if w <= 0 or h <= 0:
            return img.resize((self.size, self.size), Image.BILINEAR)

        scale = self.size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        img = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - new_w) // 2
        top = (self.size - new_h) // 2
        canvas.paste(img, (left, top))
        return canvas


def build_transform(img_size: int):
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)
    return transforms.Compose([
        ResizeWithPad(img_size, fill=(123, 116, 103)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])


def build_model(num_classes: int, dropout_p: float = 0.3):
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # 保险：如果你训练时改过 dropout
    try:
        model.classifier[0].p = float(dropout_p)
    except Exception:
        pass

    return model


def load_class_map(run_dir: Path) -> Dict[int, str]:
    p = run_dir / CLASS_MAP_FILE
    if not p.exists():
        raise FileNotFoundError(f"找不到 {CLASS_MAP_FILE}: {p}")
    class_to_idx = json.loads(p.read_text(encoding="utf-8"))
    return {int(v): str(k) for k, v in class_to_idx.items()}


def load_checkpoint_model(run_dir: Path, device: torch.device):
    ckpt_path = run_dir / MODEL_FILE
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "class_to_idx" in ckpt and isinstance(ckpt["class_to_idx"], dict):
        num_classes = len(ckpt["class_to_idx"])
    else:
        num_classes = len(load_class_map(run_dir))

    model = build_model(num_classes=num_classes, dropout_p=0.3)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_topk(model, img: Image.Image, tfm, idx_to_class: Dict[int, str],
                 device: torch.device, topk: int = 5) -> List[Dict[str, Any]]:
    x = tfm(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(topk, probs.numel())
    vals, inds = torch.topk(probs, k=k)

    out = []
    for score, idx in zip(vals.tolist(), inds.tolist()):
        out.append({
            "class_id": int(idx),
            "class_name": idx_to_class.get(int(idx), str(idx)),
            "confidence": float(score),
        })
    return out


# =========================
# FastAPI + 简单“结果存储”（内存 + 可选落盘）
# =========================
app = FastAPI(title="Fish Classifier API", version="1.0")

if not RUN_DIR:
    raise ValueError("INFER_RUN_DIR 为空，请在环境变量中设置模型运行目录。")

RUN_DIR_PATH = Path(RUN_DIR)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TFM = build_transform(IMG_SIZE)
IDX_TO_CLASS = load_class_map(RUN_DIR_PATH)
MODEL = load_checkpoint_model(RUN_DIR_PATH, DEVICE)

RECORD_DIR = RUN_DIR_PATH / SAVE_DIRNAME
RECORD_DIR.mkdir(parents=True, exist_ok=True)

# 内存存储：request_id -> record
RECORDS: Dict[str, Dict[str, Any]] = {}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "num_classes": len(IDX_TO_CLASS),
        "model_file": str((RUN_DIR_PATH / MODEL_FILE)),
        "record_dir": str(RECORD_DIR),
    }


# ---------- 输入接口：上传图片 ----------
@app.post("/input")
async def input_image(file: UploadFile = File(...)):
    """
    前端用 multipart/form-data 上传图片：
    key: file
    value: binary image

    ✅ 方法A：直接返回预测结果（predictions），Django 调一次就够
    """
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")

        request_id = uuid.uuid4().hex
        ts = int(time.time())

        # 推理
        preds = predict_topk(
            model=MODEL,
            img=img,
            tfm=TFM,
            idx_to_class=IDX_TO_CLASS,
            device=DEVICE,
            topk=TOPK
        )

        # 可选：保存图片
        saved_image_path: Optional[str] = None
        if SAVE_INPUT_IMAGE:
            ext = Path(file.filename).suffix.lower()
            if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                ext = ".jpg"
            img_path = RECORD_DIR / f"{ts}_{request_id}{ext}"
            img.save(img_path)
            saved_image_path = str(img_path)

        record = {
            "request_id": request_id,
            "filename": file.filename,
            "timestamp": ts,
            "topk": TOPK,
            "predictions": preds,               # ✅ 直接返回
            "saved_image_path": saved_image_path,
        }

        # 内存记录
        RECORDS[request_id] = record

        # 落盘 json（可选）
        (RECORD_DIR / f"{ts}_{request_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # ✅ 方法A：直接把 record 返回给调用方（Django）
        return JSONResponse({"message": "ok", **record})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": repr(e)})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": repr(e)})


# ---------- 输出接口：按 request_id 取结果 ----------
@app.get("/output/{request_id}")
def output_result(request_id: str):
    rec = RECORDS.get(request_id)
    if rec is None:
        return JSONResponse(status_code=404, content={"error": "request_id not found"})
    return JSONResponse(rec)


# ---------- 输出列表：最近 N 条 ----------
@app.get("/outputs")
def outputs(limit: int = 50):
    limit = max(1, min(int(limit), 500))
    items = list(RECORDS.values())
    items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return JSONResponse({
        "count": min(limit, len(items)),
        "items": items[:limit],
    })


if __name__ == "__main__":
    import uvicorn
    if HOST and PORT:
        uvicorn.run("infer_api:app", host=HOST, port=int(PORT), reload=False)
    else:
        raise ValueError("INFER_HOST/INFER_PORT 为空，请设置环境变量或用 uvicorn 指定。")
