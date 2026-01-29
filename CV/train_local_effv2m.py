# -*- coding: utf-8 -*-
"""
EfficientNetV2-S 图像分类训练脚本（train/val/test 三集合）
✅ batch 级进度打印（含 data/compute/显存）
✅ 显示剩余时间 ETA（batch级：本epoch剩余；epoch级：全训练剩余）
✅ 断点续训（resume）：中断后可继续（PyTorch 2.6+ 兼容：weights_only=False）
✅ 强增强 + MixUp/CutMix：缓解小样本多类过拟合

数据目录格式（必须）：
DATA_ROOT/
  train/classA/*.jpg
  val/classA/*.jpg
  test/classA/*.jpg
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import random
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image


# =========================================================
# ✅ 超参集中在这里
# =========================================================

# 数据与输出
DATA_ROOT = os.getenv("TRAIN_DATA_ROOT", "CHANGE_ME_TRAIN_DATA_ROOT")
OUT_DIR   = os.getenv("TRAIN_OUT_DIR", "CHANGE_ME_TRAIN_OUT_DIR")

# 输入尺寸
IMG_SIZE = 224

# 训练轮数与 batch
EPOCHS = 30
BATCH_SIZE = 16
DROPOUT_P = 0.3

# DataLoader（Windows 稳定优先：NUM_WORKERS=0 最稳）
NUM_WORKERS = 2
PIN_MEMORY = True               # CUDA 建议 True（Windows 也可以）
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# 优化器与学习率
LR = 3e-4
WEIGHT_DECAY = 1e-3
OPTIMIZER = "adamw"             # "adamw" 或 "sgd"
MOMENTUM = 0.9                  # 只对 SGD 有效

# 学习率调度器
SCHEDULER = "cosine"            # "cosine" / "step" / "none"
COSINE_TMAX = EPOCHS
STEP_SIZE = 10
GAMMA = 0.1

# 损失函数
LABEL_SMOOTHING = 0.1           # 0~0.2：防过拟合常用

# 训练技巧
USE_AMP = True                  # 混合精度（CUDA 才会启用）
GRAD_CLIP_NORM = 1.0            # 0 表示不开

# 早停
USE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 4
EARLY_STOP_METRIC = "val_loss"  # "val_acc" 或 "val_loss"

# ========= 强增强（小样本建议开）=========
USE_STRONG_AUG = False
HFLIP_PROB = 0.5
COLOR_JITTER = 0.1              # 可选：0~0.2
USE_TRIVIAL_AUG = True          # TrivialAugmentWide（非常推荐）
RRC_SCALE = (0.6, 1.0)          # RandomResizedCrop scale
RRC_RATIO = (0.75, 1.33)        # RandomResizedCrop ratio
RANDOM_ERASING_PROB = 0.25      # 0~0.25 常用

# ========= MixUp / CutMix（小样本多类强烈建议开）=========
USE_MIXUP_CUTMIX = False
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
MIX_PROB = 1.0                  # 每个 batch 都做混合
SWITCH_PROB = 0.5               # 0.5：MixUp/CutMix 各一半

# 训练过程打印（每多少个 batch 打一次）
PRINT_EVERY = 20

# 随机种子
SEED = 42

# ===== 断点续训（resume）相关 =====
RESUME = True
RESUME_PATH = ""
SAVE_EVERY_EPOCH = True

# =========================================================


# ------------------ 小工具：格式化时间 ------------------
def format_seconds(sec: float) -> str:
    if sec is None or sec < 0:
        return "--:--:--"
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ------------------ 固定随机性 ------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ------------------ 指标：accuracy + rmse ------------------
@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


@torch.no_grad()
def rmse_prob_vs_onehot(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """
    分类 RMSE：softmax 概率分布 vs one-hot 标签（越低越好）
    注意：MixUp/CutMix 训练时这个指标参考意义不大，主要看 val。
    """
    probs = torch.softmax(logits, dim=1)
    onehot = torch.zeros((labels.size(0), num_classes), device=logits.device, dtype=probs.dtype)
    onehot.scatter_(1, labels.unsqueeze(1), 1.0)
    mse = torch.mean((probs - onehot) ** 2)
    return float(torch.sqrt(mse).item())


def confusion_matrix_np(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, y in zip(preds, labels):
        cm[y, p] += 1
    return cm


def metrics_from_confusion_matrix(cm: np.ndarray):
    tp = np.diag(cm).astype(np.float64)
    fp = np.sum(cm, axis=0).astype(np.float64) - tp
    fn = np.sum(cm, axis=1).astype(np.float64) - tp

    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(tp + fn, 1e-12)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    macro_p = float(np.mean(precision))
    macro_r = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))
    acc = float(tp.sum() / np.maximum(cm.sum(), 1))

    return {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "accuracy": acc,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
    }


# ------------------ 模型：EfficientNetV2-S ------------------
def build_model(num_classes: int):
    try:
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

        # 调 dropout
        try:
            model.classifier[0].p = float(DROPOUT_P)
        except Exception:
            pass

        return model, "torchvision_efficientnet_v2_s_imagenet1k"
    except Exception as e:
        try:
            import timm
            model = timm.create_model("tf_efficientnetv2_s", pretrained=True, num_classes=num_classes)
            return model, "timm_tf_efficientnetv2_s"
        except Exception:
            raise RuntimeError(
                "构建 EfficientNetV2-S 失败。\n"
                "请升级 torchvision（包含 efficientnet_v2_s），或安装 timm。\n"
                f"torchvision error: {repr(e)}"
            )


class ResizeWithPad:
    """
    等比例缩放 + padding 成正方形（letterbox）
    - 输入：PIL Image
    - 输出：PIL Image（正方形）
    """
    def __init__(self, size: int, fill=(0, 0, 0)):
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


# ------------------ transforms ------------------
def build_transforms():
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    # padding 颜色：用 mean*255 更中性；不过 val/test 影响很小
    pad_fill_rgb = (123, 116, 103)

    if USE_STRONG_AUG:
        t_train = [
            transforms.RandomResizedCrop(
                IMG_SIZE, scale=RRC_SCALE, ratio=RRC_RATIO,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip(p=HFLIP_PROB),
        ]

        if USE_TRIVIAL_AUG:
            t_train.append(transforms.TrivialAugmentWide())

        if COLOR_JITTER and COLOR_JITTER > 0:
            t_train.append(
                transforms.ColorJitter(
                    brightness=COLOR_JITTER,
                    contrast=COLOR_JITTER,
                    saturation=COLOR_JITTER,
                    hue=min(0.05, COLOR_JITTER),
                )
            )

        t_train += [
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]

        if RANDOM_ERASING_PROB and RANDOM_ERASING_PROB > 0:
            t_train.append(transforms.RandomErasing(p=RANDOM_ERASING_PROB, value=0))

        train_tf = transforms.Compose(t_train)
    else:
        train_tf = transforms.Compose([
            ResizeWithPad(IMG_SIZE, fill=pad_fill_rgb),
            transforms.RandomHorizontalFlip(p=HFLIP_PROB),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

    val_tf = transforms.Compose([
        ResizeWithPad(IMG_SIZE, fill=pad_fill_rgb),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    return train_tf, val_tf


# ------------------ MixUp/CutMix ------------------
def rand_bbox(W, H, lam):
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def mixup_cutmix(images, labels):
    """
    返回：mixed_images, y_a, y_b, lam, mode
    loss = lam * CE(pred, y_a) + (1-lam) * CE(pred, y_b)
    """
    if (not USE_MIXUP_CUTMIX) or (MIX_PROB <= 0):
        return images, labels, labels, 1.0, "none"

    if np.random.rand() > MIX_PROB:
        return images, labels, labels, 1.0, "none"

    bs = images.size(0)
    index = torch.randperm(bs, device=images.device)
    y_a = labels
    y_b = labels[index]

    use_cutmix = (np.random.rand() < SWITCH_PROB)

    if use_cutmix and CUTMIX_ALPHA and CUTMIX_ALPHA > 0:
        lam = np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA)
        W = images.size(3)
        H = images.size(2)
        x1, y1, x2, y2 = rand_bbox(W, H, lam)
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(W * H))
        return images, y_a, y_b, float(lam), "cutmix"

    if MIXUP_ALPHA and MIXUP_ALPHA > 0:
        lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
        mixed = lam * images + (1.0 - lam) * images[index]
        return mixed, y_a, y_b, float(lam), "mixup"

    return images, labels, labels, 1.0, "none"


# ------------------ 断点续训：保存/加载 checkpoint ------------------
def save_checkpoint(path: Path,
                    epoch: int,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    history: dict,
                    best_metric,
                    patience_counter: int,
                    model_tag: str,
                    class_to_idx: dict,
                    epoch_time_sec: float):
    if "epoch_time_sec" not in history:
        history["epoch_time_sec"] = []
    history["epoch_time_sec"].append(float(epoch_time_sec))

    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": (scheduler.state_dict() if scheduler is not None else None),
        "scaler_state": (scaler.state_dict() if scaler is not None else None),
        "history": history,
        "best_metric": best_metric,
        "patience_counter": patience_counter,
        "model_tag": model_tag,
        "class_to_idx": class_to_idx,

        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }
    torch.save(ckpt, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler, device):
    # PyTorch 2.6+ 默认 weights_only=True，会导致你这种包含 numpy/random 状态的 ckpt 报错
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    try:
        random.setstate(ckpt["rng_python"])
        np.random.set_state(ckpt["rng_numpy"])
        torch.set_rng_state(ckpt["rng_torch"])
        if torch.cuda.is_available() and ckpt.get("rng_torch_cuda") is not None:
            torch.cuda.set_rng_state_all(ckpt["rng_torch_cuda"])
    except Exception:
        pass

    start_epoch = int(ckpt["epoch"]) + 1
    history = ckpt.get("history", None)
    best_metric = ckpt.get("best_metric", None)
    patience_counter = int(ckpt.get("patience_counter", 0))

    return start_epoch, history, best_metric, patience_counter


# ------------------ 训练/验证：带进度打印 + epoch ETA ------------------
def run_one_epoch(model, loader, criterion, optimizer, device, num_classes,
                  scaler=None, train=True, epoch_idx=1, total_epochs=1, print_every=20):
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_rmse = 0.0
    n_samples = 0

    n_batches = len(loader)
    t_epoch_start = time.time()
    t_prev = time.time()

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        t_after_load = time.time()
        data_time = t_after_load - t_prev

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        bs = labels.size(0)
        n_samples += bs

        if train:
            optimizer.zero_grad(set_to_none=True)

            # ✅ MixUp/CutMix（仅训练）
            images, y_a, y_b, lam, mix_mode = mixup_cutmix(images, labels)

            amp_enabled = (USE_AMP and device.type == "cuda" and scaler is not None)
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    logits = model(images)
                    loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)

                scaler.scale(loss).backward()

                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
                loss.backward()

                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                optimizer.step()

            # 训练 acc：mix 情况下仅参考（按 y_a）
            if USE_MIXUP_CUTMIX and lam < 0.999:
                batch_acc = float(accuracy_from_logits(logits, y_a))
                batch_rmse = float(rmse_prob_vs_onehot(logits, y_a, num_classes))
            else:
                batch_acc = float(accuracy_from_logits(logits, labels))
                batch_rmse = float(rmse_prob_vs_onehot(logits, labels, num_classes))

        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, labels)

            batch_acc = float(accuracy_from_logits(logits, labels))
            batch_rmse = float(rmse_prob_vs_onehot(logits, labels, num_classes))

        batch_loss = float(loss.item())

        epoch_loss += batch_loss * bs
        epoch_acc  += batch_acc * bs
        epoch_rmse += batch_rmse * bs

        t_after_compute = time.time()
        compute_time = t_after_compute - t_after_load

        elapsed_epoch = t_after_compute - t_epoch_start
        avg_batch_time = elapsed_epoch / max(batch_idx, 1)
        eta_epoch_sec = avg_batch_time * max(n_batches - batch_idx, 0)

        if (batch_idx == 1) or (batch_idx == n_batches) or (batch_idx % max(1, print_every) == 0):
            if device.type == "cuda":
                mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                mem_resv  = torch.cuda.memory_reserved() / (1024 ** 3)
                mem_str = f" | cuda_mem={mem_alloc:.2f}G alloc, {mem_resv:.2f}G resv"
            else:
                mem_str = ""

            ips = (batch_idx * bs) / max(elapsed_epoch, 1e-6)
            mode = "TRAIN" if train else "VAL"

            print(
                f"[{mode}] ep {epoch_idx}/{total_epochs} | "
                f"batch {batch_idx}/{n_batches} | "
                f"loss={batch_loss:.4f} acc={batch_acc:.4f} rmse={batch_rmse:.4f} | "
                f"data={data_time:.3f}s compute={compute_time:.3f}s | "
                f"{ips:.1f} img/s | "
                f"ETA(epoch)={format_seconds(eta_epoch_sec)}{mem_str}"
            )

        t_prev = time.time()

    epoch_loss /= max(n_samples, 1)
    epoch_acc  /= max(n_samples, 1)
    epoch_rmse /= max(n_samples, 1)
    epoch_time_sec = time.time() - t_epoch_start
    return epoch_loss, epoch_acc, epoch_rmse, float(epoch_time_sec)


@torch.no_grad()
def evaluate_test(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    n_samples = 0

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        n_samples += bs
        total_loss += float(loss.item()) * bs
        total_rmse += float(rmse_prob_vs_onehot(logits, labels, num_classes)) * bs

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        ys = labels.detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(ys)

    total_loss /= max(n_samples, 1)
    total_rmse /= max(n_samples, 1)

    all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([], dtype=np.int64)
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([], dtype=np.int64)

    cm = confusion_matrix_np(all_preds, all_labels, num_classes)
    metrics = metrics_from_confusion_matrix(cm)

    return {
        "test_loss": float(total_loss),
        "test_rmse": float(total_rmse),
        **metrics,
        "confusion_matrix": cm,
    }


def save_curves(history: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "curve_loss.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "curve_accuracy.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_rmse"], label="train_rmse")
    plt.plot(epochs, history["val_rmse"], label="val_rmse")
    plt.xlabel("Epoch"); plt.ylabel("RMSE (prob vs one-hot)"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "curve_rmse.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["lr"], label="lr")
    plt.xlabel("Epoch"); plt.ylabel("Learning Rate"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "curve_lr.png", dpi=160)
    plt.close()


def main():
    if "CHANGE_ME" in str(DATA_ROOT) or "CHANGE_ME" in str(OUT_DIR):
        raise ValueError("请设置 TRAIN_DATA_ROOT / TRAIN_OUT_DIR 环境变量，或直接修改脚本路径。")
    seed_everything(SEED)

    data_root = Path(DATA_ROOT)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存超参快照
    hparams = {k: v for k, v in globals().items() if k.isupper()}
    (out_dir / "hparams.json").write_text(json.dumps(hparams, ensure_ascii=False, indent=2), encoding="utf-8")

    train_dir = data_root / "train"
    val_dir   = data_root / "val"
    test_dir  = data_root / "test"
    assert train_dir.exists() and val_dir.exists() and test_dir.exists(), \
        f"找不到 train/val/test 目录，请检查：{DATA_ROOT}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    train_tf, val_tf = build_transforms()

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(val_dir), transform=val_tf)
    test_ds  = datasets.ImageFolder(str(test_dir), transform=val_tf)

    num_classes = len(train_ds.classes)
    print(f"[INFO] Classes: {num_classes}")
    print(f"[INFO] Example classes: {train_ds.classes[:10]}{'...' if len(train_ds.classes) > 10 else ''}")

    (out_dir / "class_to_idx.json").write_text(
        json.dumps(train_ds.class_to_idx, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = PERSISTENT_WORKERS
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    model, model_tag = build_model(num_classes)
    model = model.to(device)
    print(f"[INFO] Model: {model_tag}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    if OPTIMIZER.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                              weight_decay=WEIGHT_DECAY, nesterov=True)
    else:
        raise ValueError("OPTIMIZER 只能是 'adamw' 或 'sgd'")

    if SCHEDULER.lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=COSINE_TMAX)
    elif SCHEDULER.lower() == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    elif SCHEDULER.lower() == "none":
        scheduler = None
    else:
        raise ValueError("SCHEDULER 只能是 'cosine'/'step'/'none'")

    scaler = torch.amp.GradScaler('cuda', enabled=(USE_AMP and device.type == "cuda"))

    best_metric = None
    patience_counter = 0
    history = None
    start_epoch = 1

    best_path = out_dir / "best_model.pt"
    last_path = out_dir / "last_model.pt"
    last_ckpt_path = Path(RESUME_PATH) if RESUME_PATH else (out_dir / "last_checkpoint.pt")

    if RESUME and last_ckpt_path.exists():
        print(f"[INFO] Resume enabled. Loading checkpoint: {last_ckpt_path}")
        start_epoch, history, best_metric, patience_counter = load_checkpoint(
            last_ckpt_path, model, optimizer, scheduler, scaler, device
        )
        print(f"[INFO] Resumed from epoch {start_epoch}. best_metric={best_metric}, patience={patience_counter}")
    else:
        print("[INFO] Resume not used (no checkpoint found or RESUME=False).")

    if history is None:
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [],  "val_acc": [],
            "train_rmse": [], "val_rmse": [],
            "lr": [],
            "epoch_time_sec": [],
        }
    else:
        if "epoch_time_sec" not in history:
            history["epoch_time_sec"] = []

    def estimate_total_eta_seconds(current_epoch: int) -> float:
        remaining_epochs = max(EPOCHS - current_epoch, 0)
        times = history.get("epoch_time_sec", [])
        if len(times) >= 1:
            avg_epoch = float(sum(times) / len(times))
            return avg_epoch * remaining_epochs
        return None

    t_train_start = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()

        lr_now = optimizer.param_groups[0]["lr"]
        history["lr"].append(lr_now)

        train_loss, train_acc, train_rmse, train_time = run_one_epoch(
            model, train_loader, criterion, optimizer, device, num_classes,
            scaler=scaler, train=True, epoch_idx=epoch, total_epochs=EPOCHS, print_every=PRINT_EVERY
        )
        val_loss, val_acc, val_rmse, val_time = run_one_epoch(
            model, val_loader, criterion, optimizer, device, num_classes,
            scaler=None, train=False, epoch_idx=epoch, total_epochs=EPOCHS, print_every=PRINT_EVERY
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)

        if scheduler is not None:
            scheduler.step()

        if EARLY_STOP_METRIC == "val_acc":
            current = val_acc
            improved = (best_metric is None) or (current > best_metric + 1e-8)
        elif EARLY_STOP_METRIC == "val_loss":
            current = val_loss
            improved = (best_metric is None) or (current < best_metric - 1e-8)
        else:
            raise ValueError("EARLY_STOP_METRIC 只能是 'val_acc' 或 'val_loss'")

        if improved:
            best_metric = current
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_metric": best_metric,
                "class_to_idx": train_ds.class_to_idx,
                "img_size": IMG_SIZE,
                "model_tag": model_tag,
            }, best_path)
        else:
            patience_counter += 1

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_metric": best_metric,
            "class_to_idx": train_ds.class_to_idx,
            "img_size": IMG_SIZE,
            "model_tag": model_tag,
        }, last_path)

        epoch_time_total = time.time() - epoch_start

        if SAVE_EVERY_EPOCH:
            save_checkpoint(
                path=last_ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                history=history,
                best_metric=best_metric,
                patience_counter=patience_counter,
                model_tag=model_tag,
                class_to_idx=train_ds.class_to_idx,
                epoch_time_sec=epoch_time_total
            )

        total_eta = estimate_total_eta_seconds(epoch)
        elapsed_total = time.time() - t_train_start

        print(
            f"[EPOCH {epoch:03d}/{EPOCHS}] lr={lr_now:.2e} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} rmse={train_rmse:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} rmse={val_rmse:.4f} | "
            f"best({EARLY_STOP_METRIC})={best_metric:.6f} | "
            f"patience={patience_counter}/{EARLY_STOP_PATIENCE} | "
            f"epoch_time={format_seconds(epoch_time_total)} | "
            f"ETA(total)={format_seconds(total_eta)} | "
            f"elapsed={format_seconds(elapsed_total)}"
        )

        if USE_EARLY_STOP and patience_counter >= EARLY_STOP_PATIENCE:
            print("[INFO] Early stopping triggered.")
            break

    save_curves(history, out_dir)
    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[INFO] Evaluating on TEST set with best checkpoint...")
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    else:
        print("[WARN] best_model.pt 不存在，将用当前模型进行测试评估。")

    test_result = evaluate_test(model, test_loader, criterion, device, num_classes)
    cm = test_result.pop("confusion_matrix")

    np.savetxt(out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    (out_dir / "test_metrics.json").write_text(json.dumps(test_result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[TEST RESULT]")
    print(json.dumps(test_result, ensure_ascii=False, indent=2))
    print(f"\n[DONE] All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
