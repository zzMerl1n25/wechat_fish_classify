# -*- coding: utf-8 -*-
"""
EfficientNetV2-M 图像分类训练脚本（train/val/test 三集合）
✅ DDP 多卡训练（torchrun 启动）
✅ 每个进程绑定自己的 GPU（LOCAL_RANK）
✅ DistributedSampler（避免重复采样）
✅ 指标 all_reduce 汇总（loss/acc/rmse）
✅ 只让 rank0 打印/保存/画图（避免重复写文件）
✅ 断点续训（resume）：中断后可继续
✅ 记录并绘制：
   - train/val loss/acc/rmse/lr
   - train-val gap（val_loss - train_loss）
   - epoch_time（每轮耗时）
"""

import os
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# =========================================================
# ✅ 超参集中在这里
# =========================================================

# 数据与输出（相对脚本所在目录最稳）
DATA_ROOT = "training_data/WildFish++_split3_stem_keepaug"
OUT_DIR   = "runs/effv2s_run02"  # 手动修改路径：run0x 便于对比多次实验

# 输入尺寸（显存/速度不够可改 320/256/224）
IMG_SIZE = 384

# 训练轮数与 batch（DDP 下：BATCH_SIZE 是“每张卡”的 batch）
EPOCHS = 30
BATCH_SIZE = 32

# DataLoader
NUM_WORKERS = 16
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 16

# 优化器与学习率
LR = 3e-4
WEIGHT_DECAY = 1e-3
OPTIMIZER = "adamw"             # "adamw" 或 "sgd"
MOMENTUM = 0.9

# 学习率调度器
SCHEDULER = "cosine"            # "cosine" / "step" / "none"
COSINE_TMAX = EPOCHS
STEP_SIZE = 10
GAMMA = 0.1
 
# 损失函数
LABEL_SMOOTHING = 0.05

# ✅ 抗过拟合：分类头 Dropout（建议 0.2~0.4）
DROPOUT_P = 0.3

# 训练技巧
USE_AMP = True
GRAD_CLIP_NORM = 1.0            # 0 表示不开

# 早停
USE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 3
EARLY_STOP_METRIC = "val_loss"   # "val_acc" 或 "val_loss"

# 在线增强（轻量）
USE_LIGHT_AUG = True
HFLIP_PROB = 0.5
COLOR_JITTER = 0.1
RANDOM_ERASING_PROB = 0.0

# 打印频率（只 rank0 打印）
PRINT_EVERY = 20

# 随机种子
SEED = 42

# ===== 断点续训（resume）相关 =====
RESUME = True
RESUME_PATH = ""
SAVE_EVERY_EPOCH = True

# =========================================================


# ------------------ DDP 工具 ------------------
def ddp_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if ddp_is_on() else 0

def get_world_size() -> int:
    return dist.get_world_size() if ddp_is_on() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def setup_ddp():
    """
    torchrun 会提供环境变量：RANK / WORLD_SIZE / LOCAL_RANK
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # env:// 默认从环境变量读取地址/端口/排名
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return True, local_rank, device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, device

def cleanup_ddp():
    if ddp_is_on():
        dist.destroy_process_group()


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
def seed_everything(seed: int, rank: int = 0):
    # 每个 rank 用不同 seed，避免 DataLoader/增强完全一致
    seed = int(seed) + int(rank) * 1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ------------------ 指标 ------------------
@torch.no_grad()
def rmse_prob_vs_onehot(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
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

        # torchvision 的 v2_s classifier 是 Sequential(Dropout, Linear)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

        # ✅ 可选：调整 dropout（不想改就删掉这几行）
        if "DROPOUT_P" in globals():
            try:
                model.classifier[0].p = float(DROPOUT_P)
            except Exception:
                pass

        return model, "torchvision_efficientnet_v2_s_imagenet1k"

    except Exception as e:
        try:
            import timm
            # timm 的 v2-s 版本（名字有多种，这个一般可用）
            model = timm.create_model("tf_efficientnetv2_s", pretrained=True, num_classes=num_classes)
            return model, "timm_tf_efficientnetv2_s"
        except Exception:
            raise RuntimeError(
                "构建 EfficientNetV2-S 失败。\n"
                "请升级 torchvision（包含 efficientnet_v2_s），或安装 timm。\n"
                f"torchvision error: {repr(e)}"
            )

# ------------------ transforms ------------------
def build_transforms():
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    if USE_LIGHT_AUG:
        t_train = [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=HFLIP_PROB),
        ]
        if COLOR_JITTER > 0:
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
        if RANDOM_ERASING_PROB > 0:
            t_train.append(transforms.RandomErasing(p=RANDOM_ERASING_PROB, value=0))
        train_tf = transforms.Compose(t_train)
    else:
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    return train_tf, val_tf


# ------------------ 断点续训：保存/加载 ------------------
def unwrap_model(m):
    return m.module if isinstance(m, DDP) else m

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
    """
    ⚠️ 注意：epoch_time_sec 不再在这里 append 到 history（避免重复记录）
    """
    m = unwrap_model(model)

    ckpt = {
        "epoch": epoch,
        "model_state": m.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": (scheduler.state_dict() if scheduler is not None else None),
        "scaler_state": (scaler.state_dict() if scaler is not None else None),
        "history": history,
        "best_metric": best_metric,
        "patience_counter": patience_counter,
        "model_tag": model_tag,
        "class_to_idx": class_to_idx,
        "epoch_time_sec_last": float(epoch_time_sec),
    }
    torch.save(ckpt, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    start_epoch = int(ckpt["epoch"]) + 1
    history = ckpt.get("history", None)
    best_metric = ckpt.get("best_metric", None)
    patience_counter = int(ckpt.get("patience_counter", 0))

    return start_epoch, history, best_metric, patience_counter


# ------------------ 训练/验证：DDP 汇总指标 ------------------
def all_reduce_sums(loss_sum, correct_sum, rmse_sum, n_samples, device):
    """
    汇总四个标量（double），DDP 下所有进程 all_reduce 求和
    """
    if not ddp_is_on():
        return loss_sum, correct_sum, rmse_sum, n_samples

    t = torch.tensor(
        [loss_sum, correct_sum, rmse_sum, n_samples],
        device=device,
        dtype=torch.float64
    )
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0].item()), float(t[1].item()), float(t[2].item()), float(t[3].item())


def run_one_epoch(model, loader, criterion, optimizer, device, num_classes,
                  scaler=None, train=True, epoch_idx=1, total_epochs=1, print_every=20):

    if train:
        model.train()
    else:
        model.eval()

    # 用“sum”累计，然后 DDP all_reduce 汇总
    loss_sum = 0.0
    correct_sum = 0.0
    rmse_sum = 0.0
    n_samples = 0.0

    n_batches = len(loader)
    t_epoch_start = time.time()
    t_prev = time.time()

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        t_after_load = time.time()
        data_time = t_after_load - t_prev

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        bs = labels.size(0)
        n_samples += float(bs)

        if train:
            optimizer.zero_grad(set_to_none=True)

            amp_enabled = (USE_AMP and device.type == "cuda" and scaler is not None)
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    logits = model(images)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()

                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()

                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, labels)

        # batch 指标（用于 rank0 打印趋势）
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            batch_correct = (preds == labels).sum().item()
            batch_acc = batch_correct / labels.size(0)
            batch_rmse = rmse_prob_vs_onehot(logits, labels, num_classes)

        batch_loss = float(loss.item())

        # ✅ epoch 累计：注意 correct_sum 应累计“正确个数”，不是acc
        loss_sum += batch_loss * bs
        correct_sum += float(batch_correct)
        rmse_sum += float(batch_rmse) * bs

        t_after_compute = time.time()
        compute_time = t_after_compute - t_after_load

        # epoch ETA（rank0 打印即可）
        elapsed_epoch = t_after_compute - t_epoch_start
        avg_batch_time = elapsed_epoch / max(batch_idx, 1)
        eta_epoch_sec = avg_batch_time * max(n_batches - batch_idx, 0)

        if is_main_process() and ((batch_idx == 1) or (batch_idx == n_batches) or (batch_idx % max(1, print_every) == 0)):
            if device.type == "cuda":
                mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
                mem_resv  = torch.cuda.memory_reserved(device) / (1024 ** 3)
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
                f"{ips:.1f} img/s | ETA(epoch)={format_seconds(eta_epoch_sec)}{mem_str}"
            )

        t_prev = time.time()

    # ✅ DDP 汇总得到全局指标
    loss_sum, correct_sum, rmse_sum, n_samples = all_reduce_sums(loss_sum, correct_sum, rmse_sum, n_samples, device)

    epoch_loss = loss_sum / max(n_samples, 1.0)
    epoch_acc  = correct_sum / max(n_samples, 1.0)
    epoch_rmse = rmse_sum / max(n_samples, 1.0)
    epoch_time_sec = time.time() - t_epoch_start

    return float(epoch_loss), float(epoch_acc), float(epoch_rmse), float(epoch_time_sec)


@torch.no_grad()
def evaluate_test_ddp(model, loader, criterion, device, num_classes):
    model.eval()

    loss_sum = 0.0
    rmse_sum = 0.0
    n_samples = 0.0

    # confusion matrix：每 rank 统计一份，然后 all_reduce 求和
    cm_local = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        n_samples += float(bs)
        loss_sum += float(loss.item()) * bs
        rmse_sum += float(rmse_prob_vs_onehot(logits, labels, num_classes)) * bs

        preds = torch.argmax(logits, dim=1)

        # 更新 confusion matrix（矢量化更新）
        idx = labels * num_classes + preds
        binc = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        cm_local += binc.to(torch.int64)

    # 汇总 loss/rmse/n_samples
    loss_sum, _, rmse_sum, n_samples = all_reduce_sums(loss_sum, 0.0, rmse_sum, n_samples, device)

    # 汇总 confusion matrix
    if ddp_is_on():
        dist.all_reduce(cm_local, op=dist.ReduceOp.SUM)

    test_loss = loss_sum / max(n_samples, 1.0)
    test_rmse = rmse_sum / max(n_samples, 1.0)

    cm = cm_local.detach().cpu().numpy()
    metrics = metrics_from_confusion_matrix(cm)

    return {
        "test_loss": float(test_loss),
        "test_rmse": float(test_rmse),
        **metrics,
        "confusion_matrix": cm,
    }


def save_epoch_table_csv(history: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics_epoch.csv"

    n = len(history["train_loss"])
    # ✅ 增加 gap_loss
    lines = ["epoch,lr,train_loss,train_acc,train_rmse,val_loss,val_acc,val_rmse,gap_loss,epoch_time_sec\n"]
    for i in range(n):
        epoch = i + 1
        lr = history["lr"][i] if i < len(history["lr"]) else ""
        gap = history.get("gap_loss", [])
        et = history.get("epoch_time_sec", [])
        gapv = gap[i] if i < len(gap) else ""
        tsec = et[i] if i < len(et) else ""
        lines.append(
            f"{epoch},{lr},{history['train_loss'][i]},{history['train_acc'][i]},{history['train_rmse'][i]},"
            f"{history['val_loss'][i]},{history['val_acc'][i]},{history['val_rmse'][i]},{gapv},{tsec}\n"
        )

    csv_path.write_text("".join(lines), encoding="utf-8")


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
    plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "curve_rmse.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["lr"], label="lr")
    plt.xlabel("Epoch"); plt.ylabel("Learning Rate"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "curve_lr.png", dpi=160)
    plt.close()

    # ✅ 新增：train-val gap（val_loss - train_loss）
    if "gap_loss" in history and len(history["gap_loss"]) == len(epochs):
        plt.figure()
        plt.plot(epochs, history["gap_loss"], label="val_loss - train_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss Gap"); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / "curve_gap_loss.png", dpi=160)
        plt.close()

    # ✅ 新增：epoch_time
    if "epoch_time_sec" in history and len(history["epoch_time_sec"]) == len(epochs):
        plt.figure()
        plt.plot(epochs, history["epoch_time_sec"], label="epoch_time_sec")
        plt.xlabel("Epoch"); plt.ylabel("Seconds"); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / "curve_epoch_time.png", dpi=160)
        plt.close()


def main():
    ddp, local_rank, device = setup_ddp()
    rank = get_rank()
    world = get_world_size()

    seed_everything(SEED, rank=rank)

    # 路径（以当前脚本所在目录为基准最稳）
    BASE_DIR = Path(__file__).resolve().parent
    data_root = (BASE_DIR / DATA_ROOT).resolve()
    out_dir = (BASE_DIR / OUT_DIR).resolve()
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)

    # 等待 rank0 创建目录（避免竞争）
    if ddp:
        dist.barrier(device_ids=[local_rank])

    # 保存超参快照（只 rank0 写）
    if is_main_process():
        hparams = {k: v for k, v in globals().items() if k.isupper()}
        (out_dir / "hparams.json").write_text(
            json.dumps(hparams, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8"
        )

    train_dir = data_root / "train"
    val_dir   = data_root / "val"
    test_dir  = data_root / "test"
    assert train_dir.exists() and val_dir.exists() and test_dir.exists(), \
        f"找不到 train/val/test 目录，请检查：{data_root}"

    if is_main_process():
        print(f"[INFO] DDP={ddp} | rank={rank}/{world} | local_rank={local_rank} | device={device}")

    train_tf, val_tf = build_transforms()

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(val_dir), transform=val_tf)
    test_ds  = datasets.ImageFolder(str(test_dir), transform=val_tf)

    assert train_ds.class_to_idx == val_ds.class_to_idx, "❌ val 的 class_to_idx 与 train 不一致！检查 val 是否缺类/文件夹结构不同"
    assert train_ds.class_to_idx == test_ds.class_to_idx, "❌ test 的 class_to_idx 与 train 不一致！"

    num_classes = len(train_ds.classes)
    if is_main_process():
        print(f"[INFO] Classes: {num_classes}")
        print(f"[INFO] Example classes: {train_ds.classes[:10]}{'...' if len(train_ds.classes) > 10 else ''}")

    # 保存类别映射（只 rank0 写）
    if is_main_process():
        (out_dir / "class_to_idx.json").write_text(
            json.dumps(train_ds.class_to_idx, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    # Sampler（DDP 必需）
    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True) if ddp else None
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world, rank=rank, shuffle=False) if ddp else None
    test_sampler  = DistributedSampler(test_ds,  num_replicas=world, rank=rank, shuffle=False) if ddp else None

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = PERSISTENT_WORKERS
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = DataLoader(train_ds, sampler=train_sampler, shuffle=(train_sampler is None), **loader_kwargs)
    val_loader   = DataLoader(val_ds,   sampler=val_sampler,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  sampler=test_sampler,  shuffle=False, **loader_kwargs)

    # 模型
    model, model_tag = build_model(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    if OPTIMIZER.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)
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

    # ===== Resume（先 load，再 wrap DDP）=====
    best_metric = None
    patience_counter = 0
    history = None
    start_epoch = 1

    best_path = out_dir / "best_model.pt"
    last_path = out_dir / "last_model.pt"
    last_ckpt_path = Path(RESUME_PATH) if RESUME_PATH else (out_dir / "last_checkpoint.pt")

    if RESUME and last_ckpt_path.exists():
        if is_main_process():
            print(f"[INFO] Resume enabled. Loading checkpoint: {last_ckpt_path}")
        start_epoch, history, best_metric, patience_counter = load_checkpoint(
            last_ckpt_path, model, optimizer, scheduler, scaler, device
        )
        if is_main_process():
            print(f"[INFO] Resumed from epoch {start_epoch}. best_metric={best_metric}, patience={patience_counter}")
    else:
        if is_main_process():
            print("[INFO] Resume not used (no checkpoint found or RESUME=False).")

    # 现在 wrap DDP（关键）
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if is_main_process():
        print(f"[INFO] Model: {model_tag}")

    # history 初始化
    if history is None:
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [],  "val_acc": [],
            "train_rmse": [], "val_rmse": [],
            "lr": [],
            "epoch_time_sec": [],
            "gap_loss": [],
        }
    else:
        if "epoch_time_sec" not in history:
            history["epoch_time_sec"] = []
        if "gap_loss" not in history:
            history["gap_loss"] = []

    def estimate_total_eta_seconds(current_epoch: int) -> float:
        remaining_epochs = max(EPOCHS - current_epoch, 0)
        times = history.get("epoch_time_sec", [])
        if len(times) >= 1:
            avg_epoch = float(sum(times) / len(times))
            return avg_epoch * remaining_epochs
        return None

    t_train_start = time.time()

    # ===== 训练循环 =====
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()

        # DDP：每个 epoch 要 set_epoch 才会正确 shuffle
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        lr_now = optimizer.param_groups[0]["lr"]
        if is_main_process():
            history["lr"].append(lr_now)

        train_loss, train_acc, train_rmse, _ = run_one_epoch(
            model, train_loader, criterion, optimizer, device, num_classes,
            scaler=scaler, train=True, epoch_idx=epoch, total_epochs=EPOCHS, print_every=PRINT_EVERY
        )
        val_loss, val_acc, val_rmse, _ = run_one_epoch(
            model, val_loader, criterion, optimizer, device, num_classes,
            scaler=None, train=False, epoch_idx=epoch, total_epochs=EPOCHS, print_every=PRINT_EVERY
        )

        # 只 rank0 记录历史、保存文件
        if is_main_process():
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["train_rmse"].append(train_rmse)
            history["val_rmse"].append(val_rmse)

        if scheduler is not None:
            scheduler.step()

        # early stop / best（rank0 决策，然后广播）
        stop_flag = torch.tensor(0, device=device, dtype=torch.int64)

        if is_main_process():
            if EARLY_STOP_METRIC == "val_acc":
                current = val_acc
                improved = (best_metric is None) or (current > best_metric + 1e-8)
            elif EARLY_STOP_METRIC == "val_loss":
                current = val_loss
                improved = (best_metric is None) or (current < best_metric - 1e-8)
            else:
                raise ValueError("EARLY_STOP_METRIC 只能是 'val_acc' 或 'val_loss'")

            if improved:
                best_metric = float(current)
                patience_counter = 0
                # 保存 best
                torch.save({
                    "epoch": epoch,
                    "model_state": unwrap_model(model).state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "class_to_idx": train_ds.class_to_idx,
                    "img_size": IMG_SIZE,
                    "model_tag": model_tag,
                }, best_path)
            else:
                patience_counter += 1

            # 保存 last
            torch.save({
                "epoch": epoch,
                "model_state": unwrap_model(model).state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_metric": best_metric,
                "class_to_idx": train_ds.class_to_idx,
                "img_size": IMG_SIZE,
                "model_tag": model_tag,
            }, last_path)

            epoch_time_total = time.time() - epoch_start

            # ✅ gap_loss / epoch_time 只在这里记录一次
            history["gap_loss"].append(val_loss - train_loss)
            history["epoch_time_sec"].append(epoch_time_total)

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
                f"best({EARLY_STOP_METRIC})={best_metric} | "
                f"patience={patience_counter}/{EARLY_STOP_PATIENCE} | "
                f"epoch_time={format_seconds(epoch_time_total)} | "
                f"ETA(total)={format_seconds(total_eta)} | "
                f"elapsed={format_seconds(elapsed_total)}"
            )

            if USE_EARLY_STOP and patience_counter >= EARLY_STOP_PATIENCE:
                stop_flag.fill_(1)

        # 广播停止信号
        if ddp:
            dist.broadcast(stop_flag, src=0)

        if int(stop_flag.item()) == 1:
            if is_main_process():
                print("[INFO] Early stopping triggered.")
            break

        # rank0 保存/写文件后，让其他 rank 同步一下
        if ddp:
            dist.barrier(device_ids=[local_rank])

    # 训练完：只 rank0 画图/保存 history
    if is_main_process():
        save_curves(history, out_dir)
        (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        save_epoch_table_csv(history, out_dir)

    if ddp:
        dist.barrier(device_ids=[local_rank])

    # ===== Test：用 best（DDP 汇总 confusion matrix）=====
    if is_main_process():
        print("\n[INFO] Evaluating on TEST set with best checkpoint...")

    # 所有 rank 都 load best（保持一致）
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        unwrap_model(model).load_state_dict(ckpt["model_state"])
    else:
        if is_main_process():
            print("[WARN] best_model.pt 不存在，将用当前模型进行测试评估。")

    test_result = evaluate_test_ddp(model, test_loader, criterion, device, num_classes)
    cm = test_result.pop("confusion_matrix")

    if is_main_process():
        np.savetxt(out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
        (out_dir / "test_metrics.json").write_text(json.dumps(test_result, ensure_ascii=False, indent=2), encoding="utf-8")

        print("[TEST RESULT]")
        print(json.dumps(test_result, ensure_ascii=False, indent=2))
        print(f"\n[DONE] All outputs saved to: {out_dir}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
