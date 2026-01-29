import os
import re
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ========= 你只需要改这里 =========
DATASET_ROOT = Path(os.getenv("SPLIT_DATASET_ROOT", "CHANGE_ME_SPLIT_DATASET_ROOT"))
OUT_ROOT     = Path(os.getenv("SPLIT_OUT_ROOT", "CHANGE_ME_SPLIT_OUT_ROOT"))
# =================================

SEED = 42
COPY_MODE = True  # True=复制（推荐）；False=移动（会改变原数据）

# 1) test 占 stem 组的比例（每个类别单独抽）
TEST_RATIO_GROUP = 0.10
MIN_TEST_GROUPS_PER_CLASS = 1

# 2) train:val = 5:1（按 stem 组数分）
TRAIN_VAL_RATIO = (5, 1)
MIN_VAL_GROUPS_PER_CLASS = 1

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 识别末尾 _000/_001/...，原名可能自带下划线，所以只剥离最后 _ddd
PATTERN_TRAILING_INDEX = re.compile(r"^(.*)_(\d{3})$")


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def stem_group_key(p: Path) -> str:
    """
    xxx_000.jpg / xxx_001.jpg / xxx_123.jpg -> key=xxx
    不符合 _ddd 结尾则 key=原 stem
    """
    m = PATTERN_TRAILING_INDEX.match(p.stem)
    if m:
        return m.group(1)
    return p.stem


def split_group_counts(num_groups: int):
    """
    对某个类别，按 stem 组数拆成 train/val/test（严格不跨）。
    目标：
      - test = round(num_groups * TEST_RATIO_GROUP)，并保证至少 MIN_TEST（若组数够）
      - 剩余部分按 train:val = 5:1 分（按组数）
      - 尽量保证 train/val/test 都不为空（组数太少会退化）
    返回： (n_train_g, n_val_g, n_test_g)
    """
    if num_groups <= 0:
        return 0, 0, 0

    # 组数很少时的退化规则
    if num_groups == 1:
        return 1, 0, 0
    if num_groups == 2:
        return 1, 1, 0
    if num_groups == 3:
        return 1, 1, 1

    # 先分 test（至少留 2 组给 train+val）
    n_test = int(round(num_groups * TEST_RATIO_GROUP))
    n_test = max(n_test, MIN_TEST_GROUPS_PER_CLASS)
    n_test = min(n_test, num_groups - 2)

    remain = num_groups - n_test

    # 再在剩余里按 5:1 分 val
    train_w, val_w = TRAIN_VAL_RATIO
    n_val = int(round(remain * (val_w / (train_w + val_w))))
    n_val = max(n_val, MIN_VAL_GROUPS_PER_CLASS)
    n_val = min(n_val, remain - 1)  # 至少留 1 组给 train

    n_train = remain - n_val

    # 兜底（理论上不会触发，但防止极端参数）
    if n_train < 1:
        n_train = 1
        n_val = max(0, remain - n_train)

    return n_train, n_val, n_test


def copy_or_move_all(files, out_dir: Path, op):
    saved = 0
    skipped = 0
    for f in files:
        try:
            op(str(f), str(out_dir / f.name))
            saved += 1
        except Exception:
            skipped += 1
    return saved, skipped


def main():
    if "CHANGE_ME" in str(DATASET_ROOT) or "CHANGE_ME" in str(OUT_ROOT):
        raise ValueError("请设置 SPLIT_DATASET_ROOT / SPLIT_OUT_ROOT 环境变量，或直接修改脚本路径。")
    assert DATASET_ROOT.exists(), f"输入目录不存在：{DATASET_ROOT}"
    safe_mkdir(OUT_ROOT)

    rng = random.Random(SEED)
    op = shutil.copy2 if COPY_MODE else shutil.move

    train_root = OUT_ROOT / "train"
    val_root   = OUT_ROOT / "val"
    test_root  = OUT_ROOT / "test"
    safe_mkdir(train_root)
    safe_mkdir(val_root)
    safe_mkdir(test_root)

    class_dirs = sorted([d for d in DATASET_ROOT.iterdir() if d.is_dir()])

    print(f"[INFO] Found {len(class_dirs)} class folders in: {DATASET_ROOT}")
    print("[INFO] Split rule: STRICT STEM GROUPS for train/val/test (no leakage)")
    print(f"[INFO] Train:Val = {TRAIN_VAL_RATIO[0]}:{TRAIN_VAL_RATIO[1]} (by groups)")
    print(f"[INFO] Test ratio (by groups) = {TEST_RATIO_GROUP}, min test groups/class = {MIN_TEST_GROUPS_PER_CLASS}")
    print(f"[INFO] Mode: {'COPY' if COPY_MODE else 'MOVE'}")
    print(f"[INFO] Output: {OUT_ROOT}\n")

    total_groups = 0
    total_images = 0

    total_train_g = total_val_g = total_test_g = 0
    total_train_i = total_val_i = total_test_i = 0
    total_skipped = 0

    for ci, class_dir in enumerate(class_dirs, start=1):
        class_name = class_dir.name

        # 1) 按 stem 分组
        groups = defaultdict(list)
        for p in class_dir.iterdir():
            if is_image(p):
                groups[stem_group_key(p)].append(p)

        group_keys = list(groups.keys())
        g = len(group_keys)
        imgs_in_class = sum(len(v) for v in groups.values())

        total_groups += g
        total_images += imgs_in_class

        if g == 0:
            print(f"[WARN] {ci}/{len(class_dirs)} '{class_name}': 0 groups, skipped.")
            continue

        rng.shuffle(group_keys)

        n_train_g, n_val_g, n_test_g = split_group_counts(g)
        train_keys = group_keys[:n_train_g]
        val_keys   = group_keys[n_train_g:n_train_g + n_val_g]
        test_keys  = group_keys[n_train_g + n_val_g:n_train_g + n_val_g + n_test_g]

        out_train_dir = train_root / class_name
        out_val_dir   = val_root / class_name
        out_test_dir  = test_root / class_name
        safe_mkdir(out_train_dir)
        safe_mkdir(out_val_dir)
        safe_mkdir(out_test_dir)

        # 2) 重要：val/test 也把该 stem 的“所有版本”一起写入（不浪费）
        train_files = []
        val_files = []
        test_files = []

        for k in train_keys:
            train_files.extend(groups[k])
        for k in val_keys:
            val_files.extend(groups[k])
        for k in test_keys:
            test_files.extend(groups[k])

        # 3) 写入
        train_saved, train_sk = copy_or_move_all(sorted(train_files), out_train_dir, op)
        val_saved, val_sk     = copy_or_move_all(sorted(val_files), out_val_dir, op)
        test_saved, test_sk   = copy_or_move_all(sorted(test_files), out_test_dir, op)

        total_skipped += (train_sk + val_sk + test_sk)

        total_train_g += len(train_keys)
        total_val_g   += len(val_keys)
        total_test_g  += len(test_keys)

        total_train_i += train_saved
        total_val_i   += val_saved
        total_test_i  += test_saved

        ratio_str = "-"
        if len(val_keys) > 0:
            ratio_str = f"{len(train_keys) / len(val_keys):.2f}:1"

        print(
            f"[CLASS] {ci}/{len(class_dirs)} {class_name}: "
            f"groups={g} imgs={imgs_in_class} -> "
            f"trainG={len(train_keys)} valG={len(val_keys)} testG={len(test_keys)} "
            f"(train:val≈{ratio_str}) | "
            f"trainI={train_saved} valI={val_saved} testI={test_saved}"
        )

    print("\n[DONE]")
    print(f"Total groups:        {total_groups}")
    print(f"Total images found:  {total_images}")
    print(f"Train groups/images: {total_train_g} / {total_train_i}")
    print(f"Val groups/images:   {total_val_g} / {total_val_i}")
    print(f"Test groups/images:  {total_test_g} / {total_test_i}")
    print(f"Skipped (errors):    {total_skipped}")
    print(f"Output root:         {OUT_ROOT}")
    print(f"Mode:                {'COPY' if COPY_MODE else 'MOVE'}")


if __name__ == "__main__":
    main()
