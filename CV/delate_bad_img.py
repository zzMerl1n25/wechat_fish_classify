# delete_bad_images.py
from pathlib import Path

BAD_LIST = "bad_images.txt"  # 你扫描生成的文件名

paths = []
for line in Path(BAD_LIST).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    p = line.split("\t")[0].strip()
    paths.append(Path(p))

print("Will delete:", len(paths), "files")
deleted, missing, failed = 0, 0, 0

for p in paths:
    try:
        if p.exists():
            p.unlink()
            deleted += 1
        else:
            missing += 1
    except Exception as e:
        failed += 1
        print("[FAILED]", p, repr(e))

print("deleted =", deleted, "missing =", missing, "failed =", failed)