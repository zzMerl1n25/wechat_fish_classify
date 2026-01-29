# wechat_fish_classify

鱼类识别小程序 + 后端服务 + 训练/推理脚本。项目包含三部分：

- 小程序前端（`miniprogram/`）：拍照/上传图片，展示识别结果和鱼类百科信息。
- Django 后端（`backend/`）：统一 API、数据库、百科内容、对接推理服务。
- 视觉训练/推理（`CV/` + `backend/infer_api.py`）：数据预处理、训练、推理服务。

如需直观看效果，可参考根目录的 `img.png`（页面截图）。

---

## 架构与流程

```
小程序(WeChat) -> Django /api/v1/predict
                   -> FastAPI /input (推理)
                   <- TopK 预测结果
                   -> Django 根据 top1.class_name 查询 FishSpecies
                   <- 返回 prediction + species
```

核心设计点：

- 前端轻量化：只负责上传图片和展示结果。
- 推理服务独立（FastAPI）：模型加载与推理独立进程，避免与 Django 逻辑耦合。
- Django 负责业务逻辑：调用推理、落库、返回百科信息。
- `model_label` 作为“模型输出类名”的统一键，用于模型输出与百科表的映射。

---

## 目录结构

```
wechat_fish_classify/
  backend/                Django 项目
    apps/                 业务应用（api / encyclopedia / ml 等）
    config/               Django settings/urls/asgi/wsgi
    infer_api.py          FastAPI 推理服务（独立运行）
  miniprogram/            微信小程序
  CV/                     训练与数据处理脚本
  img.png                 页面截图
  阿里云认证.md            HTTPS 证书与 Nginx 参考
```

---

## 快速开始（本地开发）

### 1) 准备 Python 环境

建议 Python 3.10+（仓库内 pyc 显示为 3.11）。

```
python -m venv .venv
source .venv/bin/activate
```

依赖没有统一 requirements.txt，请按组件安装：

- Django 后端：
  - django
  - djangorestframework
  - django-cors-headers
  - python-dotenv
  - requests
  - psycopg (PostgreSQL 驱动)

- 推理服务：
  - fastapi
  - uvicorn
  - torch
  - torchvision
  - pillow

- 训练/数据处理：
  - torch / torchvision
  - opencv-python
  - numpy
  - matplotlib
  - tqdm
  - segment-anything (如使用 SAM 裁剪)

可按需执行：

```
pip install django djangorestframework django-cors-headers python-dotenv requests psycopg
pip install fastapi uvicorn torch torchvision pillow
pip install opencv-python numpy matplotlib tqdm
```

### 2) 配置后端 .env

Django 会读取 `backend/.env`（`config/settings/base.py`）。先复制示例文件：

```
cp backend/.env.example backend/.env
```

在 `backend/.env` 中填写你自己的配置（示例占位符）：

```
DJANGO_ENV=dev
DJANGO_SECRET_KEY=CHANGE_ME_SECRET_KEY
DJANGO_DEBUG=1
DJANGO_ALLOWED_HOSTS=CHANGE_ME_HOSTS

DB_NAME=CHANGE_ME_DB_NAME
DB_USER=CHANGE_ME_DB_USER
DB_PASSWORD=CHANGE_ME_DB_PASSWORD
DB_HOST=CHANGE_ME_DB_HOST
DB_PORT=CHANGE_ME_DB_PORT

INFER_API_BASE=CHANGE_ME_INFER_API_BASE
```

注意：`INFER_API_BASE` 必须和 `infer_api.py` 实际运行的 host/port 保持一致。

### 3) 初始化数据库

```
cd backend
python manage.py migrate
python manage.py createsuperuser
```

FishSpecies 模型定义在：`backend/apps/encyclopedia/models.py`。

### 4) 启动 Django API

```
cd backend
python manage.py runserver <YOUR_HOST>:<YOUR_PORT>
```

### 5) 启动 FastAPI 推理服务

```
cd backend
python infer_api.py
```

或：

```
uvicorn infer_api:app --host <YOUR_HOST> --port <YOUR_PORT>
```

`infer_api.py` 通过环境变量读取配置（可在 shell 中 export）：

- `INFER_RUN_DIR`：训练输出目录（含 `best_model.pt`、`class_to_idx.json`）
- `INFER_MODEL_FILE`：推理权重文件
- `INFER_CLASS_MAP_FILE`：类别映射文件
- `INFER_IMG_SIZE`、`INFER_TOPK`：推理参数
- `INFER_HOST`、`INFER_PORT`：仅当你用 `python infer_api.py` 直接启动时需要

### 6) 运行小程序

在微信开发者工具中导入 `miniprogram/`。

修改 API 地址（推荐方式）：

```
cp miniprogram/config.example.js miniprogram/config.js
```

然后在 `miniprogram/config.js` 里填写你自己的后端地址：

```
module.exports = {
  API_BASE: "http://<YOUR_BACKEND_HOST>:<PORT>"
}
```

---

## API 接口

Django API（`/api/v1/`）：

- `GET /api/v1/health/`：健康检查
- `GET /api/v1/species/`：物种列表，支持 `?q=关键词`
- `GET /api/v1/species/<id>/`：物种详情
- `GET /api/v1/species/by-label/<label>/`：按模型 label 查百科
- `POST /api/v1/predict/`：上传图片推理（multipart form-data，字段名 `image`）

FastAPI（`infer_api.py`）：

- `GET /health`：模型与设备信息
- `POST /input`：上传图片推理（字段名 `file`）
- `GET /output/{request_id}`：按 request_id 获取结果
- `GET /outputs?limit=50`：最近 N 条推理结果

---

## 数据与训练

### 数据准备

脚本位于 `CV/`：

- `preprocessing.py`：
  - 旋转 + 裁切增强，避免黑边影响
  - 生成 `_000/_001` 等增强版本
- `split_train_val.py`：
  - 按“stem 分组”切分 train/val/test，避免增强版本数据泄漏
- `sam_crop_dataset.py`：
  - 使用 SAM 自动裁剪主体
  - 支持阈值过滤和失败回退

### 训练脚本

- `train_local_effv2m.py`：本地单卡训练
- `train_server_effv2s.py`：DDP 多卡训练

训练输出保存到 `CV/runs/*`，包含：

- `best_model.pt` / `last_model.pt`
- `class_to_idx.json`
- `history.json` / `hparams.json`
- 训练曲线图和混淆矩阵

### 推理一致性

推理与训练需保持一致：

- 图片尺寸（`IMG_SIZE`）
- 归一化 mean/std
- 类别顺序（`class_to_idx.json`）

`infer_api.py` 内使用 `ResizeWithPad` 以保持长宽比并补边，需与训练设置一致。

---

## 数据库设计（FishSpecies）

`FishSpecies`（`apps/encyclopedia/models.py`）字段：

- `model_label`：模型输出类别名（**关键**）
- `name_cn` / `name_en` / `scientific_name`
- `aliases` / `features`（JSONField）
- `description` / `habitat` / `diet`

设计思路：用 `model_label` 把模型预测结果和百科内容对齐。

---

## 小程序设计思路

- `home` 页面只负责拍照/上传，上传后进入 `result` 页。
- `result` 页通过 `eventChannel` 传递大对象，避免 URL 参数截断。
- UI 使用渐变背景 + 卡片式信息块，强调识别结果和百科内容。

---

## 部署提示

- Django 默认允许所有 CORS（开发态），生产环境请设置白名单（`config/settings/prod.py`）。
- 若需要 HTTPS + 域名，可参考 `阿里云认证.md` 中的 DNS-01 证书流程与 Nginx 反代配置示例。
- 生产环境建议：Django + FastAPI 分开部署，分别反代到不同端口。

---

## 常见问题

1) `infer_service_unreachable`

- Django 调用不到 FastAPI：检查 `INFER_API_BASE` 端口是否一致、服务是否启动。

2) 识别到了类别但百科为空

- 数据库中缺少对应 `model_label` 的记录，需要补全 FishSpecies 表。

3) 推理结果不稳定

- 检查训练/推理的预处理是否一致（尺寸、归一化、裁剪策略）。

---

## 备注

- `backend/apps/ml/infer.py` 是本地推理示例（直接加载 .pt），当前 API 路径采用 FastAPI 服务。
- 多处脚本内使用了占位路径（`CHANGE_ME_*`），运行前需要替换为你自己的路径/环境变量。
