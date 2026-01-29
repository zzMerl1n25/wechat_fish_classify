# wechat_fish_classify

鱼类识别微信小程序 + Django 后端 + FastAPI 推理服务 + PyTorch 训练脚本。

这个仓库主要解决三件事：
1) 小程序拍照/上传图片并展示识别结果与百科信息
2) 后端统一 API、数据库和业务逻辑
3) 视觉模型训练与独立推理服务

---

## 技术栈

- 前端：微信小程序（WXML/WXSS/JS）
- 后端：Django + Django REST Framework
- 数据库：PostgreSQL
- 推理服务：FastAPI + Uvicorn
- 视觉训练/推理：PyTorch + Torchvision + PIL（可选 OpenCV / SAM）

---

## 架构与业务流程

```
小程序(WeChat)
  └─ 上传图片 → Django /api/v1/predict
                 └─ 转发 → FastAPI /input（模型推理）
                      └─ 返回 TopK 预测结果
                 └─ Django 用 top1.class_name 查询 FishSpecies
  └─ 返回：prediction + species
```

关键点：
- 推理服务独立进程，避免与 Django 业务耦合。
- `model_label` 是模型输出与百科内容的映射键。
- 小程序只负责上传与展示，业务逻辑在后端完成。

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

### 1) Python 环境

建议 Python 3.10+。

```
python -m venv .venv
source .venv/bin/activate
```

依赖未集中管理，请按组件安装：

- Django 后端：
  - django
  - djangorestframework
  - django-cors-headers
  - python-dotenv
  - requests
  - psycopg

- 推理服务：
  - fastapi
  - uvicorn
  - torch
  - torchvision
  - pillow

- 训练/数据处理（按需）：
  - opencv-python
  - numpy
  - matplotlib
  - tqdm
  - segment-anything（如使用 SAM 裁剪）

示例：
```
pip install django djangorestframework django-cors-headers python-dotenv requests psycopg
pip install fastapi uvicorn torch torchvision pillow
pip install opencv-python numpy matplotlib tqdm
```

### 2) 配置后端 .env

```
cp backend/.env.example backend/.env
```

在 `backend/.env` 中填写配置：

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

### 3) 初始化数据库

```
cd backend
python manage.py migrate
python manage.py createsuperuser
```

`FishSpecies` 模型位于 `backend/apps/encyclopedia/models.py`。

### 4) 启动 FastAPI 推理服务

需要模型运行目录（含 `best_model.pt`、`class_to_idx.json`）：

```
export INFER_RUN_DIR=/path/to/run_dir
export INFER_MODEL_FILE=best_model.pt
export INFER_CLASS_MAP_FILE=class_to_idx.json
export INFER_IMG_SIZE=224
export INFER_TOPK=5
```

启动：
```
cd backend
python infer_api.py
```
或：
```
uvicorn infer_api:app --host <YOUR_HOST> --port <YOUR_PORT>
```

### 5) 启动 Django API

```
cd backend
python manage.py runserver <YOUR_HOST>:<YOUR_PORT>
```

### 6) 运行小程序

在微信开发者工具中导入 `miniprogram/`。

```
cp miniprogram/config.example.js miniprogram/config.js
```

修改 `miniprogram/config.js`：

```
module.exports = {
  API_BASE: "http://<YOUR_BACKEND_HOST>:<PORT>"
}
```

---

## API 接口

### Django（`/api/v1/`）

- `GET /api/v1/health/`：健康检查
- `GET /api/v1/species/`：物种列表，支持 `?q=关键词`
- `GET /api/v1/species/<id>/`：物种详情
- `GET /api/v1/species/by-label/<label>/`：按模型 label 查百科
- `POST /api/v1/predict/`：上传图片推理（multipart form-data，字段名 `image`）

### FastAPI（`infer_api.py`）

- `GET /health`：模型与设备信息
- `POST /input`：上传图片推理（字段名 `file`）
- `GET /output/{request_id}`：按 request_id 获取结果
- `GET /outputs?limit=50`：最近 N 条推理结果

---

## 数据与训练流程

### 数据格式

训练脚本要求如下目录结构：

```
DATA_ROOT/
  train/classA/*.jpg
  val/classA/*.jpg
  test/classA/*.jpg
```

### 数据处理脚本（`CV/`）

- `preprocessing.py`：旋转 + 裁切增强，生成 `_000/_001` 版本
- `split_train_val.py`：按 stem 分组切分 train/val/test，避免数据泄漏
- `sam_crop_dataset.py`：使用 SAM 自动裁剪主体（可选）

### 训练脚本

- `train_local_effv2m.py`：本地单卡训练
- `train_server_effv2s.py`：多卡训练

训练输出保存在 `CV/runs/*`，包含：
- `best_model.pt` / `last_model.pt`
- `class_to_idx.json`
- `history.json` / `hparams.json`
- 训练曲线图与混淆矩阵

### 推理一致性

推理与训练必须保持一致：
- 输入尺寸（`IMG_SIZE`）
- 归一化 mean/std
- 类别顺序（`class_to_idx.json`）

`infer_api.py` 使用 `ResizeWithPad` 保持长宽比并补边。

---

## 部署提示

- 开发环境允许全部 CORS，生产环境请在 `config/settings/prod.py` 配置白名单。
- 推荐 Django 与 FastAPI 分开部署，分别反代到不同端口。
- HTTPS 与 Nginx 证书配置可参考 `阿里云认证.md`。

---

## 常见问题

1) `infer_service_unreachable`
- 检查 `INFER_API_BASE` 是否正确、FastAPI 是否已启动。

2) 识别到了类别但百科为空
- 数据库中缺少对应 `model_label` 记录，需要补全 FishSpecies。

3) 推理结果不稳定
- 检查训练与推理预处理是否一致（尺寸、归一化、裁剪策略）。

---

## 备注

- `backend/apps/ml/infer.py` 提供本地推理示例（直接加载 .pt）。
- `backend/reset_pg_password.py` 需要 DB 相关环境变量。
- 运行脚本前请替换所有 `CHANGE_ME_*` 配置项。
