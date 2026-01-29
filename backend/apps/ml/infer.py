import os
import torch
from PIL import Image
from torchvision import transforms

# 你需要把这些路径/类别映射换成你自己的
MODEL_PATH = os.getenv("ML_MODEL_PATH", "").strip()

# 例子：idx -> model_label（你训练时的类别顺序）
IDX_TO_LABEL = [
    "grass_carp",
    # "another_fish",
]

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),   # 这里必须和你训练时一致
    transforms.ToTensor(),
    # transforms.Normalize(mean=[...], std=[...])  # 如果训练用了归一化，这里也要一致
])

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH:
            raise ValueError("ML_MODEL_PATH 为空，请设置环境变量或在代码中填写模型路径。")
        m = torch.jit.load(MODEL_PATH, map_location=_device) if MODEL_PATH.endswith(".pt") else torch.load(MODEL_PATH, map_location=_device)
        # 如果你保存的是 state_dict，需要自己构建网络再 load_state_dict
        m.eval()
        _model = m
    return _model

@torch.inference_mode()
def predict_image(image_path: str):
    model = get_model()
    img = Image.open(image_path).convert("RGB")
    x = _preprocess(img).unsqueeze(0).to(_device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = torch.max(probs, dim=0)

    label = IDX_TO_LABEL[int(idx)]
    return label, float(conf)
