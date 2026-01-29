import os
import requests
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.shortcuts import get_object_or_404


from apps.api.serializers import PredictRequestSerializer
from apps.encyclopedia.models import FishSpecies
from apps.ml.infer import predict_image
from django.http import JsonResponse
from django.conf import settings

from apps.encyclopedia.models import FishSpecies
from .serializers import FishSpeciesSerializer


print("INFER_API_BASE =", settings.INFER_API_BASE)


def health(request):
    return JsonResponse({"status": "ok"})

@api_view(["GET"])
def species_list(request):
    """
    GET /api/v1/species/
    可选：?q=关键词（按中文名/英文名/学名/model_label 模糊搜索）
    """
    q = (request.GET.get("q") or "").strip()
    qs = FishSpecies.objects.all().order_by("name_cn")

    if q:
        qs = qs.filter(
            name_cn__icontains=q
        ) | qs.filter(
            name_en__icontains=q
        ) | qs.filter(
            scientific_name__icontains=q
        ) | qs.filter(
            model_label__icontains=q
        )

    data = FishSpeciesSerializer(qs, many=True).data
    return Response(data)


@api_view(["GET"])
def species_detail(request, pk: int):
    """
    GET /api/v1/species/<id>/
    """
    obj = get_object_or_404(FishSpecies, pk=pk)
    return Response(FishSpeciesSerializer(obj).data)


@api_view(["GET"])
def species_by_label(request, label: str):
    """
    GET /api/v1/species/by-label/<label>/
    给 /predict 用：模型输出 label → 查百科
    """
    obj = get_object_or_404(FishSpecies, model_label=label)
    return Response(FishSpeciesSerializer(obj).data)

@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def predict(request):
    """
    POST /api/v1/predict/
    - 小程序上传：form-data 里 name=image
    - Django 转发到 FastAPI /input（name=file）
    - FastAPI 返回 predictions
    - Django 用 top1.class_name 作为 model_label 查百科并返回
    """
    ser = PredictRequestSerializer(data=request.data)
    ser.is_valid(raise_exception=True)

    img = ser.validated_data["image"]

    # 转发给 FastAPI：注意字段名必须叫 file（和 infer_api.py 里一致）
    files = {
        "file": (img.name, img.file, img.content_type or "application/octet-stream")
    }

    try:
        r = requests.post(
            f"{settings.INFER_API_BASE}/input",
            files=files,
            timeout=60,
        )
    except requests.RequestException as e:
        return Response(
            {"ok": False, "error": "infer_service_unreachable", "detail": str(e)},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    if r.status_code != 200:
        return Response(
            {"ok": False, "error": "infer_service_error", "detail": r.text},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    infer_data = r.json()
    preds = infer_data.get("predictions") or []

    if not preds:
        return Response(
            {"ok": False, "error": "empty_predictions", "infer_data": infer_data},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    top1 = preds[0]
    model_label = top1.get("class_name")
    confidence = float(top1.get("confidence", 0.0))

    if not model_label:
        return Response(
            {"ok": False, "error": "missing_class_name", "top1": top1},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    # 用 model_label 查百科
    try:
        species = FishSpecies.objects.get(model_label=model_label)
    except FishSpecies.DoesNotExist:
        return Response({
            "ok": False,
            "error": "species_not_in_db",
            "message": "识别到了一个类别，但数据库未收录该鱼种信息",
            "prediction": {
                "model_label": model_label,
                "confidence": confidence,
                "topk": preds,
            },
            "request_id": infer_data.get("request_id"),
        }, status=status.HTTP_200_OK)

    return Response({
        "ok": True,
        "request_id": infer_data.get("request_id"),
        "prediction": {
            "model_label": model_label,
            "confidence": confidence,
            "topk": preds,
        },
        "species": {
            "id": species.id,
            "model_label": species.model_label,
            "name_cn": species.name_cn,
            "name_en": species.name_en,
            "scientific_name": species.scientific_name,
            "aliases": species.aliases,
            "features": species.features,
            "description": species.description,
            "habitat": species.habitat,
            "diet": species.diet,
        }
    })