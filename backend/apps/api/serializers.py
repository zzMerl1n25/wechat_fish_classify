from rest_framework import serializers
from apps.encyclopedia.models import FishSpecies
from rest_framework import serializers

class FishSpeciesSerializer(serializers.ModelSerializer):
    class Meta:
        model = FishSpecies
        fields = [
            "id",
            "model_label",
            "name_cn",
            "name_en",
            "scientific_name",
            "aliases",
            "features",
            "description",
            "habitat",
            "diet",
            "created_at",
            "updated_at",
        ]

class PredictRequestSerializer(serializers.Serializer):
    image = serializers.ImageField()