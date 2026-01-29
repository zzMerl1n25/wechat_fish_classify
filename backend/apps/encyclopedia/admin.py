from django.contrib import admin
from .models import FishSpecies

@admin.register(FishSpecies)
class FishSpeciesAdmin(admin.ModelAdmin):
    list_display = ("id", "name_cn", "model_label", "scientific_name", "updated_at")
    search_fields = ("name_cn", "name_en", "scientific_name", "model_label")
    ordering = ("name_cn",)
