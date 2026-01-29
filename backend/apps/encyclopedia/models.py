from django.db import models

class FishSpecies(models.Model):
    """
    鱼类百科表：一条记录代表一种鱼（或一个模型类别）
    """
    # ✅ 关键：模型输出的类别标签（建议与你训练时的 label 完全一致）
    model_label = models.CharField(max_length=128, unique=True, db_index=True)

    # 基本信息
    name_cn = models.CharField("中文名", max_length=128, db_index=True)
    name_en = models.CharField("英文名", max_length=128, blank=True, default="")
    scientific_name = models.CharField("学名", max_length=256, blank=True, default="")

    # 可扩展信息（用 JSONField 很适合存列表/结构）
    aliases = models.JSONField("别名", blank=True, default=list)       # ["黄花鱼", "大黄花"]
    features = models.JSONField("识别特征", blank=True, default=list)   # ["体侧金黄", "背鳍较长"]

    description = models.TextField("简介", blank=True, default="")
    habitat = models.TextField("栖息地", blank=True, default="")
    diet = models.TextField("食性", blank=True, default="")

    # 你也可以先不加图片字段，后面再扩展
    # cover_image = models.ImageField(upload_to="species/", null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "fish_species"
        ordering = ["name_cn"]

    def __str__(self):
        return f"{self.name_cn} ({self.model_label})"
