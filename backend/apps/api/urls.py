from django.urls import path
from .views import health
from . import views
from .views import predict

urlpatterns = [
    path("health/", health),
    path("species/", views.species_list),
    path("species/<int:pk>/", views.species_detail),
    path("species/by-label/<str:label>/", views.species_by_label),
    path("predict/", predict),
]