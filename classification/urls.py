from django.urls import path
from .views import BrainTumorPredictionView

urlpatterns = [
    path("predict/", BrainTumorPredictionView.as_view(), name="brain-tumor-prediction"),
]
