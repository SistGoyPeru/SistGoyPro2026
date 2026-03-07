from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('predict/', views.xgb_predict, name='xgb_predict'),
]
