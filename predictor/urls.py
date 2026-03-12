from django.urls import path

from .views import best_bets_by_date, dashboard


urlpatterns = [
    path("", dashboard, name="dashboard"),
    path("mejores-apuestas/", best_bets_by_date, name="best_bets_by_date"),
]