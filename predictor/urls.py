from django.urls import path

from .views import best_bets_by_date, dashboard, best_bets_pdf


urlpatterns = [
    path("", dashboard, name="dashboard"),
    path("mejores-apuestas/", best_bets_by_date, name="best_bets_by_date"),
    path("mejores-apuestas/pdf/", best_bets_pdf, name="best_bets_pdf"),
]