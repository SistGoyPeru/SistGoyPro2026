from django.urls import path

from .views import best_bets_1x2_pdf, best_bets_by_date, best_bets_double_chance_pdf, best_bets_totals_pdf, dashboard, best_bets_pdf


urlpatterns = [
    path("", dashboard, name="dashboard"),
    path("mejores-apuestas/", best_bets_by_date, name="best_bets_by_date"),
    path("mejores-apuestas/pdf/", best_bets_pdf, name="best_bets_pdf"),
    path("mejores-apuestas/pdf-1x2/", best_bets_1x2_pdf, name="best_bets_1x2_pdf"),
    path("mejores-apuestas/pdf-doble-oportunidad/", best_bets_double_chance_pdf, name="best_bets_double_chance_pdf"),
    path("mejores-apuestas/pdf-totales/", best_bets_totals_pdf, name="best_bets_totals_pdf"),
]