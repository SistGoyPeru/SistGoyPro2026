from django.urls import path

from .views import best_bets_1x2_pdf, best_bets_by_date, best_bets_double_chance_pdf, best_bets_totals_pdf, dashboard, best_bets_pdf, league_dashboard, match_prediction_page, refresh_all_leagues


urlpatterns = [
    path("", dashboard, name="dashboard"),
    path("pronostico/", match_prediction_page, name="match_prediction_page"),
    path("liga/<str:liga>/", league_dashboard, name="league_dashboard"),
    path("actualizar/", refresh_all_leagues, name="refresh_all_leagues"),
    path("mejores-apuestas/", best_bets_by_date, name="best_bets_by_date"),
    path("mejores-apuestas/pdf/", best_bets_pdf, name="best_bets_pdf"),
    path("mejores-apuestas/pdf-1x2/", best_bets_1x2_pdf, name="best_bets_1x2_pdf"),
    path("mejores-apuestas/pdf-doble-oportunidad/", best_bets_double_chance_pdf, name="best_bets_double_chance_pdf"),
    path("mejores-apuestas/pdf-totales/", best_bets_totals_pdf, name="best_bets_totals_pdf"),
]