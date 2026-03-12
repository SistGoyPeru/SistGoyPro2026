from datetime import datetime

from django.shortcuts import render

from .engine import (
	get_prediction_service_spain,
	get_prediction_service_bundesliga,
	get_prediction_service_premier,
	get_prediction_service_seriea,
	get_prediction_service_ligue1,
)
from .sync import refresh_fixture_links


LEAGUE_SERVICE_FACTORIES = {
	"spain": get_prediction_service_spain,
	"bundesliga": get_prediction_service_bundesliga,
	"premier": get_prediction_service_premier,
	"seriea": get_prediction_service_seriea,
	"ligue1": get_prediction_service_ligue1,
}


def _get_service(liga: str):
	return LEAGUE_SERVICE_FACTORIES.get(liga, get_prediction_service_spain)()


def _parse_fixture_date(value: str):
	for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d"):
		try:
			return datetime.strptime(str(value), fmt).date()
		except ValueError:
			continue
	return None


def dashboard(request):
	liga = request.GET.get("liga") or request.POST.get("liga", "spain")
	if liga not in ("spain", "bundesliga", "premier", "seriea", "ligue1"):
		liga = "spain"

	refresh_status = ""
	if request.method == "GET":
		try:
			updated_rows = refresh_fixture_links(liga)
			refresh_status = f"Enlaces actualizados y guardados en base de datos: {updated_rows} encuentros."
		except Exception as exc:
			refresh_status = f"No se pudo refrescar enlaces en este acceso: {exc}"

	service = _get_service(liga)
	fixtures = service.get_pending_fixtures()
	selected_match_key = request.POST.get("match_key") or request.GET.get("match_key", "")

	prediction = None
	error_message = ""
	if fixtures:
		if not selected_match_key:
			selected_match_key = fixtures[0]["match_key"]
		try:
			prediction = service.predict_match(selected_match_key)
		except ValueError as exc:
			error_message = str(exc)
	else:
		error_message = "No hay encuentros pendientes en el calendario actual."

	context = {
		"liga": liga,
		"league_name": service.league_name,
		"league_logo_url": service.league_logo_url,
		"datasets": service.dataset_labels,
		"fixtures": fixtures,
		"prediction": prediction,
		"selected_match_key": selected_match_key,
		"model_name": service.best_model_name,
		"validation_accuracy": round(service.validation_accuracy * 100, 2),
		"model_scores": service.model_scores,
		"error_message": error_message,
		"refresh_status": refresh_status,
	}
	return render(request, "predictor/dashboard.html", context)


def best_bets_by_date(request):
	best_entries: list[dict[str, object]] = []
	refresh_log: list[str] = []
	today = datetime.now().date()
	target_month = 5
	target_year = today.year

	refresh_enabled = request.GET.get("refresh") == "1"
	if refresh_enabled:
		for liga in LEAGUE_SERVICE_FACTORIES:
			try:
				updated_rows = refresh_fixture_links(liga)
				refresh_log.append(f"{liga}: {updated_rows} encuentros actualizados")
			except Exception as exc:
				refresh_log.append(f"{liga}: error al actualizar ({exc})")
	else:
		refresh_log.append("Modo rapido: sin refresco automatico. Usa ?refresh=1 para actualizar enlaces.")

	# Recolectar todos los encuentros pendientes de mayo del anio actual.
	eligible_fixtures: list[tuple[str, object, dict[str, object], object]] = []
	for liga, factory in LEAGUE_SERVICE_FACTORIES.items():
		service = factory()
		for fixture in service.get_pending_fixtures():
			date_label = str(fixture["fecha"])
			sort_date = _parse_fixture_date(date_label)
			if sort_date is None or sort_date < today:
				continue
			if sort_date.year != target_year or sort_date.month != target_month:
				continue
			eligible_fixtures.append((liga, service, fixture, sort_date))

	if not eligible_fixtures:
		context = {
			"best_entries": [],
			"refresh_log": refresh_log,
			"target_month_name": "mayo",
			"target_year": target_year,
		}
		return render(request, "predictor/best_bets_by_date.html", context)

	# Calcular la recomendacion ligera de cada encuentro y ordenar por probabilidad combinada.
	for liga, service, fixture, sort_date in eligible_fixtures:
		try:
			prediction = service.predict_recommended_bet_fast(fixture["match_key"])
		except ValueError:
			continue

		recommended = prediction["recommended_bet"]
		multiple = prediction.get("multiple", {})
		combined_probability = float(multiple.get("prob_combinada", 0.0))
		date_label = str(fixture["fecha"])
		entry = {
			"date_label": date_label,
			"sort_date": sort_date,
			"league_key": liga,
			"league_name": service.league_name,
			"league_logo_url": service.league_logo_url,
			"match_key": fixture["match_key"],
			"kickoff": str(fixture["hora"]),
			"home_team": str(fixture["local"]),
			"away_team": str(fixture["visitante"]),
			"home_logo": fixture.get("local_logo", ""),
			"away_logo": fixture.get("visitante_logo", ""),
			"market": str(recommended["market"]),
			"pick": str(recommended["pick"]),
			"probability": float(recommended["probability"]),
			"confidence": str(recommended["confidence"]),
			"multiple_confidence": str(multiple.get("confidence", "Sin selecciones")),
			"multiple_legs": list(multiple.get("legs", [])),
			"multiple_combined_probability": combined_probability,
			"multiple_combined_probability_text": f"{combined_probability:.2f}".replace(".", ","),
			"dashboard_url": f"/?liga={liga}&match_key={fixture['match_key']}",
		}
		best_entries.append(entry)

	best_entries = sorted(
		best_entries,
		key=lambda item: (-float(item["multiple_combined_probability"]), item["sort_date"], item["kickoff"]),
	)

	context = {
		"best_entries": best_entries,
		"refresh_log": refresh_log,
		"target_month_name": "mayo",
		"target_year": target_year,
	}
	return render(request, "predictor/best_bets_by_date.html", context)
