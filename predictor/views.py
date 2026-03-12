from datetime import datetime, timedelta
from io import BytesIO

from django.shortcuts import render
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors

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
	window_start = None
	window_end = None

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

	# Recolectar todos los encuentros pendientes para determinar la proxima fecha.
	eligible_fixtures: list[tuple[str, object, dict[str, object], object]] = []
	for liga, factory in LEAGUE_SERVICE_FACTORIES.items():
		service = factory()
		for fixture in service.get_pending_fixtures():
			date_label = str(fixture["fecha"])
			sort_date = _parse_fixture_date(date_label)
			if sort_date is None or sort_date < today:
				continue
			eligible_fixtures.append((liga, service, fixture, sort_date))

	if not eligible_fixtures:
		context = {
			"best_entries": [],
			"refresh_log": refresh_log,
			"window_label": "sin encuentros pendientes",
		}
		return render(request, "predictor/best_bets_by_date.html", context)

	window_start = min(row[3] for row in eligible_fixtures)
	window_end = window_start + timedelta(days=1)
	window_fixtures = [row for row in eligible_fixtures if window_start <= row[3] <= window_end]

	# Calcular la recomendacion ligera de cada encuentro y ordenar por probabilidad combinada.
	for liga, service, fixture, sort_date in window_fixtures:
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
		key=lambda item: (item["sort_date"], -float(item["multiple_combined_probability"]), item["kickoff"]),
	)

	date_groups: list[dict[str, object]] = []
	for entry in best_entries:
		if not date_groups or date_groups[-1]["sort_date"] != entry["sort_date"]:
			date_groups.append(
				{
					"sort_date": entry["sort_date"],
					"date_label": entry["date_label"],
					"items": [entry],
				}
			)
		else:
			date_groups[-1]["items"].append(entry)

	window_label = f"{window_start.strftime('%d/%m/%Y')} y {window_end.strftime('%d/%m/%Y')}"

	context = {
		"best_entries": best_entries,
		"date_groups": date_groups,
		"refresh_log": refresh_log,
		"window_label": window_label,
	}
	return render(request, "predictor/best_bets_by_date.html", context)


def best_bets_pdf(request):
	"""Generar PDF de mejores apuestas por fecha."""
	best_entries: list[dict[str, object]] = []
	today = datetime.now().date()
	window_start = None
	window_end = None

	# Recolectar todos los encuentros pendientes para determinar la proxima fecha.
	eligible_fixtures: list[tuple[str, object, dict[str, object], object]] = []
	for liga, factory in LEAGUE_SERVICE_FACTORIES.items():
		service = factory()
		for fixture in service.get_pending_fixtures():
			date_label = str(fixture["fecha"])
			sort_date = _parse_fixture_date(date_label)
			if sort_date is None or sort_date < today:
				continue
			eligible_fixtures.append((liga, service, fixture, sort_date))

	if not eligible_fixtures:
		return HttpResponse("No hay encuentros pendientes.", content_type="text/plain")

	window_start = min(row[3] for row in eligible_fixtures)
	window_end = window_start + timedelta(days=1)
	window_fixtures = [row for row in eligible_fixtures if window_start <= row[3] <= window_end]

	# Calcular recomendaciones
	for liga, service, fixture, sort_date in window_fixtures:
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
			"league_name": service.league_name,
			"home_team": str(fixture["local"]),
			"away_team": str(fixture["visitante"]),
			"kickoff": str(fixture["hora"]),
			"multiple_confidence": str(multiple.get("confidence", "Sin selecciones")),
			"multiple_legs": list(multiple.get("legs", [])),
			"multiple_combined_probability_text": f"{combined_probability:.2f}".replace(".", ","),
		}
		best_entries.append(entry)

	best_entries = sorted(
		best_entries,
		key=lambda item: (item["sort_date"], -float(item["multiple_combined_probability_text"].replace(",", "."))),
	)

	# Crear PDF
	buffer = BytesIO()
	doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=0.5*inch, leftMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
	story = []

	# Estilos
	styles = getSampleStyleSheet()
	title_style = ParagraphStyle(
		"CustomTitle",
		parent=styles["Heading1"],
		fontSize=18,
		textColor=colors.HexColor("#0f3d28"),
		spaceAfter=10,
		alignment=1,
	)

	date_style = ParagraphStyle(
		"DateHeader",
		parent=styles["Heading2"],
		fontSize=14,
		textColor=colors.HexColor("#145433"),
		spaceAfter=8,
		spaceBefore=12,
	)

	# Título
	story.append(Paragraph("MEJORES APUESTAS POR FECHA", title_style))
	window_label = f"{window_start.strftime('%d/%m/%Y')} al {window_end.strftime('%d/%m/%Y')}"
	story.append(Paragraph(f"Ventana: {window_label}", styles["Normal"]))
	story.append(Spacer(1, 0.3*inch))

	# Agrupar por fecha
	date_groups_dict = {}
	for entry in best_entries:
		date_key = entry["date_label"]
		if date_key not in date_groups_dict:
			date_groups_dict[date_key] = []
		date_groups_dict[date_key].append(entry)

	# Generar tabla por fecha
	for date_label in sorted(date_groups_dict.keys()):
		group_items = date_groups_dict[date_label]
		story.append(Paragraph(f"Fecha: {date_label}", date_style))

		# Detalles de cada encuentro
		for idx, item in enumerate(group_items, 1):
			# Encabezado del partido
			matchup = f"{item['home_team']} vs {item['away_team']}"
			partido_text = f"<b>#{idx} - {item['kickoff']} | {matchup}</b><br/><i>Liga: {item['league_name']}</i>"
			story.append(Paragraph(partido_text, styles["Normal"]))
			story.append(Spacer(1, 0.1*inch))

			# Detalles de la apuesta múltiple
			multi_text = f"<b>🅃 Apuesta múltiple — Solo selecciones Alta confianza</b><br/>"
			multi_text += f"Confianza: <b>{item['multiple_confidence']}</b><br/>"
			story.append(Paragraph(multi_text, styles["Normal"]))

			# Piernas/Legs
			if item["multiple_legs"]:
				legs_text = "<b>Selecciones:</b><br/>"
				for leg in item["multiple_legs"]:
					legs_text += f"• {leg}<br/>"
				story.append(Paragraph(legs_text, styles["Normal"]))
			else:
				story.append(Paragraph("<i>Sin selecciones de alta confianza para combinar.</i>", styles["Normal"]))

			# Probabilidad combinada
			prob_text = f"<b>Probabilidad combinada: {item['multiple_combined_probability_text']}%</b>"
			story.append(Paragraph(prob_text, styles["Normal"]))
			story.append(Spacer(1, 0.15*inch))

		story.append(Spacer(1, 0.25*inch))

	# Generar PDF
	doc.build(story)
	buffer.seek(0)

	response = HttpResponse(buffer.getvalue(), content_type="application/pdf")
	response["Content-Disposition"] = f'attachment; filename="mejores_apuestas_{window_start.strftime("%d_%m_%Y")}.pdf"'
	return response
