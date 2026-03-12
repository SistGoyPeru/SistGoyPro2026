from datetime import datetime, timedelta
from io import BytesIO
import re

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


def _extract_leg_probability(leg: object) -> float | None:
	if not isinstance(leg, str):
		return None
	match = re.search(r"\(([0-9]+(?:\.[0-9]+)?)%\)", leg)
	if not match:
		return None
	try:
		return float(match.group(1))
	except ValueError:
		return None


def _legs_with_probabilities(legs: list[object]) -> list[tuple[str, float]]:
	parsed: list[tuple[str, float]] = []
	for leg in legs:
		if not isinstance(leg, str):
			continue
		leg_prob = _extract_leg_probability(leg)
		if leg_prob is None:
			continue
		parsed.append((leg, leg_prob))
	return parsed


def _top_two_legs(legs: list[object]) -> list[str]:
	parsed = _legs_with_probabilities(legs)
	top_two = sorted(parsed, key=lambda item: item[1], reverse=True)[:2]
	return [leg for leg, _ in top_two]


def _legs_without_cards_corners(legs: list[object]) -> list[str]:
	filtered: list[str] = []
	for leg, _ in _legs_with_probabilities(legs):
		leg_lower = leg.lower()
		if any(token in leg_lower for token in ("tarjetas", "amarillas", "corners", "córners")):
			continue
		filtered.append(leg)
	return filtered


def _combined_top_two_probability(legs: list[object]) -> float:
	probs = [p for _, p in _legs_with_probabilities(legs)]
	if len(probs) < 2:
		return 0.0
	probs = sorted(probs, reverse=True)[:2]
	combined = (probs[0] / 100.0) * (probs[1] / 100.0)
	return round(combined * 100, 2)


def _combined_without_cards_corners_probability(legs: list[object]) -> float:
	probs = [p for _, p in _legs_with_probabilities(_legs_without_cards_corners(legs))]

	if not probs:
		return 0.0

	combined = 1.0
	for prob in probs:
		combined *= prob / 100.0
	return round(combined * 100, 2)


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
		multiple_legs = list(multiple.get("legs", []))
		top_two_legs = _top_two_legs(multiple_legs)
		no_cards_corners_legs = _legs_without_cards_corners(multiple_legs)
		combined_probability = float(multiple.get("prob_combinada", 0.0))
		top_two_combined_probability = _combined_top_two_probability(multiple_legs)
		no_cards_corners_probability = _combined_without_cards_corners_probability(multiple_legs)
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
			"multiple_legs": multiple_legs,
			"multiple_legs_general": multiple_legs,
			"multiple_legs_top2": top_two_legs,
			"multiple_legs_no_cards_corners": no_cards_corners_legs,
			"multiple_combined_probability": combined_probability,
			"multiple_combined_probability_text": f"{combined_probability:.2f}".replace(".", ","),
			"multiple_top2_combined_probability": top_two_combined_probability,
			"multiple_top2_combined_probability_text": f"{top_two_combined_probability:.2f}".replace(".", ","),
			"multiple_no_cards_corners_probability": no_cards_corners_probability,
			"multiple_no_cards_corners_probability_text": f"{no_cards_corners_probability:.2f}".replace(".", ","),
			"dashboard_url": f"/?liga={liga}&match_key={fixture['match_key']}",
		}
		best_entries.append(entry)

	best_entries = sorted(
		best_entries,
		key=lambda item: (
			item["sort_date"],
			-float(item.get("multiple_no_cards_corners_probability", 0.0)),
			-float(item.get("multiple_top2_combined_probability", 0.0)),
			-float(item["multiple_combined_probability"]),
			item["kickoff"],
		),
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
		multiple_legs = list(multiple.get("legs", []))
		top_two_legs = _top_two_legs(multiple_legs)
		no_cards_corners_legs = _legs_without_cards_corners(multiple_legs)
		combined_probability = float(multiple.get("prob_combinada", 0.0))
		top_two_combined_probability = _combined_top_two_probability(multiple_legs)
		no_cards_corners_probability = _combined_without_cards_corners_probability(multiple_legs)
		date_label = str(fixture["fecha"])
		entry = {
			"date_label": date_label,
			"sort_date": sort_date,
			"league_name": service.league_name,
			"home_team": str(fixture["local"]),
			"away_team": str(fixture["visitante"]),
			"kickoff": str(fixture["hora"]),
			"multiple_confidence": str(multiple.get("confidence", "Sin selecciones")),
			"multiple_legs": multiple_legs,
			"multiple_legs_general": multiple_legs,
			"multiple_legs_top2": top_two_legs,
			"multiple_legs_no_cards_corners": no_cards_corners_legs,
			"multiple_combined_probability": combined_probability,
			"multiple_combined_probability_text": f"{combined_probability:.2f}".replace(".", ","),
			"multiple_top2_combined_probability": top_two_combined_probability,
			"multiple_top2_combined_probability_text": f"{top_two_combined_probability:.2f}".replace(".", ","),
			"multiple_no_cards_corners_probability": no_cards_corners_probability,
			"multiple_no_cards_corners_probability_text": f"{no_cards_corners_probability:.2f}".replace(".", ","),
		}
		best_entries.append(entry)

	best_entries = sorted(
		best_entries,
		key=lambda item: (
			item["sort_date"],
			-float(item.get("multiple_no_cards_corners_probability", 0.0)),
			-float(item.get("multiple_top2_combined_probability", 0.0)),
			-float(item["multiple_combined_probability"]),
		),
	)

	# Crear PDF
	buffer = BytesIO()
	doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=0.5*inch, leftMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
	story = []

	# Estilos personalizados para mejor diseño
	styles = getSampleStyleSheet()
	title_style = ParagraphStyle(
		"CustomTitle",
		parent=styles["Heading1"],
		fontName="Helvetica-Bold",
		fontSize=24,
		textColor=colors.HexColor("#0f3d28"),
		spaceAfter=6,
		alignment=1,
	)
	subtitle_style = ParagraphStyle(
		"Subtitle",
		parent=styles["Normal"],
		fontName="Helvetica",
		fontSize=10,
		textColor=colors.HexColor("#1a7d5c"),
		spaceAfter=20,
		alignment=1,
	)
	date_header_style = ParagraphStyle(
		"DateHeader",
		parent=styles["Heading2"],
		fontName="Helvetica-Bold",
		fontSize=13,
		textColor=colors.white,
		spaceAfter=10,
		spaceBefore=12,
	)
	match_header_style = ParagraphStyle(
		"MatchHeader",
		parent=styles["Heading3"],
		fontName="Helvetica-Bold",
		fontSize=11,
		textColor=colors.HexColor("#0f3d28"),
		spaceAfter=5,
	)
	label_style = ParagraphStyle(
		"Label",
		parent=styles["Normal"],
		fontName="Helvetica-Bold",
		fontSize=9,
		textColor=colors.HexColor("#145433"),
		spaceAfter=3,
	)
	value_style = ParagraphStyle(
		"Value",
		parent=styles["Normal"],
		fontName="Helvetica",
		fontSize=9,
		textColor=colors.black,
		spaceAfter=2,
	)
	confidence_style = ParagraphStyle(
		"Confidence",
		parent=styles["Normal"],
		fontName="Helvetica-Bold",
		fontSize=10,
		textColor=colors.HexColor("#1a7d5c"),
		spaceAfter=4,
	)
	prob_style = ParagraphStyle(
		"Probability",
		parent=styles["Normal"],
		fontName="Helvetica-Bold",
		fontSize=12,
		textColor=colors.HexColor("#1a7d5c"),
		spaceAfter=0,
	)

	# Calcular etiqueta de ventana
	window_label = f"{window_start.strftime('%d/%m/%Y')} al {window_end.strftime('%d/%m/%Y')}"

	# Título principal
	story.append(Paragraph("MEJORES APUESTAS POR FECHA", title_style))
	story.append(Paragraph(f"Ventana: {window_label}", subtitle_style))
	story.append(Spacer(1, 0.15*inch))

	# Agrupar por fecha
	date_groups_dict = {}
	for entry in best_entries:
		date_key = entry["date_label"]
		if date_key not in date_groups_dict:
			date_groups_dict[date_key] = []
		date_groups_dict[date_key].append(entry)

	# Generar contenido profesional por fecha
	for date_label in sorted(date_groups_dict.keys()):
		group_items = date_groups_dict[date_label]
		
		# Cabecera de fecha con fondo de color
		date_header_table = Table([
			[Paragraph(f"📅 FECHA: {date_label}", date_header_style)]
		], colWidths=[7.5*inch])
		date_header_table.setStyle(TableStyle([
			("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1a7d5c")),
			("ALIGN", (0, 0), (-1, -1), "LEFT"),
			("PADDING", (0, 0), (-1, -1), 10),
			("LEFTPADDING", (0, 0), (-1, -1), 15),
		]))
		story.append(date_header_table)
		story.append(Spacer(1, 0.1*inch))

		# Detalles de cada encuentro en cuadros
		for idx, item in enumerate(group_items, 1):
			matchup = f"{item['home_team']} vs {item['away_team']}"

			# Colores según nivel de confianza (Alta=verde, Media=amarillo, Baja=naranja)
			_conf = item['multiple_confidence']
			if _conf == 'Alta':
				conf_accent = colors.HexColor("#1a7d5c")
				conf_bg = colors.HexColor("#d4f0e8")
				conf_text_hex = "#1a7d5c"
				conf_border = colors.HexColor("#52c99b")
			elif _conf == 'Media':
				conf_accent = colors.HexColor("#8a7010")
				conf_bg = colors.HexColor("#f5ecc0")
				conf_text_hex = "#8a7010"
				conf_border = colors.HexColor("#c9a82a")
			else:
				conf_accent = colors.HexColor("#8a3a1a")
				conf_bg = colors.HexColor("#f5d0b8")
				conf_text_hex = "#8a3a1a"
				conf_border = colors.HexColor("#c96c42")

			# Cuadro principal del partido
			main_data = [
				[
					Paragraph(f"<b>#{idx}</b>", styles["Normal"]),
					Paragraph(f"<b>{matchup}</b><br/><font size=8>{item['league_name']} • {item['kickoff']}</font>", match_header_style),
				]
			]
			main_table = Table(main_data, colWidths=[0.6*inch, 6.9*inch])
			main_table.setStyle(TableStyle([
				("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e8f5f0")),
				("BORDER", (0, 0), (-1, -1), 1.5, colors.HexColor("#1a7d5c")),
				("ALIGN", (0, 0), (0, -1), "CENTER"),
				("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
				("PADDING", (0, 0), (-1, -1), 8),
				("LEFTPADDING", (1, 0), (1, -1), 12),
			]))
			story.append(main_table)
			story.append(Spacer(1, 0.08*inch))

			# Cabecera del ticket múltiple con badge de confianza
			multi_header = [
				[
					Paragraph("🅃 <b>APUESTA MÚLTIPLE — SOLO SELECCIONES ALTA CONFIANZA</b>", confidence_style),
					Paragraph(f'<font color="{conf_text_hex}"><b>{item["multiple_confidence"]}</b></font>', confidence_style),
				]
			]
			multi_header_table = Table(multi_header, colWidths=[6.2*inch, 1.3*inch])
			multi_header_table.setStyle(TableStyle([
				("BACKGROUND", (0, 0), (-1, -1), conf_bg),
				("BORDER", (0, 0), (-1, -1), 2, conf_accent),
				("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f3d28")),
				("ALIGN", (0, 0), (0, -1), "LEFT"),
				("ALIGN", (1, 0), (1, -1), "RIGHT"),
				("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
				("PADDING", (0, 0), (-1, -1), 8),
				("LEFTPADDING", (0, 0), (0, -1), 10),
				("RIGHTPADDING", (1, 0), (1, -1), 10),
			]))
			story.append(multi_header_table)
			story.append(Spacer(1, 0.06*inch))

			# --- Grid de 3 columnas (igual que el HTML) ---
			if item["multiple_legs"]:
				def _metric_bg_border(prob):
					if prob >= 75.0:
						return (colors.HexColor("#c8f0dc"), colors.HexColor("#52c99b"), colors.HexColor("#0f5c30"), colors.HexColor("#083d1e"))
					elif prob >= 62.0:
						return (colors.HexColor("#f5ecc0"), colors.HexColor("#c9a82a"), colors.HexColor("#7a6000"), colors.HexColor("#4a3800"))
					else:
						return (colors.HexColor("#f5d0b8"), colors.HexColor("#c96c42"), colors.HexColor("#7a3010"), colors.HexColor("#4a1e08"))

				col_defs = [
					("Opción 1 · Sin tarjetas/corners",
					 item["multiple_no_cards_corners_probability_text"],
					 float(item["multiple_no_cards_corners_probability"]),
					 item["multiple_legs_no_cards_corners"]),
					("Opción 2 · Combinada (2 mejores)",
					 item["multiple_top2_combined_probability_text"],
					 float(item["multiple_top2_combined_probability"]),
					 item["multiple_legs_top2"]),
					("Opción 3 · Probabilidad general",
					 item["multiple_combined_probability_text"],
					 float(item["multiple_combined_probability"]),
					 item["multiple_legs_general"]),
				]
				COL_W = 2.45 * inch
				cells_row = []
				bg_cols = []
				border_cols = []
				for col_title, prob_text, prob_raw, legs in col_defs:
					cbg, cborder, ctitle_c, cvalue_c = _metric_bg_border(prob_raw)
					bg_cols.append(cbg)
					border_cols.append(cborder)
					m_title_st = ParagraphStyle("MT", parent=styles["Normal"], fontName="Helvetica-Bold",
												 fontSize=7, leading=9, textColor=ctitle_c, spaceAfter=3)
					m_value_st = ParagraphStyle("MV", parent=styles["Normal"], fontName="Helvetica-Bold",
												 fontSize=16, leading=18, textColor=cvalue_c, spaceAfter=4)
					m_leg_st = ParagraphStyle("ML", parent=styles["Normal"], fontName="Helvetica",
											   fontSize=7, leading=9, textColor=ctitle_c, spaceAfter=1)
					cell = [
						Paragraph(col_title.upper(), m_title_st),
						Paragraph(f"{prob_text}%", m_value_st),
					]
					if legs:
						for leg in legs:
							cell.append(Paragraph(f"\u2022 {leg}", m_leg_st))
					else:
						cell.append(Paragraph("Sin selecciones válidas", m_leg_st))
					cells_row.append(cell)

				metrics_table = Table([cells_row], colWidths=[COL_W, COL_W, COL_W])
				ts_cmds = [
					("VALIGN", (0, 0), (-1, -1), "TOP"),
					("TOPPADDING", (0, 0), (-1, -1), 8),
					("BOTTOMPADDING", (0, 0), (-1, -1), 8),
					("LEFTPADDING", (0, 0), (-1, -1), 8),
					("RIGHTPADDING", (0, 0), (-1, -1), 8),
				]
				for ci, (cbg, cborder) in enumerate(zip(bg_cols, border_cols)):
					ts_cmds.extend([
						("BACKGROUND", (ci, 0), (ci, 0), cbg),
						("BOX", (ci, 0), (ci, 0), 1.5, cborder),
					])
				metrics_table.setStyle(TableStyle(ts_cmds))
				story.append(metrics_table)
			else:
				no_sel_data = [[Paragraph("Ninguna selección supera el 75% de confianza en este partido.", value_style)]]
				no_sel_table = Table(no_sel_data, colWidths=[7.5*inch])
				no_sel_table.setStyle(TableStyle([
					("BACKGROUND", (0, 0), (-1, -1), colors.white),
					("BORDER", (0, 0), (-1, -1), 1, conf_border),
					("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#555555")),
					("PADDING", (0, 0), (-1, -1), 8),
					("LEFTPADDING", (0, 0), (-1, -1), 10),
				]))
				story.append(no_sel_table)
			story.append(Spacer(1, 0.2*inch))

	# Generar PDF
	doc.build(story)
	buffer.seek(0)

	response = HttpResponse(buffer.getvalue(), content_type="application/pdf")
	response["Content-Disposition"] = f'attachment; filename="mejores_apuestas_{window_start.strftime("%d_%m_%Y")}.pdf"'
	return response
