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
	get_prediction_service_primeiraliga,
	get_prediction_service_proleague,
	get_prediction_service_eredivisie,
)
from .sync import refresh_fixture_links


LEAGUE_SERVICE_FACTORIES = {
	"spain": get_prediction_service_spain,
	"bundesliga": get_prediction_service_bundesliga,
	"premier": get_prediction_service_premier,
	"seriea": get_prediction_service_seriea,
	"ligue1": get_prediction_service_ligue1,
	"primeiraliga": get_prediction_service_primeiraliga,
	"proleague": get_prediction_service_proleague,
	"eredivisie": get_prediction_service_eredivisie,
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


def _fair_odds(probability_pct: float) -> float:
	if probability_pct <= 0:
		return 0.0
	return round(100.0 / probability_pct, 2)


def _parse_decimal_odd(value: str | None) -> float | None:
	if value is None:
		return None
	clean = str(value).strip().replace(",", ".")
	if not clean:
		return None
	try:
		odd = float(clean)
	except ValueError:
		return None
	if odd <= 1.0:
		return None
	return odd


def _value_signal(probability_pct: float, offered_odd: float | None) -> dict[str, object]:
	fair_odd = _fair_odds(probability_pct)
	has_value = bool(offered_odd is not None and fair_odd > 0 and offered_odd > fair_odd)
	edge = round((offered_odd - fair_odd), 2) if offered_odd is not None else 0.0
	return {
		"probability": round(probability_pct, 2),
		"fair_odd": fair_odd,
		"offered_odd": offered_odd,
		"has_value": has_value,
		"edge": edge,
	}


def _build_league_stats(service) -> dict[str, object]:
	df = service.historical_df
	total_matches = int(len(df))
	if total_matches <= 0:
		return {
			"total_matches": 0,
			"avg_goals": 0.0,
			"home_win_pct": 0.0,
			"draw_pct": 0.0,
			"away_win_pct": 0.0,
			"btts_pct": 0.0,
			"over25_pct": 0.0,
			"avg_corners": 0.0,
			"avg_cards": 0.0,
			"red_match_pct": 0.0,
			"market_tone": "Sin datos",
			"recommendations": ["No hay suficientes partidos historicos para clasificar tendencias."],
			"top_table": [],
		}

	home_win_pct = round(float((df["FTR"] == "H").mean() * 100), 2)
	draw_pct = round(float((df["FTR"] == "D").mean() * 100), 2)
	away_win_pct = round(float((df["FTR"] == "A").mean() * 100), 2)
	avg_goals = round(float((df["FTHG"] + df["FTAG"]).mean()), 2)
	btts_pct = round(float(((df["FTHG"] > 0) & (df["FTAG"] > 0)).mean() * 100), 2)
	over25_pct = round(float(((df["FTHG"] + df["FTAG"]) > 2).mean() * 100), 2)
	avg_corners = round(float((df["HC"] + df["AC"]).mean()), 2)
	avg_cards = round(float((df["HY"] + df["AY"] + df["HR"] + df["AR"]).mean()), 2)
	red_match_pct = round(float(((df["HR"] + df["AR"]) > 0).mean() * 100), 2)

	if avg_goals >= 2.85:
		market_tone = "Liga de ritmo alto"
	elif avg_goals >= 2.45:
		market_tone = "Liga equilibrada"
	else:
		market_tone = "Liga de control tactico"

	recommendations: list[str] = []
	if over25_pct >= 56:
		recommendations.append("Prioriza mercados de goles (Over 2.5 / BTTS) en prepartido.")
	elif over25_pct <= 44:
		recommendations.append("Prioriza unders y lineas conservadoras de goles.")
	else:
		recommendations.append("Mercado de goles mixto: conviene filtrar por forma del partido.")

	if home_win_pct - away_win_pct >= 12:
		recommendations.append("Existe sesgo local fuerte: prioriza 1X o local DNB en equipos top.")
	elif away_win_pct >= home_win_pct:
		recommendations.append("Sin sesgo local marcado: evita sobrepagar cuotas del favorito local.")
	else:
		recommendations.append("Sesgo local moderado: combina 1X2 con contexto de forma reciente.")

	if avg_cards >= 4.8 or red_match_pct >= 20:
		recommendations.append("Perfil de contacto alto: hay valor en tarjetas over y props disciplinarios.")
	else:
		recommendations.append("Disciplina estable: mejor usar tarjetas como mercado complementario.")

	top_table_raw = sorted(
		service.standings_snapshot.values(),
		key=lambda row: (row.get("points", 0), row.get("gd", 0), row.get("gf", 0)),
		reverse=True,
	)[:5]
	top_table = [
		{
			"position": team.get("position", 0),
			"team": team.get("team", ""),
			"points": team.get("points", 0),
			"gd": team.get("gd", 0),
		}
		for team in top_table_raw
	]

	return {
		"total_matches": total_matches,
		"avg_goals": avg_goals,
		"home_win_pct": home_win_pct,
		"draw_pct": draw_pct,
		"away_win_pct": away_win_pct,
		"btts_pct": btts_pct,
		"over25_pct": over25_pct,
		"avg_corners": avg_corners,
		"avg_cards": avg_cards,
		"red_match_pct": red_match_pct,
		"market_tone": market_tone,
		"recommendations": recommendations,
		"top_table": top_table,
	}


def _best_bets_snapshot(max_items: int = 8) -> dict[str, object]:
	today = datetime.now().date()
	eligible_fixtures: list[tuple[str, object, dict[str, object], object]] = []

	for liga_key, factory in LEAGUE_SERVICE_FACTORIES.items():
		service = factory()
		for fixture in service.get_pending_fixtures():
			date_label = str(fixture.get("fecha", ""))
			sort_date = _parse_fixture_date(date_label)
			if sort_date is None or sort_date < today:
				continue
			eligible_fixtures.append((liga_key, service, fixture, sort_date))

	if not eligible_fixtures:
		return {
			"window_label": "sin encuentros pendientes",
			"entries": [],
			"total_count": 0,
		}

	window_start = min(row[3] for row in eligible_fixtures)
	window_end = window_start + timedelta(days=1)
	window_fixtures = [row for row in eligible_fixtures if window_start <= row[3] <= window_end]

	entries: list[dict[str, object]] = []
	for liga_key, service, fixture, sort_date in window_fixtures:
		try:
			quick_prediction = service.predict_recommended_bet_fast(str(fixture.get("match_key", "")))
		except Exception:
			continue

		recommended = quick_prediction.get("recommended_bet", {})
		multiple = quick_prediction.get("multiple", {})
		entries.append(
			{
				"date_label": str(fixture.get("fecha", "")),
				"sort_date": sort_date,
				"league_key": liga_key,
				"league_name": service.league_name,
				"league_logo_url": service.league_logo_url,
				"match_key": str(fixture.get("match_key", "")),
				"kickoff": str(fixture.get("hora", "")),
				"home_team": str(fixture.get("local", "")),
				"away_team": str(fixture.get("visitante", "")),
				"market": str(recommended.get("market", "Sin datos")),
				"pick": str(recommended.get("pick", "Sin recomendacion")),
				"probability": float(recommended.get("probability", 0.0)),
				"confidence": str(recommended.get("confidence", "Baja")),
				"multiple_confidence": str(multiple.get("confidence", "Baja")),
				"multiple_combined_probability": float(multiple.get("prob_combinada", 0.0)),
				"dashboard_url": f"/?liga={liga_key}&match_key={fixture.get('match_key', '')}",
			}
		)

	entries = sorted(
		entries,
		key=lambda item: (
			item["sort_date"],
			-float(item.get("multiple_combined_probability", 0.0)),
			-float(item.get("probability", 0.0)),
			item.get("kickoff", ""),
		),
	)

	window_label = f"{window_start.strftime('%d/%m/%Y')} y {window_end.strftime('%d/%m/%Y')}"
	return {
		"window_label": window_label,
		"entries": entries[:max_items],
		"total_count": len(entries),
	}


def _build_multileague_home(selected_date_raw: str | None) -> dict[str, object]:
	today = datetime.now().date()
	eligible: list[tuple[str, object, dict[str, object], object]] = []

	for liga_key, factory in LEAGUE_SERVICE_FACTORIES.items():
		service = factory()
		for fixture in service.get_pending_fixtures():
			sort_date = _parse_fixture_date(str(fixture.get("fecha", "")))
			if sort_date is None or sort_date < today:
				continue
			eligible.append((liga_key, service, fixture, sort_date))

	if not eligible:
		return {
			"selected_date_label": "sin fecha",
			"date_options": [],
			"league_cards": [],
			"entries": [],
		}

	available_dates = sorted({row[3] for row in eligible})
	selected_date = None
	if selected_date_raw:
		try:
			selected_date = datetime.strptime(selected_date_raw, "%Y-%m-%d").date()
		except ValueError:
			selected_date = None
	if selected_date is None or selected_date not in available_dates:
		selected_date = available_dates[0]

	target_fixtures = [row for row in eligible if row[3] == selected_date]
	entries: list[dict[str, object]] = []
	for liga_key, service, fixture, sort_date in target_fixtures:
		try:
			quick_prediction = service.predict_recommended_bet_fast(str(fixture.get("match_key", "")))
		except Exception:
			continue

		recommended = quick_prediction.get("recommended_bet", {})
		multiple = quick_prediction.get("multiple", {})
		entries.append(
			{
				"date_label": str(fixture.get("fecha", "")),
				"league_key": liga_key,
				"league_name": service.league_name,
				"league_logo_url": service.league_logo_url,
				"match_key": str(fixture.get("match_key", "")),
				"kickoff": str(fixture.get("hora", "")),
				"home_team": str(fixture.get("local", "")),
				"away_team": str(fixture.get("visitante", "")),
				"market": str(recommended.get("market", "Sin datos")),
				"pick": str(recommended.get("pick", "Sin recomendacion")),
				"probability": float(recommended.get("probability", 0.0)),
				"confidence": str(recommended.get("confidence", "Baja")),
				"multiple_confidence": str(multiple.get("confidence", "Baja")),
				"multiple_combined_probability": float(multiple.get("prob_combinada", 0.0)),
				"dashboard_url": f"/?liga={liga_key}&match_key={fixture.get('match_key', '')}",
			}
		)

	def _kickoff_sort_value(value: object) -> tuple[int, int]:
		text = str(value or "").strip()
		if ":" not in text:
			return (99, 99)
		parts = text.split(":", 1)
		try:
			hour = int(parts[0])
			minute = int(parts[1])
		except ValueError:
			return (99, 99)
		return (hour, minute)

	entries = sorted(
		entries,
		key=lambda item: (
			_kickoff_sort_value(item.get("kickoff", "")),
			-float(item.get("multiple_combined_probability", 0.0)),
			-float(item.get("probability", 0.0)),
		),
	)

	league_cards_map: dict[str, dict[str, object]] = {}
	for row in entries:
		card = league_cards_map.setdefault(
			str(row["league_key"]),
			{
				"league_key": row["league_key"],
				"league_name": row["league_name"],
				"league_logo_url": row["league_logo_url"],
				"match_count": 0,
				"avg_probability": 0.0,
				"top_pick": "Sin pick",
				"top_pick_probability": 0.0,
			},
		)
		card["match_count"] = int(card["match_count"]) + 1
		card["avg_probability"] = float(card["avg_probability"]) + float(row.get("probability", 0.0))
		if float(row.get("probability", 0.0)) > float(card.get("top_pick_probability", 0.0)):
			card["top_pick_probability"] = float(row.get("probability", 0.0))
			card["top_pick"] = str(row.get("pick", "Sin pick"))

	league_cards = []
	for card in league_cards_map.values():
		count = int(card["match_count"])
		avg_prob = (float(card["avg_probability"]) / count) if count else 0.0
		league_cards.append(
			{
				**card,
				"avg_probability": round(avg_prob, 2),
			}
		)

	league_cards = sorted(league_cards, key=lambda item: (-float(item["avg_probability"]), -int(item["match_count"])))
	date_options = [
		{
			"value": date_item.strftime("%Y-%m-%d"),
			"label": date_item.strftime("%d/%m/%Y"),
			"is_selected": date_item == selected_date,
		}
		for date_item in available_dates
	]

	return {
		"selected_date_label": selected_date.strftime("%d/%m/%Y"),
		"date_options": date_options,
		"league_cards": league_cards,
		"entries": entries,
	}


def _build_home_rankings() -> dict[str, list[dict[str, object]]]:
	rank_1x2: list[dict[str, object]] = []
	rank_over15: list[dict[str, object]] = []
	rank_under45: list[dict[str, object]] = []
	rank_avg_goals: list[dict[str, object]] = []
	rank_over25: list[dict[str, object]] = []
	rank_under35: list[dict[str, object]] = []
	rank_btts: list[dict[str, object]] = []

	for _, factory in LEAGUE_SERVICE_FACTORIES.items():
		service = factory()
		df = service.historical_df
		total_matches = int(len(df))
		if total_matches <= 0:
			continue

		results = df["FTR"].astype(str)
		one_x_two_top = max(
			float((results == "H").mean() * 100),
			float((results == "D").mean() * 100),
			float((results == "A").mean() * 100),
		)
		over15 = float(((df["FTHG"] + df["FTAG"]) > 1).mean() * 100)
		over25 = float(((df["FTHG"] + df["FTAG"]) > 2).mean() * 100)
		under35 = float(((df["FTHG"] + df["FTAG"]) < 4).mean() * 100)
		under45 = float(((df["FTHG"] + df["FTAG"]) < 5).mean() * 100)
		btts = float(((df["FTHG"] > 0) & (df["FTAG"] > 0)).mean() * 100)
		avg_goals = float((df["FTHG"] + df["FTAG"]).mean())

		base = {"league": service.league_name}
		rank_1x2.append({**base, "value": round(one_x_two_top, 2)})
		rank_over15.append({**base, "value": round(over15, 2)})
		rank_over25.append({**base, "value": round(over25, 2)})
		rank_under35.append({**base, "value": round(under35, 2)})
		rank_under45.append({**base, "value": round(under45, 2)})
		rank_btts.append({**base, "value": round(btts, 2)})
		rank_avg_goals.append({**base, "value": round(avg_goals, 2)})

	return {
		"top_1x2": sorted(rank_1x2, key=lambda item: float(item["value"]), reverse=True),
		"top_over15": sorted(rank_over15, key=lambda item: float(item["value"]), reverse=True),
		"top_under45": sorted(rank_under45, key=lambda item: float(item["value"]), reverse=True),
		"top_avg_goals": sorted(rank_avg_goals, key=lambda item: float(item["value"]), reverse=True),
		"top_over25": sorted(rank_over25, key=lambda item: float(item["value"]), reverse=True),
		"top_under35": sorted(rank_under35, key=lambda item: float(item["value"]), reverse=True),
		"top_btts": sorted(rank_btts, key=lambda item: float(item["value"]), reverse=True),
	}


def dashboard(request):
	liga = request.GET.get("liga") or request.POST.get("liga", "spain")
	if liga not in ("spain", "bundesliga", "premier", "seriea", "ligue1", "primeiraliga", "proleague", "eredivisie"):
		liga = "spain"

	refresh_status = ""
	if request.method == "GET":
		try:
			updated_rows = refresh_fixture_links(liga)
			refresh_status = f"Enlaces actualizados y guardados en base de datos: {updated_rows} encuentros."
		except Exception as exc:
			refresh_status = f"No se pudo refrescar enlaces en este acceso: {exc}"

	service = _get_service(liga)
	league_stats = _build_league_stats(service)
	best_bets_snapshot = _best_bets_snapshot(max_items=8)
	multileague_home = _build_multileague_home(request.GET.get("home_date"))
	home_rankings = _build_home_rankings()
	fixtures = service.get_pending_fixtures()
	selected_match_key = request.POST.get("match_key") or request.GET.get("match_key", "")
	sportsbook_rows: list[dict[str, object]] = []
	for fixture in fixtures[:8]:
		try:
			match_prediction = service.predict_match(str(fixture.get("match_key", "")))
			totals_market = match_prediction.get("markets", {}).get("totales", {})
			over_1_5 = float(totals_market.get("over_1_5", 0.0))
			under_3_5 = float(totals_market.get("under_3_5", 0.0))
			under_4_5 = float(totals_market.get("under_4_5", 0.0))
			sportsbook_rows.append(
				{
					"local": fixture.get("local", ""),
					"visitante": fixture.get("visitante", ""),
					"hora": fixture.get("hora", ""),
					"odd_over_1_5": _fair_odds(over_1_5),
					"odd_under_3_5": _fair_odds(under_3_5),
					"odd_under_4_5": _fair_odds(under_4_5),
				}
			)
		except Exception:
			continue

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
		"validation_metrics": service.validation_metrics,
		"no_bet_min_prob": round(service.no_bet_min_prob, 2),
		"market_min_prob": service.market_min_prob,
		"model_scores": service.model_scores,
		"error_message": error_message,
		"refresh_status": refresh_status,
		"league_stats": league_stats,
		"sportsbook_rows": sportsbook_rows,
		"best_bets_snapshot": best_bets_snapshot,
		"multileague_home": multileague_home,
		"home_rankings": home_rankings,
	}
	context["market_min_prob_rows"] = [
		{"label": "Umbral 1X2", "value": service.market_min_prob.get("1X2", service.no_bet_min_prob)},
		{"label": "Umbral doble oportunidad", "value": service.market_min_prob.get("Doble oportunidad", service.no_bet_min_prob)},
		{"label": "Umbral totales", "value": service.market_min_prob.get("Totales", service.no_bet_min_prob)},
		{"label": "Umbral BTTS", "value": service.market_min_prob.get("Ambos marcan", service.no_bet_min_prob)},
		{"label": "Umbral corners", "value": service.market_min_prob.get("Corners", service.no_bet_min_prob)},
		{"label": "Umbral tarjetas", "value": service.market_min_prob.get("Tarjetas totales", service.no_bet_min_prob)},
	]
	return render(request, "predictor/home_dashboard.html", context)


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

	# Calcular recomendaciones con el mismo pipeline del dashboard para evitar diferencias.
	for liga, service, fixture, sort_date in window_fixtures:
		try:
			prediction = service.predict_match(fixture["match_key"])
		except ValueError:
			continue

		recommended = prediction.get("match_report", {}).get("recommended_bet", {})
		multiple = prediction.get("categorized_bets", {}).get("multiple", {})
		multiple_legs = list(multiple.get("legs", []))
		top_two_legs = _top_two_legs(multiple_legs)
		no_cards_corners_legs = _legs_without_cards_corners(multiple_legs)
		combined_probability = float(multiple.get("prob_combinada", 0.0))
		top_two_combined_probability = _combined_top_two_probability(multiple_legs)
		no_cards_corners_probability = _combined_without_cards_corners_probability(multiple_legs)
		combined_fair_odds = _fair_odds(combined_probability)
		top_two_fair_odds = _fair_odds(top_two_combined_probability)
		no_cards_corners_fair_odds = _fair_odds(no_cards_corners_probability)
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
			"market": str(recommended.get("market", "Sin datos")),
			"pick": str(recommended.get("pick", "Sin recomendacion")),
			"probability": float(recommended.get("probability", 0.0)),
			"confidence": str(recommended.get("confidence", "Baja")),
			"multiple_confidence": str(multiple.get("confidence", "Sin selecciones")),
			"multiple_legs": multiple_legs,
			"multiple_legs_general": multiple_legs,
			"multiple_legs_top2": top_two_legs,
			"multiple_legs_no_cards_corners": no_cards_corners_legs,
			"multiple_combined_probability": combined_probability,
			"multiple_combined_probability_text": f"{combined_probability:.2f}".replace(".", ","),
			"multiple_combined_fair_odds": combined_fair_odds,
			"multiple_combined_fair_odds_text": f"{combined_fair_odds:.2f}".replace(".", ","),
			"multiple_top2_combined_probability": top_two_combined_probability,
			"multiple_top2_combined_probability_text": f"{top_two_combined_probability:.2f}".replace(".", ","),
			"multiple_top2_combined_fair_odds": top_two_fair_odds,
			"multiple_top2_combined_fair_odds_text": f"{top_two_fair_odds:.2f}".replace(".", ","),
			"multiple_no_cards_corners_probability": no_cards_corners_probability,
			"multiple_no_cards_corners_probability_text": f"{no_cards_corners_probability:.2f}".replace(".", ","),
			"multiple_no_cards_corners_fair_odds": no_cards_corners_fair_odds,
			"multiple_no_cards_corners_fair_odds_text": f"{no_cards_corners_fair_odds:.2f}".replace(".", ","),
			"dashboard_url": f"/?liga={liga}&match_key={fixture['match_key']}",
		}
		best_entries.append(entry)

	best_entries = sorted(
		best_entries,
		key=lambda item: (
			item["sort_date"],
			-float(item["multiple_combined_probability"]),
			-float(item.get("multiple_top2_combined_probability", 0.0)),
			-float(item.get("multiple_no_cards_corners_probability", 0.0)),
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

	# Calcular recomendaciones con el mismo pipeline del dashboard para evitar diferencias.
	for liga, service, fixture, sort_date in window_fixtures:
		try:
			prediction = service.predict_match(fixture["match_key"])
		except ValueError:
			continue

		multiple = prediction.get("categorized_bets", {}).get("multiple", {})
		multiple_legs = list(multiple.get("legs", []))
		top_two_legs = _top_two_legs(multiple_legs)
		no_cards_corners_legs = _legs_without_cards_corners(multiple_legs)
		combined_probability = float(multiple.get("prob_combinada", 0.0))
		top_two_combined_probability = _combined_top_two_probability(multiple_legs)
		no_cards_corners_probability = _combined_without_cards_corners_probability(multiple_legs)
		combined_fair_odds = _fair_odds(combined_probability)
		top_two_fair_odds = _fair_odds(top_two_combined_probability)
		no_cards_corners_fair_odds = _fair_odds(no_cards_corners_probability)
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
			"multiple_combined_fair_odds": combined_fair_odds,
			"multiple_combined_fair_odds_text": f"{combined_fair_odds:.2f}".replace(".", ","),
			"multiple_top2_combined_probability": top_two_combined_probability,
			"multiple_top2_combined_probability_text": f"{top_two_combined_probability:.2f}".replace(".", ","),
			"multiple_top2_combined_fair_odds": top_two_fair_odds,
			"multiple_top2_combined_fair_odds_text": f"{top_two_fair_odds:.2f}".replace(".", ","),
			"multiple_no_cards_corners_probability": no_cards_corners_probability,
			"multiple_no_cards_corners_probability_text": f"{no_cards_corners_probability:.2f}".replace(".", ","),
			"multiple_no_cards_corners_fair_odds": no_cards_corners_fair_odds,
			"multiple_no_cards_corners_fair_odds_text": f"{no_cards_corners_fair_odds:.2f}".replace(".", ","),
		}
		best_entries.append(entry)

	best_entries = sorted(
		best_entries,
		key=lambda item: (
			item["sort_date"],
			-float(item["multiple_combined_probability"]),
			-float(item.get("multiple_top2_combined_probability", 0.0)),
			-float(item.get("multiple_no_cards_corners_probability", 0.0)),
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
					 item["multiple_no_cards_corners_fair_odds_text"],
					 float(item["multiple_no_cards_corners_probability"]),
					 item["multiple_legs_no_cards_corners"]),
					("Opción 2 · Combinada (2 mejores)",
					 item["multiple_top2_combined_probability_text"],
					 item["multiple_top2_combined_fair_odds_text"],
					 float(item["multiple_top2_combined_probability"]),
					 item["multiple_legs_top2"]),
					("Opción 3 · Probabilidad general",
					 item["multiple_combined_probability_text"],
					 item["multiple_combined_fair_odds_text"],
					 float(item["multiple_combined_probability"]),
					 item["multiple_legs_general"]),
				]
				COL_W = 2.45 * inch
				cells_row = []
				bg_cols = []
				border_cols = []
				for col_title, prob_text, odds_text, prob_raw, legs in col_defs:
					cbg, cborder, ctitle_c, cvalue_c = _metric_bg_border(prob_raw)
					bg_cols.append(cbg)
					border_cols.append(cborder)
					m_title_st = ParagraphStyle("MT", parent=styles["Normal"], fontName="Helvetica-Bold",
												 fontSize=7, leading=9, textColor=ctitle_c, spaceAfter=3)
					m_value_st = ParagraphStyle("MV", parent=styles["Normal"], fontName="Helvetica-Bold",
												 fontSize=16, leading=18, textColor=cvalue_c, spaceAfter=4)
					m_odds_st = ParagraphStyle("MO", parent=styles["Normal"], fontName="Helvetica-Bold",
											 fontSize=8, leading=10, textColor=ctitle_c, spaceAfter=4)
					m_leg_st = ParagraphStyle("ML", parent=styles["Normal"], fontName="Helvetica",
											   fontSize=7, leading=9, textColor=ctitle_c, spaceAfter=1)
					cell = [
						Paragraph(col_title.upper(), m_title_st),
						Paragraph(f"{prob_text}%", m_value_st),
						Paragraph(f"Cuota justa: {odds_text}", m_odds_st),
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


def best_bets_1x2_pdf(request):
	"""Generar PDF de mejores apuestas 1X2 por fecha."""
	entries: list[dict[str, object]] = []
	today = datetime.now().date()

	eligible_fixtures: list[tuple[str, object, dict[str, object], object]] = []
	for liga, factory in LEAGUE_SERVICE_FACTORIES.items():
		service = factory()
		for fixture in service.get_pending_fixtures():
			sort_date = _parse_fixture_date(str(fixture["fecha"]))
			if sort_date is None or sort_date < today:
				continue
			eligible_fixtures.append((liga, service, fixture, sort_date))

	if not eligible_fixtures:
		return HttpResponse("No hay encuentros pendientes.", content_type="text/plain")

	window_start = min(row[3] for row in eligible_fixtures)
	window_end = window_start + timedelta(days=1)
	window_fixtures = [row for row in eligible_fixtures if window_start <= row[3] <= window_end]

	for _liga, service, fixture, sort_date in window_fixtures:
		try:
			prediction = service.predict_match(fixture["match_key"])
		except ValueError:
			continue

		pick_1x2 = prediction.get("categorized_bets", {}).get("resultado_1x2", {})
		prob = float(pick_1x2.get("prob", 0.0))
		fair_odds = _fair_odds(prob)
		entries.append(
			{
				"sort_date": sort_date,
				"date_label": str(fixture["fecha"]),
				"league_name": service.league_name,
				"home_team": str(fixture["local"]),
				"away_team": str(fixture["visitante"]),
				"kickoff": str(fixture["hora"]),
				"market": str(pick_1x2.get("market", "1X2")),
				"pick": str(pick_1x2.get("pick", "Sin dato")),
				"probability": prob,
				"probability_text": f"{prob:.2f}".replace(".", ","),
				"fair_odds": fair_odds,
				"fair_odds_text": f"{fair_odds:.2f}".replace(".", ","),
			}
		)

	entries = sorted(entries, key=lambda item: (item["sort_date"], -float(item["probability"]), item["kickoff"]))

	buffer = BytesIO()
	doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=0.5*inch, leftMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
	styles = getSampleStyleSheet()
	story = []

	title_style = ParagraphStyle("Title1X2", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=22, textColor=colors.HexColor("#0f3d28"), alignment=1, spaceAfter=6)
	subtitle_style = ParagraphStyle("Sub1X2", parent=styles["Normal"], fontName="Helvetica", fontSize=10, textColor=colors.HexColor("#1a7d5c"), alignment=1, spaceAfter=16)
	header_style = ParagraphStyle("Header1X2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, textColor=colors.white, spaceAfter=6, spaceBefore=10)
	row_style = ParagraphStyle("Row1X2", parent=styles["Normal"], fontName="Helvetica", fontSize=9, leading=11)

	window_label = f"{window_start.strftime('%d/%m/%Y')} al {window_end.strftime('%d/%m/%Y')}"
	story.append(Paragraph("MEJORES APUESTAS 1X2 POR FECHA", title_style))
	story.append(Paragraph(f"Ventana: {window_label}", subtitle_style))

	grouped: dict[str, list[dict[str, object]]] = {}
	for item in entries:
		grouped.setdefault(str(item["date_label"]), []).append(item)

	for date_label in sorted(grouped.keys()):
		head = Table([[Paragraph(f"FECHA: {date_label}", header_style)]], colWidths=[7.5*inch])
		head.setStyle(TableStyle([
			("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1a7d5c")),
			("LEFTPADDING", (0, 0), (-1, -1), 10),
			("TOPPADDING", (0, 0), (-1, -1), 8),
			("BOTTOMPADDING", (0, 0), (-1, -1), 8),
		]))
		story.append(head)
		story.append(Spacer(1, 0.08*inch))

		rows = [["#", "Encuentro", "Liga", "1X2", "Prob.", "Cuota justa"]]
		for idx, item in enumerate(grouped[date_label], 1):
			rows.append([
				str(idx),
				f"{item['home_team']} vs {item['away_team']} ({item['kickoff']})",
				str(item["league_name"]),
				str(item["pick"]),
				f"{item['probability_text']}%",
				str(item["fair_odds_text"]),
			])

		table = Table(rows, colWidths=[0.4*inch, 2.9*inch, 1.35*inch, 0.8*inch, 0.9*inch, 1.1*inch])
		table.setStyle(TableStyle([
			("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f5f0")),
			("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f3d28")),
			("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
			("FONTSIZE", (0, 0), (-1, -1), 8),
			("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#a7d7c4")),
			("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
			("LEFTPADDING", (0, 0), (-1, -1), 5),
			("RIGHTPADDING", (0, 0), (-1, -1), 5),
		]))
		story.append(table)
		story.append(Spacer(1, 0.15*inch))

	doc.build(story)
	buffer.seek(0)
	response = HttpResponse(buffer.getvalue(), content_type="application/pdf")
	response["Content-Disposition"] = f'attachment; filename="mejores_apuestas_1x2_{window_start.strftime("%d_%m_%Y")}.pdf"'
	return response


def best_bets_double_chance_pdf(request):
	"""Generar PDF de mejores apuestas Doble Oportunidad por fecha."""
	entries: list[dict[str, object]] = []
	today = datetime.now().date()

	eligible_fixtures: list[tuple[str, object, dict[str, object], object]] = []
	for liga, factory in LEAGUE_SERVICE_FACTORIES.items():
		service = factory()
		for fixture in service.get_pending_fixtures():
			sort_date = _parse_fixture_date(str(fixture["fecha"]))
			if sort_date is None or sort_date < today:
				continue
			eligible_fixtures.append((liga, service, fixture, sort_date))

	if not eligible_fixtures:
		return HttpResponse("No hay encuentros pendientes.", content_type="text/plain")

	window_start = min(row[3] for row in eligible_fixtures)
	window_end = window_start + timedelta(days=1)
	window_fixtures = [row for row in eligible_fixtures if window_start <= row[3] <= window_end]

	for _liga, service, fixture, sort_date in window_fixtures:
		try:
			prediction = service.predict_match(fixture["match_key"])
		except ValueError:
			continue

		pick_dc = prediction.get("categorized_bets", {}).get("doble_oportunidad", {})
		prob = float(pick_dc.get("prob", 0.0))
		fair_odds = _fair_odds(prob)
		entries.append(
			{
				"sort_date": sort_date,
				"date_label": str(fixture["fecha"]),
				"league_name": service.league_name,
				"home_team": str(fixture["local"]),
				"away_team": str(fixture["visitante"]),
				"kickoff": str(fixture["hora"]),
				"market": str(pick_dc.get("market", "Doble oportunidad")),
				"pick": str(pick_dc.get("pick", "Sin dato")),
				"probability": prob,
				"probability_text": f"{prob:.2f}".replace(".", ","),
				"fair_odds": fair_odds,
				"fair_odds_text": f"{fair_odds:.2f}".replace(".", ","),
			}
		)

	entries = sorted(entries, key=lambda item: (item["sort_date"], -float(item["probability"]), item["kickoff"]))

	buffer = BytesIO()
	doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=0.5*inch, leftMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
	styles = getSampleStyleSheet()
	story = []

	title_style = ParagraphStyle("TitleDC", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=22, textColor=colors.HexColor("#0f3d28"), alignment=1, spaceAfter=6)
	subtitle_style = ParagraphStyle("SubDC", parent=styles["Normal"], fontName="Helvetica", fontSize=10, textColor=colors.HexColor("#1a7d5c"), alignment=1, spaceAfter=16)
	header_style = ParagraphStyle("HeaderDC", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, textColor=colors.white, spaceAfter=6, spaceBefore=10)

	window_label = f"{window_start.strftime('%d/%m/%Y')} al {window_end.strftime('%d/%m/%Y')}"
	story.append(Paragraph("MEJORES APUESTAS DOBLE OPORTUNIDAD", title_style))
	story.append(Paragraph(f"Ventana: {window_label}", subtitle_style))

	grouped: dict[str, list[dict[str, object]]] = {}
	for item in entries:
		grouped.setdefault(str(item["date_label"]), []).append(item)

	for date_label in sorted(grouped.keys()):
		head = Table([[Paragraph(f"FECHA: {date_label}", header_style)]], colWidths=[7.5*inch])
		head.setStyle(TableStyle([
			("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1a7d5c")),
			("LEFTPADDING", (0, 0), (-1, -1), 10),
			("TOPPADDING", (0, 0), (-1, -1), 8),
			("BOTTOMPADDING", (0, 0), (-1, -1), 8),
		]))
		story.append(head)
		story.append(Spacer(1, 0.08*inch))

		rows = [["#", "Encuentro", "Liga", "Doble oportunidad", "Prob.", "Cuota justa"]]
		for idx, item in enumerate(grouped[date_label], 1):
			rows.append([
				str(idx),
				f"{item['home_team']} vs {item['away_team']} ({item['kickoff']})",
				str(item["league_name"]),
				str(item["pick"]),
				f"{item['probability_text']}%",
				str(item["fair_odds_text"]),
			])

		table = Table(rows, colWidths=[0.4*inch, 2.8*inch, 1.35*inch, 1.1*inch, 0.8*inch, 1.05*inch])
		table.setStyle(TableStyle([
			("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f5f0")),
			("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f3d28")),
			("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
			("FONTSIZE", (0, 0), (-1, -1), 8),
			("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#a7d7c4")),
			("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
			("LEFTPADDING", (0, 0), (-1, -1), 5),
			("RIGHTPADDING", (0, 0), (-1, -1), 5),
		]))
		story.append(table)
		story.append(Spacer(1, 0.15*inch))

	doc.build(story)
	buffer.seek(0)
	response = HttpResponse(buffer.getvalue(), content_type="application/pdf")
	response["Content-Disposition"] = f'attachment; filename="mejores_apuestas_doble_oportunidad_{window_start.strftime("%d_%m_%Y")}.pdf"'
	return response


def best_bets_totals_pdf(request):
	"""Generar PDF de mercados Totales (Over 1.5/2.5 y Under 3.5/4.5) por fecha."""
	entries: list[dict[str, object]] = []
	today = datetime.now().date()

	eligible_fixtures: list[tuple[str, object, dict[str, object], object]] = []
	for liga, factory in LEAGUE_SERVICE_FACTORIES.items():
		service = factory()
		for fixture in service.get_pending_fixtures():
			sort_date = _parse_fixture_date(str(fixture["fecha"]))
			if sort_date is None or sort_date < today:
				continue
			eligible_fixtures.append((liga, service, fixture, sort_date))

	if not eligible_fixtures:
		return HttpResponse("No hay encuentros pendientes.", content_type="text/plain")

	window_start = min(row[3] for row in eligible_fixtures)
	window_end = window_start + timedelta(days=1)
	window_fixtures = [row for row in eligible_fixtures if window_start <= row[3] <= window_end]

	for _liga, service, fixture, sort_date in window_fixtures:
		try:
			prediction = service.predict_match(fixture["match_key"])
		except ValueError:
			continue

		totals = prediction.get("markets", {}).get("totales", {})
		o15 = float(totals.get("over_1_5", 0.0))
		o25 = float(totals.get("over_2_5", 0.0))
		u35 = float(totals.get("under_3_5", 0.0))
		u45 = float(totals.get("under_4_5", 0.0))
		options = [
			("Over 1.5", o15),
			("Over 2.5", o25),
			("Under 3.5", u35),
			("Under 4.5", u45),
		]
		best_pick, best_prob = max(options, key=lambda item: item[1])

		entries.append(
			{
				"sort_date": sort_date,
				"date_label": str(fixture["fecha"]),
				"league_name": service.league_name,
				"home_team": str(fixture["local"]),
				"away_team": str(fixture["visitante"]),
				"kickoff": str(fixture["hora"]),
				"over_1_5": o15,
				"over_2_5": o25,
				"under_3_5": u35,
				"under_4_5": u45,
				"best_pick": best_pick,
				"best_prob": best_prob,
			}
		)

	entries = sorted(entries, key=lambda item: (item["sort_date"], -float(item["best_prob"]), item["kickoff"]))

	buffer = BytesIO()
	doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=0.35*inch, leftMargin=0.35*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
	styles = getSampleStyleSheet()
	story = []

	title_style = ParagraphStyle("TitleTotals", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=20, textColor=colors.HexColor("#0f3d28"), alignment=1, spaceAfter=6)
	subtitle_style = ParagraphStyle("SubTotals", parent=styles["Normal"], fontName="Helvetica", fontSize=10, textColor=colors.HexColor("#1a7d5c"), alignment=1, spaceAfter=14)
	header_style = ParagraphStyle("HeaderTotals", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10, textColor=colors.white, spaceAfter=6, spaceBefore=8)

	window_label = f"{window_start.strftime('%d/%m/%Y')} al {window_end.strftime('%d/%m/%Y')}"
	story.append(Paragraph("REPORTE TOTALES: O1.5, O2.5, U3.5, U4.5", title_style))
	story.append(Paragraph(f"Ventana: {window_label}", subtitle_style))

	grouped: dict[str, list[dict[str, object]]] = {}
	for item in entries:
		grouped.setdefault(str(item["date_label"]), []).append(item)

	for date_label in sorted(grouped.keys()):
		head = Table([[Paragraph(f"FECHA: {date_label}", header_style)]], colWidths=[7.7*inch])
		head.setStyle(TableStyle([
			("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1a7d5c")),
			("LEFTPADDING", (0, 0), (-1, -1), 10),
			("TOPPADDING", (0, 0), (-1, -1), 8),
			("BOTTOMPADDING", (0, 0), (-1, -1), 8),
		]))
		story.append(head)
		story.append(Spacer(1, 0.08*inch))

		rows = [["#", "Encuentro", "Liga", "O1.5", "O2.5", "U3.5", "U4.5", "Mejor pick"]]
		for idx, item in enumerate(grouped[date_label], 1):
			rows.append([
				str(idx),
				f"{item['home_team']} vs {item['away_team']} ({item['kickoff']})",
				str(item["league_name"]),
				f"{float(item['over_1_5']):.2f}%".replace(".", ","),
				f"{float(item['over_2_5']):.2f}%".replace(".", ","),
				f"{float(item['under_3_5']):.2f}%".replace(".", ","),
				f"{float(item['under_4_5']):.2f}%".replace(".", ","),
				f"{item['best_pick']} ({float(item['best_prob']):.2f}%)".replace(".", ","),
			])

		table = Table(rows, colWidths=[0.35*inch, 2.15*inch, 1.05*inch, 0.58*inch, 0.58*inch, 0.58*inch, 0.58*inch, 1.73*inch])
		table.setStyle(TableStyle([
			("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f5f0")),
			("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f3d28")),
			("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
			("FONTSIZE", (0, 0), (-1, -1), 7.5),
			("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#a7d7c4")),
			("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
			("LEFTPADDING", (0, 0), (-1, -1), 4),
			("RIGHTPADDING", (0, 0), (-1, -1), 4),
		]))
		story.append(table)
		story.append(Spacer(1, 0.15*inch))

	doc.build(story)
	buffer.seek(0)
	response = HttpResponse(buffer.getvalue(), content_type="application/pdf")
	response["Content-Disposition"] = f'attachment; filename="reporte_totales_{window_start.strftime("%d_%m_%Y")}.pdf"'
	return response
