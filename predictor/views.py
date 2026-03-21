from datetime import datetime, timedelta
from io import BytesIO
import re
from collections import defaultdict
from urllib.parse import urlencode

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
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

VALID_LEAGUES = tuple(LEAGUE_SERVICE_FACTORIES.keys())

LEAGUE_NAMES = {
	"spain": "LaLiga",
	"bundesliga": "Bundesliga",
	"premier": "Premier League",
	"seriea": "Serie A",
	"ligue1": "Ligue 1",
	"primeiraliga": "Primeira Liga",
	"proleague": "Pro League",
	"eredivisie": "Eredivisie",
}

PDF_MIN_PROBABILITY = 80.0


def _meets_pdf_threshold(probability_pct: float) -> bool:
	return float(probability_pct) >= PDF_MIN_PROBABILITY


def _team_logo(service, team_name: str) -> str:
	team = str(team_name or "").strip()
	if not team:
		return ""
	logo_getter = getattr(service, "_team_logo_url", None)
	if callable(logo_getter):
		return str(logo_getter(team) or "")
	return ""


def _get_service(liga: str):
	return LEAGUE_SERVICE_FACTORIES.get(liga, get_prediction_service_spain)()


def _build_sportsbook_rows(service, fixtures: list[dict[str, object]], limit: int | None = 8) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	selected_fixtures = fixtures if limit is None else fixtures[:limit]
	for fixture in selected_fixtures:
		try:
			match_prediction = service.predict_match(str(fixture.get("match_key", "")))
			totals_market = match_prediction.get("markets", {}).get("totales", {})
			over_1_5 = float(totals_market.get("over_1_5", 0.0))
			under_3_5 = float(totals_market.get("under_3_5", 0.0))
			under_4_5 = float(totals_market.get("under_4_5", 0.0))
			rows.append(
				{
					"local": fixture.get("local", ""),
					"visitante": fixture.get("visitante", ""),
					"hora": fixture.get("hora", ""),
					"local_logo": fixture.get("local_logo", ""),
					"visitante_logo": fixture.get("visitante_logo", ""),
					"odd_over_1_5": _fair_odds(over_1_5),
					"odd_under_3_5": _fair_odds(under_3_5),
					"odd_under_4_5": _fair_odds(under_4_5),
				}
			)
		except Exception:
			continue
	return rows


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


def _safe_float(value: object, default: float = 0.0) -> float:
	try:
		return float(value)
	except (TypeError, ValueError):
		return default


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


def _prediction_url(liga: str, match_key: object, home_date: str | None = None, predict: bool = True) -> str:
	params = {
		"liga": str(liga),
		"match_key": str(match_key or ""),
	}
	if home_date:
		params["home_date"] = str(home_date)
	if predict:
		params["predict"] = "1"
	return f"/pronostico/?{urlencode(params)}"


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


def _build_full_prediction_rows(prediction: dict[str, object] | None) -> list[dict[str, object]]:
	if not isinstance(prediction, dict):
		return []

	probabilities = prediction.get("probabilities", {}) if isinstance(prediction.get("probabilities", {}), dict) else {}
	markets = prediction.get("markets", {}) if isinstance(prediction.get("markets", {}), dict) else {}

	def add_row(container: list[dict[str, object]], market: str, pick: str, value: object) -> None:
		prob = _safe_float(value)
		container.append(
			{
				"market": market,
				"pick": pick,
				"probability": round(prob, 2),
				"fair_odd": _fair_odds(prob),
			}
		)

	rows: list[dict[str, object]] = []
	add_row(rows, "1X2", "Local", probabilities.get("local", 0.0))
	add_row(rows, "1X2", "Empate", probabilities.get("empate", 0.0))
	add_row(rows, "1X2", "Visitante", probabilities.get("visitante", 0.0))

	dc = markets.get("doble_oportunidad", {}) if isinstance(markets.get("doble_oportunidad", {}), dict) else {}
	add_row(rows, "Doble oportunidad", "1X", dc.get("uno_x", 0.0))
	add_row(rows, "Doble oportunidad", "12", dc.get("uno_dos", 0.0))
	add_row(rows, "Doble oportunidad", "X2", dc.get("x_dos", 0.0))

	btts = markets.get("btts", {}) if isinstance(markets.get("btts", {}), dict) else {}
	add_row(rows, "Ambos marcan", "Si", btts.get("si", 0.0))
	add_row(rows, "Ambos marcan", "No", btts.get("no", 0.0))

	totals = markets.get("totales", {}) if isinstance(markets.get("totales", {}), dict) else {}
	add_row(rows, "Totales", "Over 1.5", totals.get("over_1_5", 0.0))
	add_row(rows, "Totales", "Over 2.5", totals.get("over_2_5", 0.0))
	add_row(rows, "Totales", "Under 3.5", totals.get("under_3_5", 0.0))
	add_row(rows, "Totales", "Under 4.5", totals.get("under_4_5", 0.0))

	corners = markets.get("corners_8_5", {}) if isinstance(markets.get("corners_8_5", {}), dict) else {}
	add_row(rows, "Corners 8.5", "Over 8.5", corners.get("over", 0.0))
	add_row(rows, "Corners 8.5", "Under 8.5", corners.get("under", 0.0))

	cards = markets.get("tarjetas", {}) if isinstance(markets.get("tarjetas", {}), dict) else {}
	yellow = cards.get("amarillas", {}) if isinstance(cards.get("amarillas", {}), dict) else {}
	add_row(rows, "Tarjetas amarillas", "Over 3.5", yellow.get("over_3_5", 0.0))
	add_row(rows, "Tarjetas amarillas", "Under 3.5", yellow.get("under_3_5", 0.0))

	red = cards.get("rojas", {}) if isinstance(cards.get("rojas", {}), dict) else {}
	add_row(rows, "Tarjetas rojas", "Over 0.5", red.get("over_0_5", 0.0))
	add_row(rows, "Tarjetas rojas", "Under 0.5", red.get("under_0_5", 0.0))

	total_cards = cards.get("totales", {}) if isinstance(cards.get("totales", {}), dict) else {}
	add_row(rows, "Tarjetas totales", "Over 3.5", total_cards.get("over_3_5", 0.0))
	add_row(rows, "Tarjetas totales", "Under 3.5", total_cards.get("under_3_5", 0.0))
	add_row(rows, "Tarjetas totales", "Over 4.5", total_cards.get("over_4_5", 0.0))
	add_row(rows, "Tarjetas totales", "Under 4.5", total_cards.get("under_4_5", 0.0))

	return sorted(rows, key=lambda row: (float(row.get("probability", 0.0)), str(row.get("market", ""))), reverse=True)


def _build_league_stats(service) -> dict[str, object]:
	df = service.historical_df
	total_matches = int(len(df))
	if total_matches <= 0:
		return {
			"league_name": service.league_name,
			"league_logo_url": service.league_logo_url,
			"total_matches": 0,
			"avg_goals": 0.0,
			"avg_home_goals": 0.0,
			"avg_away_goals": 0.0,
			"home_win_pct": 0.0,
			"draw_pct": 0.0,
			"away_win_pct": 0.0,
			"btts_pct": 0.0,
			"over15_pct": 0.0,
			"over25_pct": 0.0,
			"under35_pct": 0.0,
			"under45_pct": 0.0,
			"avg_corners": 0.0,
			"avg_cards": 0.0,
			"red_match_pct": 0.0,
			"market_tone": "Sin datos",
			"recommendations": ["No hay suficientes partidos historicos para clasificar tendencias."],
			"top_table": [],
			"table_rows": [],
			"attack_leaders": [],
			"defense_leaders": [],
		}

	home_win_pct = round(float((df["FTR"] == "H").mean() * 100), 2)
	draw_pct = round(float((df["FTR"] == "D").mean() * 100), 2)
	away_win_pct = round(float((df["FTR"] == "A").mean() * 100), 2)
	avg_goals = round(float((df["FTHG"] + df["FTAG"]).mean()), 2)
	avg_home_goals = round(float(df["FTHG"].mean()), 2)
	avg_away_goals = round(float(df["FTAG"].mean()), 2)
	btts_pct = round(float(((df["FTHG"] > 0) & (df["FTAG"] > 0)).mean() * 100), 2)
	over15_pct = round(float(((df["FTHG"] + df["FTAG"]) > 1).mean() * 100), 2)
	over25_pct = round(float(((df["FTHG"] + df["FTAG"]) > 2).mean() * 100), 2)
	under35_pct = round(float(((df["FTHG"] + df["FTAG"]) < 4).mean() * 100), 2)
	under45_pct = round(float(((df["FTHG"] + df["FTAG"]) < 5).mean() * 100), 2)
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
	form_by_team: dict[str, list[str]] = defaultdict(list)
	for _, row in df.sort_values(["Date", "Time"]).iterrows():
		home_team = str(row.get("HomeTeam", "")).strip()
		away_team = str(row.get("AwayTeam", "")).strip()
		result = str(row.get("FTR", "")).strip().upper()
		home_form = "D"
		away_form = "D"
		if result == "H":
			home_form = "W"
			away_form = "L"
		elif result == "A":
			home_form = "L"
			away_form = "W"
		if home_team:
			form_by_team[home_team].append(home_form)
		if away_team:
			form_by_team[away_team].append(away_form)
	top_table = [
		{
			"position": team.get("position", 0),
			"team": team.get("team", ""),
			"logo": _team_logo(service, str(team.get("team", ""))),
			"form": form_by_team.get(str(team.get("team", "")), [])[-5:],
			"points": team.get("points", 0),
			"gd": team.get("gd", 0),
		}
		for team in top_table_raw
	]
	table_rows = [
		{
			"position": team.get("position", 0),
			"team": team.get("team", ""),
			"logo": _team_logo(service, str(team.get("team", ""))),
			"form": form_by_team.get(str(team.get("team", "")), [])[-5:],
			"played": team.get("played", 0),
			"points": team.get("points", 0),
			"ppp": round((float(team.get("points", 0)) / float(team.get("played", 1))) if float(team.get("played", 0)) > 0 else 0.0, 2),
			"gf": team.get("gf", 0),
			"ga": team.get("ga", 0),
			"gd": team.get("gd", 0),
		}
		for team in sorted(
			service.standings_snapshot.values(),
			key=lambda row: (row.get("position", 999), -row.get("points", 0)),
		)
	]
	attack_leaders = [
		{
			"team": team.get("team", ""),
			"logo": _team_logo(service, str(team.get("team", ""))),
			"value": team.get("gf", 0),
		}
		for team in sorted(service.standings_snapshot.values(), key=lambda row: row.get("gf", 0), reverse=True)[:5]
	]
	defense_leaders = [
		{
			"team": team.get("team", ""),
			"logo": _team_logo(service, str(team.get("team", ""))),
			"value": team.get("ga", 0),
		}
		for team in sorted(service.standings_snapshot.values(), key=lambda row: row.get("ga", 999))[:5]
	]

	return {
		"league_name": service.league_name,
		"league_logo_url": service.league_logo_url,
		"total_matches": total_matches,
		"avg_goals": avg_goals,
		"avg_home_goals": avg_home_goals,
		"avg_away_goals": avg_away_goals,
		"home_win_pct": home_win_pct,
		"draw_pct": draw_pct,
		"away_win_pct": away_win_pct,
		"btts_pct": btts_pct,
		"over15_pct": over15_pct,
		"over25_pct": over25_pct,
		"under35_pct": under35_pct,
		"under45_pct": under45_pct,
		"avg_corners": avg_corners,
		"avg_cards": avg_cards,
		"red_match_pct": red_match_pct,
		"market_tone": market_tone,
		"recommendations": recommendations,
		"top_table": top_table,
		"table_rows": table_rows,
		"attack_leaders": attack_leaders,
		"defense_leaders": defense_leaders,
	}


def _build_team_detail(service, team_name: str | None) -> dict[str, object] | None:
	team = str(team_name or "").strip()
	if not team:
		return None

	df = service.historical_df.copy()
	team_matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()
	if team_matches.empty:
		return None

	stats = {
		"matches": 0,
		"wins": 0,
		"draws": 0,
		"losses": 0,
		"points": 0,
		"gf": 0,
		"ga": 0,
		"shots": 0.0,
		"shots_on_target": 0.0,
		"corners": 0.0,
		"yellow_cards": 0.0,
		"red_cards": 0.0,
		"clean_sheets": 0,
		"failed_to_score": 0,
		"btts": 0,
		"over15": 0,
		"over25": 0,
		"under35": 0,
		"under45": 0,
	}
	home = {"matches": 0, "points": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0}
	away = {"matches": 0, "points": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0}
	recent_matches: list[dict[str, object]] = []

	for _, row in team_matches.sort_values(["Date", "Time"]).iterrows():
		is_home = row["HomeTeam"] == team
		goals_for = int(row["FTHG"] if is_home else row["FTAG"])
		goals_against = int(row["FTAG"] if is_home else row["FTHG"])
		shots = float(row["HS"] if is_home else row["AS"])
		shots_on_target = float(row["HST"] if is_home else row["AST"])
		corners = float(row["HC"] if is_home else row["AC"])
		yellow_cards = float(row["HY"] if is_home else row["AY"])
		red_cards = float(row["HR"] if is_home else row["AR"])
		if goals_for > goals_against:
			result = "W"
			points = 3
		elif goals_for == goals_against:
			result = "D"
			points = 1
		else:
			result = "L"
			points = 0

		stats["matches"] += 1
		stats["wins"] += 1 if result == "W" else 0
		stats["draws"] += 1 if result == "D" else 0
		stats["losses"] += 1 if result == "L" else 0
		stats["points"] += points
		stats["gf"] += goals_for
		stats["ga"] += goals_against
		stats["shots"] += shots
		stats["shots_on_target"] += shots_on_target
		stats["corners"] += corners
		stats["yellow_cards"] += yellow_cards
		stats["red_cards"] += red_cards
		stats["clean_sheets"] += 1 if goals_against == 0 else 0
		stats["failed_to_score"] += 1 if goals_for == 0 else 0
		stats["btts"] += 1 if goals_for > 0 and goals_against > 0 else 0
		total_goals = goals_for + goals_against
		stats["over15"] += 1 if total_goals > 1 else 0
		stats["over25"] += 1 if total_goals > 2 else 0
		stats["under35"] += 1 if total_goals < 4 else 0
		stats["under45"] += 1 if total_goals < 5 else 0

		split = home if is_home else away
		split["matches"] += 1
		split["points"] += points
		split["wins"] += 1 if result == "W" else 0
		split["draws"] += 1 if result == "D" else 0
		split["losses"] += 1 if result == "L" else 0
		split["gf"] += goals_for
		split["ga"] += goals_against

		opponent = str(row["AwayTeam"] if is_home else row["HomeTeam"])
		recent_matches.append(
			{
				"date": row["Date"].strftime("%d/%m/%Y") if hasattr(row["Date"], "strftime") else "",
				"venue": "Local" if is_home else "Visitante",
				"opponent": opponent,
				"opponent_logo": _team_logo(service, opponent),
				"score": f"{goals_for}-{goals_against}",
				"result": result,
			}
		)

	matches = max(int(stats["matches"]), 1)
	standing = service.standings_snapshot.get(team, {})
	return {
		"team": team,
		"team_logo": _team_logo(service, team),
		"position": int(standing.get("position", 0)),
		"points": int(stats["points"]),
		"ppp": round(float(stats["points"]) / matches, 2),
		"matches": int(stats["matches"]),
		"wins": int(stats["wins"]),
		"draws": int(stats["draws"]),
		"losses": int(stats["losses"]),
		"gf": int(stats["gf"]),
		"ga": int(stats["ga"]),
		"gd": int(stats["gf"] - stats["ga"]),
		"avg_goals_for": round(float(stats["gf"]) / matches, 2),
		"avg_goals_against": round(float(stats["ga"]) / matches, 2),
		"avg_shots": round(float(stats["shots"]) / matches, 2),
		"avg_shots_on_target": round(float(stats["shots_on_target"]) / matches, 2),
		"avg_corners": round(float(stats["corners"]) / matches, 2),
		"avg_yellow_cards": round(float(stats["yellow_cards"]) / matches, 2),
		"avg_red_cards": round(float(stats["red_cards"]) / matches, 2),
		"clean_sheet_pct": round((float(stats["clean_sheets"]) / matches) * 100, 2),
		"failed_to_score_pct": round((float(stats["failed_to_score"]) / matches) * 100, 2),
		"btts_pct": round((float(stats["btts"]) / matches) * 100, 2),
		"over15_pct": round((float(stats["over15"]) / matches) * 100, 2),
		"over25_pct": round((float(stats["over25"]) / matches) * 100, 2),
		"under35_pct": round((float(stats["under35"]) / matches) * 100, 2),
		"under45_pct": round((float(stats["under45"]) / matches) * 100, 2),
		"home": {
			"matches": int(home["matches"]),
			"points": int(home["points"]),
			"ppp": round((float(home["points"]) / float(home["matches"])) if float(home["matches"]) > 0 else 0.0, 2),
			"wins": int(home["wins"]),
			"draws": int(home["draws"]),
			"losses": int(home["losses"]),
			"gf": int(home["gf"]),
			"ga": int(home["ga"]),
		},
		"away": {
			"matches": int(away["matches"]),
			"points": int(away["points"]),
			"ppp": round((float(away["points"]) / float(away["matches"])) if float(away["matches"]) > 0 else 0.0, 2),
			"wins": int(away["wins"]),
			"draws": int(away["draws"]),
			"losses": int(away["losses"]),
			"gf": int(away["gf"]),
			"ga": int(away["ga"]),
		},
		"recent_matches": recent_matches[-8:][::-1],
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
				"dashboard_url": f"/pronostico/?liga={liga_key}&match_key={fixture.get('match_key', '')}&home_date={selected_date.strftime('%Y-%m-%d')}",
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
				"home_team_logo": str(fixture.get("local_logo", "")) or _team_logo(service, str(fixture.get("local", ""))),
				"away_team_logo": str(fixture.get("visitante_logo", "")) or _team_logo(service, str(fixture.get("visitante", ""))),
				"market": str(recommended.get("market", "Sin datos")),
				"pick": str(recommended.get("pick", "Sin recomendacion")),
				"probability": float(recommended.get("probability", 0.0)),
				"confidence": str(recommended.get("confidence", "Baja")),
				"multiple_confidence": str(multiple.get("confidence", "Baja")),
				"multiple_combined_probability": float(multiple.get("prob_combinada", 0.0)),
				"dashboard_url": _prediction_url(
					liga=liga_key,
					match_key=fixture.get("match_key", ""),
					home_date=selected_date.strftime("%Y-%m-%d"),
					predict=True,
				),
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
		"selected_date_value": selected_date.strftime("%Y-%m-%d"),
		"date_options": date_options,
		"league_cards": league_cards,
		"entries": entries,
	}


def _get_pdf_target_fixtures(request) -> tuple[list[tuple[str, object, dict[str, object], object]], object | None]:
	today = datetime.now().date()
	selected_date = None
	selected_date_raw = str(request.GET.get("date", "")).strip()
	if selected_date_raw:
		try:
			selected_date = datetime.strptime(selected_date_raw, "%Y-%m-%d").date()
		except ValueError:
			selected_date = None

	eligible_fixtures: list[tuple[str, object, dict[str, object], object]] = []
	for liga, factory in LEAGUE_SERVICE_FACTORIES.items():
		service = factory()
		for fixture in service.get_pending_fixtures():
			sort_date = _parse_fixture_date(str(fixture.get("fecha", "")))
			if sort_date is None or sort_date < today:
				continue
			eligible_fixtures.append((liga, service, fixture, sort_date))

	if not eligible_fixtures:
		return [], None

	available_dates = sorted({row[3] for row in eligible_fixtures})
	if selected_date is None or selected_date not in available_dates:
		selected_date = available_dates[0]

	target_fixtures = [row for row in eligible_fixtures if row[3] == selected_date]
	return target_fixtures, selected_date


def _build_home_rankings() -> dict[str, list[dict[str, object]]]:
	rank_1x2: list[dict[str, object]] = []
	rank_over15: list[dict[str, object]] = []
	rank_under45: list[dict[str, object]] = []
	rank_avg_goals: list[dict[str, object]] = []
	rank_over25: list[dict[str, object]] = []
	rank_under35: list[dict[str, object]] = []
	rank_btts: list[dict[str, object]] = []
	rank_corners: list[dict[str, object]] = []
	rank_total_cards: list[dict[str, object]] = []

	for league_key, factory in LEAGUE_SERVICE_FACTORIES.items():
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
		avg_corners = float((df["HC"] + df["AC"]).mean())
		avg_total_cards = float((df["HY"] + df["AY"] + df["HR"] + df["AR"]).mean())

		base = {"league": service.league_name, "key": league_key, "league_logo_url": service.league_logo_url}
		rank_1x2.append({**base, "value": round(one_x_two_top, 2)})
		rank_over15.append({**base, "value": round(over15, 2)})
		rank_over25.append({**base, "value": round(over25, 2)})
		rank_under35.append({**base, "value": round(under35, 2)})
		rank_under45.append({**base, "value": round(under45, 2)})
		rank_btts.append({**base, "value": round(btts, 2)})
		rank_avg_goals.append({**base, "value": round(avg_goals, 2)})
		rank_corners.append({**base, "value": round(avg_corners, 2)})
		rank_total_cards.append({**base, "value": round(avg_total_cards, 2)})

	return {
		"top_1x2": sorted(rank_1x2, key=lambda item: float(item["value"]), reverse=True),
		"top_over15": sorted(rank_over15, key=lambda item: float(item["value"]), reverse=True),
		"top_under45": sorted(rank_under45, key=lambda item: float(item["value"]), reverse=True),
		"top_avg_goals": sorted(rank_avg_goals, key=lambda item: float(item["value"]), reverse=True),
		"top_over25": sorted(rank_over25, key=lambda item: float(item["value"]), reverse=True),
		"top_under35": sorted(rank_under35, key=lambda item: float(item["value"]), reverse=True),
		"top_btts": sorted(rank_btts, key=lambda item: float(item["value"]), reverse=True),
		"top_corners": sorted(rank_corners, key=lambda item: float(item["value"]), reverse=True),
		"top_total_cards": sorted(rank_total_cards, key=lambda item: float(item["value"]), reverse=True),
	}


def dashboard(request):
	liga = request.GET.get("liga") or request.POST.get("liga", "spain")
	if liga not in VALID_LEAGUES:
		liga = "spain"

	refresh_status = ""
	refresh_enabled = request.GET.get("refresh") == "1"
	if request.method == "GET" and refresh_enabled:
		try:
			updated_rows = refresh_fixture_links(liga)
			refresh_status = f"Enlaces actualizados y guardados en base de datos: {updated_rows} encuentros."
		except Exception as exc:
			refresh_status = f"No se pudo refrescar enlaces en este acceso: {exc}"
	elif request.method == "GET":
		refresh_status = "Modo rapido: sin refresco automatico. Usa ?refresh=1 para actualizar enlaces."

	service = _get_service(liga)
	league_stats = _build_league_stats(service)
	selected_team = request.GET.get("team", "")
	team_stats = _build_team_detail(service, selected_team)
	multileague_home = _build_multileague_home(request.GET.get("home_date"))
	home_rankings = _build_home_rankings()
	fixtures = service.get_pending_fixtures()
	selected_match_key = request.POST.get("match_key") or request.GET.get("match_key", "")
	sportsbook_rows = _build_sportsbook_rows(service, fixtures, limit=8)

	prediction = None
	error_message = ""
	if fixtures and selected_match_key:
		try:
			prediction = service.predict_match(selected_match_key)
		except ValueError as exc:
			error_message = str(exc)
	elif not fixtures:
		error_message = "No hay encuentros pendientes en el calendario actual."

	selected_fixture = next((fixture for fixture in fixtures if str(fixture.get("match_key", "")) == str(selected_match_key)), None)
	full_prediction_rows = _build_full_prediction_rows(prediction)

	context = {
		"liga": liga,
		"league_name": service.league_name,
		"league_logo_url": service.league_logo_url,
		"datasets": service.dataset_labels,
		"fixtures": fixtures,
		"prediction": prediction,
		"selected_fixture": selected_fixture,
		"full_prediction_rows": full_prediction_rows,
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
		"team_stats": team_stats,
		"selected_team": selected_team,
		"sportsbook_rows": sportsbook_rows,
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


def league_dashboard(request, liga: str):
	if liga not in VALID_LEAGUES:
		liga = "spain"

	refresh_status = ""
	refresh_enabled = request.GET.get("refresh") == "1"
	if request.method == "GET" and refresh_enabled:
		try:
			updated_rows = refresh_fixture_links(liga)
			refresh_status = f"Enlaces actualizados: {updated_rows} encuentros."
		except Exception as exc:
			refresh_status = f"No se pudo refrescar la liga: {exc}"
	elif request.method == "GET":
		refresh_status = "Modo rapido: sin refresco automatico. Usa ?refresh=1 para actualizar enlaces."

	service = _get_service(liga)
	league_stats = _build_league_stats(service)
	selected_team = request.GET.get("team", "")
	team_stats = _build_team_detail(service, selected_team)
	fixtures = service.get_pending_fixtures()
	selected_match_key = request.POST.get("match_key") or request.GET.get("match_key", "")
	predict_enabled = bool(selected_match_key) or request.GET.get("predict") == "1"
	sportsbook_rows = _build_sportsbook_rows(service, fixtures, limit=8) if predict_enabled else []

	prediction = None
	error_message = ""
	if fixtures and selected_match_key and predict_enabled:
		try:
			prediction = service.predict_match(selected_match_key)
		except ValueError as exc:
			error_message = str(exc)
	elif not fixtures:
		error_message = "No hay encuentros pendientes en el calendario actual."

	context = {
		"liga": liga,
		"league_name": service.league_name,
		"league_logo_url": service.league_logo_url,
		"league_stats": league_stats,
		"team_stats": team_stats,
		"selected_team": selected_team,
		"fixtures": fixtures,
		"selected_match_key": selected_match_key,
		"prediction": prediction,
		"sportsbook_rows": sportsbook_rows,
		"refresh_status": refresh_status,
		"error_message": error_message,
	}
	return render(request, "predictor/league_dashboard.html", context)


def match_prediction_page(request):
	liga = request.GET.get("liga") or request.POST.get("liga", "spain")
	if liga not in VALID_LEAGUES:
		liga = "spain"

	selected_match_key = request.POST.get("match_key") or request.GET.get("match_key", "")
	predict_enabled = request.GET.get("predict") == "1"
	fixtures: list[dict[str, object]] = []
	service = None
	league_name = LEAGUE_NAMES.get(liga, liga)
	league_logo_url = ""

	prediction = None
	error_message = ""
	if selected_match_key and not predict_enabled:
		error_message = "Modo rapido activo: abre el cálculo completo con el botón de pronóstico."

	if predict_enabled:
		service = _get_service(liga)
		league_name = service.league_name
		league_logo_url = service.league_logo_url
		fixtures = service.get_pending_fixtures()
		if fixtures and selected_match_key:
			try:
				prediction = service.predict_match(selected_match_key)
			except ValueError as exc:
				error_message = str(exc)
		elif not fixtures:
			error_message = "No hay encuentros pendientes en el calendario actual."
		else:
			error_message = "Selecciona un encuentro para ver el pronóstico completo."

	selected_fixture = next((fixture for fixture in fixtures if str(fixture.get("match_key", "")) == str(selected_match_key)), None)
	if selected_fixture is None and selected_match_key:
		parts = str(selected_match_key).split("|")
		if len(parts) >= 4:
			selected_fixture = {
				"fecha": parts[0],
				"hora": parts[1],
				"local": parts[2],
				"visitante": parts[3],
				"local_logo": "",
				"visitante_logo": "",
			}
	full_prediction_rows = _build_full_prediction_rows(prediction)

	context = {
		"liga": liga,
		"league_name": league_name,
		"league_logo_url": league_logo_url,
		"fixtures": fixtures,
		"selected_match_key": selected_match_key,
		"selected_fixture": selected_fixture,
		"prediction": prediction,
		"full_prediction_rows": full_prediction_rows,
		"error_message": error_message,
		"predict_enabled": predict_enabled,
		"home_date": str(request.GET.get("home_date", "")).strip(),
	}
	return render(request, "predictor/match_prediction.html", context)


def refresh_all_leagues(request):
	"""Actualiza CSVs históricos, CSV de encuentros y caché de enlaces para todas las ligas."""
	import subprocess
	import sys
	from pathlib import Path

	if request.method not in ("GET", "POST"):
		from django.http import HttpResponseNotAllowed
		return HttpResponseNotAllowed(["GET", "POST"])

	run_requested = request.method == "POST" or request.GET.get("run") == "1"
	if not run_requested:
		return render(
			request,
			"predictor/refresh_all.html",
			{"results": [], "ok": False, "ok_count": 0, "total": 0, "ran": False},
		)

	base_dir = Path(__file__).resolve().parent.parent
	results: list[dict[str, object]] = []

	# ── 1. CSVs históricos (football-data.co.uk) ────────────────────────
	csv_script = base_dir / "actualizar_csv.py"
	try:
		proc = subprocess.run(
			[sys.executable, str(csv_script)],
			cwd=str(base_dir),
			capture_output=True,
			text=True,
			timeout=180,
		)
		ok = proc.returncode == 0
		results.append({"step": "CSV histórico (todas las ligas)", "ok": ok, "detail": (proc.stdout + proc.stderr).strip()[-400:]})
	except Exception as exc:
		results.append({"step": "CSV histórico (todas las ligas)", "ok": False, "detail": str(exc)})

	# ── 2. Encuentros + caché de enlaces para cada liga ────────────────
	for liga_key in LEAGUE_SERVICE_FACTORIES:
		try:
			from .sync import refresh_fixture_links
			n = refresh_fixture_links(liga_key)
			results.append({"step": f"Encuentros & enlaces: {LEAGUE_NAMES.get(liga_key, liga_key)}", "ok": True, "detail": f"{n} encuentros actualizados"})
		except Exception as exc:
			results.append({"step": f"Encuentros & enlaces: {LEAGUE_NAMES.get(liga_key, liga_key)}", "ok": False, "detail": str(exc)[:300]})

	ok_count = sum(1 for r in results if r["ok"])
	total = len(results)
	payload = {"ok": ok_count == total, "results": results, "ok_count": ok_count, "total": total, "ran": True}

	accept_header = str(request.headers.get("Accept", ""))
	if request.method == "POST" and "application/json" in accept_header:
		return JsonResponse(payload)

	return render(request, "predictor/refresh_all.html", payload)


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
			"dashboard_url": _prediction_url(
				liga=liga,
				match_key=fixture["match_key"],
				home_date=sort_date.strftime("%Y-%m-%d"),
				predict=True,
			),
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
	window_fixtures, target_date = _get_pdf_target_fixtures(request)
	if not window_fixtures or target_date is None:
		return HttpResponse("No hay encuentros pendientes.", content_type="text/plain")

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
		if _meets_pdf_threshold(max(combined_probability, top_two_combined_probability, no_cards_corners_probability)):
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
	window_label = target_date.strftime('%d/%m/%Y')

	# Título principal
	story.append(Paragraph("MEJORES APUESTAS POR FECHA (>= 80%)", title_style))
	story.append(Paragraph(f"Fecha seleccionada: {window_label} | Filtro mínimo: {PDF_MIN_PROBABILITY:.0f}%", subtitle_style))
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
				col_defs = [row for row in col_defs if _meets_pdf_threshold(row[3])]
				if not col_defs:
					no_sel_data = [[Paragraph(f"Ninguna métrica alcanza el {PDF_MIN_PROBABILITY:.0f}% en este partido.", value_style)]]
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
					continue

				COL_W = (7.35 * inch) / max(len(col_defs), 1)
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

				metrics_table = Table([cells_row], colWidths=[COL_W] * len(col_defs))
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
				no_sel_data = [[Paragraph(f"Ninguna selección supera el {PDF_MIN_PROBABILITY:.0f}% de confianza en este partido.", value_style)]]
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
	response["Content-Disposition"] = f'attachment; filename="mejores_apuestas_{target_date.strftime("%d_%m_%Y")}.pdf"'
	return response


def best_bets_1x2_pdf(request):
	"""Generar PDF de mejores apuestas 1X2 por fecha."""
	entries: list[dict[str, object]] = []
	window_fixtures, target_date = _get_pdf_target_fixtures(request)
	if not window_fixtures or target_date is None:
		return HttpResponse("No hay encuentros pendientes.", content_type="text/plain")

	for _liga, service, fixture, sort_date in window_fixtures:
		try:
			prediction = service.predict_match(fixture["match_key"])
		except ValueError:
			continue

		pick_1x2 = prediction.get("categorized_bets", {}).get("resultado_1x2", {})
		prob = float(pick_1x2.get("prob", 0.0))
		if not _meets_pdf_threshold(prob):
			continue
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

	window_label = target_date.strftime('%d/%m/%Y')
	story.append(Paragraph("MEJORES APUESTAS 1X2 POR FECHA (>= 80%)", title_style))
	story.append(Paragraph(f"Fecha seleccionada: {window_label} | Filtro mínimo: {PDF_MIN_PROBABILITY:.0f}%", subtitle_style))

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
	response["Content-Disposition"] = f'attachment; filename="mejores_apuestas_1x2_{target_date.strftime("%d_%m_%Y")}.pdf"'
	return response


def best_bets_double_chance_pdf(request):
	"""Generar PDF de mejores apuestas Doble Oportunidad por fecha."""
	entries: list[dict[str, object]] = []
	window_fixtures, target_date = _get_pdf_target_fixtures(request)
	if not window_fixtures or target_date is None:
		return HttpResponse("No hay encuentros pendientes.", content_type="text/plain")

	for _liga, service, fixture, sort_date in window_fixtures:
		try:
			prediction = service.predict_match(fixture["match_key"])
		except ValueError:
			continue

		pick_dc = prediction.get("categorized_bets", {}).get("doble_oportunidad", {})
		prob = float(pick_dc.get("prob", 0.0))
		if not _meets_pdf_threshold(prob):
			continue
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

	window_label = target_date.strftime('%d/%m/%Y')
	story.append(Paragraph("MEJORES APUESTAS DOBLE OPORTUNIDAD (>= 80%)", title_style))
	story.append(Paragraph(f"Fecha seleccionada: {window_label} | Filtro mínimo: {PDF_MIN_PROBABILITY:.0f}%", subtitle_style))

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
	response["Content-Disposition"] = f'attachment; filename="mejores_apuestas_doble_oportunidad_{target_date.strftime("%d_%m_%Y")}.pdf"'
	return response


def best_bets_totals_pdf(request):
	"""Generar PDF de mercados Totales (Over 1.5/2.5 y Under 3.5/4.5) por fecha."""
	entries: list[dict[str, object]] = []
	window_fixtures, target_date = _get_pdf_target_fixtures(request)
	if not window_fixtures or target_date is None:
		return HttpResponse("No hay encuentros pendientes.", content_type="text/plain")

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
	market_header_style = ParagraphStyle("MarketHeaderTotals", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=9, textColor=colors.white, spaceAfter=4, spaceBefore=6)

	window_label = target_date.strftime('%d/%m/%Y')
	story.append(Paragraph("REPORTE TOTALES: O1.5, O2.5, U3.5, U4.5 (>= 80%)", title_style))
	story.append(Paragraph(f"Fecha seleccionada: {window_label} | Filtro mínimo: {PDF_MIN_PROBABILITY:.0f}%", subtitle_style))

	grouped: dict[str, list[dict[str, object]]] = {}
	for item in entries:
		grouped.setdefault(str(item["date_label"]), []).append(item)

	market_sections = [
		("over_1_5", "INFORME SEPARADO: OVER 1.5"),
		("over_2_5", "INFORME SEPARADO: OVER 2.5"),
		("under_3_5", "INFORME SEPARADO: UNDER 3.5"),
		("under_4_5", "INFORME SEPARADO: UNDER 4.5"),
	]

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

		date_items = list(grouped[date_label])
		for market_key, market_title in market_sections:
			market_head = Table([[Paragraph(market_title, market_header_style)]], colWidths=[7.7*inch])
			market_head.setStyle(TableStyle([
				("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#145433")),
				("LEFTPADDING", (0, 0), (-1, -1), 10),
				("TOPPADDING", (0, 0), (-1, -1), 6),
				("BOTTOMPADDING", (0, 0), (-1, -1), 6),
			]))
			story.append(market_head)
			story.append(Spacer(1, 0.05*inch))

			market_items = sorted(
				date_items,
				key=lambda row: (
					-float(row.get(market_key, 0.0)),
					str(row.get("kickoff", "")),
				),
			)
			market_items = [row for row in market_items if _meets_pdf_threshold(float(row.get(market_key, 0.0)))]
			if not market_items:
				empty_table = Table(
					[[f"Sin selecciones con probabilidad >= {PDF_MIN_PROBABILITY:.0f}% para este mercado."]],
					colWidths=[7.7*inch],
				)
				empty_table.setStyle(TableStyle([
					("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fbf9")),
					("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#4a5a52")),
					("FONTNAME", (0, 0), (-1, -1), "Helvetica-Oblique"),
					("FONTSIZE", (0, 0), (-1, -1), 8),
					("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#c5d8cf")),
					("LEFTPADDING", (0, 0), (-1, -1), 8),
					("TOPPADDING", (0, 0), (-1, -1), 6),
					("BOTTOMPADDING", (0, 0), (-1, -1), 6),
				]))
				story.append(empty_table)
				story.append(Spacer(1, 0.11*inch))
				continue

			rows = [["#", "Encuentro", "Liga", "Prob.", "Cuota justa"]]
			for idx, item in enumerate(market_items, 1):
				probability = float(item.get(market_key, 0.0))
				rows.append([
					str(idx),
					f"{item['home_team']} vs {item['away_team']} ({item['kickoff']})",
					str(item["league_name"]),
					f"{probability:.2f}%".replace(".", ","),
					f"{_fair_odds(probability):.2f}".replace(".", ","),
				])

			table = Table(rows, colWidths=[0.35*inch, 3.55*inch, 1.55*inch, 0.95*inch, 1.3*inch])
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
			story.append(Spacer(1, 0.11*inch))

		story.append(Spacer(1, 0.08*inch))

	doc.build(story)
	buffer.seek(0)
	response = HttpResponse(buffer.getvalue(), content_type="application/pdf")
	response["Content-Disposition"] = f'attachment; filename="reporte_totales_{target_date.strftime("%d_%m_%Y")}.pdf"'
	return response


def best_bets_summary_pdf(request):
	"""PDF consolidado: 1X2 | Doble Oportunidad | Goles | Tarjetas | Corners | Multiple (encuentros con 2+ mercados)."""
	entries: list[dict[str, object]] = []
	window_fixtures, target_date = _get_pdf_target_fixtures(request)
	if not window_fixtures or target_date is None:
		return HttpResponse("No hay encuentros pendientes.", content_type="text/plain")

	for _liga, service, fixture, sort_date in window_fixtures:
		try:
			prediction = service.predict_match(fixture["match_key"])
		except ValueError:
			continue

		markets = prediction.get("markets", {})

		# 1X2
		pick_1x2 = prediction.get("categorized_bets", {}).get("resultado_1x2", {})
		prob_1x2 = float(pick_1x2.get("prob", 0.0))

		# Doble oportunidad
		pick_dc = prediction.get("categorized_bets", {}).get("doble_oportunidad", {})
		prob_dc = float(pick_dc.get("prob", 0.0))

		# Goles (Over/Under de goles totales)
		totales = markets.get("totales", {}) if isinstance(markets.get("totales", {}), dict) else {}
		goles_candidates = [
			("Over 1.5", float(totales.get("over_1_5", 0.0))),
			("Over 2.5", float(totales.get("over_2_5", 0.0))),
			("Under 3.5", float(totales.get("under_3_5", 0.0))),
			("Under 4.5", float(totales.get("under_4_5", 0.0))),
		]
		goles_pick, goles_prob = max(goles_candidates, key=lambda x: x[1])

		# Tarjetas
		cards = markets.get("tarjetas", {}) if isinstance(markets.get("tarjetas", {}), dict) else {}
		total_cards = cards.get("totales", {}) if isinstance(cards.get("totales", {}), dict) else {}
		tarjetas_candidates = [
			("Total Over 3.5", float(total_cards.get("over_3_5", 0.0))),
			("Total Under 3.5", float(total_cards.get("under_3_5", 0.0))),
			("Total Over 4.5", float(total_cards.get("over_4_5", 0.0))),
			("Total Under 4.5", float(total_cards.get("under_4_5", 0.0))),
		]
		tarjetas_pick, tarjetas_prob = max(tarjetas_candidates, key=lambda x: x[1])

		# Corners
		corners_mkt = markets.get("corners_8_5", {}) if isinstance(markets.get("corners_8_5", {}), dict) else {}
		corners_candidates = [
			("Over 8.5", float(corners_mkt.get("over", 0.0))),
			("Under 8.5", float(corners_mkt.get("under", 0.0))),
		]
		corners_pick, corners_prob = max(corners_candidates, key=lambda x: x[1])

		has_any = any([
			_meets_pdf_threshold(prob_1x2),
			_meets_pdf_threshold(prob_dc),
			_meets_pdf_threshold(goles_prob),
			_meets_pdf_threshold(tarjetas_prob),
			_meets_pdf_threshold(corners_prob),
		])
		if not has_any:
			continue

		entries.append({
			"sort_date": sort_date,
			"date_label": str(fixture["fecha"]),
			"league_name": service.league_name,
			"kickoff": str(fixture["hora"]),
			"match": f"{fixture['local']} vs {fixture['visitante']}",
			"one_x_two_pick": str(pick_1x2.get("pick", "N/D")),
			"one_x_two_prob": prob_1x2,
			"dc_pick": str(pick_dc.get("pick", "N/D")),
			"dc_prob": prob_dc,
			"goles_pick": goles_pick,
			"goles_prob": goles_prob,
			"tarjetas_pick": tarjetas_pick,
			"tarjetas_prob": tarjetas_prob,
			"corners_pick": corners_pick,
			"corners_prob": corners_prob,
		})

	if not entries:
		return HttpResponse(
			f"No hay selecciones con probabilidad mayor o igual a {PDF_MIN_PROBABILITY:.0f}% para la fecha elegida.",
			content_type="text/plain",
		)

	entries = sorted(
		entries,
		key=lambda item: (item["sort_date"], item["kickoff"]),
	)

	# Conteos para portada
	window_label = target_date.strftime("%d/%m/%Y")
	total_matches = len(entries)
	one_x_two_count = sum(1 for r in entries if _meets_pdf_threshold(float(r["one_x_two_prob"])))
	dc_count = sum(1 for r in entries if _meets_pdf_threshold(float(r["dc_prob"])))
	goles_count = sum(1 for r in entries if _meets_pdf_threshold(float(r["goles_prob"])))
	tarjetas_count = sum(1 for r in entries if _meets_pdf_threshold(float(r["tarjetas_prob"])))
	corners_count = sum(1 for r in entries if _meets_pdf_threshold(float(r["corners_prob"])))

	# Múltiple: encuentros con 2+ mercados que pasan el umbral
	def _collect_valid_picks(row: dict) -> list[tuple[str, str, float]]:
		picks = []
		if _meets_pdf_threshold(float(row["one_x_two_prob"])):
			picks.append(("1X2", str(row["one_x_two_pick"]), float(row["one_x_two_prob"])))
		if _meets_pdf_threshold(float(row["dc_prob"])):
			picks.append(("Doble Oport.", str(row["dc_pick"]), float(row["dc_prob"])))
		if _meets_pdf_threshold(float(row["goles_prob"])):
			picks.append(("Goles", str(row["goles_pick"]), float(row["goles_prob"])))
		if _meets_pdf_threshold(float(row["tarjetas_prob"])):
			picks.append(("Tarjetas", str(row["tarjetas_pick"]), float(row["tarjetas_prob"])))
		if _meets_pdf_threshold(float(row["corners_prob"])):
			picks.append(("Corners", str(row["corners_pick"]), float(row["corners_prob"])))
		return picks

	multiple_entries = [(row, _collect_valid_picks(row)) for row in entries if len(_collect_valid_picks(row)) >= 2]
	multiple_count = len(multiple_entries)

	# PDF setup
	buffer = BytesIO()
	doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=0.35 * inch, leftMargin=0.35 * inch, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
	styles = getSampleStyleSheet()
	story = []

	cover_title = ParagraphStyle("CoverTitle", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=24, textColor=colors.HexColor("#0f3d28"), alignment=1, spaceAfter=10)
	cover_subtitle = ParagraphStyle("CoverSubtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=11, textColor=colors.HexColor("#1a7d5c"), alignment=1, spaceAfter=20)
	section_title = ParagraphStyle("SectionTitle", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=15, textColor=colors.HexColor("#145433"), spaceAfter=8)
	header_style = ParagraphStyle("HeaderSummary", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10, textColor=colors.white, spaceAfter=6, spaceBefore=8)
	body_text = ParagraphStyle("BodyTextSummary", parent=styles["Normal"], fontName="Helvetica", fontSize=9, textColor=colors.HexColor("#21352b"), leading=12)
	leg_style = ParagraphStyle("LegStyle", parent=styles["Normal"], fontName="Helvetica", fontSize=7, leading=10, textColor=colors.HexColor("#1a4a35"))
	match_title_st = ParagraphStyle("MatchTitleSt", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, textColor=colors.HexColor("#0f3d28"), leading=11)

	# ── Portada ──────────────────────────────────────────────────────────────
	story.append(Spacer(1, 1.2 * inch))
	story.append(Paragraph("RESUMEN GENERAL DE REPORTES", cover_title))
	story.append(Paragraph("1X2  ·  Doble Oportunidad  ·  Goles  ·  Tarjetas  ·  Corners  ·  Múltiple", cover_subtitle))

	cover_data = [
		["Fecha de reporte", window_label],
		["Filtro mínimo", f"{PDF_MIN_PROBABILITY:.0f}%"],
		["Encuentros con al menos 1 mercado válido", str(total_matches)],
		["Picks 1X2 válidos", str(one_x_two_count)],
		["Picks Doble Oportunidad válidos", str(dc_count)],
		["Picks Goles válidos", str(goles_count)],
		["Picks Tarjetas válidos", str(tarjetas_count)],
		["Picks Corners válidos", str(corners_count)],
		["Encuentros con Apuesta Múltiple (2+ mercados)", str(multiple_count)],
	]
	cover_table = Table(cover_data, colWidths=[3.8 * inch, 3.7 * inch])
	cover_table.setStyle(TableStyle([
		("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3faf6")),
		("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#a7d7c4")),
		("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cfe8dd")),
		("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
		("FONTNAME", (1, 0), (1, -1), "Helvetica"),
		("FONTSIZE", (0, 0), (-1, -1), 10),
		("LEFTPADDING", (0, 0), (-1, -1), 8),
		("RIGHTPADDING", (0, 0), (-1, -1), 8),
		("TOPPADDING", (0, 0), (-1, -1), 7),
		("BOTTOMPADDING", (0, 0), (-1, -1), 7),
	]))
	story.append(cover_table)
	story.append(PageBreak())

	# ── Helper secciones simples ──────────────────────────────────────────────
	def _simple_section(section_label: str, prob_key: str, pick_key: str) -> None:
		section_items = [row for row in entries if _meets_pdf_threshold(float(row[prob_key]))]
		if not section_items:
			return

		story.append(PageBreak())
		story.append(Paragraph(section_label, section_title))
		story.append(Paragraph(f"Solo picks con probabilidad >= {PDF_MIN_PROBABILITY:.0f}%.", body_text))
		story.append(Spacer(1, 0.08 * inch))

		date_groups: dict[str, list] = {}
		for row in section_items:
			date_groups.setdefault(str(row["date_label"]), []).append(row)

		for date_label in sorted(date_groups.keys()):
			head = Table([[Paragraph(f"FECHA: {date_label}", header_style)]], colWidths=[7.7 * inch])
			head.setStyle(TableStyle([
				("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1a7d5c")),
				("LEFTPADDING", (0, 0), (-1, -1), 10),
				("TOPPADDING", (0, 0), (-1, -1), 8),
				("BOTTOMPADDING", (0, 0), (-1, -1), 8),
			]))
			story.append(head)
			story.append(Spacer(1, 0.06 * inch))

			tbl_rows = [["#", "Encuentro", "Liga", "Pick", "Prob.", "Cuota justa"]]
			for idx, row in enumerate(sorted(date_groups[date_label], key=lambda r: r["kickoff"]), 1):
				prob = float(row[prob_key])
				tbl_rows.append([
					str(idx),
					f"{row['match']} ({row['kickoff']})",
					str(row["league_name"]),
					str(row[pick_key]),
					f"{prob:.2f}%".replace(".", ","),
					f"{_fair_odds(prob):.2f}".replace(".", ","),
				])

			table = Table(tbl_rows, colWidths=[0.35 * inch, 3.05 * inch, 1.45 * inch, 1.55 * inch, 0.7 * inch, 0.6 * inch])
			table.setStyle(TableStyle([
				("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f5f0")),
				("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f3d28")),
				("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
				("FONTSIZE", (0, 0), (-1, -1), 7),
				("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#a7d7c4")),
				("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
				("LEFTPADDING", (0, 0), (-1, -1), 4),
				("RIGHTPADDING", (0, 0), (-1, -1), 4),
			]))
			story.append(table)
			story.append(Spacer(1, 0.1 * inch))

	_simple_section("Seccion 1: Mercado 1X2", "one_x_two_prob", "one_x_two_pick")
	_simple_section("Seccion 2: Doble Oportunidad", "dc_prob", "dc_pick")
	_simple_section("Seccion 3: Goles (Over/Under)", "goles_prob", "goles_pick")
	_simple_section("Seccion 4: Tarjetas", "tarjetas_prob", "tarjetas_pick")
	_simple_section("Seccion 5: Corners", "corners_prob", "corners_pick")

	# ── Sección 6: Apuesta Múltiple ───────────────────────────────────────────
	if multiple_entries:
		multi_date_groups: dict[str, list] = {}
		for row, picks in multiple_entries:
			multi_date_groups.setdefault(str(row["date_label"]), []).append((row, picks))

		for date_label in sorted(multi_date_groups.keys()):
			head = Table([[Paragraph(f"FECHA: {date_label}", header_style)]], colWidths=[7.7 * inch])
			head.setStyle(TableStyle([
				("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#145433")),
				("LEFTPADDING", (0, 0), (-1, -1), 10),
				("TOPPADDING", (0, 0), (-1, -1), 8),
				("BOTTOMPADDING", (0, 0), (-1, -1), 8),
			]))
			story.append(head)
			story.append(Spacer(1, 0.06 * inch))

			def _comb_prob(picks):
				r = 1.0
				for _, _, p in picks:
					r *= p / 100.0
				return r

			for idx, (row, picks) in enumerate(
				sorted(multi_date_groups[date_label], key=lambda t: -_comb_prob(t[1])), 1
			):
				combined_prob = 1.0
				for _, _, p in picks:
					combined_prob *= p / 100.0
				combined_prob_pct = round(combined_prob * 100, 2)

				header_row = Table(
					[[
						Paragraph(f"<b>#{idx} {row['match']}</b><br/><font size=7>{row['league_name']} - {row['kickoff']}</font>", match_title_st),
						Paragraph(f"<b>{len(picks)} picks</b><br/>Prob. combinada: <b>{combined_prob_pct:.2f}%</b>", body_text),
						Paragraph(f"Cuota comb.:<br/><b>{_fair_odds(combined_prob_pct):.2f}</b>", body_text),
					]],
					colWidths=[3.4 * inch, 2.7 * inch, 1.6 * inch],
				)
				header_row.setStyle(TableStyle([
					("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e0f2eb")),
					("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor("#145433")),
					("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
					("LEFTPADDING", (0, 0), (-1, -1), 6),
					("TOPPADDING", (0, 0), (-1, -1), 6),
					("BOTTOMPADDING", (0, 0), (-1, -1), 6),
				]))
				story.append(header_row)

				ph_st = ParagraphStyle("PH", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=7, textColor=colors.HexColor("#0f3d28"))
				picks_data = [[
					Paragraph("MERCADO", ph_st),
					Paragraph("PICK", ph_st),
					Paragraph("PROB.", ph_st),
					Paragraph("CUOTA JUSTA", ph_st),
				]]
				for mkt_name, mkt_pick, mkt_prob in sorted(picks, key=lambda x: -x[2]):
					picks_data.append([
						Paragraph(mkt_name, leg_style),
						Paragraph(mkt_pick, leg_style),
						Paragraph(f"{mkt_prob:.2f}%".replace(".", ","), leg_style),
						Paragraph(f"{_fair_odds(mkt_prob):.2f}".replace(".", ","), leg_style),
					])
				picks_table = Table(picks_data, colWidths=[1.5 * inch, 3.2 * inch, 1.0 * inch, 2.0 * inch])
				picks_table.setStyle(TableStyle([
					("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#c8e8d4")),
					("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5fcf9")),
					("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#a7d7c4")),
					("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8e8dc")),
					("LEFTPADDING", (0, 0), (-1, -1), 6),
					("TOPPADDING", (0, 0), (-1, -1), 4),
					("BOTTOMPADDING", (0, 0), (-1, -1), 4),
					("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
				]))
				story.append(picks_table)
				story.append(Spacer(1, 0.12 * inch))

	doc.build(story)
	buffer.seek(0)
	response = HttpResponse(buffer.getvalue(), content_type="application/pdf")
	response["Content-Disposition"] = f'attachment; filename="resumen_general_reportes_{target_date.strftime("%d_%m_%Y")}.pdf"'
	return response
