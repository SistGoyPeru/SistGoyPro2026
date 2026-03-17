from __future__ import annotations

from typing import Callable

from django.db import transaction

from actualizar_encuentros import download_page as spain_download_page
from actualizar_encuentros import extract_matches as spain_extract_matches
from actualizar_encuentros import write_csv as spain_write_csv
from actualizar_encuentros_bundesliga import download_page as bundesliga_download_page
from actualizar_encuentros_bundesliga import extract_matches as bundesliga_extract_matches
from actualizar_encuentros_bundesliga import write_csv as bundesliga_write_csv
from actualizar_encuentros_ligue1 import download_page as ligue1_download_page
from actualizar_encuentros_ligue1 import extract_matches as ligue1_extract_matches
from actualizar_encuentros_ligue1 import write_csv as ligue1_write_csv
from actualizar_encuentros_premierleague import download_page as premier_download_page
from actualizar_encuentros_premierleague import extract_matches as premier_extract_matches
from actualizar_encuentros_premierleague import write_csv as premier_write_csv
from actualizar_encuentros_seriea import download_page as seriea_download_page
from actualizar_encuentros_seriea import extract_matches as seriea_extract_matches
from actualizar_encuentros_seriea import write_csv as seriea_write_csv
from actualizar_encuentros_primeiraliga import download_page as primeiraliga_download_page
from actualizar_encuentros_primeiraliga import extract_matches as primeiraliga_extract_matches
from actualizar_encuentros_primeiraliga import write_csv as primeiraliga_write_csv
from actualizar_encuentros_proleague import download_page as proleague_download_page
from actualizar_encuentros_proleague import extract_matches as proleague_extract_matches
from actualizar_encuentros_proleague import write_csv as proleague_write_csv
from actualizar_encuentros_eredivisie import download_page as eredivisie_download_page
from actualizar_encuentros_eredivisie import extract_matches as eredivisie_extract_matches
from actualizar_encuentros_eredivisie import write_csv as eredivisie_write_csv

from .engine import (
    get_prediction_service_bundesliga,
    get_prediction_service_ligue1,
    get_prediction_service_premier,
    get_prediction_service_seriea,
    get_prediction_service_spain,
    get_prediction_service_primeiraliga,
    get_prediction_service_proleague,
    get_prediction_service_eredivisie,
)
from .models import FixtureLinkCache


Updater = tuple[Callable[[], str], Callable[[str], list[dict[str, str]]], Callable[[list[dict[str, str]]], None]]


LEAGUE_UPDATERS: dict[str, Updater] = {
    "spain": (spain_download_page, spain_extract_matches, spain_write_csv),
    "bundesliga": (bundesliga_download_page, bundesliga_extract_matches, bundesliga_write_csv),
    "premier": (premier_download_page, premier_extract_matches, premier_write_csv),
    "seriea": (seriea_download_page, seriea_extract_matches, seriea_write_csv),
    "ligue1": (ligue1_download_page, ligue1_extract_matches, ligue1_write_csv),
    "primeiraliga": (primeiraliga_download_page, primeiraliga_extract_matches, primeiraliga_write_csv),
    "proleague": (proleague_download_page, proleague_extract_matches, proleague_write_csv),
    "eredivisie": (eredivisie_download_page, eredivisie_extract_matches, eredivisie_write_csv),
}


def clear_service_cache(liga: str) -> None:
    if liga == "bundesliga":
        get_prediction_service_bundesliga.cache_clear()
    elif liga == "premier":
        get_prediction_service_premier.cache_clear()
    elif liga == "seriea":
        get_prediction_service_seriea.cache_clear()
    elif liga == "ligue1":
        get_prediction_service_ligue1.cache_clear()
    elif liga == "primeiraliga":
        get_prediction_service_primeiraliga.cache_clear()
    elif liga == "proleague":
        get_prediction_service_proleague.cache_clear()
    elif liga == "eredivisie":
        get_prediction_service_eredivisie.cache_clear()
    else:
        get_prediction_service_spain.cache_clear()


@transaction.atomic
def refresh_fixture_links(liga: str) -> int:
    updater = LEAGUE_UPDATERS.get(liga)
    if updater is None:
        return 0

    download_page, extract_matches, write_csv = updater
    html = download_page()
    rows = extract_matches(html)
    write_csv(rows)

    current_keys: set[str] = set()
    for row in rows:
        match_key = "|".join(
            [
                str(row.get("fecha", "")),
                str(row.get("hora", "")),
                str(row.get("local", "")),
                str(row.get("visitante", "")),
            ]
        )
        current_keys.add(match_key)
        FixtureLinkCache.objects.update_or_create(
            league_key=liga,
            match_key=match_key,
            defaults={
                "competition": str(row.get("competicion", "")),
                "round_name": str(row.get("jornada", "")),
                "match_date": str(row.get("fecha", "")),
                "match_time": str(row.get("hora", "")),
                "home_team": str(row.get("local", "")),
                "away_team": str(row.get("visitante", "")),
                "result": str(row.get("resultado", "")),
                "status": str(row.get("estado", "")),
                "match_link": str(row.get("enlace_partido", "")),
            },
        )

    FixtureLinkCache.objects.filter(league_key=liga).exclude(match_key__in=current_keys).delete()
    clear_service_cache(liga)
    return len(rows)
