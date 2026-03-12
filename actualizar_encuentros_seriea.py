from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag


SOURCE_URL = "https://www.livefutbol.com/competition/co111/italia-serie-a/all-matches/"
BASE_URL = "https://www.livefutbol.com"
OUTPUT_FILE = Path(__file__).resolve().parent / "seriea_italia_encuentros.csv"
FIELDNAMES = [
    "competicion",
    "jornada",
    "fecha",
    "hora",
    "local",
    "visitante",
    "resultado",
    "estado",
    "enlace_partido",
]


def download_page() -> str:
    response = requests.get(
        SOURCE_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    response.encoding = "utf-8"
    return response.text


def get_text(node: Tag | None, selector: str) -> str:
    if node is None:
        return ""
    selected = node.select_one(selector)
    return selected.get_text(strip=True) if selected else ""


def get_match_link(node: Tag) -> str:
    result_link = node.select_one(".match-result a")
    if result_link and result_link.get("href"):
        return urljoin(BASE_URL, result_link["href"])

    more_link = node.select_one(".match-more a")
    if more_link and more_link.get("href"):
        return urljoin(BASE_URL, more_link["href"])

    return ""


def extract_matches(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one("div.module-gameplan > div")
    if container is None:
        raise ValueError("No se encontro el bloque de calendario en la pagina.")

    current_round = ""
    current_date = ""
    matches: list[dict[str, str]] = []

    for child in container.find_all(recursive=False):
        classes = child.get("class") or []
        if "round-head" in classes:
            current_round = child.get_text(strip=True)
            continue

        if "match-date" in classes:
            current_date = child.get_text(strip=True)
            continue

        if "match" not in classes:
            continue

        matches.append(
            {
                "competicion": "Serie A Italia",
                "jornada": current_round,
                "fecha": current_date,
                "hora": get_text(child, ".match-time"),
                "local": get_text(child, ".team-name-home"),
                "visitante": get_text(child, ".team-name-away"),
                "resultado": get_text(child, ".match-result"),
                "estado": get_text(child, ".match-status"),
                "enlace_partido": get_match_link(child),
            }
        )

    return matches


def write_csv(rows: list[dict[str, str]]) -> None:
    output_dir = OUTPUT_FILE.parent
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8-sig",
        newline="",
        dir=output_dir,
        delete=False,
        suffix=".tmp",
    ) as temp_file:
        writer = csv.DictWriter(temp_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(temp_file.name)

    temp_path.replace(OUTPUT_FILE)


def main() -> None:
    html = download_page()
    rows = extract_matches(html)
    write_csv(rows)
    print(f"Archivo actualizado: {OUTPUT_FILE}")
    print(f"Encuentros procesados: {len(rows)}")


if __name__ == "__main__":
    main()
