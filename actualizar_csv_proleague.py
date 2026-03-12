from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


SOURCE_URL = "https://www.football-data.co.uk/mmz4281/2526/B1.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "proleague_belgica.csv"

TEAM_NAME_MAP = {
    "Anderlecht":          "RSC Anderlecht",
    "Antwerp":             "Royal Antwerp FC",
    "Cercle Brugge":       "Cercle Brugge",
    "Charleroi":           "Sporting Charleroi",
    "Club Brugge":         "Club Brugge KV",
    "Dender":              "FCV Dender EH",
    "Genk":                "KRC Genk",
    "Gent":                "KAA Gent",
    "Mechelen":            "KV Mechelen",
    "Oud-Heverlee Leuven": "Oud-Heverlee Leuven",
    "RAAL La Louviere":    "RAAL La Louviére",
    "St Truiden":          "Sint-Truidense VV",
    "St. Gilloise":        "Union Saint-Gilloise",
    "Standard":            "Standard Lieja",
    "Waregem":             "SV Zulte Waregem",
    "Westerlo":            "KVC Westerlo",
}


def normalize_team_name(team_name: str) -> str:
    return TEAM_NAME_MAP.get(team_name, team_name)


def download_rows() -> tuple[list[str], list[dict[str, str]]]:
    request = Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response:
        raw_csv = response.read().decode("utf-8-sig")

    reader = csv.DictReader(raw_csv.splitlines())
    rows = list(reader)
    if reader.fieldnames is None:
        raise ValueError("No se pudieron leer los encabezados del CSV remoto.")

    return reader.fieldnames, rows


def normalize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        normalized_row = dict(row)
        if row.get("Div") == "B1":
            normalized_row["Div"] = "Pro League"
        normalized_row["HomeTeam"] = normalize_team_name(row.get("HomeTeam", ""))
        normalized_row["AwayTeam"] = normalize_team_name(row.get("AwayTeam", ""))
        normalized_rows.append(normalized_row)

    return normalized_rows


def write_output(fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output_dir = OUTPUT_FILE.parent
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8-sig",
        newline="",
        dir=output_dir,
        delete=False,
        suffix=".tmp",
    ) as temp_file:
        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(temp_file.name)

    temp_path.replace(OUTPUT_FILE)


def main() -> None:
    fieldnames, rows = download_rows()
    normalized = normalize_rows(rows)
    write_output(fieldnames, normalized)
    print(f"Archivo actualizado: {OUTPUT_FILE}")
    print(f"Partidos procesados: {len(normalized)}")


if __name__ == "__main__":
    main()
