from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


SOURCE_URL = "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "bundesliga_1_alemania.csv"

TEAM_NAME_MAP = {
    "Bayern Munich":  "Bayern Munich",
    "Dortmund":       "Borussia Dortmund",
    "Hoffenheim":     "1899 Hoffenheim",
    "RB Leipzig":     "RB Leipzig",
    "Leverkusen":     "Bayer Leverkusen",
    "Freiburg":       "SC Friburgo",
    "Ein Frankfurt":  "Eintracht Francfort",
    "Stuttgart":      "VfB Stuttgart",
    "FC Koln":        "1. FC Colonia",
    "Wolfsburg":      "VfL Wolfsburgo",
    "Hamburg":        "Hamburgo SV",
    "Union Berlin":   "Union Berlín",
    "Augsburg":       "FC Augsburgo",
    "Werder Bremen":  "Werder Bremen",
    "M'gladbach":     "Bor. Mönchengladbach",
    "Heidenheim":     "1. FC Heidenheim 1846",
    "Mainz":          "1. FSV Mainz 05",
    "St Pauli":       "FC St. Pauli",
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
        if row.get("Div") == "D1":
            normalized_row["Div"] = "Bundesliga 1"
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
    normalized_rows = normalize_rows(rows)
    write_output(fieldnames, normalized_rows)
    print(f"Archivo actualizado: {OUTPUT_FILE}")
    print(f"Partidos procesados: {len(normalized_rows)}")


if __name__ == "__main__":
    main()
