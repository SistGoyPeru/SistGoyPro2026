from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from functools import lru_cache
from math import exp, factorial
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


BASE_DIR = Path(__file__).resolve().parent.parent
HISTORICAL_CSV = BASE_DIR / "liga_1_españa.csv"
FIXTURES_CSV = BASE_DIR / "liga1_españa_encuentros.csv"
ENV_FILE = BASE_DIR / ".env"
TEAM_COORDS = {
    "FC Barcelona": (41.3851, 2.1734),
    "Real Madrid": (40.4168, -3.7038),
    "Atlético de Madrid": (40.4168, -3.7038),
    "Real Betis": (37.3891, -5.9845),
    "Rayo Vallecano": (40.4168, -3.7038),
    "Athletic Club": (43.2630, -2.9350),
    "Villarreal CF": (39.9384, -0.1009),
    "Valencia CF": (39.4699, -0.3763),
    "Elche CF": (38.2699, -0.7126),
    "CA Osasuna": (42.8125, -1.6458),
    "Espanyol Barcelona": (41.3851, 2.1734),
    "CD Alavés": (42.8467, -2.6716),
    "Real Sociedad": (43.3183, -1.9812),
    "Levante UD": (39.4699, -0.3763),
    "RC Celta": (42.2406, -8.7207),
    "RCD Mallorca": (39.5696, 2.6502),
    "Sevilla FC": (37.3891, -5.9845),
    "Girona FC": (41.9794, 2.8214),
    "Real Oviedo": (43.3614, -5.8494),
    "Getafe CF": (40.3083, -3.7327),
}
TEAM_API_ALIASES = {
    "Atlético de Madrid": "Atletico Madrid",
    "Espanyol Barcelona": "Espanyol",
    "RC Celta": "Celta Vigo",
    "RCD Mallorca": "Mallorca",
}
TEAM_SCRAPE_ALIASES = {
    "Athletic Club": "Athletic Bilbao",
    "Atlético de Madrid": "Atletico de Madrid",
    "CD Alavés": "Deportivo Alaves",
    "RC Celta": "Celta Vigo",
    "Espanyol Barcelona": "Espanyol",
    "RCD Mallorca": "Mallorca",
}
TEAM_TRANSFERMARKT_IDS = {
    "FC Barcelona": 131,
    "Real Madrid": 418,
    "Atlético de Madrid": 13,
    "Real Betis": 150,
    "Rayo Vallecano": 367,
    "Athletic Club": 621,
    "Villarreal CF": 1050,
    "Valencia CF": 1049,
    "Elche CF": 1532,
    "CA Osasuna": 331,
    "Espanyol Barcelona": 714,
    "CD Alavés": 1108,
    "Real Sociedad": 681,
    "Levante UD": 336,
    "RC Celta": 940,
    "RCD Mallorca": 237,
    "Sevilla FC": 368,
    "Girona FC": 12321,
    "Real Oviedo": 1141,
    "Getafe CF": 3709,
}
APIFOOTBALL_BASE = "https://v3.football.api-sports.io"
APIFOOTBALL_LEAGUE_ID = 140
TRANSFERMARKT_BASE = "https://www.transfermarkt.com"

# ── Bundesliga 1 ─────────────────────────────────────────────────────────────
BL_TEAM_COORDS = {
    "Bayern Munich":         (48.2188, 11.6247),
    "Borussia Dortmund":     (51.4926,  7.4517),
    "1899 Hoffenheim":       (49.2386,  8.8893),
    "RB Leipzig":            (51.3459, 12.3484),
    "Bayer Leverkusen":      (51.0381,  6.9836),
    "SC Friburgo":           (48.0217,  7.8927),
    "Eintracht Francfort":   (50.0685,  8.6452),
    "VfB Stuttgart":         (48.7924,  9.2320),
    "1. FC Colonia":         (50.9333,  6.8749),
    "VfL Wolfsburgo":        (52.4341, 10.8025),
    "Hamburgo SV":           (53.5870,  9.8988),
    "Union Berlín":          (52.4573, 13.5679),
    "FC Augsburgo":          (48.3233, 10.8842),
    "Werder Bremen":         (53.0664,  8.8381),
    "Bor. Mönchengladbach":  (51.1745,  6.3855),
    "1. FC Heidenheim 1846": (48.6778, 10.1556),
    "1. FSV Mainz 05":       (49.9841,  8.2241),
    "FC St. Pauli":          (53.5547,  9.9655),
}
BL_TEAM_API_ALIASES = {
    "SC Friburgo":           "Freiburg",
    "Eintracht Francfort":   "Eintracht Frankfurt",
    "VfL Wolfsburgo":        "Wolfsburg",
    "Hamburgo SV":           "Hamburg",
    "Union Berlín":          "Union Berlin",
    "FC Augsburgo":          "Augsburg",
    "Bor. Mönchengladbach":  "Borussia M'gladbach",
    "1. FC Heidenheim 1846": "Heidenheim",
    "1. FSV Mainz 05":       "Mainz",
    "FC St. Pauli":          "St. Pauli",
    "1. FC Colonia":         "FC Koln",
}
BL_TEAM_SCRAPE_ALIASES = {
    "SC Friburgo":           "Freiburg",
    "Eintracht Francfort":   "Eintracht Frankfurt",
    "VfL Wolfsburgo":        "VfL Wolfsburg",
    "Hamburgo SV":           "Hamburger SV",
    "Union Berlín":          "Union Berlin",
    "FC Augsburgo":          "FC Augsburg",
    "Bor. Mönchengladbach":  "Borussia Monchengladbach",
    "1. FC Heidenheim 1846": "FC Heidenheim",
    "1. FSV Mainz 05":       "FSV Mainz 05",
    "FC St. Pauli":          "FC St. Pauli",
    "1. FC Colonia":         "1. FC Koln",
}
BL_TEAM_TRANSFERMARKT_IDS = {
    "Bayern Munich":         27,
    "Borussia Dortmund":     16,
    "1899 Hoffenheim":       533,
    "RB Leipzig":            23826,
    "Bayer Leverkusen":      15,
    "SC Friburgo":           17,
    "Eintracht Francfort":   24,
    "VfB Stuttgart":         79,
    "1. FC Colonia":         3,
    "VfL Wolfsburgo":        82,
    "Hamburgo SV":           41,
    "Union Berlín":          89,
    "FC Augsburgo":          167,
    "Werder Bremen":         86,
    "Bor. Mönchengladbach":  18,
    "1. FC Heidenheim 1846": 2036,
    "1. FSV Mainz 05":       39,
    "FC St. Pauli":          35,
}
BL_APIFOOTBALL_LEAGUE_ID = 78

# ── Premier League ───────────────────────────────────────────────────────────
PL_TEAM_COORDS = {
    "Arsenal FC":                  (51.5549, -0.1084),
    "Aston Villa":                 (52.5092, -1.8847),
    "AFC Bournemouth":             (50.7352, -1.8380),
    "Brentford FC":                (51.4907, -0.2887),
    "Brighton & Hove Albion":      (50.8618, -0.0834),
    "Burnley FC":                  (53.7889, -2.2303),
    "Chelsea FC":                  (51.4816, -0.1910),
    "Crystal Palace":              (51.3983, -0.0855),
    "Everton FC":                  (53.4388, -2.9664),
    "Fulham FC":                   (51.4749, -0.2214),
    "Leeds United":                (53.7775, -1.5722),
    "Liverpool FC":                (53.4308, -2.9608),
    "Manchester City":             (53.4831, -2.2004),
    "Manchester United":           (53.4631, -2.2913),
    "Newcastle United":            (54.9756, -1.6217),
    "Nottingham Forest":           (52.9399, -1.1327),
    "Sunderland AFC":              (54.9147, -1.3882),
    "Tottenham Hotspur":           (51.6043, -0.0665),
    "West Ham United":             (51.5386, -0.0164),
    "Wolverhampton Wanderers":     (52.5901, -2.1302),
}
PL_TEAM_API_ALIASES = {
    "AFC Bournemouth":             "Bournemouth",
    "Brighton & Hove Albion":      "Brighton",
    "Wolverhampton Wanderers":     "Wolves",
}
PL_TEAM_SCRAPE_ALIASES = {
    "AFC Bournemouth":             "Bournemouth",
    "Brighton & Hove Albion":      "Brighton & Hove Albion",
    "Wolverhampton Wanderers":     "Wolverhampton Wanderers",
    "Nottingham Forest":           "Nottingham Forest",
}
PL_TEAM_TRANSFERMARKT_IDS = {
    "Arsenal FC":                  11,
    "Aston Villa":                 405,
    "AFC Bournemouth":             989,
    "Brentford FC":                1148,
    "Brighton & Hove Albion":      1237,
    "Burnley FC":                  1132,
    "Chelsea FC":                  631,
    "Crystal Palace":              873,
    "Everton FC":                  29,
    "Fulham FC":                   931,
    "Leeds United":                399,
    "Liverpool FC":                31,
    "Manchester City":             281,
    "Manchester United":           985,
    "Newcastle United":            762,
    "Nottingham Forest":           703,
    "Sunderland AFC":              289,
    "Tottenham Hotspur":           148,
    "West Ham United":             379,
    "Wolverhampton Wanderers":     543,
}
PL_APIFOOTBALL_LEAGUE_ID = 39

# ── Serie A Italia ───────────────────────────────────────────────────────────
SA_TEAM_COORDS = {
    "Atalanta":           (45.7089, 9.6757),
    "Bologna FC":         (44.4929, 11.3128),
    "Cagliari Calcio":    (39.2084, 9.1274),
    "Como 1907":          (45.8097, 9.0852),
    "US Cremonese":       (45.1372, 10.0291),
    "ACF Fiorentina":     (43.7800, 11.2828),
    "Genoa CFC":          (44.4187, 8.9519),
    "Inter":              (45.4781, 9.1240),
    "Juventus":           (45.1096, 7.6412),
    "Lazio Roma":         (41.9341, 12.4547),
    "US Lecce":           (40.3606, 18.1750),
    "AC Milan":           (45.4781, 9.1240),
    "SSC Napoli":         (40.8279, 14.1931),
    "Parma Calcio 1913":  (44.7978, 10.3383),
    "Pisa SC":            (43.7105, 10.3912),
    "AS Roma":            (41.9341, 12.4547),
    "Sassuolo Calcio":    (44.5398, 10.7856),
    "Torino FC":          (45.0408, 7.6505),
    "Udinese Calcio":     (46.0756, 13.1874),
    "Hellas Verona":      (45.4384, 10.9916),
}
SA_TEAM_API_ALIASES: dict[str, str] = {}
SA_TEAM_SCRAPE_ALIASES: dict[str, str] = {
    "ACF Fiorentina": "Fiorentina",
    "SSC Napoli":     "Napoli",
    "AC Milan":       "Milan",
    "AS Roma":        "Roma",
    "Lazio Roma":     "Lazio",
    "Inter":          "Internazionale",
}
SA_TEAM_TRANSFERMARKT_IDS = {
    "Atalanta":           800,
    "Bologna FC":         1025,
    "Cagliari Calcio":    1390,
    "Como 1907":          3634,
    "US Cremonese":       3614,
    "ACF Fiorentina":     430,
    "Genoa CFC":          252,
    "Inter":              46,
    "Juventus":           506,
    "Lazio Roma":         398,
    "US Lecce":           1825,
    "AC Milan":           5,
    "SSC Napoli":         6195,
    "Parma Calcio 1913":  161,
    "Pisa SC":            3617,
    "AS Roma":            12,
    "Sassuolo Calcio":    6574,
    "Torino FC":          416,
    "Udinese Calcio":     410,
    "Hellas Verona":      276,
}
SA_APIFOOTBALL_LEAGUE_ID = 135

# ── Ligue 1 Francia ──────────────────────────────────────────────────────────
L1_TEAM_COORDS = {
    "Angers SCO":          (47.4603, -0.5304),
    "AJ Auxerre":          (47.7982, 3.5735),
    "Stade Brestois 29":   (48.4047, -4.4937),
    "Havre AC":            (49.4988, 0.1696),
    "RC Lens":             (50.4332, 2.8158),
    "Lille OSC":           (50.6119, 3.1300),
    "FC Lorient":          (47.7481, -3.3699),
    "Olympique Lyonnais":  (45.7650, 4.9820),
    "Olympique Marseille": (43.2698, 5.3959),
    "FC Metz":             (49.1086, 6.1597),
    "AS Monaco":           (43.7274, 7.4157),
    "FC Nantes":           (47.2556, -1.5253),
    "OGC Nice":            (43.7068, 7.1928),
    "Paris FC":            (48.8199, 2.3794),
    "Paris Saint-Germain": (48.8414, 2.2530),
    "Stade Rennais":       (48.1076, -1.7138),
    "Racing Strasbourg":   (48.5602, 7.7560),
    "Toulouse FC":         (43.5834, 1.4340),
}
L1_TEAM_API_ALIASES = {
    "Paris Saint-Germain": "Paris SG",
    "Olympique Marseille": "Marseille",
    "Olympique Lyonnais":  "Lyon",
}
L1_TEAM_SCRAPE_ALIASES = {
    "Paris Saint-Germain": "Paris Saint-Germain",
}
L1_TEAM_TRANSFERMARKT_IDS = {
    "Angers SCO":          1420,
    "AJ Auxerre":          290,
    "Stade Brestois 29":   3911,
    "Havre AC":            738,
    "RC Lens":             826,
    "Lille OSC":           1082,
    "FC Lorient":          1158,
    "Olympique Lyonnais":  1041,
    "Olympique Marseille": 244,
    "FC Metz":             347,
    "AS Monaco":           162,
    "FC Nantes":           995,
    "OGC Nice":            417,
    "Paris FC":            736,
    "Paris Saint-Germain": 583,
    "Stade Rennais":       273,
    "Racing Strasbourg":   667,
    "Toulouse FC":         415,
}
L1_APIFOOTBALL_LEAGUE_ID = 61

# ── Configs de liga ──────────────────────────────────────────────────────────
SPAIN_CONFIG: dict = {
    "historical_csv":         BASE_DIR / "liga_1_españa.csv",
    "fixtures_csv":           BASE_DIR / "liga1_españa_encuentros.csv",
    "team_coords":            TEAM_COORDS,
    "team_api_aliases":       TEAM_API_ALIASES,
    "team_scrape_aliases":    TEAM_SCRAPE_ALIASES,
    "team_transfermarkt_ids": TEAM_TRANSFERMARKT_IDS,
    "apifootball_league_id":  APIFOOTBALL_LEAGUE_ID,
    "league_name":            "Liga 1 España",
}
BUNDESLIGA_CONFIG: dict = {
    "historical_csv":         BASE_DIR / "bundesliga_1_alemania.csv",
    "fixtures_csv":           BASE_DIR / "bundesliga1_alemania_encuentros.csv",
    "team_coords":            BL_TEAM_COORDS,
    "team_api_aliases":       BL_TEAM_API_ALIASES,
    "team_scrape_aliases":    BL_TEAM_SCRAPE_ALIASES,
    "team_transfermarkt_ids": BL_TEAM_TRANSFERMARKT_IDS,
    "apifootball_league_id":  BL_APIFOOTBALL_LEAGUE_ID,
    "league_name":            "Bundesliga 1 Alemania",
}
PREMIER_LEAGUE_CONFIG: dict = {
    "historical_csv":         BASE_DIR / "premierleague_inglaterra.csv",
    "fixtures_csv":           BASE_DIR / "premierleague_inglaterra_encuentros.csv",
    "team_coords":            PL_TEAM_COORDS,
    "team_api_aliases":       PL_TEAM_API_ALIASES,
    "team_scrape_aliases":    PL_TEAM_SCRAPE_ALIASES,
    "team_transfermarkt_ids": PL_TEAM_TRANSFERMARKT_IDS,
    "apifootball_league_id":  PL_APIFOOTBALL_LEAGUE_ID,
    "league_name":            "Premier League Inglaterra",
}
SERIEA_CONFIG: dict = {
    "historical_csv":         BASE_DIR / "seriea_italia.csv",
    "fixtures_csv":           BASE_DIR / "seriea_italia_encuentros.csv",
    "team_coords":            SA_TEAM_COORDS,
    "team_api_aliases":       SA_TEAM_API_ALIASES,
    "team_scrape_aliases":    SA_TEAM_SCRAPE_ALIASES,
    "team_transfermarkt_ids": SA_TEAM_TRANSFERMARKT_IDS,
    "apifootball_league_id":  SA_APIFOOTBALL_LEAGUE_ID,
    "league_name":            "Serie A Italia",
}
LIGUE1_CONFIG: dict = {
    "historical_csv":         BASE_DIR / "ligue1_francia.csv",
    "fixtures_csv":           BASE_DIR / "ligue1_francia_encuentros.csv",
    "team_coords":            L1_TEAM_COORDS,
    "team_api_aliases":       L1_TEAM_API_ALIASES,
    "team_scrape_aliases":    L1_TEAM_SCRAPE_ALIASES,
    "team_transfermarkt_ids": L1_TEAM_TRANSFERMARKT_IDS,
    "apifootball_league_id":  L1_APIFOOTBALL_LEAGUE_ID,
    "league_name":            "Ligue 1 Francia",
}

# ── Primeira Liga Portugal ────────────────────────────────────────────────────
PT_TEAM_COORDS = {
    "FC Alverca":        (38.9037, -9.0300),
    "FC Arouca":         (40.9333, -8.2500),
    "AVS":               (41.3631, -8.7455),
    "SL Benfica":        (38.7523, -9.1839),
    "Casa Pia AC":       (38.7290, -9.1583),
    "GD Estoril":        (38.7075, -9.3967),
    "Estrela Amadora":   (38.7538, -9.2302),
    "FC Famalicão":      (41.4000, -8.5167),
    "Gil Vicente":       (41.5333, -8.3000),
    "Vitória Guimarães": (41.4500, -8.2833),
    "Moreirense FC":     (41.4167, -8.3000),
    "CD Nacional":       (32.6500, -16.9000),
    "FC Porto":          (41.1620, -8.5834),
    "Rio Ave FC":        (41.3667, -8.6167),
    "CD Santa Clara":    (37.7333, -25.6667),
    "Sporting Braga":    (41.5500, -8.4167),
    "Sporting CP":       (38.7612, -9.1602),
    "CD Tondela":        (40.5167, -8.0833),
}
PT_TEAM_API_ALIASES = {
    "FC Famalicão":      "Famalicao",
    "Vitória Guimarães": "Vitoria Guimaraes",
    "Sporting CP":       "Sporting CP",
    "Sporting Braga":    "Braga",
}
PT_TEAM_SCRAPE_ALIASES = {
    "FC Famalicão":      "FC Famalicao",
    "Vitória Guimarães": "Vitoria SC",
    "Sporting Braga":    "SC Braga",
}
PT_TEAM_TRANSFERMARKT_IDS = {
    "FC Alverca":        10520,
    "FC Arouca":         13602,
    "AVS":               50191,
    "SL Benfica":        294,
    "Casa Pia AC":       21530,
    "GD Estoril":        3080,
    "Estrela Amadora":   24040,
    "FC Famalicão":      10296,
    "Gil Vicente":       4827,
    "Vitória Guimarães": 2751,
    "Moreirense FC":     10534,
    "CD Nacional":       3473,
    "FC Porto":          720,
    "Rio Ave FC":        3474,
    "CD Santa Clara":    15417,
    "Sporting Braga":    1070,
    "Sporting CP":       336,
    "CD Tondela":        13644,
}
PT_APIFOOTBALL_LEAGUE_ID = 94
PRIMEIRALIGA_CONFIG: dict = {
    "historical_csv":         BASE_DIR / "primeiraliga_portugal.csv",
    "fixtures_csv":           BASE_DIR / "primeiraliga_portugal_encuentros.csv",
    "team_coords":            PT_TEAM_COORDS,
    "team_api_aliases":       PT_TEAM_API_ALIASES,
    "team_scrape_aliases":    PT_TEAM_SCRAPE_ALIASES,
    "team_transfermarkt_ids": PT_TEAM_TRANSFERMARKT_IDS,
    "apifootball_league_id":  PT_APIFOOTBALL_LEAGUE_ID,
    "league_name":            "Primeira Liga Portugal",
}

# ── Pro League Bélgica ────────────────────────────────────────────────────────
BE_TEAM_COORDS = {
    "RSC Anderlecht":      (50.8359, 4.2978),
    "Royal Antwerp FC":    (51.2250, 4.4170),
    "Cercle Brugge":       (51.1978, 3.2183),
    "Sporting Charleroi":  (50.4051, 4.4435),
    "Club Brugge KV":      (51.1978, 3.2183),
    "FCV Dender EH":       (50.9167, 4.0500),
    "KRC Genk":            (50.9667, 5.5000),
    "KAA Gent":            (51.0500, 3.7333),
    "KV Mechelen":         (51.0167, 4.4833),
    "Oud-Heverlee Leuven": (50.8833, 4.7000),
    "RAAL La Louviére":    (50.4667, 4.1833),
    "Sint-Truidense VV":   (50.8167, 5.1833),
    "Union Saint-Gilloise":(50.8503, 4.3517),
    "Standard Lieja":      (50.6000, 5.5500),
    "SV Zulte Waregem":    (50.8833, 3.4333),
    "KVC Westerlo":        (51.0833, 4.9167),
}
BE_TEAM_API_ALIASES = {
    "RSC Anderlecht":      "Anderlecht",
    "Royal Antwerp FC":    "Antwerp",
    "Club Brugge KV":      "Club Brugge",
    "KRC Genk":            "Genk",
    "KAA Gent":            "Gent",
    "Union Saint-Gilloise": "Union Saint Gilloise",
    "Standard Lieja":      "Standard Liege",
    "SV Zulte Waregem":    "Zulte Waregem",
}
BE_TEAM_SCRAPE_ALIASES = {
    "RSC Anderlecht":      "Anderlecht",
    "Royal Antwerp FC":    "Antwerp",
    "Club Brugge KV":      "Club Brugge",
    "KRC Genk":            "Genk",
    "KAA Gent":            "Gent",
    "RAAL La Louviére":    "RAAL La Louviere",
    "Sint-Truidense VV":   "Saint-Trond",
    "Union Saint-Gilloise": "Union SG",
    "Standard Lieja":      "Standard",
    "SV Zulte Waregem":    "Zulte-Waregem",
}
BE_TEAM_TRANSFERMARKT_IDS = {
    "RSC Anderlecht":      59,
    "Royal Antwerp FC":    749,
    "Cercle Brugge":       1437,
    "Sporting Charleroi":  276,
    "Club Brugge KV":      2282,
    "FCV Dender EH":       10453,
    "KRC Genk":            3304,
    "KAA Gent":            2566,
    "KV Mechelen":         6006,
    "Oud-Heverlee Leuven": 24064,
    "RAAL La Louviére":    4618,
    "Sint-Truidense VV":   3297,
    "Union Saint-Gilloise": 8239,
    "Standard Lieja":      276,
    "SV Zulte Waregem":    2473,
    "KVC Westerlo":        5696,
}
BE_APIFOOTBALL_LEAGUE_ID = 144
PROLEAGUE_CONFIG: dict = {
    "historical_csv":         BASE_DIR / "proleague_belgica.csv",
    "fixtures_csv":           BASE_DIR / "proleague_belgica_encuentros.csv",
    "team_coords":            BE_TEAM_COORDS,
    "team_api_aliases":       BE_TEAM_API_ALIASES,
    "team_scrape_aliases":    BE_TEAM_SCRAPE_ALIASES,
    "team_transfermarkt_ids": BE_TEAM_TRANSFERMARKT_IDS,
    "apifootball_league_id":  BE_APIFOOTBALL_LEAGUE_ID,
    "league_name":            "Pro League Bélgica",
}

# ── Eredivisie Holanda ───────────────────────────────────────────────────────
NL_TEAM_COORDS = {
    "AFC Ajax":          (52.3144, 4.9414),
    "AZ Alkmaar":        (52.6317, 4.7486),
    "SBV Excelsior":     (51.9194, 4.5239),
    "Feyenoord":         (51.8939, 4.5231),
    "Fortuna Sittard":   (50.9983, 5.8694),
    "Go Ahead Eagles":   (52.2550, 6.1639),
    "FC Groningen":      (53.2194, 6.5665),
    "Sc Heerenveen":     (52.9600, 5.9200),
    "Heracles Almelo":   (52.3567, 6.6625),
    "NAC Breda":         (51.5719, 4.7683),
    "NEC Nijmegen":      (51.8425, 5.8528),
    "PSV Eindhoven":     (51.4416, 5.4697),
    "Sparta Rotterdam":  (51.9225, 4.4792),
    "Telstar":           (52.4625, 4.6211),
    "FC Twente":         (52.2215, 6.8937),
    "FC Utrecht":        (52.0907, 5.1214),
    "FC Volendam":       (52.4950, 5.0708),
    "PEC Zwolle":        (52.5168, 6.0830),
}
NL_TEAM_API_ALIASES = {
    "AFC Ajax":         "Ajax",
    "SBV Excelsior":    "Excelsior",
    "Fortuna Sittard":  "Fortuna Sittard",
    "FC Groningen":     "Groningen",
    "Sc Heerenveen":    "Heerenveen",
    "Heracles Almelo":  "Heracles",
    "NEC Nijmegen":     "Nijmegen",
    "FC Twente":        "Twente",
    "FC Utrecht":       "Utrecht",
    "FC Volendam":      "Volendam",
    "PEC Zwolle":       "Zwolle",
}
NL_TEAM_SCRAPE_ALIASES = {
    "AFC Ajax":         "Ajax",
    "SBV Excelsior":    "Excelsior",
    "FC Groningen":     "Groningen",
    "Sc Heerenveen":    "Heerenveen",
    "Heracles Almelo":  "Heracles",
    "NEC Nijmegen":     "NEC",
    "FC Twente":        "Twente",
    "FC Utrecht":       "Utrecht",
    "FC Volendam":      "Volendam",
    "PEC Zwolle":       "Zwolle",
}
NL_TEAM_TRANSFERMARKT_IDS = {
    "AFC Ajax":         610,
    "AZ Alkmaar":       1090,
    "SBV Excelsior":    798,
    "Feyenoord":        234,
    "Fortuna Sittard":  385,
    "Go Ahead Eagles":  1435,
    "FC Groningen":     202,
    "Sc Heerenveen":    306,
    "Heracles Almelo":  1304,
    "NAC Breda":        132,
    "NEC Nijmegen":     467,
    "PSV Eindhoven":    383,
    "Sparta Rotterdam": 468,
    "Telstar":          1573,
    "FC Twente":        317,
    "FC Utrecht":       481,
    "FC Volendam":      1128,
    "PEC Zwolle":       3839,
}
NL_APIFOOTBALL_LEAGUE_ID = 88
EREDIVISIE_CONFIG: dict = {
    "historical_csv":         BASE_DIR / "eredivisie_holanda.csv",
    "fixtures_csv":           BASE_DIR / "eredivisie_holanda_encuentros.csv",
    "team_coords":            NL_TEAM_COORDS,
    "team_api_aliases":       NL_TEAM_API_ALIASES,
    "team_scrape_aliases":    NL_TEAM_SCRAPE_ALIASES,
    "team_transfermarkt_ids": NL_TEAM_TRANSFERMARKT_IDS,
    "apifootball_league_id":  NL_APIFOOTBALL_LEAGUE_ID,
    "league_name":            "Eredivisie Holanda",
}

# ── Süper Lig Turquía ─────────────────────────────────────────────────────────
TR_TEAM_COORDS = {
    "Alanyaspor":           (36.5444, 32.0058),
    "Antalyaspor":          (36.8824, 30.6956),
    "Beşiktaş":             (41.0437, 29.0073),
    "İstanbul Başakşehir":  (41.0830, 28.7981),
    "Eyüpspor":             (41.0519, 28.9317),
    "Fenerbahçe":           (40.9997, 29.0416),
    "Galatasaray":          (41.0731, 28.9900),
    "Gaziantep FK":         (37.0594, 37.3825),
    "Gençlerbirliği":       (39.8794, 32.8597),
    "Göztepe":              (38.4192, 27.1397),
    "Fatih Karagümrük":     (41.0180, 28.9536),
    "Kasımpaşa SK":         (41.0608, 28.9519),
    "Kayserispor":          (38.6931, 35.5031),
    "Kocaelispor":          (40.7867, 29.9167),
    "Konyaspor":            (37.8667, 32.5000),
    "Çaykur Rizespor":      (41.0208, 40.5236),
    "Samsunspor":           (41.2867, 36.3333),
    "Trabzonspor":          (40.9975, 39.7425),
}
TR_TEAM_API_ALIASES: dict[str, str] = {
    "İstanbul Başakşehir": "Istanbul Basaksehir",
    "Beşiktaş":            "Besiktas",
    "Fenerbahçe":          "Fenerbahce",
    "Galatasaray":         "Galatasaray",
    "Gençlerbirliği":      "Genclerbirligi",
    "Göztepe":             "Goztepe",
    "Fatih Karagümrük":    "Fatih Karagumruk",
    "Kasımpaşa SK":        "Kasimpasa",
    "Çaykur Rizespor":     "Caykur Rizespor",
    "Eyüpspor":            "Eyupspor",
}
TR_TEAM_SCRAPE_ALIASES: dict[str, str] = {
    "İstanbul Başakşehir": "Basaksehir",
    "Beşiktaş":            "Besiktas",
    "Fenerbahçe":          "Fenerbahce",
    "Gençlerbirliği":      "Genclerbirligi",
    "Göztepe":             "Goztepe",
    "Fatih Karagümrük":    "Karagumruk",
    "Kasımpaşa SK":        "Kasimpasa",
    "Çaykur Rizespor":     "Rizespor",
    "Eyüpspor":            "Eyupspor",
}
TR_TEAM_TRANSFERMARKT_IDS: dict[str, int] = {
    "Galatasaray":          141,
    "Fenerbahçe":           36,
    "Beşiktaş":             114,
    "Trabzonspor":          449,
    "İstanbul Başakşehir":  11965,
    "Alanyaspor":           26624,
    "Antalyaspor":          389,
    "Eyüpspor":             7613,
    "Göztepe":              680,
    "Fatih Karagümrük":     635,
    "Kasımpaşa SK":         7890,
    "Kayserispor":          2434,
    "Gençlerbirliği":       2289,
    "Konyaspor":            1258,
    "Çaykur Rizespor":      2235,
    "Samsunspor":           1072,
    "Gaziantep FK":         6012,
    "Kocaelispor":          1024,
}
TR_APIFOOTBALL_LEAGUE_ID = 203
SUPERLIG_TURQUIA_CONFIG: dict = {
    "historical_csv":         BASE_DIR / "superlig_turquia.csv",
    "fixtures_csv":           BASE_DIR / "superlig_turquia_encuentros.csv",
    "team_coords":            TR_TEAM_COORDS,
    "team_api_aliases":       TR_TEAM_API_ALIASES,
    "team_scrape_aliases":    TR_TEAM_SCRAPE_ALIASES,
    "team_transfermarkt_ids": TR_TEAM_TRANSFERMARKT_IDS,
    "apifootball_league_id":  TR_APIFOOTBALL_LEAGUE_ID,
    "league_name":            "Süper Lig Turquía",
}

# ── Super League Grecia ───────────────────────────────────────────────────────
GR_TEAM_COORDS = {
    "Aris Saloniki":       (40.6236, 22.9655),
    "Olympiakos Piraeus":  (37.9667, 23.6667),
    "Panetolikos":         (38.6239, 21.4150),
    "AEK Athen":           (37.9994, 23.7294),
    "PAOK Saloniki":       (40.6297, 22.9419),
    "Levadiakos":          (38.4400, 22.8700),
    "Volos NFC":           (39.3667, 22.9333),
    "Panserraikos":        (41.0833, 23.5500),
    "AE Lárissa":          (39.6361, 22.4194),
    "Panathinaikos":       (37.9833, 23.7333),
    "Atromitos":           (38.0000, 23.6833),
    "Asteras Tripolis":    (37.5094, 22.3764),
    "AE Kifisias":         (38.0681, 23.8186),
    "OFI Heraklion":       (35.3297, 25.1331),
}
GR_TEAM_API_ALIASES: dict[str, str] = {
    "Aris Saloniki":      "Aris",
    "Olympiakos Piraeus": "Olympiakos",
    "AEK Athen":          "AEK Athens",
    "PAOK Saloniki":      "PAOK",
    "Levadiakos":         "Levadeiakos",
    "AE Lárissa":         "Larissa",
    "AE Kifisias":        "Kifisia",
    "OFI Heraklion":      "OFI Crete",
}
GR_TEAM_SCRAPE_ALIASES: dict[str, str] = {
    "Aris Saloniki":      "Aris",
    "Olympiakos Piraeus": "Olympiakos",
    "AEK Athen":          "AEK",
    "PAOK Saloniki":      "PAOK",
    "Levadiakos":         "Levadeiakos",
    "AE Lárissa":         "Larissa",
    "AE Kifisias":        "Kifisia",
    "OFI Heraklion":      "OFI",
}
GR_TEAM_TRANSFERMARKT_IDS: dict[str, int] = {
    "Olympiakos Piraeus":  2686,
    "Panathinaikos":       519,
    "AEK Athen":           269,
    "PAOK Saloniki":       1067,
    "Aris Saloniki":       1088,
    "Asteras Tripolis":    13682,
    "Atromitos":           5636,
    "Panetolikos":         14682,
    "Levadiakos":          17609,
    "Volos NFC":           45543,
    "Panserraikos":        11015,
    "AE Lárissa":          14025,
    "AE Kifisias":         48462,
    "OFI Heraklion":       3536,
}
GR_APIFOOTBALL_LEAGUE_ID = 197
SUPERLEAGUE_GRECIA_CONFIG: dict = {
    "historical_csv":         BASE_DIR / "superleague_grecia.csv",
    "fixtures_csv":           BASE_DIR / "superleague_grecia_encuentros.csv",
    "team_coords":            GR_TEAM_COORDS,
    "team_api_aliases":       GR_TEAM_API_ALIASES,
    "team_scrape_aliases":    GR_TEAM_SCRAPE_ALIASES,
    "team_transfermarkt_ids": GR_TEAM_TRANSFERMARKT_IDS,
    "apifootball_league_id":  GR_APIFOOTBALL_LEAGUE_ID,
    "league_name":            "Super League Grecia",
}

# ── Premiership Escocia ───────────────────────────────────────────────────────
SC_TEAM_COORDS = {
    "Kilmarnock FC":       (55.6117, -4.4972),
    "Motherwell FC":       (55.7778, -3.9778),
    "Falkirk FC":          (56.0000, -3.7833),
    "Dundee FC":           (56.4667, -2.9667),
    "Celtic FC":           (55.8500, -4.2000),
    "Heart of Midlothian": (55.9281, -3.2317),
    "Livingston FC":       (55.8833, -3.5167),
    "St. Mirren FC":       (55.8500, -4.4333),
    "Rangers FC":          (55.8553, -4.3086),
    "Aberdeen FC":         (57.1561, -2.0972),
    "Dundee United":       (56.4667, -2.9667),
    "Hibernian FC":        (55.9617, -3.1644),
}
SC_TEAM_API_ALIASES: dict[str, str] = {
    "Kilmarnock FC":       "Kilmarnock",
    "Motherwell FC":       "Motherwell",
    "Falkirk FC":          "Falkirk",
    "Dundee FC":           "Dundee",
    "Celtic FC":           "Celtic",
    "Heart of Midlothian": "Hearts",
    "Livingston FC":       "Livingston",
    "St. Mirren FC":       "St Mirren",
    "Rangers FC":          "Rangers",
    "Aberdeen FC":         "Aberdeen",
    "Dundee United":       "Dundee United",
    "Hibernian FC":        "Hibernian",
}
SC_TEAM_SCRAPE_ALIASES: dict[str, str] = {
    "Kilmarnock FC":       "Kilmarnock",
    "Motherwell FC":       "Motherwell",
    "Falkirk FC":          "Falkirk",
    "Dundee FC":           "Dundee",
    "Celtic FC":           "Celtic",
    "Heart of Midlothian": "Hearts",
    "Livingston FC":       "Livingston",
    "St. Mirren FC":       "St Mirren",
    "Rangers FC":          "Rangers",
    "Aberdeen FC":         "Aberdeen",
    "Hibernian FC":        "Hibernian",
}
SC_TEAM_TRANSFERMARKT_IDS: dict[str, int] = {
    "Celtic FC":           371,
    "Rangers FC":          1764,
    "Heart of Midlothian": 1449,
    "Hibernian FC":        1450,
    "Aberdeen FC":         1452,
    "Motherwell FC":       1451,
    "Dundee United":       2276,
    "Kilmarnock FC":       1455,
    "St. Mirren FC":       1457,
    "Livingston FC":       2415,
    "Dundee FC":           1453,
    "Falkirk FC":          1456,
}
SC_APIFOOTBALL_LEAGUE_ID = 179
PREMIERSHIP_ESCOCIA_CONFIG: dict = {
    "historical_csv":         BASE_DIR / "premierleague_escocia.csv",
    "fixtures_csv":           BASE_DIR / "premierleague_escocia_encuentros.csv",
    "team_coords":            SC_TEAM_COORDS,
    "team_api_aliases":       SC_TEAM_API_ALIASES,
    "team_scrape_aliases":    SC_TEAM_SCRAPE_ALIASES,
    "team_transfermarkt_ids": SC_TEAM_TRANSFERMARKT_IDS,
    "apifootball_league_id":  SC_APIFOOTBALL_LEAGUE_ID,
    "league_name":            "Premiership Escocia",
}


def _read_env_value(key: str) -> str:
    if not ENV_FILE.exists():
        return ""
    try:
        for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            env_key, env_value = line.split("=", 1)
            if env_key.strip() != key:
                continue
            return env_value.strip().strip('"').strip("'")
    except OSError:
        return ""
    return ""
FEATURE_COLUMNS = [
    "home_points_recent",
    "away_points_recent",
    "points_recent_diff",
    "home_goals_for_recent",
    "away_goals_for_recent",
    "goals_for_diff",
    "home_goals_against_recent",
    "away_goals_against_recent",
    "defensive_diff",
    "home_shots_recent",
    "away_shots_recent",
    "home_sot_recent",
    "away_sot_recent",
    "home_win_rate",
    "away_win_rate",
    "home_draw_rate",
    "away_draw_rate",
    "cards_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    # Nuevas variables derivadas
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_win_streak",
    "away_win_streak",
    "h2h_home_win_rate",
    "h2h_draw_rate",
]
ZERO_STATS = {
    "points": 0.0,
    "goals_for": 0.0,
    "goals_against": 0.0,
    "shots": 0.0,
    "shots_on_target": 0.0,
    "corners": 0.0,
    "cards": 0.0,
    "yellow_cards": 0.0,
    "red_cards": 0.0,
    "win_rate": 0.0,
    "draw_rate": 0.0,
    "loss_rate": 0.0,
    "matches": 0,
}

CONTEXT_PROFILES = {
    "conservador": {
        "position_weight": 0.055,
        "ppg_weight": 0.07,
        "gdpg_weight": 0.04,
        "form_weight": 0.045,
        "elo_weight": 0.04,
        "streak_weight": 0.03,
        "table_shift_cap": 0.1,
        "injury_load_weight": 0.013,
        "injury_count_weight": 0.002,
        "injury_shift_cap": 0.1,
        "combined_shift_cap": 0.13,
    },
    "balanceado": {
        "position_weight": 0.07,
        "ppg_weight": 0.09,
        "gdpg_weight": 0.05,
        "form_weight": 0.06,
        "elo_weight": 0.05,
        "streak_weight": 0.04,
        "table_shift_cap": 0.14,
        "injury_load_weight": 0.018,
        "injury_count_weight": 0.003,
        "injury_shift_cap": 0.14,
        "combined_shift_cap": 0.18,
    },
    "agresivo": {
        "position_weight": 0.085,
        "ppg_weight": 0.11,
        "gdpg_weight": 0.06,
        "form_weight": 0.075,
        "elo_weight": 0.06,
        "streak_weight": 0.05,
        "table_shift_cap": 0.18,
        "injury_load_weight": 0.024,
        "injury_count_weight": 0.004,
        "injury_shift_cap": 0.18,
        "combined_shift_cap": 0.24,
    },
}

ACTIVE_CONTEXT_PROFILE = os.getenv("CONTEXT_PROFILE", "agresivo").strip().lower()
CONTEXT_TUNING = CONTEXT_PROFILES.get(ACTIVE_CONTEXT_PROFILE, CONTEXT_PROFILES["agresivo"])


class MatchPredictionService:
    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = SPAIN_CONFIG
        self._historical_csv: Path = config["historical_csv"]
        self._fixtures_csv: Path = config["fixtures_csv"]
        self._team_coords: dict = config["team_coords"]
        self._team_api_aliases: dict = config["team_api_aliases"]
        self._team_scrape_aliases: dict = config["team_scrape_aliases"]
        self._team_transfermarkt_ids: dict = config["team_transfermarkt_ids"]
        self._apifootball_league_id: int = config["apifootball_league_id"]
        self.league_name: str = config.get("league_name", "")
        self.league_logo_url: str = f"https://media.api-sports.io/football/leagues/{self._apifootball_league_id}.png"
        self.dataset_labels = {
            "historico": self._historical_csv.name,
            "encuentros": self._fixtures_csv.name,
        }
        self.historical_df = self._load_historical_data()
        self.fixtures_df = self._load_fixtures_data()
        self.label_encoder = LabelEncoder()
        self.best_model_name = ""
        self.validation_accuracy = 0.0
        self.validation_metrics: dict[str, float | str] = {
            "accuracy": 0.0,
            "brier": 0.0,
            "log_loss": 0.0,
            "calibration": "none",
        }
        self.model_scores: list[dict[str, float | str]] = []
        self.model = None
        self.calibrated_model = None
        self.totals_model = None
        self.totals_model_blend = float(os.getenv("TOTALS_MODEL_BLEND", "0.35"))
        self.btts_model = None
        self.btts_model_blend = float(os.getenv("BTTS_MODEL_BLEND", "0.3"))
        self.double_chance_models: dict[str, HistGradientBoostingClassifier] = {}
        self.double_chance_model_blend = float(os.getenv("DOUBLE_CHANCE_MODEL_BLEND", "0.25"))
        self.auto_blend_tuning = os.getenv("AUTO_BLEND_TUNING", "1").strip().lower() not in ("0", "false", "no")
        self.auto_threshold_tuning = os.getenv("AUTO_THRESHOLD_TUNING", "1").strip().lower() not in ("0", "false", "no")
        self.auto_threshold_min_coverage = float(os.getenv("AUTO_THRESHOLD_MIN_COVERAGE", "0.12"))
        self.no_bet_min_prob = float(os.getenv("NO_BET_MIN_PROB", "62"))
        self.market_min_prob = {
            "1X2": float(os.getenv("NO_BET_MIN_PROB_1X2", "62")),
            "Doble oportunidad": float(os.getenv("NO_BET_MIN_PROB_DOBLE", "70")),
            "Totales": float(os.getenv("NO_BET_MIN_PROB_TOTALES", "74")),
            "Ambos marcan": float(os.getenv("NO_BET_MIN_PROB_BTTS", "70")),
            "Corners": float(os.getenv("NO_BET_MIN_PROB_CORNERS", "72")),
            "Tarjetas amarillas": float(os.getenv("NO_BET_MIN_PROB_TARJETAS", "72")),
            "Tarjetas totales": float(os.getenv("NO_BET_MIN_PROB_TARJETAS", "72")),
        }
        self.team_snapshots: dict[str, dict[str, float]] = {}
        self.team_elo_snapshots: dict[str, float] = {}
        self.team_streak_snapshots: dict[str, int] = {}
        self.team_last_match_date_snapshot: dict[str, pd.Timestamp] = {}
        self.h2h_history_snapshot: dict[tuple, list[str]] = {}
        self.standings_snapshot: dict[str, dict[str, float | int | str]] = {}
        self._team_id_cache: dict[str, int] = {}
        self._transfermarkt_url_cache: dict[str, str | None] = {}
        self._injuries_cache: dict[tuple[str, str], dict[str, str | list[dict[str, str]]]] = {}
        self._injuries_cache_timestamp: dict[tuple[str, str], datetime] = {}
        self.league_home_goals = float(self.historical_df["FTHG"].mean())
        self.league_away_goals = float(self.historical_df["FTAG"].mean())
        self._train_model()

    def _load_historical_data(self) -> pd.DataFrame:
        df = pd.read_csv(self._historical_csv)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df["Time"] = df["Time"].fillna("00:00")
        numeric_columns = ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
        return df.sort_values(["Date", "Time", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    def _load_fixtures_data(self) -> pd.DataFrame:
        df = pd.read_csv(self._fixtures_csv)
        df["fecha_dt"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
        df["hora"] = df["hora"].fillna("00:00")
        df["match_key"] = (
            df["fecha"].fillna("")
            + "|"
            + df["hora"].fillna("")
            + "|"
            + df["local"].fillna("")
            + "|"
            + df["visitante"].fillna("")
        )
        return df.sort_values(["fecha_dt", "hora", "local", "visitante"]).reset_index(drop=True)

    def _recent_stats(self, team_history: dict[str, list[dict[str, float]]], team: str, window: int = 5) -> dict[str, float]:
        history = team_history.get(team, [])[-window:]
        if not history:
            return dict(ZERO_STATS)

        total_matches = len(history)
        return {
            "points": sum(item["points"] for item in history) / total_matches,
            "goals_for": sum(item["goals_for"] for item in history) / total_matches,
            "goals_against": sum(item["goals_against"] for item in history) / total_matches,
            "shots": sum(item["shots"] for item in history) / total_matches,
            "shots_on_target": sum(item["shots_on_target"] for item in history) / total_matches,
            "corners": sum(item["corners"] for item in history) / total_matches,
            "cards": sum(item["cards"] for item in history) / total_matches,
            "yellow_cards": sum(item["yellow_cards"] for item in history) / total_matches,
            "red_cards": sum(item["red_cards"] for item in history) / total_matches,
            "win_rate": sum(1 for item in history if item["points"] == 3) / total_matches,
            "draw_rate": sum(1 for item in history if item["points"] == 1) / total_matches,
            "loss_rate": sum(1 for item in history if item["points"] == 0) / total_matches,
            "matches": total_matches,
        }

    def _build_features(
        self,
        home_stats: dict[str, float],
        away_stats: dict[str, float],
        home_rest_days: float = 7.0,
        away_rest_days: float = 7.0,
        home_elo: float = 1500.0,
        away_elo: float = 1500.0,
        home_streak: int = 0,
        away_streak: int = 0,
        h2h_stats: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if h2h_stats is None:
            h2h_stats = {"home_win_rate": 0.33, "draw_rate": 0.33}
        return {
            "home_points_recent": home_stats["points"],
            "away_points_recent": away_stats["points"],
            "points_recent_diff": home_stats["points"] - away_stats["points"],
            "home_goals_for_recent": home_stats["goals_for"],
            "away_goals_for_recent": away_stats["goals_for"],
            "goals_for_diff": home_stats["goals_for"] - away_stats["goals_for"],
            "home_goals_against_recent": home_stats["goals_against"],
            "away_goals_against_recent": away_stats["goals_against"],
            "defensive_diff": away_stats["goals_against"] - home_stats["goals_against"],
            "home_shots_recent": home_stats["shots"],
            "away_shots_recent": away_stats["shots"],
            "home_sot_recent": home_stats["shots_on_target"],
            "away_sot_recent": away_stats["shots_on_target"],
            "home_win_rate": home_stats["win_rate"],
            "away_win_rate": away_stats["win_rate"],
            "home_draw_rate": home_stats["draw_rate"],
            "away_draw_rate": away_stats["draw_rate"],
            "cards_diff": home_stats["cards"] - away_stats["cards"],
            "home_rest_days": float(home_rest_days),
            "away_rest_days": float(away_rest_days),
            "rest_days_diff": float(home_rest_days - away_rest_days),
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "home_win_streak": float(home_streak),
            "away_win_streak": float(away_streak),
            "h2h_home_win_rate": h2h_stats["home_win_rate"],
            "h2h_draw_rate": h2h_stats["draw_rate"],
        }

    def _update_team_history(self, team_history: dict[str, list[dict[str, float]]], match: pd.Series) -> None:
        home_points = 3 if match["FTR"] == "H" else 1 if match["FTR"] == "D" else 0
        away_points = 3 if match["FTR"] == "A" else 1 if match["FTR"] == "D" else 0

        team_history.setdefault(match["HomeTeam"], []).append(
            {
                "points": home_points,
                "goals_for": float(match["FTHG"]),
                "goals_against": float(match["FTAG"]),
                "shots": float(match["HS"]),
                "shots_on_target": float(match["HST"]),
                "corners": float(match["HC"]),
                "cards": float(match["HY"] + match["HR"] * 1.5),
                "yellow_cards": float(match["HY"]),
                "red_cards": float(match["HR"]),
            }
        )
        team_history.setdefault(match["AwayTeam"], []).append(
            {
                "points": away_points,
                "goals_for": float(match["FTAG"]),
                "goals_against": float(match["FTHG"]),
                "shots": float(match["AS"]),
                "shots_on_target": float(match["AST"]),
                "corners": float(match["AC"]),
                "cards": float(match["AY"] + match["AR"] * 1.5),
                "yellow_cards": float(match["AY"]),
                "red_cards": float(match["AR"]),
            }
        )

    def _build_standings_snapshot(self) -> None:
        table: dict[str, dict[str, float | int | str]] = {}
        for _, row in self.historical_df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            for team in (home, away):
                if team not in table:
                    table[team] = {
                        "team": team,
                        "played": 0,
                        "points": 0,
                        "gf": 0,
                        "ga": 0,
                    }

            table[home]["played"] += 1
            table[away]["played"] += 1
            table[home]["gf"] += int(row["FTHG"])
            table[home]["ga"] += int(row["FTAG"])
            table[away]["gf"] += int(row["FTAG"])
            table[away]["ga"] += int(row["FTHG"])

            if row["FTR"] == "H":
                table[home]["points"] += 3
            elif row["FTR"] == "A":
                table[away]["points"] += 3
            else:
                table[home]["points"] += 1
                table[away]["points"] += 1

        sorted_teams = sorted(
            table.values(),
            key=lambda item: (item["points"], item["gf"] - item["ga"], item["gf"]),
            reverse=True,
        )
        snapshot: dict[str, dict[str, float | int | str]] = {}
        for index, team_data in enumerate(sorted_teams, start=1):
            snapshot[team_data["team"]] = {
                **team_data,
                "position": index,
                "gd": team_data["gf"] - team_data["ga"],
            }
        self.standings_snapshot = snapshot

    def _weather_context(self, team_name: str, date_str: str, hour_str: str) -> dict[str, str | float]:
        coords = self._team_coords.get(team_name)
        if not coords:
            return {"status": "No disponible", "reason": "Sin coordenadas del estadio"}

        lat, lon = coords
        try:
            date_obj = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
            if pd.isna(date_obj):
                return {"status": "No disponible", "reason": "Fecha invalida"}

            hour = int(str(hour_str).split(":")[0]) if ":" in str(hour_str) else 12
            url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation_probability,windspeed_10m"
                f"&start_date={date_obj.strftime('%Y-%m-%d')}&end_date={date_obj.strftime('%Y-%m-%d')}&timezone=Europe%2FMadrid"
            )
            data = requests.get(url, timeout=8).json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            target_prefix = f"{date_obj.strftime('%Y-%m-%d')}T{hour:02d}:"
            idx = next((i for i, t in enumerate(times) if t.startswith(target_prefix)), None)
            if idx is None:
                return {"status": "No disponible", "reason": "Fuera de rango pronosticable"}

            temp = hourly.get("temperature_2m", [None])[idx]
            rain = hourly.get("precipitation_probability", [None])[idx]
            wind = hourly.get("windspeed_10m", [None])[idx]
            return {
                "status": "Disponible",
                "temperature": float(temp) if temp is not None else 0.0,
                "rain_probability": float(rain) if rain is not None else 0.0,
                "wind_speed": float(wind) if wind is not None else 0.0,
            }
        except Exception:
            return {"status": "No disponible", "reason": "Error consultando clima"}

    def _api_headers(self) -> dict[str, str]:
        api_key = os.getenv("APIFOOTBALL_API_KEY", "").strip() or _read_env_value("APIFOOTBALL_API_KEY")
        if not api_key:
            return {}
        return {"x-apisports-key": api_key}

    def _browser_headers(self) -> dict[str, str]:
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
        }

    def _api_get(self, path: str, params: dict[str, str | int]) -> dict:
        headers = self._api_headers()
        if not headers:
            return {}
        try:
            response = requests.get(
                f"{APIFOOTBALL_BASE}{path}",
                headers=headers,
                params=params,
                timeout=10,
            )
            if response.status_code != 200:
                return {}
            return response.json() if response.content else {}
        except Exception:
            return {}

    def _season_from_date(self, date_str: str) -> int:
        date_obj = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
        if pd.isna(date_obj):
            return pd.Timestamp.now().year
        return date_obj.year if date_obj.month >= 7 else date_obj.year - 1

    def _team_api_name(self, team_name: str) -> str:
        return self._team_api_aliases.get(team_name, team_name)

    def _team_id_from_api(self, team_name: str) -> int | None:
        if team_name in self._team_id_cache:
            return self._team_id_cache[team_name]

        season = pd.Timestamp.now().year
        search_name = self._team_api_name(team_name)
        payload = self._api_get(
            "/teams",
            {
                "search": search_name,
                "league": self._apifootball_league_id,
                "season": season,
            },
        )
        for item in payload.get("response", []):
            team = item.get("team", {})
            team_id = team.get("id")
            if isinstance(team_id, int):
                self._team_id_cache[team_name] = team_id
                return team_id
        return None

    def _team_search_name(self, team_name: str) -> str:
        return self._team_scrape_aliases.get(team_name, self._team_api_aliases.get(team_name, team_name))

    def _transfermarkt_injuries_url(self, team_name: str) -> str | None:
        if team_name in self._transfermarkt_url_cache:
            return self._transfermarkt_url_cache[team_name]

        team_id = self._team_transfermarkt_ids.get(team_name)
        if isinstance(team_id, int):
            url = f"{TRANSFERMARKT_BASE}/x/sperrenundverletzungen/verein/{team_id}"
            self._transfermarkt_url_cache[team_name] = url
            return url

        query = self._team_search_name(team_name)
        try:
            response = requests.get(
                f"{TRANSFERMARKT_BASE}/schnellsuche/ergebnis/schnellsuche",
                params={"query": query},
                headers=self._browser_headers(),
                timeout=(3, 4),
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.select("a[href*='/startseite/verein/']"):
                href = str(link.get("href", ""))
                if not href:
                    continue
                base_href = href.split("?", 1)[0]
                parts = base_href.strip("/").split("/")
                if len(parts) < 4 or parts[-2] != "verein":
                    continue
                team_id = parts[-1]
                url = f"{TRANSFERMARKT_BASE}/x/sperrenundverletzungen/verein/{team_id}"
                self._transfermarkt_url_cache[team_name] = url
                return url
        except Exception:
            pass

        self._transfermarkt_url_cache[team_name] = None
        return None

    def _normalize_return_value(self, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized or normalized in {"0", "0.0", "-", "--", "N/A", "n/a"}:
            return "No confirmado"
        return normalized

    def _is_plausible_absence_item(self, player: str, reason: str, until: str) -> bool:
        player_clean = str(player or "").strip()
        reason_clean = str(reason or "").strip()
        until_clean = self._normalize_return_value(until)

        if len(player_clean) < 4 or player_clean.isdigit():
            return False
        if len(reason_clean) < 4:
            return False
        if re.fullmatch(r"\d{1,2}", until_clean):
            return False
        if player_clean.lower() in {"unknown", "player", "jugador"}:
            return False
        return True

    def _transfermarkt_players_context(self, home: str, away: str) -> dict[str, str | list[dict[str, str]]]:
        """
        Fetch injury info from Transfermarkt with intelligent caching and timeout handling.
        Returns cached data if available and not expired (6 hour TTL).
        """
        cache_key = (home, away)
        now = datetime.now()
        cache_ttl = timedelta(hours=6)
        
        # Check in-memory cache with timestamp
        if cache_key in self._injuries_cache:
            cached_time = self._injuries_cache_timestamp.get(cache_key)
            if cached_time and (now - cached_time) < cache_ttl:
                return self._injuries_cache[cache_key]
        
        injuries: list[dict[str, str]] = []
        for team in [home, away]:
            injuries_url = self._transfermarkt_injuries_url(team)
            if not injuries_url:
                continue
            try:
                # Timeout: 10s connect, 12s read (total ~12s per request, 2 teams = ~24s max)
                response = requests.get(
                    injuries_url, 
                    headers=self._browser_headers(), 
                    timeout=(10, 12)
                )
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                valid_tables = []
                for table in soup.select("table.items"):
                    table_text = table.get_text(" | ", strip=True).lower()
                    if "reason" in table_text and "expected return" in table_text:
                        valid_tables.append(table)
                if not valid_tables:
                    continue
                for table in valid_tables[:1]:
                    for row in table.select("tbody tr.odd, tbody tr.even")[:8]:
                        parts = [part.strip() for part in row.get_text(" | ", strip=True).split(" | ") if part.strip()]
                        if len(parts) < 6:
                            continue
                        market_value_text = next((part for part in parts if "€" in part), "")
                        market_value_million = self._parse_market_value_million(market_value_text)
                        importance_score = self._injury_importance_score(
                            reason=parts[3],
                            market_value_million=market_value_million,
                        )
                        until_value = self._normalize_return_value(parts[5] if len(parts) > 5 else "")
                        if not self._is_plausible_absence_item(parts[0], parts[3], until_value):
                            continue
                        injuries.append(
                            {
                                "team": team,
                                "player": parts[0],
                                "reason": parts[3],
                                "until": until_value,
                                "market_value": market_value_text,
                                "importance_score": str(round(importance_score, 2)),
                            }
                        )
            except (requests.Timeout, requests.ConnectionError):
                # Network timeout: skip this team, use previously cached data if available
                if cache_key in self._injuries_cache:
                    return self._injuries_cache[cache_key]
                continue
            except requests.RequestException:
                # Other request errors (403, 404, etc.): skip
                continue
            except Exception:
                # Any parsing or other errors: skip
                continue

        result = (
            {
                "status": "Disponible",
                "note": f"{len(injuries)} bajas detectadas via scraping publico.",
                "source": "Transfermarkt",
                "items": injuries,
            }
            if injuries
            else {
                "status": "Disponible",
                "note": "No hay bajas reportadas.",
                "source": "Transfermarkt",
                "items": [],
            }
        )
        
        # Store in cache with timestamp
        self._injuries_cache[cache_key] = result
        self._injuries_cache_timestamp[cache_key] = now
        return result
        return {
            "status": "No disponible",
            "note": "No se pudieron obtener bajas desde scraping publico para estos equipos.",
            "source": "Transfermarkt",
            "items": [],
        }

    def _players_context(self, home: str, away: str, date_str: str) -> dict[str, str | list[dict[str, str]]]:
        from datetime import timedelta
        from django.utils import timezone as tz
        from predictor.models import InjuryCache

        cache_key = (home, away, date_str)
        if cache_key in self._injuries_cache:
            return self._injuries_cache[cache_key]

        # Cache persistente en DB (valida 6 horas)
        try:
            cached = InjuryCache.objects.get(home_team=home, away_team=away)
            if tz.now() - cached.fetched_at < timedelta(hours=6):
                self._injuries_cache[cache_key] = cached.data
                return cached.data
        except Exception:
            pass

        if not self._api_headers():
            result = self._transfermarkt_players_context(home, away)
        else:
            season = self._season_from_date(date_str)
            injuries: list[dict[str, str]] = []
            for team in [home, away]:
                team_id = self._team_id_from_api(team)
                if team_id is None:
                    continue
                payload = self._api_get(
                    "/injuries",
                    {
                        "team": team_id,
                        "league": self._apifootball_league_id,
                        "season": season,
                    },
                )
                for item in payload.get("response", [])[:8]:
                    player = item.get("player", {})
                    fixture = item.get("fixture", {})
                    details = item.get("player", {}).get("reason", "Sin detalle")
                    until_value = self._normalize_return_value(str(fixture.get("date", ""))[:10])
                    player_name = str(player.get("name", "Jugador"))
                    if not self._is_plausible_absence_item(player_name, str(details), until_value):
                        continue
                    injuries.append(
                        {
                            "team": team,
                            "player": player_name,
                            "reason": str(details),
                            "until": until_value,
                            "importance_score": "1.0",
                        }
                    )

            if injuries:
                result = {
                    "status": "Disponible",
                    "note": f"{len(injuries)} bajas detectadas desde API externa.",
                    "source": "API-Football",
                    "items": injuries,
                }
            else:
                result = self._transfermarkt_players_context(home, away)

        # Guardar en DB para proximas consultas
        try:
            InjuryCache.objects.update_or_create(
                home_team=home,
                away_team=away,
                defaults={
                    "source": result.get("source", ""),
                    "data": result,
                    "fetched_at": tz.now(),
                },
            )
        except Exception:
            pass

        self._injuries_cache[cache_key] = result
        return result

    def _parse_market_value_million(self, value_text: str) -> float:
        text = str(value_text or "").strip().lower().replace(" ", "")
        if not text or "€" not in text:
            return 0.0

        match = re.search(r"([\d\.,]+)([mk])", text)
        if not match:
            return 0.0

        raw_number = match.group(1)
        suffix = match.group(2)

        normalized = raw_number
        if "," in normalized and "." in normalized:
            if normalized.rfind(",") > normalized.rfind("."):
                normalized = normalized.replace(".", "").replace(",", ".")
            else:
                normalized = normalized.replace(",", "")
        elif "," in normalized:
            normalized = normalized.replace(",", ".")

        try:
            base = float(normalized)
        except ValueError:
            return 0.0

        return base if suffix == "m" else base / 1000.0

    def _injury_importance_score(self, reason: str, market_value_million: float) -> float:
        score = 1.0
        if market_value_million > 0:
            # Jugadores con mayor valor de mercado suelen tener mas impacto competitivo.
            score += min(1.8, market_value_million / 20.0)

        reason_l = str(reason).lower()
        severe_tokens = (
            "cruciate",
            "ligament",
            "meniscus",
            "achilles",
            "fracture",
            "surgery",
            "tendon",
        )
        if any(token in reason_l for token in severe_tokens):
            score += 0.35
        elif "muscle" in reason_l or "hamstring" in reason_l:
            score += 0.18
        elif "suspension" in reason_l:
            score += 0.1

        return max(0.7, min(score, 3.5))

    def _injury_item_weight(self, item: dict[str, str]) -> float:
        raw_score = item.get("importance_score", 1.0)
        try:
            score = float(str(raw_score))
        except (ValueError, TypeError):
            score = 1.0

        # Reforzar por razon, incluso cuando no hay valor de mercado.
        reason_l = str(item.get("reason", "")).lower()
        if any(token in reason_l for token in ("cruciate", "ligament", "fracture", "achilles", "surgery")):
            score += 0.2
        return max(0.7, min(score, 3.5))

    def _injury_counts(self, players_status: dict[str, str | list[dict[str, str]]], home: str, away: str) -> tuple[int, int, float, float]:
        items = players_status.get("items", []) if isinstance(players_status, dict) else []
        if not isinstance(items, list):
            return 0, 0, 0.0, 0.0

        home_count = 0
        away_count = 0
        home_load = 0.0
        away_load = 0.0
        for item in items:
            if not isinstance(item, dict):
                continue
            team = str(item.get("team", "")).strip()
            weight = self._injury_item_weight(item)
            if team == home:
                home_count += 1
                home_load += weight
            elif team == away:
                away_count += 1
                away_load += weight
        return home_count, away_count, round(home_load, 3), round(away_load, 3)

    def _group_players_by_team(
        self,
        players_status: dict[str, str | list[dict[str, str]]],
        home: str,
        away: str,
    ) -> dict[str, dict[str, list[dict[str, str]]]]:
        grouped = {
            "home": {"injuries": [], "suspensions": []},
            "away": {"injuries": [], "suspensions": []},
        }
        items = players_status.get("items", []) if isinstance(players_status, dict) else []
        if not isinstance(items, list):
            return grouped

        for item in items:
            if not isinstance(item, dict):
                continue
            reason = str(item.get("reason", "Sin detalle"))
            reason_l = reason.lower()
            category = "suspensions" if any(token in reason_l for token in ("susp", "suspension", "ban", "red card", "yellow card")) else "injuries"
            try:
                importance_score = float(str(item.get("importance_score", "1.0")))
            except (TypeError, ValueError):
                importance_score = 1.0

            if importance_score >= 2.5:
                importance_tier = "critica"
            elif importance_score >= 1.6:
                importance_tier = "alta"
            else:
                importance_tier = "media"

            normalized = {
                "player": str(item.get("player", "Jugador")),
                "reason": reason,
                "until": str(item.get("until", "N/D")),
                "market_value": str(item.get("market_value", "")),
                "importance_score": f"{importance_score:.2f}",
                "importance_tier": importance_tier,
            }
            team = str(item.get("team", "")).strip()
            if team == home:
                grouped["home"][category].append(normalized)
            elif team == away:
                grouped["away"][category].append(normalized)

        for side in ("home", "away"):
            for category in ("injuries", "suspensions"):
                grouped[side][category].sort(
                    key=lambda item: float(item.get("importance_score", "0") or 0),
                    reverse=True,
                )
        return grouped

    def _players_source_reliability(self, players_status: dict[str, str | list[dict[str, str]]]) -> tuple[float, float]:
        source = str(players_status.get("source", "")).strip().lower() if isinstance(players_status, dict) else ""
        items = players_status.get("items", []) if isinstance(players_status, dict) else []
        item_count = len(items) if isinstance(items, list) else 0

        if "api-football" in source:
            base_score = 0.92
        elif "transfermarkt" in source:
            base_score = 0.68
        elif source:
            base_score = 0.6
        else:
            base_score = 0.45

        completeness_bonus = min(0.08, item_count * 0.01)
        reliability_score = min(1.0, base_score + completeness_bonus)
        source_weight = 0.55 + (reliability_score * 0.45)
        return round(reliability_score, 3), round(source_weight, 3)

    def _context_weather_severity(self, weather: dict[str, str | float]) -> float:
        if str(weather.get("status", "")) != "Disponible":
            return 0.0
        rain = float(weather.get("rain_probability") or 0.0)
        wind = float(weather.get("wind_speed") or 0.0)
        temp = float(weather.get("temperature") or 0.0)

        rain_term = min(max(rain, 0.0), 100.0) / 100.0 * 0.08
        wind_term = min(max(wind - 18.0, 0.0), 25.0) / 25.0 * 0.04
        cold_term = min(max(10.0 - temp, 0.0), 15.0) / 15.0 * 0.03
        return min(0.14, rain_term + wind_term + cold_term)

    def _table_context_shift(
        self,
        home_table: dict[str, float | int | str],
        away_table: dict[str, float | int | str],
        home_stats: dict[str, float],
        away_stats: dict[str, float],
        home_elo: float,
        away_elo: float,
        home_streak: int,
        away_streak: int,
    ) -> tuple[float, dict[str, float]]:
        n_teams = max(len(self.standings_snapshot), 2)

        def _to_float(value: float | int | str | None, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        home_pos = _to_float(home_table.get("position"), (n_teams + 1) / 2)
        away_pos = _to_float(away_table.get("position"), (n_teams + 1) / 2)
        home_points = _to_float(home_table.get("points"), 0.0)
        away_points = _to_float(away_table.get("points"), 0.0)
        home_played = max(1.0, _to_float(home_table.get("played"), 1.0))
        away_played = max(1.0, _to_float(away_table.get("played"), 1.0))
        home_gf = _to_float(home_table.get("gf"), 0.0)
        away_gf = _to_float(away_table.get("gf"), 0.0)
        home_ga = _to_float(home_table.get("ga"), 0.0)
        away_ga = _to_float(away_table.get("ga"), 0.0)

        pos_gap = (away_pos - home_pos) / max(1.0, (n_teams - 1))
        home_ppg = home_points / home_played
        away_ppg = away_points / away_played
        ppg_gap = (home_ppg - away_ppg) / 3.0
        home_gdpg = (home_gf - home_ga) / home_played
        away_gdpg = (away_gf - away_ga) / away_played
        gdpg_gap = (home_gdpg - away_gdpg) / 2.5
        recent_points_gap = (home_stats["points"] - away_stats["points"]) / 3.0
        elo_gap = (home_elo - away_elo) / 400.0
        streak_gap = (home_streak - away_streak) / 6.0

        pos_shift = max(-0.07, min(0.07, pos_gap * CONTEXT_TUNING["position_weight"]))
        ppg_shift = max(-0.06, min(0.06, ppg_gap * CONTEXT_TUNING["ppg_weight"]))
        gd_shift = max(-0.05, min(0.05, gdpg_gap * CONTEXT_TUNING["gdpg_weight"]))
        form_shift = max(-0.05, min(0.05, recent_points_gap * CONTEXT_TUNING["form_weight"]))
        elo_shift = max(-0.05, min(0.05, elo_gap * CONTEXT_TUNING["elo_weight"]))
        streak_shift = max(-0.04, min(0.04, streak_gap * CONTEXT_TUNING["streak_weight"]))

        total_shift = pos_shift + ppg_shift + gd_shift + form_shift + elo_shift + streak_shift
        total_shift = max(-CONTEXT_TUNING["table_shift_cap"], min(CONTEXT_TUNING["table_shift_cap"], total_shift))

        return total_shift, {
            "table_pos_shift": round(pos_shift, 4),
            "table_ppg_shift": round(ppg_shift, 4),
            "table_gd_shift": round(gd_shift, 4),
            "form_shift": round(form_shift, 4),
            "elo_shift": round(elo_shift, 4),
            "streak_shift": round(streak_shift, 4),
            "table_total_shift": round(total_shift, 4),
        }

    def _apply_context_adjustments(
        self,
        class_map: dict[str, float],
        weather: dict[str, str | float],
        players_status: dict[str, str | list[dict[str, str]]],
        home: str,
        away: str,
        home_table: dict[str, float | int | str],
        away_table: dict[str, float | int | str],
        home_stats: dict[str, float],
        away_stats: dict[str, float],
        home_elo: float,
        away_elo: float,
        home_streak: int,
        away_streak: int,
    ) -> tuple[dict[str, float], dict[str, float | int]]:
        p_home = float(class_map.get("H", 0.0))
        p_draw = float(class_map.get("D", 0.0))
        p_away = float(class_map.get("A", 0.0))

        weather_severity = self._context_weather_severity(weather)
        home_injuries, away_injuries, home_injury_load, away_injury_load = self._injury_counts(players_status, home, away)
        injury_reliability, injury_source_weight = self._players_source_reliability(players_status)
        table_shift, table_impact = self._table_context_shift(
            home_table,
            away_table,
            home_stats,
            away_stats,
            home_elo,
            away_elo,
            home_streak,
            away_streak,
        )
        injury_shift = (away_injury_load - home_injury_load) * CONTEXT_TUNING["injury_load_weight"]
        injury_shift += (away_injuries - home_injuries) * CONTEXT_TUNING["injury_count_weight"]
        injury_shift *= injury_source_weight
        injury_shift = max(-CONTEXT_TUNING["injury_shift_cap"], min(CONTEXT_TUNING["injury_shift_cap"], injury_shift))
        combined_shift = max(-CONTEXT_TUNING["combined_shift_cap"], min(CONTEXT_TUNING["combined_shift_cap"], injury_shift + table_shift))

        # Clima adverso: tiende a subir empate y bajar extremos.
        home_weather_factor = 1.0 - (weather_severity * 0.25)
        away_weather_factor = 1.0 - (weather_severity * 0.25)
        draw_weather_factor = 1.0 + weather_severity

        # Mas bajas de un equipo reducen su probabilidad de victoria.
        home_injury_factor = 1.0 + combined_shift
        away_injury_factor = 1.0 - combined_shift
        draw_injury_factor = 1.0 - (abs(combined_shift) * 0.25)

        raw_home = max(0.001, p_home * home_weather_factor * home_injury_factor)
        raw_draw = max(0.001, p_draw * draw_weather_factor * draw_injury_factor)
        raw_away = max(0.001, p_away * away_weather_factor * away_injury_factor)
        total = raw_home + raw_draw + raw_away

        adjusted = {
            "H": raw_home / total,
            "D": raw_draw / total,
            "A": raw_away / total,
        }
        impact = {
            "weather_severity": round(weather_severity, 4),
            "home_injuries": home_injuries,
            "away_injuries": away_injuries,
            "home_injury_load": round(home_injury_load, 3),
            "away_injury_load": round(away_injury_load, 3),
            "injury_source_reliability": injury_reliability,
            "injury_source_weight": injury_source_weight,
            "injury_shift": round(injury_shift, 4),
            "combined_shift": round(combined_shift, 4),
        }
        impact.update(table_impact)
        return adjusted, impact

    def _score_projection_from_expected(
        self,
        expected_home: float,
        expected_away: float,
    ) -> dict[str, float | str | list[dict[str, float | str]]]:
        expected_home = min(max(expected_home, 0.2), 4.0)
        expected_away = min(max(expected_away, 0.2), 4.0)

        score_probabilities = []
        for home_goals in range(6):
            for away_goals in range(6):
                home_prob = exp(-expected_home) * (expected_home ** home_goals) / factorial(home_goals)
                away_prob = exp(-expected_away) * (expected_away ** away_goals) / factorial(away_goals)
                score_probabilities.append(
                    {
                        "score": f"{home_goals}-{away_goals}",
                        "probability": round(home_prob * away_prob * 100, 2),
                    }
                )
        score_probabilities.sort(key=lambda item: item["probability"], reverse=True)
        return {
            "expected_home": round(expected_home, 2),
            "expected_away": round(expected_away, 2),
            "total_expected": round(expected_home + expected_away, 2),
            "best_score": score_probabilities[0]["score"],
            "top_scores": score_probabilities[:3],
        }

    def _poisson_draw_probability(self, expected_home: float, expected_away: float, max_goals: int = 6) -> float:
        probability = 0.0
        for goals in range(max_goals + 1):
            home_prob = exp(-expected_home) * (expected_home ** goals) / factorial(goals)
            away_prob = exp(-expected_away) * (expected_away ** goals) / factorial(goals)
            probability += home_prob * away_prob
        return max(0.0, min(1.0, probability))

    def _stabilize_probabilities(
        self,
        class_map: dict[str, float],
        score_projection: dict[str, float | str | list[dict[str, float | str]]],
    ) -> dict[str, float]:
        p_home = float(class_map.get("H", 0.0))
        p_draw = float(class_map.get("D", 0.0))
        p_away = float(class_map.get("A", 0.0))
        expected_home = float(score_projection.get("expected_home", 1.0))
        expected_away = float(score_projection.get("expected_away", 1.0))
        total_expected = float(score_projection.get("total_expected", expected_home + expected_away))
        xg_gap = abs(expected_home - expected_away)

        if xg_gap <= 0.35 and total_expected <= 3.2:
            poisson_draw = self._poisson_draw_probability(expected_home, expected_away)
            min_draw = min(0.28, max(0.12, poisson_draw * 0.72))
            if p_draw < min_draw:
                remaining = max(0.001, 1.0 - min_draw)
                non_draw_total = max(0.001, p_home + p_away)
                p_home = (p_home / non_draw_total) * remaining
                p_away = (p_away / non_draw_total) * remaining
                p_draw = min_draw

        total = p_home + p_draw + p_away
        return {
            "H": p_home / total,
            "D": p_draw / total,
            "A": p_away / total,
        }

    def _build_context_explanation(self, impact: dict[str, float | int]) -> dict[str, str]:
        table_total = float(impact.get("table_total_shift", 0.0) or 0.0)
        injury_shift = float(impact.get("injury_shift", 0.0) or 0.0)
        weather_severity = float(impact.get("weather_severity", 0.0) or 0.0)
        elo_shift = float(impact.get("elo_shift", 0.0) or 0.0)
        form_shift = float(impact.get("form_shift", 0.0) or 0.0)

        drivers = {
            "tabla": abs(table_total),
            "lesiones": abs(injury_shift),
            "clima": abs(weather_severity),
            "elo": abs(elo_shift),
            "forma": abs(form_shift),
        }
        main_driver = max(drivers, key=drivers.get)

        explanations = {
            "tabla": "La clasificación y el rendimiento acumulado de la temporada empujan el pronóstico.",
            "lesiones": "Las bajas detectadas son el factor que más mueve este pronóstico.",
            "clima": "El clima esperado está condicionando el partido y moderando el modelo.",
            "elo": "La diferencia de fuerza ELO entre ambos equipos está influyendo en el pronóstico.",
            "forma": "La forma reciente de los equipos es el factor más relevante en este partido.",
        }
        return {
            "main_driver": main_driver,
            "summary": explanations[main_driver],
        }

    def _apply_context_to_score_projection(
        self,
        score_projection: dict[str, float | str | list[dict[str, float | str]]],
        weather: dict[str, str | float],
        players_status: dict[str, str | list[dict[str, str]]],
        home: str,
        away: str,
        table_shift: float,
    ) -> dict[str, float | str | list[dict[str, float | str]]]:
        expected_home = float(score_projection.get("expected_home", 1.0))
        expected_away = float(score_projection.get("expected_away", 1.0))

        weather_severity = self._context_weather_severity(weather)
        _, _, home_injury_load, away_injury_load = self._injury_counts(players_status, home, away)
        _, injury_source_weight = self._players_source_reliability(players_status)

        weather_goal_factor = max(0.82, 1.0 - (weather_severity * 0.9))
        home_injury_goal_factor = max(0.72, 1.0 - ((home_injury_load * 0.035) * injury_source_weight))
        away_injury_goal_factor = max(0.72, 1.0 - ((away_injury_load * 0.035) * injury_source_weight))
        home_table_goal_factor = max(0.86, min(1.14, 1.0 + (table_shift * 0.45)))
        away_table_goal_factor = max(0.86, min(1.14, 1.0 - (table_shift * 0.45)))

        adjusted_home = expected_home * weather_goal_factor * home_injury_goal_factor * home_table_goal_factor
        adjusted_away = expected_away * weather_goal_factor * away_injury_goal_factor * away_table_goal_factor

        return self._score_projection_from_expected(adjusted_home, adjusted_away)

    def _update_elo(self, elo_ratings: dict[str, float], match: pd.Series) -> None:
        K = 20
        home = match["HomeTeam"]
        away = match["AwayTeam"]
        home_elo = elo_ratings.get(home, 1500.0)
        away_elo = elo_ratings.get(away, 1500.0)
        expected_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400.0))
        actual_home = 1.0 if match["FTR"] == "H" else 0.5 if match["FTR"] == "D" else 0.0
        elo_ratings[home] = home_elo + K * (actual_home - expected_home)
        elo_ratings[away] = away_elo + K * ((1.0 - actual_home) - (1.0 - expected_home))

    def _update_streak(self, win_streaks: dict[str, int], match: pd.Series) -> None:
        home = match["HomeTeam"]
        away = match["AwayTeam"]
        if match["FTR"] == "H":
            win_streaks[home] = max(win_streaks.get(home, 0), 0) + 1
            win_streaks[away] = min(win_streaks.get(away, 0), 0) - 1
        elif match["FTR"] == "A":
            win_streaks[away] = max(win_streaks.get(away, 0), 0) + 1
            win_streaks[home] = min(win_streaks.get(home, 0), 0) - 1
        else:
            win_streaks[home] = 0
            win_streaks[away] = 0

    def _update_h2h(self, h2h_history: dict[tuple, list[str]], match: pd.Series) -> None:
        home = match["HomeTeam"]
        away = match["AwayTeam"]
        key = (min(home, away), max(home, away))
        outcome = match["FTR"]
        if home != key[0]:  # Normalizar: 'H' = victoria del equipo key[0]
            outcome = "A" if outcome == "H" else ("H" if outcome == "A" else "D")
        h2h_history.setdefault(key, []).append(outcome)

    def _h2h_stats(self, h2h_history: dict[tuple, list[str]], home: str, away: str, window: int = 5) -> dict[str, float]:
        key = (min(home, away), max(home, away))
        history = h2h_history.get(key, [])[-window:]
        if not history:
            return {"home_win_rate": 0.33, "draw_rate": 0.33}
        total = len(history)
        if home == key[0]:
            wins = sum(1 for o in history if o == "H")
        else:
            wins = sum(1 for o in history if o == "A")
        draws = sum(1 for o in history if o == "D")
        return {"home_win_rate": wins / total, "draw_rate": draws / total}

    def _build_training_frame(self) -> pd.DataFrame:
        team_history: dict[str, list[dict[str, float]]] = {}
        elo_ratings: dict[str, float] = {}
        win_streaks: dict[str, int] = {}
        team_last_played: dict[str, pd.Timestamp] = {}
        h2h_history: dict[tuple, list[str]] = {}
        rows: list[dict[str, float | str]] = []
        for _, match in self.historical_df.iterrows():
            home = match["HomeTeam"]
            away = match["AwayTeam"]
            match_date = pd.to_datetime(match["Date"], errors="coerce")

            default_rest = 7.0
            home_last_date = team_last_played.get(home)
            away_last_date = team_last_played.get(away)
            home_rest_days = default_rest
            away_rest_days = default_rest
            if home_last_date is not None and not pd.isna(match_date):
                home_rest_days = max(1.0, float((match_date - home_last_date).days))
            if away_last_date is not None and not pd.isna(match_date):
                away_rest_days = max(1.0, float((match_date - away_last_date).days))

            home_stats = self._recent_stats(team_history, home)
            away_stats = self._recent_stats(team_history, away)
            home_elo = elo_ratings.get(home, 1500.0)
            away_elo = elo_ratings.get(away, 1500.0)
            home_streak = win_streaks.get(home, 0)
            away_streak = win_streaks.get(away, 0)
            h2h_stats = self._h2h_stats(h2h_history, home, away)
            feature_row = self._build_features(
                home_stats,
                away_stats,
                home_rest_days,
                away_rest_days,
                home_elo,
                away_elo,
                home_streak,
                away_streak,
                h2h_stats,
            )
            feature_row["target"] = match["FTR"]
            total_goals = int(float(match.get("FTHG", 0)) + float(match.get("FTAG", 0)))
            feature_row["target_total_goals_cap"] = min(total_goals, 5)
            feature_row["target_btts"] = 1 if (float(match.get("FTHG", 0)) > 0 and float(match.get("FTAG", 0)) > 0) else 0
            feature_row["target_dc_uno_x"] = 1 if str(match.get("FTR", "")) in ("H", "D") else 0
            feature_row["target_dc_uno_dos"] = 1 if str(match.get("FTR", "")) in ("H", "A") else 0
            feature_row["target_dc_x_dos"] = 1 if str(match.get("FTR", "")) in ("D", "A") else 0
            rows.append(feature_row)
            self._update_team_history(team_history, match)
            self._update_elo(elo_ratings, match)
            self._update_streak(win_streaks, match)
            self._update_h2h(h2h_history, match)
            if not pd.isna(match_date):
                team_last_played[home] = match_date
                team_last_played[away] = match_date

        self.team_snapshots = {
            team: self._recent_stats(team_history, team)
            for team in sorted(team_history)
        }
        self.team_elo_snapshots = {team: elo_ratings.get(team, 1500.0) for team in sorted(team_history)}
        self.team_streak_snapshots = {team: win_streaks.get(team, 0) for team in sorted(team_history)}
        self.team_last_match_date_snapshot = {
            team: team_last_played[team]
            for team in sorted(team_last_played)
            if team_last_played.get(team) is not None
        }
        self.h2h_history_snapshot = h2h_history
        self._build_standings_snapshot()
        return pd.DataFrame(rows)

    def _rest_days_for_fixture(self, team_name: str, fixture_date: pd.Timestamp) -> float:
        last_played = self.team_last_match_date_snapshot.get(team_name)
        if last_played is None or pd.isna(fixture_date):
            return 7.0
        return max(1.0, float((fixture_date - last_played).days))

    def _build_model_factories(self):
        factories = {
            "Regresion logistica": lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=3000)),
                ]
            ),
            "Random forest": lambda: RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=42,
            ),
            "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
                random_state=42,
                max_depth=5,
            ),
        }
        if xgb is not None:
            factories["XGBoost"] = lambda: xgb.XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )
        return factories

    def _multiclass_brier_score(self, y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
        if len(y_true) == 0:
            return 0.0
        one_hot = np.zeros((len(y_true), num_classes), dtype=float)
        one_hot[np.arange(len(y_true)), y_true] = 1.0
        return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))

    def _blend_probability(self, base_prob: float, specialist_prob: float | None, blend_weight: float) -> float:
        if specialist_prob is None:
            return base_prob
        blend = min(0.8, max(0.0, blend_weight))
        mixed = ((1.0 - blend) * base_prob) + (blend * specialist_prob)
        return max(0.0, min(1.0, mixed))

    def _totals_goal_distribution(self, feature_frame: pd.DataFrame) -> dict[int, float] | None:
        if self.totals_model is None:
            return None
        try:
            probs = self.totals_model.predict_proba(feature_frame)[0]
            classes = self.totals_model.classes_
        except Exception:
            return None
        return {int(c): float(p) for c, p in zip(classes, probs)}

    def _btts_probability(self, feature_frame: pd.DataFrame) -> float | None:
        if self.btts_model is None:
            return None
        try:
            probs = self.btts_model.predict_proba(feature_frame)[0]
            classes = list(self.btts_model.classes_)
        except Exception:
            return None
        if 1 in classes:
            return float(probs[classes.index(1)])
        return None

    def _double_chance_probabilities(self, feature_frame: pd.DataFrame) -> dict[str, float] | None:
        if not self.double_chance_models:
            return None
        results: dict[str, float] = {}
        for key, model in self.double_chance_models.items():
            try:
                probs = model.predict_proba(feature_frame)[0]
                classes = list(model.classes_)
            except Exception:
                continue
            if 1 in classes:
                results[key] = float(probs[classes.index(1)])
        return results if results else None

    def _predict_probabilities(self, feature_frame: pd.DataFrame) -> dict[str, float]:
        estimator = self.calibrated_model if self.calibrated_model is not None else self.model
        probabilities = estimator.predict_proba(feature_frame)[0]
        return dict(zip(self.label_encoder.classes_, probabilities))

    def _train_model(self) -> None:
        training_df = self._build_training_frame()
        X = training_df[FEATURE_COLUMNS]
        y = self.label_encoder.fit_transform(training_df["target"])
        y_totals = training_df["target_total_goals_cap"].astype(int).to_numpy()
        y_btts = training_df["target_btts"].astype(int).to_numpy()
        y_dc_uno_x = training_df["target_dc_uno_x"].astype(int).to_numpy()
        y_dc_uno_dos = training_df["target_dc_uno_dos"].astype(int).to_numpy()
        y_dc_x_dos = training_df["target_dc_x_dos"].astype(int).to_numpy()
        split_index = max(20, int(len(X) * 0.8))
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]
        y_totals_train = y_totals[:split_index]
        y_totals_test = y_totals[split_index:]
        y_btts_train = y_btts[:split_index]
        y_btts_test = y_btts[split_index:]
        y_dc_uno_x_train = y_dc_uno_x[:split_index]
        y_dc_uno_x_test = y_dc_uno_x[split_index:]
        y_dc_uno_dos_train = y_dc_uno_dos[:split_index]
        y_dc_uno_dos_test = y_dc_uno_dos[split_index:]
        y_dc_x_dos_train = y_dc_x_dos[:split_index]
        y_dc_x_dos_test = y_dc_x_dos[split_index:]

        best_factory = None
        best_name = ""
        best_accuracy = -1.0
        self.model_scores = []
        for name, factory in self._build_model_factories().items():
            model = factory()
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
            self.model_scores.append({"name": name, "score": round(score * 100, 2)})
            if score > best_accuracy:
                best_accuracy = score
                best_name = name
                best_factory = factory

        self.best_model_name = best_name
        self.validation_accuracy = best_accuracy

        # Modelo principal: entrenado con todo el historico.
        self.model = best_factory()
        self.model.fit(X, y)

        # Calibracion holdout: mejora calidad de probabilidades para decidir stake/no-bet.
        calibration_label = "none"
        y_prob_eval = self.model.predict_proba(X_test)
        self.calibrated_model = None
        if len(X_test) >= 15 and len(set(y_test)) == len(set(y)):
            try:
                prefit_model = best_factory()
                prefit_model.fit(X_train, y_train)
                calibrator = CalibratedClassifierCV(prefit_model, cv="prefit", method="sigmoid")
                calibrator.fit(X_test, y_test)
                self.calibrated_model = calibrator
                y_prob_eval = calibrator.predict_proba(X_test)
                calibration_label = "sigmoid-holdout"
            except Exception:
                self.calibrated_model = None
                calibration_label = "none"

        eval_brier = self._multiclass_brier_score(y_test, y_prob_eval, len(self.label_encoder.classes_))
        try:
            eval_log_loss = float(log_loss(y_test, y_prob_eval, labels=list(range(len(self.label_encoder.classes_)))))
        except ValueError:
            eval_log_loss = 0.0

        # Baselines de mercado para comparar contra especialistas.
        totals_base_accuracy = 0.0
        btts_base_accuracy = 0.0
        dc_base_accuracy = 0.0
        totals_base_prob_rows: list[list[float]] = []
        btts_base_prob_yes: list[float] = []
        dc_base_prob_map: dict[str, np.ndarray] = {}
        if len(X_test):
            totals_true: list[int] = []
            totals_pred: list[int] = []
            btts_true: list[int] = []
            btts_pred: list[int] = []

            for row in X_test.to_dict(orient="records"):
                expected_home = max(
                    0.2,
                    0.55 * self.league_home_goals + 0.45 * ((float(row["home_goals_for_recent"]) + float(row["away_goals_against_recent"])) / 2),
                )
                expected_away = max(
                    0.2,
                    0.55 * self.league_away_goals + 0.45 * ((float(row["away_goals_for_recent"]) + float(row["home_goals_against_recent"])) / 2),
                )
                expected_home += 0.06 * (float(row["home_sot_recent"]) - float(row["away_sot_recent"]))
                expected_away += 0.04 * (float(row["away_sot_recent"]) - float(row["home_sot_recent"]))
                total_lambda = expected_home + expected_away

                p0 = exp(-total_lambda)
                p1 = p0 * total_lambda
                p2 = p1 * total_lambda / 2
                p3 = p2 * total_lambda / 3
                p4 = p3 * total_lambda / 4
                p5_plus = max(0.0, 1.0 - (p0 + p1 + p2 + p3 + p4))
                totals_dist = [p0, p1, p2, p3, p4, p5_plus]
                totals_base_prob_rows.append(totals_dist)
                totals_pred.append(int(max(range(6), key=lambda i: totals_dist[i])))

                p_home_zero = exp(-expected_home)
                p_away_zero = exp(-expected_away)
                p_btts_yes = 1 - p_home_zero - p_away_zero + (p_home_zero * p_away_zero)
                btts_base_prob_yes.append(float(p_btts_yes))
                btts_pred.append(1 if p_btts_yes >= 0.5 else 0)

            totals_true = list(y_totals_test.astype(int))
            btts_true = list(y_btts_test.astype(int))
            totals_base_accuracy = float(accuracy_score(totals_true, totals_pred)) if totals_true else 0.0
            btts_base_accuracy = float(accuracy_score(btts_true, btts_pred)) if btts_true else 0.0

            h_idx = list(self.label_encoder.classes_).index("H") if "H" in self.label_encoder.classes_ else None
            d_idx = list(self.label_encoder.classes_).index("D") if "D" in self.label_encoder.classes_ else None
            a_idx = list(self.label_encoder.classes_).index("A") if "A" in self.label_encoder.classes_ else None
            if h_idx is not None and d_idx is not None and a_idx is not None:
                p_home_arr = y_prob_eval[:, h_idx]
                p_draw_arr = y_prob_eval[:, d_idx]
                p_away_arr = y_prob_eval[:, a_idx]
                dc_base_prob_map = {
                    "uno_x": p_home_arr + p_draw_arr,
                    "uno_dos": p_home_arr + p_away_arr,
                    "x_dos": p_draw_arr + p_away_arr,
                }
                dc1x_pred = (p_home_arr + p_draw_arr >= 0.5).astype(int)
                dc12_pred = (p_home_arr + p_away_arr >= 0.5).astype(int)
                dcx2_pred = (p_draw_arr + p_away_arr >= 0.5).astype(int)
                dc_scores = [
                    float(accuracy_score(y_dc_uno_x_test.astype(int), dc1x_pred)),
                    float(accuracy_score(y_dc_uno_dos_test.astype(int), dc12_pred)),
                    float(accuracy_score(y_dc_x_dos_test.astype(int), dcx2_pred)),
                ]
                dc_base_accuracy = sum(dc_scores) / len(dc_scores)

        self.validation_metrics = {
            "accuracy": round(best_accuracy * 100, 2),
            "brier": round(eval_brier, 4),
            "log_loss": round(eval_log_loss, 4),
            "calibration": calibration_label,
            "totals_base_accuracy": round(totals_base_accuracy * 100, 2),
            "btts_base_accuracy": round(btts_base_accuracy * 100, 2),
            "double_chance_base_accuracy": round(dc_base_accuracy * 100, 2),
        }

        self.totals_model = None
        if len(X_train) >= 30 and len(set(y_totals_train)) >= 3:
            totals_model = HistGradientBoostingClassifier(random_state=42, max_depth=4)
            totals_model.fit(X_train, y_totals_train)
            self.totals_model = totals_model
            totals_score = float(accuracy_score(y_totals_test, totals_model.predict(X_test))) if len(X_test) else 0.0
            self.validation_metrics["totals_model_accuracy"] = round(totals_score * 100, 2)
            self.validation_metrics["totals_model_blend"] = round(self.totals_model_blend, 2)

        self.btts_model = None
        if len(X_train) >= 30 and len(set(y_btts_train)) >= 2:
            btts_model = HistGradientBoostingClassifier(random_state=42, max_depth=4)
            btts_model.fit(X_train, y_btts_train)
            self.btts_model = btts_model
            btts_score = float(accuracy_score(y_btts_test, btts_model.predict(X_test))) if len(X_test) else 0.0
            self.validation_metrics["btts_model_accuracy"] = round(btts_score * 100, 2)
            self.validation_metrics["btts_model_blend"] = round(self.btts_model_blend, 2)

        self.double_chance_models = {}
        dc_scores: list[float] = []
        dc_targets = {
            "uno_x": (y_dc_uno_x_train, y_dc_uno_x_test),
            "uno_dos": (y_dc_uno_dos_train, y_dc_uno_dos_test),
            "x_dos": (y_dc_x_dos_train, y_dc_x_dos_test),
        }
        for key, (y_train_dc, y_test_dc) in dc_targets.items():
            if len(X_train) < 30 or len(set(y_train_dc)) < 2:
                continue
            dc_model = HistGradientBoostingClassifier(random_state=42, max_depth=4)
            dc_model.fit(X_train, y_train_dc)
            self.double_chance_models[key] = dc_model
            dc_score = float(accuracy_score(y_test_dc, dc_model.predict(X_test))) if len(X_test) else 0.0
            dc_scores.append(dc_score)
        if dc_scores:
            self.validation_metrics["double_chance_model_accuracy"] = round((sum(dc_scores) / len(dc_scores)) * 100, 2)
            self.validation_metrics["double_chance_model_blend"] = round(self.double_chance_model_blend, 2)

        # Autoajuste de blends por mercado usando la ventana de validacion temporal.
        if self.auto_blend_tuning and len(X_test):
            candidate_blends = [i / 10 for i in range(0, 9)]

            if self.totals_model is not None and totals_base_prob_rows:
                totals_probs_raw = self.totals_model.predict_proba(X_test)
                totals_classes = [int(c) for c in self.totals_model.classes_]
                spec_rows: list[list[float]] = []
                for row in totals_probs_raw:
                    dist = [0.0] * 6
                    for idx, cls in enumerate(totals_classes):
                        dist[min(max(cls, 0), 5)] += float(row[idx])
                    spec_rows.append(dist)

                best_blend = self.totals_model_blend
                best_acc = -1.0
                for blend in candidate_blends:
                    mixed_pred: list[int] = []
                    for base_row, spec_row in zip(totals_base_prob_rows, spec_rows):
                        mixed = [(1.0 - blend) * b + blend * s for b, s in zip(base_row, spec_row)]
                        mixed_pred.append(int(max(range(6), key=lambda i: mixed[i])))
                    acc = float(accuracy_score(y_totals_test.astype(int), mixed_pred))
                    if acc > best_acc:
                        best_acc = acc
                        best_blend = blend
                self.totals_model_blend = best_blend
                self.validation_metrics["totals_model_blend"] = round(best_blend, 2)
                self.validation_metrics["totals_blend_auto_accuracy"] = round(best_acc * 100, 2)

            if self.btts_model is not None and btts_base_prob_yes:
                btts_probs_raw = self.btts_model.predict_proba(X_test)
                btts_classes = list(self.btts_model.classes_)
                if 1 in btts_classes:
                    one_idx = btts_classes.index(1)
                    spec_yes = [float(row[one_idx]) for row in btts_probs_raw]
                    best_blend = self.btts_model_blend
                    best_acc = -1.0
                    for blend in candidate_blends:
                        mixed_pred = [
                            1 if (((1.0 - blend) * base_p) + (blend * spec_p)) >= 0.5 else 0
                            for base_p, spec_p in zip(btts_base_prob_yes, spec_yes)
                        ]
                        acc = float(accuracy_score(y_btts_test.astype(int), mixed_pred))
                        if acc > best_acc:
                            best_acc = acc
                            best_blend = blend
                    self.btts_model_blend = best_blend
                    self.validation_metrics["btts_model_blend"] = round(best_blend, 2)
                    self.validation_metrics["btts_blend_auto_accuracy"] = round(best_acc * 100, 2)

            if self.double_chance_models and dc_base_prob_map:
                dc_spec_prob_map: dict[str, np.ndarray] = {}
                for key, model in self.double_chance_models.items():
                    probs_raw = model.predict_proba(X_test)
                    classes = list(model.classes_)
                    if 1 in classes:
                        dc_spec_prob_map[key] = probs_raw[:, classes.index(1)]

                if dc_spec_prob_map:
                    best_blend = self.double_chance_model_blend
                    best_acc = -1.0
                    dc_truth_map = {
                        "uno_x": y_dc_uno_x_test.astype(int),
                        "uno_dos": y_dc_uno_dos_test.astype(int),
                        "x_dos": y_dc_x_dos_test.astype(int),
                    }
                    for blend in candidate_blends:
                        local_scores: list[float] = []
                        for key in ("uno_x", "uno_dos", "x_dos"):
                            if key not in dc_base_prob_map or key not in dc_spec_prob_map:
                                continue
                            mixed = ((1.0 - blend) * dc_base_prob_map[key]) + (blend * dc_spec_prob_map[key])
                            pred = (mixed >= 0.5).astype(int)
                            local_scores.append(float(accuracy_score(dc_truth_map[key], pred)))
                        if local_scores:
                            acc = sum(local_scores) / len(local_scores)
                            if acc > best_acc:
                                best_acc = acc
                                best_blend = blend
                    self.double_chance_model_blend = best_blend
                    self.validation_metrics["double_chance_model_blend"] = round(best_blend, 2)
                    self.validation_metrics["double_chance_blend_auto_accuracy"] = round(best_acc * 100, 2)

        # Autoajuste de umbrales por mercado usando precision y cobertura.
        if self.auto_threshold_tuning and len(X_test):
            h_idx = list(self.label_encoder.classes_).index("H") if "H" in self.label_encoder.classes_ else None
            d_idx = list(self.label_encoder.classes_).index("D") if "D" in self.label_encoder.classes_ else None
            a_idx = list(self.label_encoder.classes_).index("A") if "A" in self.label_encoder.classes_ else None

            market_samples: dict[str, list[tuple[float, bool]]] = {
                "1X2": [],
                "Doble oportunidad": [],
                "Totales": [],
                "Ambos marcan": [],
            }

            if h_idx is not None and d_idx is not None and a_idx is not None:
                p_home_arr = y_prob_eval[:, h_idx]
                p_draw_arr = y_prob_eval[:, d_idx]
                p_away_arr = y_prob_eval[:, a_idx]
                outcomes = ["H", "D", "A"]
                for i, y_true_val in enumerate(y_test):
                    probs = [float(p_home_arr[i]), float(p_draw_arr[i]), float(p_away_arr[i])]
                    pred_idx = int(max(range(3), key=lambda j: probs[j]))
                    pred_outcome = outcomes[pred_idx]
                    true_outcome = str(self.label_encoder.classes_[int(y_true_val)])
                    market_samples["1X2"].append((probs[pred_idx] * 100.0, pred_outcome == true_outcome))

                dc_spec_map: dict[str, np.ndarray] = {}
                for key, model in self.double_chance_models.items():
                    probs_raw = model.predict_proba(X_test)
                    classes = list(model.classes_)
                    if 1 in classes:
                        dc_spec_map[key] = probs_raw[:, classes.index(1)]

                for i in range(len(X_test)):
                    dc_probs = {
                        "uno_x": float(p_home_arr[i] + p_draw_arr[i]),
                        "uno_dos": float(p_home_arr[i] + p_away_arr[i]),
                        "x_dos": float(p_draw_arr[i] + p_away_arr[i]),
                    }
                    for key in ("uno_x", "uno_dos", "x_dos"):
                        if key in dc_spec_map:
                            dc_probs[key] = self._blend_probability(dc_probs[key], float(dc_spec_map[key][i]), self.double_chance_model_blend)

                    best_dc_key = max(dc_probs, key=dc_probs.get)
                    true_outcome = str(self.label_encoder.classes_[int(y_test[i])])
                    truth = (
                        (best_dc_key == "uno_x" and true_outcome in ("H", "D"))
                        or (best_dc_key == "uno_dos" and true_outcome in ("H", "A"))
                        or (best_dc_key == "x_dos" and true_outcome in ("D", "A"))
                    )
                    market_samples["Doble oportunidad"].append((dc_probs[best_dc_key] * 100.0, truth))

            if totals_base_prob_rows:
                totals_spec_prob_rows: list[list[float]] = []
                if self.totals_model is not None:
                    totals_probs_raw = self.totals_model.predict_proba(X_test)
                    totals_classes = [int(c) for c in self.totals_model.classes_]
                    for row in totals_probs_raw:
                        dist = [0.0] * 6
                        for idx, cls in enumerate(totals_classes):
                            dist[min(max(cls, 0), 5)] += float(row[idx])
                        totals_spec_prob_rows.append(dist)

                for i, total_true in enumerate(y_totals_test.astype(int)):
                    base = totals_base_prob_rows[i]
                    spec = totals_spec_prob_rows[i] if i < len(totals_spec_prob_rows) else None
                    probs = {
                        "Over 1.5": 1.0 - (base[0] + base[1]),
                        "Over 2.5": 1.0 - (base[0] + base[1] + base[2]),
                        "Under 3.5": base[0] + base[1] + base[2] + base[3],
                        "Under 4.5": base[0] + base[1] + base[2] + base[3] + base[4],
                    }
                    if spec is not None:
                        spec_probs = {
                            "Over 1.5": 1.0 - (spec[0] + spec[1]),
                            "Over 2.5": 1.0 - (spec[0] + spec[1] + spec[2]),
                            "Under 3.5": spec[0] + spec[1] + spec[2] + spec[3],
                            "Under 4.5": spec[0] + spec[1] + spec[2] + spec[3] + spec[4],
                        }
                        for key in probs:
                            probs[key] = self._blend_probability(probs[key], spec_probs[key], self.totals_model_blend)

                    best_key = max(probs, key=probs.get)
                    truth = (
                        (best_key == "Over 1.5" and total_true >= 2)
                        or (best_key == "Over 2.5" and total_true >= 3)
                        or (best_key == "Under 3.5" and total_true <= 3)
                        or (best_key == "Under 4.5" and total_true <= 4)
                    )
                    market_samples["Totales"].append((probs[best_key] * 100.0, truth))

            if btts_base_prob_yes:
                btts_spec_yes: list[float] = []
                if self.btts_model is not None:
                    probs_raw = self.btts_model.predict_proba(X_test)
                    classes = list(self.btts_model.classes_)
                    if 1 in classes:
                        btts_spec_yes = [float(row[classes.index(1)]) for row in probs_raw]

                for i, y_true_btts in enumerate(y_btts_test.astype(int)):
                    p_yes = float(btts_base_prob_yes[i])
                    if i < len(btts_spec_yes):
                        p_yes = self._blend_probability(p_yes, btts_spec_yes[i], self.btts_model_blend)
                    p_no = 1.0 - p_yes
                    if p_yes >= p_no:
                        market_samples["Ambos marcan"].append((p_yes * 100.0, y_true_btts == 1))
                    else:
                        market_samples["Ambos marcan"].append((p_no * 100.0, y_true_btts == 0))

            threshold_candidates = list(range(55, 86))
            for market_name in ("1X2", "Doble oportunidad", "Totales", "Ambos marcan"):
                samples = market_samples.get(market_name, [])
                if not samples:
                    continue
                best_threshold = float(self.market_min_prob.get(market_name, self.no_bet_min_prob))
                best_score = -1.0
                best_precision = 0.0
                best_coverage = 0.0

                for th in threshold_candidates:
                    accepted = [ok for prob, ok in samples if prob >= th]
                    coverage = (len(accepted) / len(samples)) if samples else 0.0
                    if not accepted or coverage < self.auto_threshold_min_coverage:
                        continue
                    precision = sum(1 for ok in accepted if ok) / len(accepted)
                    score = precision * (0.5 + 0.5 * coverage)
                    if score > best_score:
                        best_score = score
                        best_threshold = float(th)
                        best_precision = precision
                        best_coverage = coverage

                self.market_min_prob[market_name] = best_threshold
                key_prefix = market_name.lower().replace(" ", "_")
                self.validation_metrics[f"{key_prefix}_threshold_auto"] = round(best_threshold, 2)
                if best_score >= 0:
                    self.validation_metrics[f"{key_prefix}_precision_auto"] = round(best_precision * 100, 2)
                    self.validation_metrics[f"{key_prefix}_coverage_auto"] = round(best_coverage * 100, 2)

    def _pending_mask(self) -> pd.Series:
        result_series = self.fixtures_df["resultado"].fillna("").astype(str).str.strip()
        # Solo devolver partidos sin resultado real (resultado = "-:-" o vacío)
        return result_series.eq("-:-") | result_series.eq("")

    def get_pending_fixtures(self) -> list[dict[str, str]]:
        pending = self.fixtures_df[self._pending_mask()].copy()
        return [
            {
                "match_key": row["match_key"],
                "label": f'{row["fecha"]} {row["hora"]} | {row["local"]} vs {row["visitante"]}',
                "jornada": row["jornada"],
                "fecha": row["fecha"],
                "hora": row["hora"],
                "local": row["local"],
                "visitante": row["visitante"],
                "local_logo": self._team_logo_url(str(row["local"])),
                "visitante_logo": self._team_logo_url(str(row["visitante"])),
            }
            for _, row in pending.iterrows()
        ]

    def _team_logo_url(self, team_name: str) -> str:
        team_id = self._team_transfermarkt_ids.get(team_name)
        if not team_id:
            return ""
        return f"https://tmssl.akamaized.net/images/wappen/head/{team_id}.png"

    def _team_stats(self, team_name: str) -> dict[str, float]:
        return self.team_snapshots.get(team_name, dict(ZERO_STATS))

    def _predict_scoreline(self, home_stats: dict[str, float], away_stats: dict[str, float]) -> dict[str, float | str | list[dict[str, float | str]]]:
        expected_home = max(
            0.2,
            0.55 * self.league_home_goals + 0.45 * ((home_stats["goals_for"] + away_stats["goals_against"]) / 2),
        )
        expected_away = max(
            0.2,
            0.55 * self.league_away_goals + 0.45 * ((away_stats["goals_for"] + home_stats["goals_against"]) / 2),
        )
        expected_home += 0.06 * (home_stats["shots_on_target"] - away_stats["shots_on_target"])
        expected_away += 0.04 * (away_stats["shots_on_target"] - home_stats["shots_on_target"])
        return self._score_projection_from_expected(expected_home, expected_away)

    def _market_probabilities(
        self,
        class_map: dict[str, float],
        score_projection: dict[str, float | str | list[dict[str, float | str]]],
        home_stats: dict[str, float],
        away_stats: dict[str, float],
        feature_frame: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        p_home = float(class_map.get("H", 0.0))
        p_draw = float(class_map.get("D", 0.0))
        p_away = float(class_map.get("A", 0.0))

        expected_home = float(score_projection["expected_home"])
        expected_away = float(score_projection["expected_away"])
        total_lambda = expected_home + expected_away

        p0 = exp(-total_lambda)
        p1 = p0 * total_lambda
        p2 = p1 * total_lambda / 2
        p3 = p2 * total_lambda / 3
        p4 = p3 * total_lambda / 4

        p_over_1_5 = 1 - (p0 + p1)
        p_over_2_5 = 1 - (p0 + p1 + p2)
        p_under_3_5 = p0 + p1 + p2 + p3
        p_under_4_5 = p0 + p1 + p2 + p3 + p4

        p_dc_uno_x = p_home + p_draw
        p_dc_uno_dos = p_home + p_away
        p_dc_x_dos = p_draw + p_away

        totals_dist = self._totals_goal_distribution(feature_frame) if feature_frame is not None else None
        if totals_dist is not None:
            model_over_1_5 = sum(p for g, p in totals_dist.items() if g >= 2)
            model_over_2_5 = sum(p for g, p in totals_dist.items() if g >= 3)
            model_under_3_5 = sum(p for g, p in totals_dist.items() if g <= 3)
            model_under_4_5 = sum(p for g, p in totals_dist.items() if g <= 4)
            p_over_1_5 = self._blend_probability(p_over_1_5, model_over_1_5, self.totals_model_blend)
            p_over_2_5 = self._blend_probability(p_over_2_5, model_over_2_5, self.totals_model_blend)
            p_under_3_5 = self._blend_probability(p_under_3_5, model_under_3_5, self.totals_model_blend)
            p_under_4_5 = self._blend_probability(p_under_4_5, model_under_4_5, self.totals_model_blend)

        dc_model_probs = self._double_chance_probabilities(feature_frame) if feature_frame is not None else None
        if dc_model_probs is not None:
            p_dc_uno_x = self._blend_probability(p_dc_uno_x, dc_model_probs.get("uno_x"), self.double_chance_model_blend)
            p_dc_uno_dos = self._blend_probability(p_dc_uno_dos, dc_model_probs.get("uno_dos"), self.double_chance_model_blend)
            p_dc_x_dos = self._blend_probability(p_dc_x_dos, dc_model_probs.get("x_dos"), self.double_chance_model_blend)

        p_home_zero = exp(-expected_home)
        p_away_zero = exp(-expected_away)
        p_btts_yes = 1 - p_home_zero - p_away_zero + (p_home_zero * p_away_zero)
        model_btts_yes = self._btts_probability(feature_frame) if feature_frame is not None else None
        p_btts_yes = self._blend_probability(p_btts_yes, model_btts_yes, self.btts_model_blend)

        expected_corners = max(3.0, home_stats["corners"] + away_stats["corners"])
        p_corners_under_8_5 = sum(
            exp(-expected_corners) * (expected_corners ** k) / factorial(k)
            for k in range(0, 9)
        )
        p_corners_over_8_5 = 1 - p_corners_under_8_5

        expected_yellow_cards = max(1.0, home_stats["yellow_cards"] + away_stats["yellow_cards"])
        p_yellow_under_3_5 = sum(
            exp(-expected_yellow_cards) * (expected_yellow_cards ** k) / factorial(k)
            for k in range(0, 4)
        )
        p_yellow_over_3_5 = 1 - p_yellow_under_3_5

        expected_red_cards = max(0.05, home_stats["red_cards"] + away_stats["red_cards"])
        p_red_under_0_5 = exp(-expected_red_cards)
        p_red_over_0_5 = 1 - p_red_under_0_5

        expected_total_cards = expected_yellow_cards + expected_red_cards
        p_total_cards_under_3_5 = sum(
            exp(-expected_total_cards) * (expected_total_cards ** k) / factorial(k)
            for k in range(0, 4)
        )
        p_total_cards_over_3_5 = 1 - p_total_cards_under_3_5
        p_total_cards_under_4_5 = sum(
            exp(-expected_total_cards) * (expected_total_cards ** k) / factorial(k)
            for k in range(0, 5)
        )
        p_total_cards_over_4_5 = 1 - p_total_cards_under_4_5

        return {
            "doble_oportunidad": {
                "uno_x": round(p_dc_uno_x * 100, 2),
                "uno_dos": round(p_dc_uno_dos * 100, 2),
                "x_dos": round(p_dc_x_dos * 100, 2),
            },
            "btts": {
                "si": round(p_btts_yes * 100, 2),
                "no": round((1 - p_btts_yes) * 100, 2),
            },
            "corners_8_5": {
                "over": round(p_corners_over_8_5 * 100, 2),
                "under": round(p_corners_under_8_5 * 100, 2),
                "media_esperada": round(expected_corners, 2),
            },
            "tarjetas": {
                "amarillas": {
                    "over_3_5": round(p_yellow_over_3_5 * 100, 2),
                    "under_3_5": round(p_yellow_under_3_5 * 100, 2),
                    "media_esperada": round(expected_yellow_cards, 2),
                },
                "rojas": {
                    "over_0_5": round(p_red_over_0_5 * 100, 2),
                    "under_0_5": round(p_red_under_0_5 * 100, 2),
                    "media_esperada": round(expected_red_cards, 2),
                },
                "totales": {
                    "over_3_5": round(p_total_cards_over_3_5 * 100, 2),
                    "under_3_5": round(p_total_cards_under_3_5 * 100, 2),
                    "over_4_5": round(p_total_cards_over_4_5 * 100, 2),
                    "under_4_5": round(p_total_cards_under_4_5 * 100, 2),
                    "media_esperada": round(expected_total_cards, 2),
                },
            },
            "totales": {
                "over_1_5": round(p_over_1_5 * 100, 2),
                "over_2_5": round(p_over_2_5 * 100, 2),
                "under_3_5": round(p_under_3_5 * 100, 2),
                "under_4_5": round(p_under_4_5 * 100, 2),
            },
        }

    def _bet_candidates(self, probabilities: dict[str, float], markets: dict[str, object]) -> list[dict[str, float | str]]:
        candidates: list[dict[str, float | str]] = []
        candidates.extend(
            [
                {"market": "1X2", "pick": "Local", "prob": float(probabilities["local"])},
                {"market": "1X2", "pick": "Empate", "prob": float(probabilities["empate"])},
                {"market": "1X2", "pick": "Visitante", "prob": float(probabilities["visitante"])},
                {
                    "market": "Doble oportunidad",
                    "pick": "1X",
                    "prob": float(markets["doble_oportunidad"]["uno_x"]),
                },
                {
                    "market": "Doble oportunidad",
                    "pick": "12",
                    "prob": float(markets["doble_oportunidad"]["uno_dos"]),
                },
                {
                    "market": "Doble oportunidad",
                    "pick": "X2",
                    "prob": float(markets["doble_oportunidad"]["x_dos"]),
                },
                {"market": "Ambos marcan", "pick": "Si", "prob": float(markets["btts"]["si"])} ,
                {"market": "Ambos marcan", "pick": "No", "prob": float(markets["btts"]["no"])} ,
                {"market": "Totales", "pick": "Over 1.5", "prob": float(markets["totales"]["over_1_5"])} ,
                {"market": "Totales", "pick": "Over 2.5", "prob": float(markets["totales"]["over_2_5"])} ,
                {"market": "Totales", "pick": "Under 3.5", "prob": float(markets["totales"]["under_3_5"])} ,
                {"market": "Totales", "pick": "Under 4.5", "prob": float(markets["totales"]["under_4_5"])} ,
                {"market": "Corners", "pick": "Over 8.5", "prob": float(markets["corners_8_5"]["over"])} ,
                {"market": "Corners", "pick": "Under 8.5", "prob": float(markets["corners_8_5"]["under"])} ,
                {
                    "market": "Tarjetas amarillas",
                    "pick": "Over 3.5",
                    "prob": float(markets["tarjetas"]["amarillas"]["over_3_5"]),
                },
                {
                    "market": "Tarjetas amarillas",
                    "pick": "Under 3.5",
                    "prob": float(markets["tarjetas"]["amarillas"]["under_3_5"]),
                },
                {
                    "market": "Tarjetas totales",
                    "pick": "Over 3.5",
                    "prob": float(markets["tarjetas"]["totales"]["over_3_5"]),
                },
                {
                    "market": "Tarjetas totales",
                    "pick": "Under 3.5",
                    "prob": float(markets["tarjetas"]["totales"]["under_3_5"]),
                },
                {
                    "market": "Tarjetas totales",
                    "pick": "Over 4.5",
                    "prob": float(markets["tarjetas"]["totales"]["over_4_5"]),
                },
                {
                    "market": "Tarjetas totales",
                    "pick": "Under 4.5",
                    "prob": float(markets["tarjetas"]["totales"]["under_4_5"]),
                },
            ]
        )
        candidates.sort(key=lambda item: float(item["prob"]), reverse=True)
        return candidates

    def _categorized_bets(
        self,
        probabilities: dict[str, float],
        markets: dict[str, object],
    ) -> dict[str, object]:
        # --- 1X2 (Local / Empate / Visitante) ---
        resultado_1x2_pool = [
            {"market": "1X2", "pick": "Local",     "prob": float(probabilities["local"])},
            {"market": "1X2", "pick": "Empate",    "prob": float(probabilities["empate"])},
            {"market": "1X2", "pick": "Visitante", "prob": float(probabilities["visitante"])},
        ]
        best_1x2 = max(resultado_1x2_pool, key=lambda x: x["prob"])

        # --- Doble oportunidad ---
        resultado_pool = [
            {"market": "Doble op.", "pick": "1X", "prob": float(markets["doble_oportunidad"]["uno_x"])},
            {"market": "Doble op.", "pick": "12", "prob": float(markets["doble_oportunidad"]["uno_dos"])},
            {"market": "Doble op.", "pick": "X2", "prob": float(markets["doble_oportunidad"]["x_dos"])},
        ]
        best_resultado = max(resultado_pool, key=lambda x: x["prob"])

        # --- Goles (totales + BTTS) ---
        goles_pool = [
            {"market": "Ambos marcan", "pick": "Sí",        "prob": float(markets["btts"]["si"])},
            {"market": "Ambos marcan", "pick": "No",        "prob": float(markets["btts"]["no"])},
            {"market": "Totales",      "pick": "Over 1.5",  "prob": float(markets["totales"]["over_1_5"])},
            {"market": "Totales",      "pick": "Over 2.5",  "prob": float(markets["totales"]["over_2_5"])},
            {"market": "Totales",      "pick": "Under 3.5", "prob": float(markets["totales"]["under_3_5"])},
            {"market": "Totales",      "pick": "Under 4.5", "prob": float(markets["totales"]["under_4_5"])},
        ]
        best_goles = max(goles_pool, key=lambda x: x["prob"])

        # --- Corners ---
        corners_pool = [
            {"market": "Corners", "pick": "Over 8.5",  "prob": float(markets["corners_8_5"]["over"])},
            {"market": "Corners", "pick": "Under 8.5", "prob": float(markets["corners_8_5"]["under"])},
        ]
        best_corners = max(corners_pool, key=lambda x: x["prob"])

        # --- Tarjetas ---
        tarjetas_pool = [
            {"market": "Amarillas",        "pick": "Over 3.5",  "prob": float(markets["tarjetas"]["amarillas"]["over_3_5"])},
            {"market": "Amarillas",        "pick": "Under 3.5", "prob": float(markets["tarjetas"]["amarillas"]["under_3_5"])},
            {"market": "Tarjetas totales", "pick": "Over 3.5",  "prob": float(markets["tarjetas"]["totales"]["over_3_5"])},
            {"market": "Tarjetas totales", "pick": "Under 3.5", "prob": float(markets["tarjetas"]["totales"]["under_3_5"])},
            {"market": "Tarjetas totales", "pick": "Over 4.5",  "prob": float(markets["tarjetas"]["totales"]["over_4_5"])},
            {"market": "Tarjetas totales", "pick": "Under 4.5", "prob": float(markets["tarjetas"]["totales"]["under_4_5"])},
        ]
        best_tarjetas = max(tarjetas_pool, key=lambda x: x["prob"])

        # --- Apuesta múltiple (solo selecciones con confianza Alta >= 75%) ---
        # Evitar duplicar mercados equivalentes: usar solo la mejor entre 1X2 y Doble oportunidad.
        result_leg = best_1x2 if best_1x2["prob"] >= best_resultado["prob"] else best_resultado
        all_legs = [result_leg, best_goles, best_corners, best_tarjetas]
        legs = sorted([leg for leg in all_legs if leg["prob"] >= 75.0], key=lambda x: x["prob"], reverse=True)
        if not legs:
            combined_pct = 0.0
            combined_confidence = "Sin selecciones"
        else:
            combined = 1.0
            for leg in legs:
                combined *= leg["prob"] / 100.0
            combined_pct = round(combined * 100, 2)
            if combined_pct >= 30:
                combined_confidence = "Alta"
            elif combined_pct >= 18:
                combined_confidence = "Media"
            else:
                combined_confidence = "Baja"

        def _fair_odds(prob: float) -> float:
            if prob <= 0:
                return 0.0
            return round(100.0 / prob, 2)

        def _fmt(leg: dict) -> str:
            return f"{leg['market']} — {leg['pick']}  ({leg['prob']:.1f}%)"

        return {
            "resultado_1x2": {
                "market": best_1x2["market"],
                "pick": best_1x2["pick"],
                "prob": round(best_1x2["prob"], 2),
                "fair_odds": _fair_odds(float(best_1x2["prob"])),
            },
            "doble_oportunidad": {
                "market": best_resultado["market"],
                "pick": best_resultado["pick"],
                "prob": round(best_resultado["prob"], 2),
                "fair_odds": _fair_odds(float(best_resultado["prob"])),
            },
            "goles": {
                "market": best_goles["market"],
                "pick": best_goles["pick"],
                "prob": round(best_goles["prob"], 2),
                "fair_odds": _fair_odds(float(best_goles["prob"])),
            },
            "corners": {
                "market": best_corners["market"],
                "pick": best_corners["pick"],
                "prob": round(best_corners["prob"], 2),
                "fair_odds": _fair_odds(float(best_corners["prob"])),
            },
            "tarjetas": {
                "market": best_tarjetas["market"],
                "pick": best_tarjetas["pick"],
                "prob": round(best_tarjetas["prob"], 2),
                "fair_odds": _fair_odds(float(best_tarjetas["prob"])),
            },
            "multiple": {
                "legs": [_fmt(l) for l in legs],
                "prob_combinada": combined_pct,
                "confidence": combined_confidence,
                "fair_odds": _fair_odds(float(combined_pct)),
            },
        }

    def _recommended_bet(self, candidates: list[dict[str, float | str]]) -> dict[str, object]:
        if not candidates:
            return {
                "market": "Sin datos",
                "pick": "Sin recomendacion",
                "probability": 0.0,
                "confidence": "Baja",
                "risk": "Alto",
                "note": "No hay informacion suficiente para una recomendacion.",
            }

        eligible_candidates = []
        for candidate in candidates:
            market = str(candidate.get("market", ""))
            threshold = float(self.market_min_prob.get(market, self.no_bet_min_prob))
            if float(candidate["prob"]) >= threshold:
                eligible_candidates.append(candidate)

        if not eligible_candidates:
            best_overall = candidates[0]
            prob = float(best_overall["prob"])
            return {
                "market": "No bet zone",
                "pick": "Evitar apuesta",
                "probability": round(prob, 2),
                "confidence": "Baja",
                "risk": "Controlado",
                "note": "Ningun mercado supera su umbral minimo configurable.",
            }

        best = eligible_candidates[0]
        prob = float(best["prob"])

        if prob >= 75:
            confidence = "Alta"
            risk = "Bajo"
        elif prob >= 62:
            confidence = "Media"
            risk = "Moderado"
        else:
            confidence = "Baja"
            risk = "Alto"

        note = "Apuesta recomendada por mayor probabilidad del modelo."
        if prob < 55:
            note = "No hay ventaja clara; conviene stake bajo o evitar apuesta."

        return {
            "market": str(best["market"]),
            "pick": str(best["pick"]),
            "probability": round(prob, 2),
            "confidence": confidence,
            "risk": risk,
            "note": note,
        }

    def _build_match_report(
        self,
        recommended_bet: dict[str, object],
        candidates: list[dict[str, float | str]],
        weather: dict[str, str | float],
        players_status: dict[str, str],
        home_table: dict[str, float | int | str],
        away_table: dict[str, float | int | str],
        home_elo: float,
        away_elo: float,
        home_streak: int,
        away_streak: int,
        h2h_stats: dict[str, float],
        kickoff_label: str,
    ) -> dict[str, object]:
        return {
            "recommended_bet": recommended_bet,
            "alternatives": [
                {
                    "market": str(item["market"]),
                    "pick": str(item["pick"]),
                    "probability": round(float(item["prob"]), 2),
                }
                for item in candidates[:3]
            ],
            "factors": {
                "model": self.best_model_name,
                "validated_accuracy": round(self.validation_accuracy * 100, 2),
                "kickoff_period": kickoff_label,
                "weather": weather,
                "players": players_status,
                "table": {
                    "home_position": home_table.get("position", "-"),
                    "away_position": away_table.get("position", "-"),
                    "home_points": home_table.get("points", "-"),
                    "away_points": away_table.get("points", "-"),
                },
                "elo": {
                    "home": round(home_elo),
                    "away": round(away_elo),
                },
                "streaks": {
                    "home": home_streak,
                    "away": away_streak,
                },
                "h2h": {
                    "home_win_rate": round(h2h_stats["home_win_rate"] * 100, 2),
                    "draw_rate": round(h2h_stats["draw_rate"] * 100, 2),
                },
            },
            "disclaimer": "Recomendacion estadistica, no garantia de resultado final.",
        }

    def predict_match(self, match_key: str) -> dict[str, object]:
        match_df = self.fixtures_df[self.fixtures_df["match_key"] == match_key]
        if match_df.empty:
            raise ValueError("No se encontro el encuentro seleccionado.")

        match = match_df.iloc[0]
        home = match["local"]
        away = match["visitante"]
        home_stats = self._team_stats(home)
        away_stats = self._team_stats(away)
        home_elo = self.team_elo_snapshots.get(home, 1500.0)
        away_elo = self.team_elo_snapshots.get(away, 1500.0)
        home_streak = self.team_streak_snapshots.get(home, 0)
        away_streak = self.team_streak_snapshots.get(away, 0)
        h2h_stats = self._h2h_stats(self.h2h_history_snapshot, home, away)
        fixture_date = pd.to_datetime(str(match["fecha"]), dayfirst=True, errors="coerce")
        home_rest_days = self._rest_days_for_fixture(home, fixture_date)
        away_rest_days = self._rest_days_for_fixture(away, fixture_date)
        features = self._build_features(
            home_stats,
            away_stats,
            home_rest_days,
            away_rest_days,
            home_elo,
            away_elo,
            home_streak,
            away_streak,
            h2h_stats,
        )
        feature_frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        class_map = self._predict_probabilities(feature_frame)

        weather = self._weather_context(home, match["fecha"], match["hora"])
        players_status = self._players_context(home, away, str(match["fecha"]))
        players_by_team = self._group_players_by_team(players_status, home, away)
        home_table = self.standings_snapshot.get(home, {})
        away_table = self.standings_snapshot.get(away, {})
        class_map, context_impact = self._apply_context_adjustments(
            class_map,
            weather,
            players_status,
            home,
            away,
            home_table,
            away_table,
            home_stats,
            away_stats,
            home_elo,
            away_elo,
            home_streak,
            away_streak,
        )

        outcome_labels = {
            "H": "Gana local",
            "D": "Empate",
            "A": "Gana visitante",
        }
        top_label = max(class_map, key=class_map.get)
        raw_score_projection = self._predict_scoreline(home_stats, away_stats)
        score_projection = self._apply_context_to_score_projection(
            raw_score_projection,
            weather,
            players_status,
            home,
            away,
            float(context_impact.get("table_total_shift", 0.0)),
        )
        class_map = self._stabilize_probabilities(class_map, score_projection)
        top_label = max(class_map, key=class_map.get)
        markets = self._market_probabilities(class_map, score_projection, home_stats, away_stats, feature_frame)
        kickoff_hour = int(str(match["hora"]).split(":")[0]) if ":" in str(match["hora"]) else 12
        kickoff_label = "Noche" if kickoff_hour >= 20 else "Tarde" if kickoff_hour >= 14 else "Mañana"
        context_explanation = self._build_context_explanation(context_impact)
        probabilities = {
            "local": round(class_map.get("H", 0.0) * 100, 2),
            "empate": round(class_map.get("D", 0.0) * 100, 2),
            "visitante": round(class_map.get("A", 0.0) * 100, 2),
        }
        candidates = self._bet_candidates(probabilities, markets)
        recommendation = self._recommended_bet(candidates)
        categorized_bets = self._categorized_bets(probabilities, markets)
        match_report = self._build_match_report(
            recommendation,
            candidates,
            weather,
            players_status,
            home_table,
            away_table,
            home_elo,
            away_elo,
            home_streak,
            away_streak,
            h2h_stats,
            kickoff_label,
        )

        return {
            "fixture": {
                "jornada": match["jornada"],
                "fecha": match["fecha"],
                "hora": match["hora"],
                "local": match["local"],
                "visitante": match["visitante"],
                "local_logo": self._team_logo_url(str(match["local"])),
                "visitante_logo": self._team_logo_url(str(match["visitante"])),
                "liga_logo": self.league_logo_url,
            },
            "outcome": outcome_labels[top_label],
            "probabilities": probabilities,
            "score_projection": score_projection,
            "markets": markets,
            "home_stats": home_stats,
            "away_stats": away_stats,
            "table": {
                "home": home_table,
                "away": away_table,
            },
            "context": {
                "kickoff_period": kickoff_label,
                "weather": weather,
                "players": players_status,
                "players_by_team": players_by_team,
                "impact": context_impact,
                "explanation": context_explanation,
            },
            "match_report": match_report,
            "categorized_bets": categorized_bets,
            "home_elo": round(home_elo),
            "away_elo": round(away_elo),
            "home_streak": home_streak,
            "away_streak": away_streak,
            "rest_days": {
                "home": round(home_rest_days, 1),
                "away": round(away_rest_days, 1),
            },
            "h2h": {
                "home_win_rate": round(h2h_stats["home_win_rate"] * 100),
                "draw_rate": round(h2h_stats["draw_rate"] * 100),
            },
        }

    def predict_recommended_bet_fast(self, match_key: str) -> dict[str, object]:
        """Prediccion ligera para listados: evita llamadas de contexto externo."""
        match_df = self.fixtures_df[self.fixtures_df["match_key"] == match_key]
        if match_df.empty:
            raise ValueError("No se encontro el encuentro seleccionado.")

        match = match_df.iloc[0]
        home = match["local"]
        away = match["visitante"]

        home_stats = self._team_stats(home)
        away_stats = self._team_stats(away)
        home_elo = self.team_elo_snapshots.get(home, 1500.0)
        away_elo = self.team_elo_snapshots.get(away, 1500.0)
        home_streak = self.team_streak_snapshots.get(home, 0)
        away_streak = self.team_streak_snapshots.get(away, 0)
        h2h_stats = self._h2h_stats(self.h2h_history_snapshot, home, away)
        fixture_date = pd.to_datetime(str(match["fecha"]), dayfirst=True, errors="coerce")
        home_rest_days = self._rest_days_for_fixture(home, fixture_date)
        away_rest_days = self._rest_days_for_fixture(away, fixture_date)

        features = self._build_features(
            home_stats,
            away_stats,
            home_rest_days,
            away_rest_days,
            home_elo,
            away_elo,
            home_streak,
            away_streak,
            h2h_stats,
        )
        feature_frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        class_map = self._predict_probabilities(feature_frame)

        score_projection = self._predict_scoreline(home_stats, away_stats)
        markets = self._market_probabilities(class_map, score_projection, home_stats, away_stats, feature_frame)
        probabilities = {
            "local": round(class_map.get("H", 0.0) * 100, 2),
            "empate": round(class_map.get("D", 0.0) * 100, 2),
            "visitante": round(class_map.get("A", 0.0) * 100, 2),
        }
        candidates = self._bet_candidates(probabilities, markets)
        categorized_bets = self._categorized_bets(probabilities, markets)

        return {
            "recommended_bet": self._recommended_bet(candidates),
            "probabilities": probabilities,
            "multiple": categorized_bets["multiple"],
        }


@lru_cache(maxsize=1)
def get_prediction_service_spain() -> MatchPredictionService:
    return MatchPredictionService(SPAIN_CONFIG)


@lru_cache(maxsize=1)
def get_prediction_service_bundesliga() -> MatchPredictionService:
    return MatchPredictionService(BUNDESLIGA_CONFIG)


@lru_cache(maxsize=1)
def get_prediction_service_premier() -> MatchPredictionService:
    return MatchPredictionService(PREMIER_LEAGUE_CONFIG)


@lru_cache(maxsize=1)
def get_prediction_service_seriea() -> MatchPredictionService:
    return MatchPredictionService(SERIEA_CONFIG)


@lru_cache(maxsize=1)
def get_prediction_service_ligue1() -> MatchPredictionService:
    return MatchPredictionService(LIGUE1_CONFIG)


@lru_cache(maxsize=1)
def get_prediction_service_primeiraliga() -> MatchPredictionService:
    return MatchPredictionService(PRIMEIRALIGA_CONFIG)


@lru_cache(maxsize=1)
def get_prediction_service_proleague() -> MatchPredictionService:
    return MatchPredictionService(PROLEAGUE_CONFIG)


@lru_cache(maxsize=1)
def get_prediction_service_eredivisie() -> MatchPredictionService:
    return MatchPredictionService(EREDIVISIE_CONFIG)


@lru_cache(maxsize=1)
def get_prediction_service_superlig_turquia() -> MatchPredictionService:
    return MatchPredictionService(SUPERLIG_TURQUIA_CONFIG)


@lru_cache(maxsize=1)
def get_prediction_service_superleague_grecia() -> MatchPredictionService:
    return MatchPredictionService(SUPERLEAGUE_GRECIA_CONFIG)


@lru_cache(maxsize=1)
def get_prediction_service_premiership_escocia() -> MatchPredictionService:
    return MatchPredictionService(PREMIERSHIP_ESCOCIA_CONFIG)


# Alias para compatibilidad
get_prediction_service = get_prediction_service_spain