import csv
import urllib.request
from datetime import datetime
from stats.models import Match
from django.core.cache import cache

def fetch_and_update_matches():
    url = "https://www.football-data.co.uk/mmz4281/2526/SP1.csv"
    
    # Check if we updated recently (e.g., within the last hour = 3600 seconds)
    # Forcing an update this time because schema changed, but normally keep cache.
    # if cache.get('last_csv_update'):
    #     return False, "Data was recently updated."
        
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            lines = [line.decode('utf-8') for line in response.readlines()]
            
        reader = csv.DictReader(lines)
        created_count = 0
        
        for row in reader:
            if not row.get('Date'):
                continue # Skip empty rows
            
            date_str = row['Date']
            try:
                match_date = datetime.strptime(date_str, '%d/%m/%Y').date()
            except ValueError:
                continue
            
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            fthg = int(row['FTHG'])
            ftag = int(row['FTAG'])
            ftr = row['FTR']
            
            # Helper for safe int parsing
            def safe_int(val):
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return 0
            
            match, created = Match.objects.update_or_create(
                date=match_date,
                home_team=home_team,
                away_team=away_team,
                defaults={
                    'fthg': fthg,
                    'ftag': ftag,
                    'ftr': ftr,
                    'hs': safe_int(row.get('HS')),
                    'as_shots': safe_int(row.get('AS')),
                    'hst': safe_int(row.get('HST')),
                    'ast': safe_int(row.get('AST')),
                    'hc': safe_int(row.get('HC')),
                    'ac': safe_int(row.get('AC')),
                    'hf': safe_int(row.get('HF')),
                    'af': safe_int(row.get('AF')),
                    'hy': safe_int(row.get('HY')),
                    'ay': safe_int(row.get('AY')),
                    'hr': safe_int(row.get('HR')),
                    'ar': safe_int(row.get('AR'))
                }
            )
            
            if created:
                created_count += 1
                
        # Set cache to avoid fetching again for 1 hour (3600 seconds)
        cache.set('last_csv_update', True, 3600)
        
        return True, f"Successfully loaded new stats. Created {created_count} matches."
        
    except Exception as e:
        return False, f"Error loading data: {e}"
