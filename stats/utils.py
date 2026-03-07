import csv
import urllib.request
from datetime import datetime
from stats.models import Match
from django.core.cache import cache

def fetch_and_update_matches():
    url = "https://www.football-data.co.uk/mmz4281/2526/SP1.csv"
    
    # Check if we updated recently (e.g., within the last hour = 3600 seconds)
    # This prevents sending a request to the server on every single page load.
    if cache.get('last_csv_update'):
        return False, "Data was recently updated."
        
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
            
            match, created = Match.objects.get_or_create(
                date=match_date,
                home_team=home_team,
                away_team=away_team,
                defaults={
                    'fthg': fthg,
                    'ftag': ftag,
                    'ftr': ftr
                }
            )
            
            if created:
                created_count += 1
                
        # Set cache to avoid fetching again for 1 hour (3600 seconds)
        cache.set('last_csv_update', True, 3600)
        
        return True, f"Successfully loaded {created_count} new matches."
        
    except Exception as e:
        return False, f"Error loading data: {e}"
