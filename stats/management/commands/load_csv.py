import csv
import urllib.request
from datetime import datetime
from django.core.management.base import BaseCommand
from stats.models import Match

class Command(BaseCommand):
    help = 'Fetches Spanish League CSV data and loads it into the database'

    def handle(self, *args, **options):
        url = "https://www.football-data.co.uk/mmz4281/2526/SP1.csv"
        self.stdout.write(f"Fetching data from {url}...")
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                lines = [line.decode('utf-8') for line in response.readlines()]
                
            reader = csv.DictReader(lines)
            
            created_count = 0
            for row in reader:
                if not row.get('Date'):
                    continue # Skip empty rows
                
                # Parse date - format is usually DD/MM/YYYY
                date_str = row['Date']
                try:
                    match_date = datetime.strptime(date_str, '%d/%m/%Y').date()
                except ValueError:
                    self.stdout.write(self.style.WARNING(f"Could not parse date: {date_str}"))
                    continue
                
                # Get fields
                home_team = row['HomeTeam']
                away_team = row['AwayTeam']
                fthg = int(row['FTHG'])
                ftag = int(row['FTAG'])
                ftr = row['FTR']
                
                # Create or update match
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
            
            self.stdout.write(self.style.SUCCESS(f"Successfully loaded {created_count} new matches."))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error loading data: {e}"))
