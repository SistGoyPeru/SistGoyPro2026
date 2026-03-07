from django.core.management.base import BaseCommand
from stats.utils import fetch_and_update_matches
from django.core.cache import cache

class Command(BaseCommand):
    help = 'Fetches Spanish League CSV data and loads it into the database'

    def handle(self, *args, **options):
        # We clear the cache to force an update when running manually
        cache.delete('last_csv_update')
        
        self.stdout.write("Fetching data from https://www.football-data.co.uk/mmz4281/2526/SP1.csv ...")
        
        success, message = fetch_and_update_matches()
        
        if success:
            self.stdout.write(self.style.SUCCESS(message))
        else:
            self.stdout.write(self.style.ERROR(message))
