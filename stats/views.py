from django.shortcuts import render
from django.db.models import Count, Sum, Q, F
from django.db.models.functions import Coalesce
from .models import Match
from .utils import fetch_and_update_matches

def calculate_standings():
    # Get all matches
    matches = Match.objects.all()
    
    standings = {}
    
    for match in matches:
        # Initialize teams if not exist
        if match.home_team not in standings:
            standings[match.home_team] = {'team': match.home_team, 'played': 0, 'won': 0, 'drawn': 0, 'lost': 0, 'gf': 0, 'ga': 0, 'gd': 0, 'points': 0}
        if match.away_team not in standings:
            standings[match.away_team] = {'team': match.away_team, 'played': 0, 'won': 0, 'drawn': 0, 'lost': 0, 'gf': 0, 'ga': 0, 'gd': 0, 'points': 0}
            
        # Update played
        standings[match.home_team]['played'] += 1
        standings[match.away_team]['played'] += 1
        
        # Update goals
        standings[match.home_team]['gf'] += match.fthg
        standings[match.home_team]['ga'] += match.ftag
        standings[match.away_team]['gf'] += match.ftag
        standings[match.away_team]['ga'] += match.fthg
        
        # Update results
        if match.ftr == 'H':
            standings[match.home_team]['won'] += 1
            standings[match.home_team]['points'] += 3
            standings[match.away_team]['lost'] += 1
        elif match.ftr == 'A':
            standings[match.away_team]['won'] += 1
            standings[match.away_team]['points'] += 3
            standings[match.home_team]['lost'] += 1
        else: # Draw
            standings[match.home_team]['drawn'] += 1
            standings[match.home_team]['points'] += 1
            standings[match.away_team]['drawn'] += 1
            standings[match.away_team]['points'] += 1
            
    # Calculate goal difference and convert to list
    standings_list = []
    for team_data in standings.values():
        team_data['gd'] = team_data['gf'] - team_data['ga']
        standings_list.append(team_data)
        
    # Sort by points (desc), then goal difference (desc), then goals for (desc)
    standings_list.sort(key=lambda x: (x['points'], x['gd'], x['gf']), reverse=True)
    
    # Add position
    for i, team in enumerate(standings_list):
        team['position'] = i + 1
        
    return standings_list

def dashboard(request):
    # Try fetching updates from URL. Cached internally to prevent spamming.
    fetch_and_update_matches()

    standings = calculate_standings()
    recent_matches = Match.objects.all()[:10]
    
    # Prepare chart data (top 5 scoring teams)
    sorted_by_goals = sorted(standings, key=lambda x: x['gf'], reverse=True)[:5]
    chart_teams = [team['team'] for team in sorted_by_goals]
    chart_goals = [team['gf'] for team in sorted_by_goals]
    
    context = {
        'standings': standings,
        'recent_matches': recent_matches,
        'chart_teams': chart_teams,
        'chart_goals': chart_goals,
    }
    
    return render(request, 'stats/dashboard.html', context)
