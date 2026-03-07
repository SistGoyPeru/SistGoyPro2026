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
            standings[match.home_team] = {
                'team': match.home_team, 'played': 0, 'won': 0, 'drawn': 0, 'lost': 0, 
                'gf': 0, 'ga': 0, 'gd': 0, 'points': 0,
                'shots': 0, 'shots_on_target': 0, 'corners': 0, 'fouls': 0, 'yellow_cards': 0, 'red_cards': 0,
                'btts': 0, 'over_25': 0, 'clean_sheets': 0, 'ht_won': 0, 'ht_drawn': 0, 'ht_lost': 0,
                'home_pts': 0, 'away_pts': 0, 'avg_win_odds': 0.0, 'win_odds_sum': 0.0,
                'comebacks': 0, 'collapses': 0, 'sh_goals': 0, 'sh_conceded': 0, 'sh_gd': 0
            }
        if match.away_team not in standings:
            standings[match.away_team] = {
                'team': match.away_team, 'played': 0, 'won': 0, 'drawn': 0, 'lost': 0, 
                'gf': 0, 'ga': 0, 'gd': 0, 'points': 0,
                'shots': 0, 'shots_on_target': 0, 'corners': 0, 'fouls': 0, 'yellow_cards': 0, 'red_cards': 0,
                'btts': 0, 'over_25': 0, 'clean_sheets': 0, 'ht_won': 0, 'ht_drawn': 0, 'ht_lost': 0,
                'home_pts': 0, 'away_pts': 0, 'avg_win_odds': 0.0, 'win_odds_sum': 0.0,
                'comebacks': 0, 'collapses': 0, 'sh_goals': 0, 'sh_conceded': 0, 'sh_gd': 0
            }
            
        # Update played
        standings[match.home_team]['played'] += 1
        standings[match.away_team]['played'] += 1
        
        # Update goals
        standings[match.home_team]['gf'] += match.fthg
        standings[match.home_team]['ga'] += match.ftag
        standings[match.away_team]['gf'] += match.ftag
        standings[match.away_team]['ga'] += match.fthg
        
        # Update advanced stats
        standings[match.home_team]['shots'] += match.hs
        standings[match.away_team]['shots'] += match.as_shots
        standings[match.home_team]['shots_on_target'] += match.hst
        standings[match.away_team]['shots_on_target'] += match.ast
        standings[match.home_team]['corners'] += match.hc
        standings[match.away_team]['corners'] += match.ac
        standings[match.home_team]['fouls'] += match.hf
        standings[match.away_team]['fouls'] += match.af
        standings[match.home_team]['yellow_cards'] += match.hy
        standings[match.away_team]['yellow_cards'] += match.ay
        standings[match.home_team]['red_cards'] += match.hr
        standings[match.away_team]['red_cards'] += match.ar
        
        # Second Half Goals (Total Goals - Half Time Goals)
        standings[match.home_team]['sh_goals'] += (match.fthg - match.hthg)
        standings[match.home_team]['sh_conceded'] += (match.ftag - match.htag)
        standings[match.away_team]['sh_goals'] += (match.ftag - match.htag)
        standings[match.away_team]['sh_conceded'] += (match.fthg - match.hthg)
        
        # Update results
        if match.ftr == 'H':
            standings[match.home_team]['won'] += 1
            standings[match.home_team]['points'] += 3
            standings[match.home_team]['home_pts'] += 3
            standings[match.away_team]['lost'] += 1
            if match.avg_h > 0:
                standings[match.home_team]['win_odds_sum'] += match.avg_h
        elif match.ftr == 'A':
            standings[match.away_team]['won'] += 1
            standings[match.away_team]['points'] += 3
            standings[match.away_team]['away_pts'] += 3
            standings[match.home_team]['lost'] += 1
            if match.avg_a > 0:
                standings[match.away_team]['win_odds_sum'] += match.avg_a
        else: # Draw
            standings[match.home_team]['drawn'] += 1
            standings[match.home_team]['points'] += 1
            standings[match.home_team]['home_pts'] += 1
            standings[match.away_team]['drawn'] += 1
            standings[match.away_team]['points'] += 1
            standings[match.away_team]['away_pts'] += 1
            
        # Update Half Time results
        if match.htr == 'H':
            standings[match.home_team]['ht_won'] += 1
            standings[match.away_team]['ht_lost'] += 1
        elif match.htr == 'A':
            standings[match.away_team]['ht_won'] += 1
            standings[match.home_team]['ht_lost'] += 1
        else:
            standings[match.home_team]['ht_drawn'] += 1
            standings[match.away_team]['ht_drawn'] += 1
            
        # Comebacks and Collapses (Points Dropped/Gained from HT)
        if match.htr == 'A' and match.ftr in ['D', 'H']:
            standings[match.home_team]['comebacks'] += 1
            standings[match.away_team]['collapses'] += 1
        elif match.htr == 'H' and match.ftr in ['D', 'A']:
            standings[match.away_team]['comebacks'] += 1
            standings[match.home_team]['collapses'] += 1
            
        # BTTS (Both Teams To Score)
        if match.fthg > 0 and match.ftag > 0:
            standings[match.home_team]['btts'] += 1
            standings[match.away_team]['btts'] += 1
            
        # Over 2.5 Goals
        if (match.fthg + match.ftag) > 2:
            standings[match.home_team]['over_25'] += 1
            standings[match.away_team]['over_25'] += 1
            
        # Clean Sheets
        if match.ftag == 0:
            standings[match.home_team]['clean_sheets'] += 1
        if match.fthg == 0:
            standings[match.away_team]['clean_sheets'] += 1
            
    # Calculate goal difference and percentages
    standings_list = []
    for team_data in standings.values():
        team_data['gd'] = team_data['gf'] - team_data['ga']
        team_data['sh_gd'] = team_data['sh_goals'] - team_data['sh_conceded']
        
        if team_data['played'] > 0:
            team_data['btts_pct'] = round((team_data['btts'] / team_data['played']) * 100)
            team_data['over_25_pct'] = round((team_data['over_25'] / team_data['played']) * 100)
            team_data['cs_pct'] = round((team_data['clean_sheets'] / team_data['played']) * 100)
            team_data['shot_conversion'] = round((team_data['gf'] / team_data['shots'] * 100), 1) if team_data['shots'] > 0 else 0
            team_data['foul_per_card'] = round(team_data['fouls'] / (team_data['yellow_cards'] + team_data['red_cards']), 1) if (team_data['yellow_cards'] + team_data['red_cards']) > 0 else 0
        else:
            team_data['btts_pct'] = 0
            team_data['over_25_pct'] = 0
            team_data['cs_pct'] = 0
            team_data['shot_conversion'] = 0
            team_data['foul_per_card'] = 0
            
        if team_data['won'] > 0:
            team_data['avg_win_odds'] = round(team_data['win_odds_sum'] / team_data['won'], 2)
        else:
            team_data['avg_win_odds'] = 0.0
            
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
    
    # Yellow Cards Top 5
    sorted_by_cards = sorted(standings, key=lambda x: x['yellow_cards'] + x['red_cards'], reverse=True)[:5]
    cards_teams = [team['team'] for team in sorted_by_cards]
    cards_count = [team['yellow_cards'] + team['red_cards'] for team in sorted_by_cards]

    # Shots Top 5
    sorted_by_shots = sorted(standings, key=lambda x: x['shots'], reverse=True)[:5]
    shots_teams = [team['team'] for team in sorted_by_shots]
    shots_count = [team['shots'] for team in sorted_by_shots]
    
    context = {
        'standings': standings,
        'recent_matches': recent_matches,
        'matches_played': Match.objects.count(),
        'chart_teams': chart_teams,
        'chart_goals': chart_goals,
        'cards_teams': cards_teams,
        'cards_count': cards_count,
        'shots_teams': shots_teams,
        'shots_count': shots_count,
    }
    
    return render(request, 'stats/dashboard.html', context)
