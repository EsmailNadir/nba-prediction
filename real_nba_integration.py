#!/usr/bin/env python3
"""
Real NBA Data Integration using working APIs
This uses alternative data sources that actually work
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time

class WorkingNBADataService:
    def __init__(self):
        # Use ESPN API which is more accessible
        self.espn_base = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
        self.current_season = self.get_current_season()
        
    def get_current_season(self):
        """Get current NBA season year"""
        now = datetime.now()
        if now.month >= 10:
            return f"{now.year}-{str(now.year + 1)[2:]}"
        else:
            return f"{now.year - 1}-{str(now.year)[2:]}"
    
    def get_team_schedule(self, team_abbr):
        """Get team schedule from ESPN API"""
        try:
            # ESPN team IDs
            team_ids = {
                'ATL': '1', 'BOS': '2', 'BRK': '17', 'CHA': '30', 'CHI': '4',
                'CLE': '5', 'DAL': '6', 'DEN': '7', 'DET': '8', 'GSW': '9',
                'HOU': '10', 'IND': '11', 'LAC': '12', 'LAL': '13', 'MEM': '29',
                'MIA': '14', 'MIL': '15', 'MIN': '16', 'NOP': '3', 'NYK': '18',
                'OKC': '25', 'ORL': '19', 'PHI': '20', 'PHX': '21', 'POR': '22',
                'SAC': '23', 'SAS': '24', 'TOR': '28', 'UTA': '26', 'WAS': '27'
            }
            
            team_id = team_ids.get(team_abbr)
            if not team_id:
                return None
                
            url = f"{self.espn_base}/teams/{team_id}/schedule"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self.parse_espn_schedule(data)
            else:
                print(f"ESPN API error for {team_abbr}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error getting schedule for {team_abbr}: {e}")
            return None
    
    def parse_espn_schedule(self, data):
        """Parse ESPN schedule data"""
        games = []
        for event in data.get('events', []):
            game_data = {
                'date': event.get('date'),
                'opponent': None,
                'home_away': None
            }
            
            # Extract opponent and home/away info
            competitions = event.get('competitions', [])
            if competitions:
                comp = competitions[0]
                competitors = comp.get('competitors', [])
                if len(competitors) >= 2:
                    # Determine home/away and opponent
                    home_team = competitors[0] if competitors[0].get('homeAway') == 'home' else competitors[1]
                    away_team = competitors[1] if competitors[0].get('homeAway') == 'home' else competitors[0]
                    
                    game_data['home_team'] = home_team.get('team', {}).get('abbreviation')
                    game_data['away_team'] = away_team.get('team', {}).get('abbreviation')
            
            games.append(game_data)
        
        return games
    
    def get_next_opponent(self, team_abbr):
        """Get next opponent for a team"""
        schedule = self.get_team_schedule(team_abbr)
        if not schedule:
            return None
            
        # Find next game after today
        today = datetime.now().date()
        for game in schedule:
            if game.get('date'):
                game_date = datetime.fromisoformat(game['date'].replace('Z', '+00:00')).date()
                if game_date > today:
                    # Determine opponent
                    if game.get('home_team') == team_abbr:
                        return game.get('away_team')
                    elif game.get('away_team') == team_abbr:
                        return game.get('home_team')
        
        return None
    
    def get_player_stats(self, player_name):
        """Get player stats from ESPN (simplified)"""
        # This would need more complex implementation
        # For now, return placeholder
        return {
            'player_name': player_name,
            'games_played': 0,
            'points_per_game': 0,
            'last_updated': datetime.now().isoformat()
        }
    
    def is_preseason_active(self):
        """Check if we're in preseason"""
        now = datetime.now()
        # NBA preseason typically runs October 1-15
        if now.month == 10 and now.day <= 15:
            return True
        return False
    
    def should_include_preseason(self):
        """Determine if preseason games should be included"""
        # Include preseason if we're in preseason or early season
        now = datetime.now()
        if now.month == 10 or (now.month == 11 and now.day <= 7):
            return True
        return False

def test_real_integration():
    """Test the real NBA integration"""
    print("🏀 Testing Real NBA Integration")
    print("=" * 40)
    
    service = WorkingNBADataService()
    
    print(f"Current season: {service.current_season}")
    print(f"Preseason active: {service.is_preseason_active()}")
    print(f"Include preseason: {service.should_include_preseason()}")
    print()
    
    # Test with a few teams
    test_teams = ['LAL', 'GSW', 'BOS']
    
    for team in test_teams:
        print(f"Testing {team}...")
        next_opponent = service.get_next_opponent(team)
        if next_opponent:
            print(f"  ✅ Next opponent: {next_opponent}")
        else:
            print(f"  ❌ No schedule data available")
        time.sleep(1)  # Be respectful to API

if __name__ == "__main__":
    test_real_integration()
