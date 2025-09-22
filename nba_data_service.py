import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import os

class NBADataService:
    def __init__(self):
        self.base_url = "https://stats.nba.com/stats"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nba.com/',
            'Connection': 'keep-alive',
        }
        
    def get_current_season(self):
        """Get current NBA season year"""
        now = datetime.now()
        # NBA season typically starts in October
        if now.month >= 10:
            return f"{now.year}-{str(now.year + 1)[2:]}"
        else:
            return f"{now.year - 1}-{str(now.year)[2:]}"
    
    def get_team_schedule(self, team_id, season=None):
        """Get team schedule for the season"""
        if season is None:
            season = self.get_current_season()
            
        url = f"{self.base_url}/teamgamelog"
        params = {
            'DateFrom': '',
            'DateTo': '',
            'GameSegment': '',
            'LastNGames': 0,
            'LeagueID': '00',
            'Location': '',
            'MeasureType': 'Base',
            'Month': 0,
            'OpponentTeamID': 0,
            'Outcome': '',
            'PORound': 0,
            'PerMode': 'PerGame',
            'Period': 0,
            'PlayerID': 0,
            'Season': season,
            'SeasonSegment': '',
            'SeasonType': 'Regular Season',
            'ShotClockRange': '',
            'TeamID': team_id,
            'VsConference': '',
            'VsDivision': ''
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                headers = data['resultSets'][0]['headers']
                rows = data['resultSets'][0]['rowSet']
                return pd.DataFrame(rows, columns=headers)
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching schedule for team {team_id}: {e}")
            return pd.DataFrame()
    
    def get_player_game_logs(self, player_id, season=None):
        """Get player game logs for the season"""
        if season is None:
            season = self.get_current_season()
            
        url = f"{self.base_url}/playergamelog"
        params = {
            'DateFrom': '',
            'DateTo': '',
            'GameSegment': '',
            'LastNGames': 0,
            'LeagueID': '00',
            'Location': '',
            'MeasureType': 'Base',
            'Month': 0,
            'OpponentTeamID': 0,
            'Outcome': '',
            'PORound': 0,
            'PerMode': 'PerGame',
            'Period': 0,
            'PlayerID': player_id,
            'Season': season,
            'SeasonSegment': '',
            'SeasonType': 'Regular Season',
            'ShotClockRange': '',
            'TeamID': 0,
            'VsConference': '',
            'VsDivision': ''
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                headers = data['resultSets'][0]['headers']
                rows = data['resultSets'][0]['rowSet']
                df = pd.DataFrame(rows, columns=headers)
                
                # Convert date column
                if 'GAME_DATE' in df.columns:
                    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
                
                return df
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching game logs for player {player_id}: {e}")
            return pd.DataFrame()
    
    def get_all_players(self, season=None):
        """Get all active players for the season"""
        if season is None:
            season = self.get_current_season()
            
        url = f"{self.base_url}/leaguedashplayerstats"
        params = {
            'College': '',
            'Conference': '',
            'Country': '',
            'DateFrom': '',
            'DateTo': '',
            'Division': '',
            'DraftPick': '',
            'DraftYear': '',
            'GameScope': '',
            'GameSegment': '',
            'Height': '',
            'LastNGames': 0,
            'LeagueID': '00',
            'Location': '',
            'MeasureType': 'Base',
            'Month': 0,
            'OpponentTeamID': 0,
            'Outcome': '',
            'PORound': 0,
            'PerMode': 'PerGame',
            'Period': 0,
            'PlayerExperience': '',
            'PlayerPosition': '',
            'Season': season,
            'SeasonSegment': '',
            'SeasonType': 'Regular Season',
            'ShotClockRange': '',
            'StarterBench': '',
            'TeamID': 0,
            'VsConference': '',
            'VsDivision': '',
            'Weight': ''
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                headers = data['resultSets'][0]['headers']
                rows = data['resultSets'][0]['rowSet']
                return pd.DataFrame(rows, columns=headers)
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching players: {e}")
            return pd.DataFrame()
    
    def get_team_roster(self, team_id, season=None):
        """Get team roster for the season"""
        if season is None:
            season = self.get_current_season()
            
        url = f"{self.base_url}/commonteamroster"
        params = {
            'LeagueID': '00',
            'Season': season,
            'TeamID': team_id
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                headers = data['resultSets'][0]['headers']
                rows = data['resultSets'][0]['rowSet']
                return pd.DataFrame(rows, columns=headers)
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching roster for team {team_id}: {e}")
            return pd.DataFrame()
    
    def get_upcoming_games(self, team_id=None, days_ahead=7):
        """Get upcoming games for a team or all teams"""
        # This would integrate with NBA schedule API
        # For now, return placeholder structure
        return {
            'team_id': team_id,
            'upcoming_games': [],
            'next_opponent': None,
            'next_game_date': None
        }
    
    def update_player_data(self, player_id, season=None):
        """Update player data with latest games"""
        if season is None:
            season = self.get_current_season()
            
        # Get latest game logs
        game_logs = self.get_player_game_logs(player_id, season)
        
        if not game_logs.empty:
            # Save to local cache
            cache_file = f"player_{player_id}_{season}.csv"
            game_logs.to_csv(cache_file, index=False)
            print(f"Updated data for player {player_id}: {len(game_logs)} games")
            return game_logs
        
        return pd.DataFrame()

# Example usage and testing
if __name__ == "__main__":
    nba_service = NBADataService()
    
    print(f"Current season: {nba_service.get_current_season()}")
    
    # Test with a known player (LeBron James - ID: 2544)
    print("Testing with LeBron James...")
    lebron_data = nba_service.get_player_game_logs(2544)
    
    if not lebron_data.empty:
        print(f"Found {len(lebron_data)} games for LeBron")
        print("Recent games:")
        print(lebron_data[['GAME_DATE', 'MATCHUP', 'PTS', 'MIN']].tail(3))
    else:
        print("No data found - this might be due to API restrictions or season timing")
