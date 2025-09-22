#!/usr/bin/env python3
"""
Real NBA Data Collection Script
This script fetches actual NBA data from the official NBA API
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import os
from nba_data_service import NBADataService

def collect_real_nba_data():
    """Collect real NBA data for the current season"""
    nba_service = NBADataService()
    current_season = nba_service.get_current_season()
    
    print(f"Collecting real NBA data for season: {current_season}")
    
    # Get all players
    print("Fetching all players...")
    all_players = nba_service.get_all_players(current_season)
    
    if all_players.empty:
        print("No players found. This might be because:")
        print("1. The season hasn't started yet")
        print("2. API rate limiting")
        print("3. Network issues")
        return None
    
    print(f"Found {len(all_players)} players")
    
    # Get game logs for each player
    all_game_logs = []
    player_count = 0
    
    for _, player in all_players.iterrows():
        player_id = player['PLAYER_ID']
        player_name = player['PLAYER_NAME']
        
        print(f"Fetching data for {player_name} (ID: {player_id})...")
        
        game_logs = nba_service.get_player_game_logs(player_id, current_season)
        
        if not game_logs.empty:
            # Add player info to each game log
            game_logs['PLAYER_NAME'] = player_name
            game_logs['PLAYER_ID'] = player_id
            all_game_logs.append(game_logs)
            player_count += 1
            
            # Rate limiting - be respectful to the API
            time.sleep(0.1)
        
        # Progress update
        if player_count % 50 == 0:
            print(f"Processed {player_count} players...")
    
    if all_game_logs:
        # Combine all game logs
        combined_data = pd.concat(all_game_logs, ignore_index=True)
        
        # Save to CSV
        filename = f"real_nba_data_{current_season}.csv"
        combined_data.to_csv(filename, index=False)
        
        print(f"\n✅ Successfully collected real NBA data!")
        print(f"📊 Total games: {len(combined_data)}")
        print(f"👥 Players with data: {player_count}")
        print(f"📅 Season: {current_season}")
        print(f"💾 Saved to: {filename}")
        
        # Show sample data
        print("\n📋 Sample data:")
        print(combined_data[['PLAYER_NAME', 'GAME_DATE', 'MATCHUP', 'PTS', 'MIN']].head())
        
        return combined_data
    else:
        print("❌ No game data collected")
        return None

def create_fallback_data():
    """Create fallback data structure for when real data isn't available"""
    print("Creating fallback data structure...")
    
    # This would contain real historical data or a more realistic dataset
    # For now, we'll create a minimal structure
    fallback_data = {
        'note': 'This is fallback data - real NBA data collection failed',
        'season': '2024-25',
        'players': [],
        'games': []
    }
    
    with open('fallback_data.json', 'w') as f:
        json.dump(fallback_data, f, indent=2)
    
    print("Fallback data structure created")

def main():
    """Main function to collect NBA data"""
    print("🏀 NBA Real Data Collection Tool")
    print("=" * 50)
    
    try:
        # Try to collect real data
        real_data = collect_real_nba_data()
        
        if real_data is None:
            print("\n⚠️  Real data collection failed")
            print("This could be because:")
            print("- The NBA season hasn't started yet")
            print("- API rate limiting or restrictions")
            print("- Network connectivity issues")
            print("- NBA API changes")
            
            create_fallback_data()
            
            print("\n💡 Recommendations:")
            print("1. Wait for the NBA season to start")
            print("2. Check your internet connection")
            print("3. Try again later (API rate limiting)")
            print("4. Use the fallback data structure")
        
    except Exception as e:
        print(f"❌ Error during data collection: {e}")
        create_fallback_data()

if __name__ == "__main__":
    main()
