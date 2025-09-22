from flask import Flask, jsonify, request
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import os
import json
warnings.filterwarnings('ignore')

# Import our real NBA data service
try:
    from real_nba_integration import WorkingNBADataService
    NBA_SERVICE_AVAILABLE = True
    print("✅ Real NBA Integration available")
except ImportError:
    NBA_SERVICE_AVAILABLE = False
    print("❌ Real NBA Integration not available - using fallback mode")

# Import advanced opponent analysis
try:
    from advanced_opponent_analysis import AdvancedOpponentAnalyzer
    ADVANCED_ANALYSIS_AVAILABLE = True
    print("✅ Advanced Opponent Analysis available")
except ImportError:
    ADVANCED_ANALYSIS_AVAILABLE = False
    print("❌ Advanced Opponent Analysis not available - using basic analysis")

app = Flask(__name__)

def detect_real_nba_data(df):
    """Detect if the data is real NBA data based on data quality indicators"""
    if df is None or len(df) == 0:
        return False
    
    # Check for real NBA data indicators
    real_data_indicators = 0
    
    # 1. Check for detailed statistics (real NBA data has many columns)
    if len(df.columns) > 50:
        real_data_indicators += 1
    
    # 2. Check for realistic date range (NBA season dates)
    if 'GAME_DATE' in df.columns:
        dates = pd.to_datetime(df['GAME_DATE'])
        if dates.min().month >= 10 or dates.max().month <= 6:  # NBA season months
            real_data_indicators += 1
    
    # 3. Check for realistic player statistics
    if 'PTS' in df.columns:
        pts_stats = df['PTS'].describe()
        if 0 <= pts_stats['min'] <= 100 and 5 <= pts_stats['mean'] <= 30:  # Realistic point ranges
            real_data_indicators += 1
    
    # 4. Check for team abbreviations (real NBA teams)
    if 'TEAM_ABBREVIATION' in df.columns:
        real_teams = {'ATL', 'BOS', 'BRK', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
                     'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
                     'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'}
        teams_in_data = set(df['TEAM_ABBREVIATION'].unique())
        if len(teams_in_data.intersection(real_teams)) >= 25:  # Most real teams present
            real_data_indicators += 1
    
    # 5. Check for detailed game statistics
    detailed_stats = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT']
    if sum(1 for col in detailed_stats if col in df.columns) >= 7:
        real_data_indicators += 1
    
    # If 4 or more indicators are present, consider it real data
    return real_data_indicators >= 4

# Initialize NBA Data Service
nba_service = None
if NBA_SERVICE_AVAILABLE:
    nba_service = WorkingNBADataService()
    current_season = nba_service.get_current_season()
    print(f"✅ Real NBA Service initialized for season: {current_season}")
    print(f"📅 Preseason active: {nba_service.is_preseason_active()}")
else:
    current_season = "2024-25"
    print("⚠️ Using fallback mode - no real NBA data service")

# Initialize Advanced Opponent Analyzer
opponent_analyzer = None
if ADVANCED_ANALYSIS_AVAILABLE:
    opponent_analyzer = AdvancedOpponentAnalyzer()
    print("✅ Advanced Opponent Analyzer initialized")
else:
    print("⚠️ Using basic opponent analysis")

# Load player data at startup
print("Loading player data...")
df = None
model = "statistical"
scaler = None
feature_columns = None

# Try to load real NBA data first
real_data_files = [f for f in os.listdir('.') if f.startswith('real_nba_data_') and f.endswith('.csv')]
if real_data_files:
    # Use the most recent real data file
    latest_file = sorted(real_data_files)[-1]
    print(f"Found real NBA data: {latest_file}")
    try:
        df = pd.read_csv(latest_file)
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        print(f"✅ Real NBA data loaded: {len(df)} games for {df['PLAYER_ID'].nunique()} players")
    except Exception as e:
        print(f"Error loading real data: {e}")
        df = None

# Check for existing NBA data file (might be real data with different name)
if df is None:
    try:
        df = pd.read_csv('player_game_logs_2024_25.csv')
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        
        # Check if this is real NBA data by examining the data quality
        is_real_data = detect_real_nba_data(df)
        
        if is_real_data:
            print(f"✅ Real NBA data loaded: {len(df)} games for {df['PLAYER_ID'].nunique()} players")
            print("✅ This appears to be real NBA data with detailed statistics")
        else:
            print(f"⚠️  Using fallback data: {len(df)} games for {df['PLAYER_ID'].nunique()} players")
            print("⚠️  WARNING: This appears to be synthetic data - predictions may not be accurate")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        df = None
        print("❌ No data available - predictions will not work")

def find_player_by_name(player_name):
    """Find player by name (fuzzy matching)"""
    if df is None:
        return None
    
    # Clean the input name
    clean_name = player_name.strip().lower()
    
    # Try exact match first
    exact_match = df[df['PLAYER_NAME'].str.lower() == clean_name]
    if not exact_match.empty:
        return exact_match['PLAYER_ID'].iloc[0], exact_match['PLAYER_NAME'].iloc[0]
    
    # Try partial match
    partial_match = df[df['PLAYER_NAME'].str.lower().str.contains(clean_name, na=False)]
    if not partial_match.empty:
        return partial_match['PLAYER_ID'].iloc[0], partial_match['PLAYER_NAME'].iloc[0]
    
    # Try last name match
    if ' ' in clean_name:
        last_name = clean_name.split()[-1]
        last_name_match = df[df['PLAYER_NAME'].str.lower().str.contains(last_name, na=False)]
        if not last_name_match.empty:
            return last_name_match['PLAYER_ID'].iloc[0], last_name_match['PLAYER_NAME'].iloc[0]
    
    return None, None

def create_basic_features(player_data):
    """Create basic features for prediction"""
    # Calculate rolling averages
    for window in [3, 5, 10]:
        for stat in ['PTS', 'FGA', 'FG_PCT', 'MIN', 'AST', 'REB']:
            player_data[f'{stat}_last_{window}_avg'] = player_data[stat].rolling(window=window, min_periods=1).mean().shift(1)
    
    # Basic game context
    player_data['is_home'] = player_data['MATCHUP'].str.contains('vs.').astype(int)
    player_data['game_number'] = range(1, len(player_data) + 1)
    
    return player_data

def get_next_opponent_from_schedule(team_abbr):
    """Get next opponent from real NBA schedule data"""
    if df is None:
        return None
    
    # First, try to get real schedule data if NBA service is available
    if nba_service and NBA_SERVICE_AVAILABLE:
        try:
            # Get real next opponent from ESPN API
            next_opponent = nba_service.get_next_opponent(team_abbr)
            if next_opponent:
                print(f"✅ Real schedule found: {team_abbr} next plays {next_opponent}")
                return next_opponent
        except Exception as e:
            print(f"Error getting real schedule: {e}")
    
    # Fallback to historical data analysis
    print(f"⚠️  Using fallback opponent prediction for {team_abbr}")
    
    # Get all games for this team, sorted by date
    team_games = df[df['TEAM_ABBREVIATION'] == team_abbr].copy()
    team_games = team_games.sort_values('GAME_DATE')
    
    if len(team_games) == 0:
        return None
    
    # Extract opponents from matchups
    team_games['opponent'] = team_games['MATCHUP'].str.extract(r'(?:@|vs\.)\s*(\w+)')
    
    # Get the last game date
    last_game_date = team_games['GAME_DATE'].max()
    
    # Use rotation logic based on recent opponents
    recent_opponents = team_games['opponent'].tail(10).dropna().tolist()
    
    # All NBA teams
    all_teams = ['ATL', 'BOS', 'BRK', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
                'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
                'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
    
    # Remove own team
    other_teams = [t for t in all_teams if t != team_abbr]
    
    # Find teams not played in last 5 games (most likely next opponents)
    not_recent = [t for t in other_teams if t not in recent_opponents[-5:]]
    
    if not_recent:
        # Use team abbreviation hash to create variety
        team_hash = sum(ord(c) for c in team_abbr)
        return not_recent[team_hash % len(not_recent)]
    else:
        # If they've played everyone recently, return least frequent opponent
        opponent_counts = {}
        for opp in recent_opponents:
            opponent_counts[opp] = opponent_counts.get(opp, 0) + 1
        
        if opponent_counts:
            # Get all teams with minimum frequency
            min_count = min(opponent_counts.values())
            least_frequent_teams = [team for team, count in opponent_counts.items() if count == min_count]
            
            # Use hash to pick from least frequent teams
            team_hash = sum(ord(c) for c in team_abbr)
            return least_frequent_teams[team_hash % len(least_frequent_teams)]
        
    return None

def get_player_prediction(player_name, specific_opponent=None):
    """Get advanced prediction for a specific player with automatic opponent detection"""
    if df is None:
        return None, "Player data not loaded"
    
    # Find player
    player_id, actual_name = find_player_by_name(player_name)
    if player_id is None:
        return None, f"Player '{player_name}' not found"
    
    # Get player's data
    player_data = df[df['PLAYER_ID'] == player_id].copy()
    if len(player_data) < 5:
        return None, f"Not enough games for {actual_name} (need at least 5)"
    
    # Create basic features
    player_data = create_basic_features(player_data)
    
    # Extract opponent from MATCHUP column first
    player_data['opponent'] = player_data['MATCHUP'].str.extract(r'(?:@|vs\.)\s*(\w+)')
    
    # Get player's team and determine next likely opponent
    player_team = player_data['TEAM_ABBREVIATION'].iloc[-1]
    next_opponent = get_next_opponent_from_schedule(player_team)
    
    # If no specific opponent provided, use the predicted next opponent
    if not specific_opponent and next_opponent:
        specific_opponent = next_opponent
    
    # ADVANCED PREDICTION ALGORITHM
    
    # 1. Base scoring averages
    recent_5_avg = player_data['PTS'].tail(5).mean()
    recent_10_avg = player_data['PTS'].tail(10).mean()
    season_avg = player_data['PTS'].mean()
    
    # 2. Minutes played impact (more minutes = more points opportunity)
    recent_minutes = player_data['MIN'].tail(5).mean()
    season_minutes = player_data['MIN'].mean()
    minutes_factor = min(1.2, recent_minutes / max(season_minutes, 1))  # Cap at 20% boost
    
    # 3. Shooting efficiency trends
    recent_fg_pct = player_data['FG_PCT'].tail(5).mean()
    season_fg_pct = player_data['FG_PCT'].mean()
    efficiency_trend = recent_fg_pct / max(season_fg_pct, 0.3) if season_fg_pct > 0 else 1.0
    efficiency_factor = min(1.15, max(0.85, efficiency_trend))  # 15% swing based on shooting
    
    # 4. Usage rate proxy (shot attempts indicate role)
    recent_fga = player_data['FGA'].tail(5).mean()
    season_fga = player_data['FGA'].mean()
    usage_trend = recent_fga / max(season_fga, 1) if season_fga > 0 else 1.0
    usage_factor = min(1.1, max(0.9, usage_trend))  # 10% swing based on shot attempts
    
    # 5. Home vs Away performance
    home_games = player_data[player_data['is_home'] == 1]
    away_games = player_data[player_data['is_home'] == 0]
    
    if len(home_games) > 0 and len(away_games) > 0:
        home_avg = home_games['PTS'].mean()
        away_avg = away_games['PTS'].mean()
        # Assume next game is neutral (could be enhanced with actual game context)
        venue_factor = (home_avg + away_avg) / (2 * season_avg) if season_avg > 0 else 1.0
    else:
        venue_factor = 1.0
    
    # 6. Opponent-specific performance analysis
    # Calculate performance against different opponent types
    opponent_factors = {}
    
    # Group opponents by defensive strength (based on points allowed to this player)
    for opp in player_data['opponent'].dropna().unique():
        opp_games = player_data[player_data['opponent'] == opp]
        if len(opp_games) >= 2:  # Need at least 2 games for reliable data
            avg_vs_opp = opp_games['PTS'].mean()
            opponent_factors[opp] = avg_vs_opp / max(season_avg, 1) if season_avg > 0 else 1.0
    
    # Calculate overall matchup difficulty factor
    if opponent_factors:
        # Categorize opponents as tough/average/favorable based on player's performance
        tough_matchups = [f for f in opponent_factors.values() if f < 0.85]  # Player scores <85% of average
        favorable_matchups = [f for f in opponent_factors.values() if f > 1.15]  # Player scores >115% of average
        
        # Create matchup difficulty context
        if len(tough_matchups) > len(favorable_matchups):
            matchup_context = "faces tough defenses"
            avg_opponent_factor = sum(opponent_factors.values()) / len(opponent_factors)
        elif len(favorable_matchups) > len(tough_matchups):
            matchup_context = "favorable matchups"
            avg_opponent_factor = sum(opponent_factors.values()) / len(opponent_factors)
        else:
            matchup_context = "mixed matchups"
            avg_opponent_factor = 1.0
    else:
        matchup_context = "limited matchup data"
        avg_opponent_factor = 1.0
    
    # Cap opponent factor to reasonable range
    opponent_factor = max(0.8, min(1.2, avg_opponent_factor))
    
    # ENHANCED OPPONENT ANALYSIS (if provided)
    specific_opponent_factor = 1.0
    specific_matchup_info = "general analysis"
    advanced_matchup_analysis = {}
    
    if specific_opponent:
        # Try advanced opponent analysis first
        if opponent_analyzer and ADVANCED_ANALYSIS_AVAILABLE:
            try:
                # Determine player position (simplified - could be enhanced with actual position data)
                player_position = 'SF'  # Default, could be determined from player data
                
                # Get enhanced opponent analysis
                enhanced_factor, enhanced_analysis, position_analysis = opponent_analyzer.get_enhanced_opponent_factor(
                    player_data, specific_opponent.upper(), player_position
                )
                
                specific_opponent_factor = enhanced_factor
                specific_matchup_info = enhanced_analysis
                advanced_matchup_analysis = position_analysis
                
                print(f"🔍 Advanced analysis: {enhanced_analysis}")
                print(f"   Defensive factor: {position_analysis.get('defensive_factor', 1.0):.3f}")
                
            except Exception as e:
                print(f"Advanced analysis failed: {e}")
                # Fallback to basic analysis
                if specific_opponent.upper() in opponent_factors:
                    specific_opponent_factor = opponent_factors[specific_opponent.upper()]
                    specific_opponent_factor = max(0.7, min(1.3, specific_opponent_factor))
                    
                    if specific_opponent_factor > 1.1:
                        specific_matchup_info = f"historically strong vs {specific_opponent.upper()}"
                    elif specific_opponent_factor < 0.9:
                        specific_matchup_info = f"historically struggles vs {specific_opponent.upper()}"
                    else:
                        specific_matchup_info = f"average performance vs {specific_opponent.upper()}"
        else:
            # Basic opponent analysis fallback
            if specific_opponent.upper() in opponent_factors:
                specific_opponent_factor = opponent_factors[specific_opponent.upper()]
                specific_opponent_factor = max(0.7, min(1.3, specific_opponent_factor))
                
                if specific_opponent_factor > 1.1:
                    specific_matchup_info = f"historically strong vs {specific_opponent.upper()}"
                elif specific_opponent_factor < 0.9:
                    specific_matchup_info = f"historically struggles vs {specific_opponent.upper()}"
                else:
                    specific_matchup_info = f"average performance vs {specific_opponent.upper()}"
        
        # Use specific opponent factor instead of general one
        opponent_factor = specific_opponent_factor
    
    # 7. Momentum factor (recent trend)
    if len(player_data) >= 10:
        first_half = player_data['PTS'].head(len(player_data)//2).mean()
        second_half = player_data['PTS'].tail(len(player_data)//2).mean()
        momentum_factor = min(1.1, max(0.9, second_half / max(first_half, 1)))
    else:
        momentum_factor = 1.0
    
    # 8. Rest days impact (players perform better with rest)
    if 'days_rest' in player_data.columns:
        avg_rest = player_data['days_rest'].tail(5).mean()
        if avg_rest >= 2:
            rest_factor = 1.05  # 5% boost for well-rested
        elif avg_rest < 1:
            rest_factor = 0.95  # 5% penalty for tired
        else:
            rest_factor = 1.0
    else:
        rest_factor = 1.0
    
    # 9. Season progression factor (early season vs established patterns)
    games_played = len(player_data)
    if games_played < 10:
        # Early season - less reliable, more volatile
        early_season_penalty = 0.95
        confidence_penalty = 0.8  # Lower confidence early in season
    elif games_played < 20:
        early_season_penalty = 0.98
        confidence_penalty = 0.9
    else:
        early_season_penalty = 1.0
        confidence_penalty = 1.0
    
    # WEIGHTED PREDICTION FORMULA
    base_prediction = (recent_5_avg * 0.4) + (recent_10_avg * 0.3) + (season_avg * 0.3)
    
    # Apply all factors with diminishing returns to prevent extreme predictions
    combined_factor = (minutes_factor * 
                      efficiency_factor * 
                      usage_factor * 
                      venue_factor * 
                      opponent_factor * 
                      momentum_factor * 
                      rest_factor * 
                      early_season_penalty)
    
    # Cap the total adjustment to prevent unrealistic predictions
    # Use square root to create diminishing returns on extreme multipliers
    if combined_factor > 1.3:
        combined_factor = 1.0 + (combined_factor - 1.0) * 0.7  # Reduce impact of extreme boosts
    elif combined_factor < 0.7:
        combined_factor = 1.0 - (1.0 - combined_factor) * 0.7  # Reduce impact of extreme penalties
    
    final_prediction = base_prediction * combined_factor
    
    # ADVANCED CONFIDENCE CALCULATION
    
    # 1. Consistency (standard deviation)
    recent_pts = player_data['PTS'].tail(10)
    std_dev = recent_pts.std() if len(recent_pts) > 1 else 5.0
    consistency_score = max(0.3, min(0.95, 1 / (1 + std_dev/8)))
    
    # 2. Sample size confidence
    sample_confidence = min(1.0, games_played / 20)  # More games = more confidence
    
    # 3. Role stability (consistent minutes = predictable role)
    minutes_std = player_data['MIN'].tail(10).std() if len(player_data) >= 10 else 10
    role_stability = max(0.5, min(1.0, 1 / (1 + minutes_std/15)))
    
    # 4. Shooting consistency
    fg_pct_std = player_data['FG_PCT'].tail(10).std() if len(player_data) >= 10 else 0.1
    shooting_consistency = max(0.7, min(1.0, 1 / (1 + fg_pct_std*5)))
    
    # Combined confidence score
    confidence = (consistency_score * 0.4 + 
                 sample_confidence * 0.3 + 
                 role_stability * 0.2 + 
                 shooting_consistency * 0.1) * confidence_penalty
    
    # Ensure confidence stays within reasonable bounds
    confidence = max(0.25, min(0.95, confidence))
    
    # Calculate additional insights
    trend_direction = "📈 Trending Up" if momentum_factor > 1.02 else "📉 Trending Down" if momentum_factor < 0.98 else "➡️ Stable"
    
    return {
        'playerName': actual_name,
        'predictedPoints': round(final_prediction, 1),
        'confidence': round(confidence, 2),
        'seasonAverage': round(season_avg, 1),
        'last5Average': round(recent_5_avg, 1),
        'last10Average': round(recent_10_avg, 1),
        'gamesPlayed': len(player_data),
        'lastGamePoints': int(player_data['PTS'].iloc[-1]),
        'recentMinutes': round(recent_minutes, 1),
        'shootingTrend': round(efficiency_factor, 3),
        'usageTrend': round(usage_factor, 3),
        'momentumFactor': round(momentum_factor, 3),
        'opponentFactor': round(opponent_factor, 3),
        'matchupContext': matchup_context,
        'specificMatchup': specific_matchup_info,
        'nextOpponent': next_opponent,
        'trend': trend_direction,
        'team': player_data['TEAM_ABBREVIATION'].iloc[-1],
        'createdAt': datetime.now().isoformat(),
        'advancedMatchupAnalysis': advanced_matchup_analysis
    }, None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        player_name = data.get('playerName', '').strip()
        
        if not player_name:
            return jsonify({'error': 'Player name is required'}), 400
        
        # System automatically determines next opponent and factors it in
        prediction, error = get_player_prediction(player_name)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/predict_next_game/<player_name>', methods=['GET'])
def predict_next_game(player_name):
    """Predict for player's next scheduled game (would need live schedule data)"""
    try:
        # This would integrate with NBA schedule API in live season
        # For now, returns general prediction
        prediction, error = get_player_prediction(player_name)
        
        if error:
            return jsonify({'error': error}), 400
            
        # Add note about live season capability
        prediction['note'] = "Live opponent analysis available when schedule data is integrated"
        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    # Check if we're using real or synthetic data
    data_source = "unknown"
    data_quality = "unknown"
    
    if df is not None:
        # Use the same detection logic as data loading
        is_real_data = detect_real_nba_data(df)
        if is_real_data:
            data_source = "real_nba_data"
            data_quality = "high"
        else:
            data_source = "synthetic_fallback"
            data_quality = "medium"
    
    status = {
        'status': 'healthy',
        'prediction_method': 'statistical' if model == "statistical" else 'none',
        'data_loaded': df is not None,
        'players_available': df['PLAYER_ID'].nunique() if df is not None else 0,
        'data_source': data_source,
        'data_quality': data_quality,
        'nba_service_available': NBA_SERVICE_AVAILABLE,
        'current_season': current_season,
        'recommendations': []
    }
    
    # Add recommendations based on data quality
    if data_quality == "low":
        status['recommendations'].append("⚠️ Using synthetic data - predictions may not be accurate")
        status['recommendations'].append("💡 Run collect_real_nba_data.py to get real NBA data")
    elif data_quality == "medium":
        status['recommendations'].append("📊 Using historical data - predictions based on past performance")
    elif data_quality == "high":
        status['recommendations'].append("✅ Using real NBA data - predictions should be accurate")
    
    if not NBA_SERVICE_AVAILABLE:
        status['recommendations'].append("🔧 NBA API service not available - using fallback methods")
    
    return jsonify(status)

@app.route('/data-info', methods=['GET'])
def data_info():
    """Get detailed information about the data being used"""
    if df is None:
        return jsonify({'error': 'No data loaded'}), 500
    
    info = {
        'total_games': len(df),
        'total_players': df['PLAYER_ID'].nunique(),
        'date_range': {
            'earliest': df['GAME_DATE'].min().isoformat() if 'GAME_DATE' in df.columns else None,
            'latest': df['GAME_DATE'].max().isoformat() if 'GAME_DATE' in df.columns else None
        },
        'teams': sorted(df['TEAM_ABBREVIATION'].unique().tolist()) if 'TEAM_ABBREVIATION' in df.columns else [],
        'sample_players': df['PLAYER_NAME'].value_counts().head(10).to_dict() if 'PLAYER_NAME' in df.columns else {},
        'data_columns': list(df.columns),
        'is_real_data': 'real_nba_data_' in str(df.columns) or (df['GAME_DATE'].max().year <= datetime.now().year if 'GAME_DATE' in df.columns else False)
    }
    
    return jsonify(info)

@app.route('/players', methods=['GET'])
def get_players():
    """Get list of available players"""
    if df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    # Get unique players with their stats
    players = df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION']).agg({
        'PTS': ['count', 'mean'],
        'GAME_DATE': 'max'
    }).round(1)
    
    players.columns = ['games_played', 'avg_points', 'last_game_date']
    players = players.reset_index()
    
    # Sort by average points (descending)
    players = players.sort_values('avg_points', ascending=False)
    
    return jsonify(players.to_dict('records'))

if __name__ == '__main__':
    app.run(port=5004, debug=True)