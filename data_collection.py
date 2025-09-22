import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

# Load your data
df = pd.read_csv('player_game_logs_2024_25.csv')
print(f"Dataset shape: {df.shape}")

# Ask for playerID
min_length = 1
while True:
    player_ID = input("Enter player ID: ")
    
    if not player_ID.isdigit():
        print("Error: Player ID must contain only digits")
    elif len(player_ID) < min_length:
        print(f"Error: Player ID must be {min_length} digits long")
    else:
        player_id = int(player_ID)
        break  

# Sort by player and date
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

# Filter for player ID
filter_player_ID = df[df['PLAYER_ID'] == player_id]
 
# Create features based on past performance
def create_lag_features(group, n_games=5):
    """Create features based on previous games for each player"""
    # Previous n games averages
    for window in [3, 5, 10]:
        for stat in ['PTS', 'FGA', 'FG_PCT', 'FG3A', 'FG3_PCT', 'FTA', 'FT_PCT', 'MIN', 'AST', 'REB']:
            group[f'{stat}_last_{window}_avg'] = group[stat].rolling(window=window, min_periods=1).mean().shift(1)
    
    # ADVANCED FEATURES
    # Variance/Consistency features
    group['PTS_variance_5'] = group['PTS'].rolling(window=5, min_periods=1).std().shift(1)
    group['MIN_variance_5'] = group['MIN'].rolling(window=5, min_periods=1).std().shift(1)
    
    # Momentum features
    group['scoring_momentum'] = (
        group['PTS'].rolling(3).mean() - 
        group['PTS'].rolling(10).mean()
    ).shift(1)
    
    # Trend (difference between last game and average of 5 games before that)
    for stat in ['PTS', 'MIN', 'REB', 'AST']:
        last_game = group[stat].shift(1)
        prev_5_avg = group[stat].shift(2).rolling(window=n_games-1, min_periods=1).mean()
        group[f'{stat}_trend'] = last_game - prev_5_avg
    
    # Hot/Cold streak
    group['last_3_vs_season_avg'] = (
        group['PTS'].rolling(3).mean() / 
        group['PTS'].expanding().mean()
    ).shift(1)
    
    # 1. DAYS OF REST (huge impact on performance)
    group['days_rest'] = group['GAME_DATE'].diff().dt.days.fillna(2)
    group['days_rest'] = group['days_rest'].shift(1)  # For previous game
    group['days_rest_capped'] = group['days_rest'].clip(upper=5)  # Cap at 5 days
    
    # More detailed rest features
    group['is_rested'] = (group['days_rest'] >= 2).astype(int)
    group['is_tired'] = (group['days_rest'] == 0).astype(int)
    
    # 2. BACK-TO-BACK GAMES (players perform worse when tired)
    group['is_back_to_back'] = (group['days_rest'] == 1).astype(int)
    
    # 3. HOME/AWAY (home court advantage)
    group['is_home'] = group['MATCHUP'].str.contains('vs.').astype(int)
    
    # Recent home/away performance
    group['home_game_streak'] = group['is_home'].groupby((group['is_home'] != group['is_home'].shift()).cumsum()).cumsum()
    
    # 4. EXTRACT OPPONENT FROM MATCHUP
    group['opponent'] = group['MATCHUP'].str.extract(r'(?:vs\.|@)\s*(\w+)')[0]
    
    # 5. GAME NUMBER IN SEASON (fatigue over time)
    group['game_number'] = range(1, len(group) + 1)
    group['pct_season_complete'] = group['game_number'] / 82
    
    # Calculate games in last 7 days
    group = group.set_index('GAME_DATE')
    group['games_in_last_week'] = group['game_number'].rolling('7D').count().shift(1)
    group = group.reset_index()
    
    # Fatigue indicator
    group['cumulative_minutes'] = group['MIN'].rolling(5).sum().shift(1)
    group['is_heavy_usage'] = (group['MIN'].rolling(3).mean() > 35).astype(int).shift(1)
    
    return group

def add_opponent_features_properly(df):
    """Add opponent strength metrics with no data leakage"""
    
    # Sort by date to ensure chronological order
    df = df.sort_values('GAME_DATE')
    
    # Initialize new columns
    df['opp_pts_allowed_L10'] = np.nan
    df['opp_fg_pct_allowed_L10'] = np.nan
    df['opp_reb_allowed_L10'] = np.nan
    df['opp_win_pct_L10'] = np.nan
    
    # For each game, calculate opponent's recent performance
    for idx, row in df.iterrows():
        game_date = row['GAME_DATE']
        opponent = row['opponent']
        
        # Get opponent's last 10 games before this date
        opp_recent = df[
            (df['TEAM_ABBREVIATION'] == opponent) & 
            (df['GAME_DATE'] < game_date)
        ].tail(10)
        
        if len(opp_recent) >= 5:  # Need at least 5 games for reliable stats
            df.at[idx, 'opp_pts_allowed_L10'] = opp_recent['PTS'].mean()
            df.at[idx, 'opp_fg_pct_allowed_L10'] = opp_recent['FG_PCT'].mean()
            df.at[idx, 'opp_reb_allowed_L10'] = opp_recent['REB'].mean()
            df.at[idx, 'opp_win_pct_L10'] = (opp_recent['WL'] == 'W').mean()
    
    # Fill NaN with league averages
    df['opp_pts_allowed_L10'].fillna(df['PTS'].mean(), inplace=True)
    df['opp_fg_pct_allowed_L10'].fillna(df['FG_PCT'].mean(), inplace=True)
    df['opp_reb_allowed_L10'].fillna(df['REB'].mean(), inplace=True)
    df['opp_win_pct_L10'].fillna(0.5, inplace=True)
    
    # Calculate if opponent is good/bad defensively
    avg_pts_allowed = df['opp_pts_allowed_L10'].mean()
    df['opp_is_good_defense'] = (df['opp_pts_allowed_L10'] < avg_pts_allowed).astype(int)
    
    # Opponent strength indicator
    df['opp_is_elite'] = (df['opp_win_pct_L10'] > 0.6).astype(int)
    df['opp_is_poor'] = (df['opp_win_pct_L10'] < 0.4).astype(int)
    
    return df

def create_advanced_features(df):
    """Create additional advanced features"""
    
    # Sort by player and date to ensure proper grouping
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
    
    # Double-double potential - simplified version
    pts_dd = df.groupby('PLAYER_ID')['PTS'].rolling(5, min_periods=1).apply(lambda x: (x >= 10).sum())
    reb_dd = df.groupby('PLAYER_ID')['REB'].rolling(5, min_periods=1).apply(lambda x: (x >= 10).sum())
    
    # Reset index to align with original dataframe
    pts_dd = pts_dd.reset_index(level=0, drop=True)
    reb_dd = reb_dd.reset_index(level=0, drop=True)
    
    # Shift and fill NaN
    df['recent_double_doubles'] = (pts_dd.shift(1) + reb_dd.shift(1)).fillna(0)
    
    # Usage rate proxy (FGA + FTA * 0.44) / MIN
    df['usage_rate_proxy'] = ((df['FGA'] + df['FTA'] * 0.44) / df['MIN']).replace([np.inf, -np.inf], 0)
    usage_l5 = df.groupby('PLAYER_ID')['usage_rate_proxy'].rolling(5, min_periods=1).mean()
    df['usage_rate_L5'] = usage_l5.reset_index(level=0, drop=True).shift(1)
    
    # Efficiency trends
    df['pts_per_fga'] = (df['PTS'] / df['FGA']).replace([np.inf, -np.inf], 0).fillna(0)
    pts_per_fga_l5 = df.groupby('PLAYER_ID')['pts_per_fga'].rolling(5, min_periods=1).mean()
    df['pts_per_fga_L5'] = pts_per_fga_l5.reset_index(level=0, drop=True).shift(1)
    
    # Blowout risk (might affect minutes)
    df['team_avg_pts'] = df.groupby('TEAM_ABBREVIATION')['PTS'].transform('mean')
    df['likely_blowout'] = (abs(df['team_avg_pts'] - df['opp_pts_allowed_L10']) > 15).astype(int)
    
    return df

# Apply function to each player's data
print("Processing player features...")
df = df.groupby('PLAYER_ID', group_keys=False).apply(create_lag_features).reset_index(drop=True)

# Add opponent features properly (no data leakage)
print("Processing opponent features...")
df = add_opponent_features_properly(df)

# Add advanced features
print("Creating advanced features...")
df = create_advanced_features(df)

# Calculate player's performance vs specific opponents
for opponent in df['opponent'].unique():
    mask = df['opponent'] == opponent
    df.loc[mask, 'avg_pts_vs_opponent'] = df[mask].groupby('PLAYER_ID')['PTS'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )

# Remove NaN values (first games for each player)
df = df.dropna()

# Define features and target
feature_cols = [col for col in df.columns if '_last_' in col or '_trend' in col or '_L5' in col or '_L10' in col]
feature_cols += [
    'TEAM_ABBREVIATION',
    'days_rest_capped',
    'is_back_to_back',
    'is_rested',
    'is_tired',
    'is_home',
    'home_game_streak',
    'game_number',
    'pct_season_complete',
    'games_in_last_week',
    'cumulative_minutes',
    'is_heavy_usage',
    'opponent',
    'opp_pts_allowed_L10',
    'opp_fg_pct_allowed_L10',
    'opp_reb_allowed_L10',
    'opp_win_pct_L10',
    'opp_is_good_defense',
    'opp_is_elite',
    'opp_is_poor',
    'avg_pts_vs_opponent',
    'PTS_variance_5',
    'MIN_variance_5',
    'scoring_momentum',
    'last_3_vs_season_avg',
    'recent_double_doubles',
    'usage_rate_L5',
    'pts_per_fga_L5',
    'likely_blowout'
]

# Remove any columns that don't exist
feature_cols = [col for col in feature_cols if col in df.columns]

X = pd.get_dummies(df[feature_cols], columns=['TEAM_ABBREVIATION', 'opponent'])
y = df['PTS']

# Train/test split (keeping chronological order)
cutoff_date = df['GAME_DATE'].quantile(0.8)
train_mask = df['GAME_DATE'] < cutoff_date
test_mask = df['GAME_DATE'] >= cutoff_date

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print(f"\nTraining on games before {cutoff_date.date()}")
print(f"Testing on games after {cutoff_date.date()}")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Number of features: {X_train.shape[1]}")

# Standardize
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# Train multiple models
print("\n" + "="*50)
print("TRAINING MODELS")
print("="*50)

# 1. Ridge regression with cross-validation
print("Training Ridge Regression...")
ridge_cv = RidgeCV(alphas=[10, 50, 100, 200, 500, 1000], cv=5)
ridge_cv.fit(X_train_scaled, y_train)
ridge_pred = ridge_cv.predict(X_test_scaled)

# 2. Random Forest
print("Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

# 3. Gradient Boosting
print("Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)

# 4. Ensemble (Voting Regressor) - without XGBoost
print("Creating Ensemble Model...")
ensemble_model = VotingRegressor([
    ('ridge', ridge_cv),
    ('rf', rf_model),
    ('gb', gb_model)
])
ensemble_model.fit(X_train_scaled, y_train)
ensemble_pred = ensemble_model.predict(X_test_scaled)

# Evaluate all models
print("\n" + "="*50)
print("MODEL PERFORMANCE COMPARISON")
print("="*50)

models = {
    'Ridge': ridge_pred,
    'Random Forest': rf_pred,
    'Gradient Boosting': gb_pred,
    'Ensemble': ensemble_pred
}

best_model = None
best_r2 = -1
best_model_obj = None

for name, predictions in models.items():
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    within_5 = sum(abs(y_test - predictions) <= 5) / len(y_test)
    
    print(f"\n{name}:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Within 5 pts: {within_5:.1%}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = name
        best_predictions = predictions
        if name == 'Random Forest':
            best_model_obj = rf_model
        elif name == 'Gradient Boosting':
            best_model_obj = gb_model
        elif name == 'Ensemble':
            best_model_obj = ensemble_model
        else:
            best_model_obj = ridge_cv

print(f"\n{'='*50}")
print(f"BEST MODEL: {best_model} with R² = {best_r2:.4f}")
print(f"{'='*50}")

# Use best model for player predictions
y_pred = best_predictions

# Find test data for specified player
player_processed = df[df['PLAYER_ID'] == player_id]

if not player_processed.empty:
    # Find which rows of player data are in the test set
    player_test_mask = (df['PLAYER_ID'] == player_id) & test_mask
    
    if player_test_mask.any():
        # Get actual vs predicted for this player
        player_actual = df[player_test_mask]['PTS'].values
        test_df_indices = df[test_mask].index.tolist()
        player_indices = df[player_test_mask].index.tolist()
        
        player_predictions = []
        for idx in player_indices:
            if idx in test_df_indices:
                position_in_test = test_df_indices.index(idx)
                player_predictions.append(y_pred[position_in_test])
        
        if player_predictions:
            print(f"\nPlayer {player_id} Test Set Performance:")
            print(f"{'Game':<8} {'Actual':<10} {'Predicted':<12} {'Error':<8}")
            print("-" * 40)
            for i, (actual, pred) in enumerate(zip(player_actual, player_predictions)):
                error = pred - actual
                print(f"{i+1:<8} {actual:<10.1f} {pred:<12.1f} {error:+.1f}")
            
            player_mae = mean_absolute_error(player_actual, player_predictions)
            print(f"\nPlayer MAE: {player_mae:.2f}")
    
    # Predict next game
    print("\n" + "="*50)
    print("NEXT GAME PREDICTION")
    print("="*50)
    
    last_game = player_processed.iloc[-1:]
    
    # Extract features
    last_features = last_game[feature_cols].copy()
    last_features_encoded = pd.get_dummies(last_features, columns=['TEAM_ABBREVIATION', 'opponent'])
    
    # Align with training features
    last_features_final = pd.DataFrame(columns=X_train.columns)
    for col in X_train.columns:
        if col in last_features_encoded.columns:
            last_features_final[col] = last_features_encoded[col].values
        else:
            last_features_final[col] = [0]
    
    last_features_scaled = scaler.transform(last_features_final)
    
    # Get predictions from all models
    predictions = {}
    for name, model in [('Ridge', ridge_cv), ('RF', rf_model), 
                        ('GB', gb_model), ('Ensemble', ensemble_model)]:
        predictions[name] = model.predict(last_features_scaled)[0]
    
    # Display predictions
    player_name = last_game['PLAYER_NAME'].values[0]
    team = last_game['TEAM_ABBREVIATION'].values[0]
    opponent = last_game['opponent'].values[0]
    is_home = 'Home' if last_game['is_home'].values[0] == 1 else 'Away'
    days_rest = last_game['days_rest'].values[0] if 'days_rest' in last_game.columns else 'N/A'
    
    print(f"Player: {player_name} (ID: {player_id})")
    print(f"Team: {team} vs {opponent} ({is_home})")
    print(f"Days Rest: {days_rest}")
    print(f"Last Game: {last_game['PTS'].values[0]:.0f} points")
    print(f"\nModel Predictions:")
    for name, pred in predictions.items():
        print(f"  {name}: {pred:.1f} points")
    
    avg_prediction = np.mean(list(predictions.values()))
    std_prediction = np.std(list(predictions.values()))
    print(f"\nConsensus: {avg_prediction:.1f} ± {std_prediction:.1f} points")

# Feature importance from best non-linear model
print("\n" + "="*50)
print("TOP 20 MOST IMPORTANT FEATURES")
print("="*50)

if best_model in ['Random Forest', 'Gradient Boosting']:
    if best_model == 'Random Forest':
        importances = rf_model.feature_importances_
    else:
        importances = gb_model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(20).iterrows():
        # Clean up feature names for display
        feature_name = row['feature']
        if feature_name.startswith('opponent_'):
            feature_name = f"vs {feature_name.replace('opponent_', '')}"
        print(f"{feature_name}: {row['importance']:.4f}")

# Cross-validation for best model
print("\n" + "="*50)
print("TIME SERIES CROSS-VALIDATION")
print("="*50)

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled), 1):
    X_cv_train, X_cv_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    if best_model == 'Random Forest':
        model = RandomForestRegressor(**rf_model.get_params())
    elif best_model == 'Gradient Boosting':
        model = GradientBoostingRegressor(**gb_model.get_params())
    elif best_model == 'Ensemble':
        # For ensemble, just use the existing one
        model = ensemble_model
    else:
        model = ridge_cv
    
    model.fit(X_cv_train, y_cv_train)
    score = model.score(X_cv_val, y_cv_val)
    cv_scores.append(score)
    print(f"Fold {fold}: R² = {score:.4f}")

print(f"\nAverage CV R²: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)

os.makedirs('models', exist_ok=True)

# Save the best model (ensemble_model in this case)
joblib.dump(ensemble_model, 'models/ensemble_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(list(X_train.columns), 'models/feature_columns.pkl')

print("\n" + "="*50)
print("MODELS SAVED")
print("="*50)
print("Saved files:")
print("  - models/ensemble_model.pkl")
print("  - models/scaler.pkl")
print("  - models/feature_columns.pkl")