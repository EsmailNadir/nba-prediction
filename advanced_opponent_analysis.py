#!/usr/bin/env python3
"""
Advanced Opponent Analysis Module
Analyzes opponent defensive metrics and position-specific matchups
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

class AdvancedOpponentAnalyzer:
    def __init__(self):
        # NBA team defensive rankings (example data - would be updated with real stats)
        self.team_defensive_metrics = {
            'ATL': {'def_rating': 110.5, 'opp_ppg': 112.3, 'opp_fg_pct': 0.456, 'opp_3pt_pct': 0.358},
            'BOS': {'def_rating': 108.2, 'opp_ppg': 108.9, 'opp_fg_pct': 0.442, 'opp_3pt_pct': 0.345},
            'BRK': {'def_rating': 112.8, 'opp_ppg': 115.1, 'opp_fg_pct': 0.468, 'opp_3pt_pct': 0.362},
            'CHA': {'def_rating': 115.3, 'opp_ppg': 118.7, 'opp_fg_pct': 0.472, 'opp_3pt_pct': 0.368},
            'CHI': {'def_rating': 109.7, 'opp_ppg': 111.2, 'opp_fg_pct': 0.448, 'opp_3pt_pct': 0.352},
            'CLE': {'def_rating': 107.8, 'opp_ppg': 109.4, 'opp_fg_pct': 0.441, 'opp_3pt_pct': 0.343},
            'DAL': {'def_rating': 111.2, 'opp_ppg': 113.8, 'opp_fg_pct': 0.461, 'opp_3pt_pct': 0.355},
            'DEN': {'def_rating': 108.9, 'opp_ppg': 110.6, 'opp_fg_pct': 0.445, 'opp_3pt_pct': 0.348},
            'DET': {'def_rating': 114.6, 'opp_ppg': 117.3, 'opp_fg_pct': 0.469, 'opp_3pt_pct': 0.365},
            'GSW': {'def_rating': 110.8, 'opp_ppg': 112.9, 'opp_fg_pct': 0.458, 'opp_3pt_pct': 0.356},
            'HOU': {'def_rating': 109.4, 'opp_ppg': 111.7, 'opp_fg_pct': 0.451, 'opp_3pt_pct': 0.349},
            'IND': {'def_rating': 113.1, 'opp_ppg': 115.8, 'opp_fg_pct': 0.464, 'opp_3pt_pct': 0.359},
            'LAC': {'def_rating': 108.5, 'opp_ppg': 110.1, 'opp_fg_pct': 0.443, 'opp_3pt_pct': 0.346},
            'LAL': {'def_rating': 109.8, 'opp_ppg': 111.5, 'opp_fg_pct': 0.449, 'opp_3pt_pct': 0.351},
            'MEM': {'def_rating': 111.7, 'opp_ppg': 114.2, 'opp_fg_pct': 0.462, 'opp_3pt_pct': 0.357},
            'MIA': {'def_rating': 108.1, 'opp_ppg': 109.8, 'opp_fg_pct': 0.440, 'opp_3pt_pct': 0.342},
            'MIL': {'def_rating': 110.3, 'opp_ppg': 112.6, 'opp_fg_pct': 0.457, 'opp_3pt_pct': 0.354},
            'MIN': {'def_rating': 107.2, 'opp_ppg': 108.7, 'opp_fg_pct': 0.438, 'opp_3pt_pct': 0.340},
            'NOP': {'def_rating': 112.4, 'opp_ppg': 114.9, 'opp_fg_pct': 0.466, 'opp_3pt_pct': 0.361},
            'NYK': {'def_rating': 108.7, 'opp_ppg': 110.3, 'opp_fg_pct': 0.444, 'opp_3pt_pct': 0.347},
            'OKC': {'def_rating': 109.1, 'opp_ppg': 111.4, 'opp_fg_pct': 0.450, 'opp_3pt_pct': 0.350},
            'ORL': {'def_rating': 108.3, 'opp_ppg': 109.9, 'opp_fg_pct': 0.442, 'opp_3pt_pct': 0.344},
            'PHI': {'def_rating': 109.6, 'opp_ppg': 111.8, 'opp_fg_pct': 0.452, 'opp_3pt_pct': 0.351},
            'PHX': {'def_rating': 111.9, 'opp_ppg': 114.5, 'opp_fg_pct': 0.463, 'opp_3pt_pct': 0.358},
            'POR': {'def_rating': 113.8, 'opp_ppg': 116.4, 'opp_fg_pct': 0.467, 'opp_3pt_pct': 0.363},
            'SAC': {'def_rating': 112.6, 'opp_ppg': 115.2, 'opp_fg_pct': 0.465, 'opp_3pt_pct': 0.360},
            'SAS': {'def_rating': 115.7, 'opp_ppg': 119.1, 'opp_fg_pct': 0.474, 'opp_3pt_pct': 0.370},
            'TOR': {'def_rating': 111.4, 'opp_ppg': 113.9, 'opp_fg_pct': 0.460, 'opp_3pt_pct': 0.356},
            'UTA': {'def_rating': 110.9, 'opp_ppg': 113.2, 'opp_fg_pct': 0.459, 'opp_3pt_pct': 0.355},
            'WAS': {'def_rating': 114.2, 'opp_ppg': 116.8, 'opp_fg_pct': 0.470, 'opp_3pt_pct': 0.366}
        }
        
        # Position-specific defensive adjustments
        self.position_defensive_adjustments = {
            'PG': {'def_rating_mult': 1.0, 'opp_fg_pct_mult': 1.0},  # Point guards
            'SG': {'def_rating_mult': 1.05, 'opp_fg_pct_mult': 1.02},  # Shooting guards
            'SF': {'def_rating_mult': 1.1, 'opp_fg_pct_mult': 1.05},  # Small forwards
            'PF': {'def_rating_mult': 1.15, 'opp_fg_pct_mult': 1.08},  # Power forwards
            'C': {'def_rating_mult': 1.2, 'opp_fg_pct_mult': 1.1}   # Centers
        }
    
    def get_opponent_defensive_factor(self, opponent_team: str, player_position: str = 'SF') -> Tuple[float, str]:
        """
        Calculate defensive factor based on opponent's defensive metrics and player position
        
        Returns:
            Tuple of (defensive_factor, analysis_description)
        """
        if opponent_team not in self.team_defensive_metrics:
            return 1.0, "No defensive data available"
        
        # Get opponent's defensive metrics
        opp_metrics = self.team_defensive_metrics[opponent_team]
        position_adj = self.position_defensive_adjustments.get(player_position, self.position_defensive_adjustments['SF'])
        
        # Calculate defensive rating (lower is better defense)
        def_rating = opp_metrics['def_rating'] * position_adj['def_rating_mult']
        opp_fg_pct = opp_metrics['opp_fg_pct'] * position_adj['opp_fg_pct_mult']
        
        # Convert to factor (1.0 = neutral, <1.0 = tough defense, >1.0 = weak defense)
        # League average defensive rating is around 110-112
        league_avg_def_rating = 111.0
        defensive_factor = league_avg_def_rating / def_rating
        
        # Adjust for opponent field goal percentage allowed
        league_avg_opp_fg_pct = 0.455
        fg_pct_factor = league_avg_opp_fg_pct / opp_fg_pct
        
        # Combine factors
        combined_factor = (defensive_factor * 0.7) + (fg_pct_factor * 0.3)
        
        # Cap the factor to reasonable range
        combined_factor = max(0.7, min(1.3, combined_factor))
        
        # Generate analysis description
        if combined_factor < 0.85:
            analysis = f"tough defensive matchup vs {opponent_team} (strong defense)"
        elif combined_factor > 1.15:
            analysis = f"favorable matchup vs {opponent_team} (weak defense)"
        else:
            analysis = f"average defensive matchup vs {opponent_team}"
        
        return combined_factor, analysis
    
    def analyze_position_specific_matchup(self, player_data: pd.DataFrame, opponent_team: str, player_position: str = 'SF') -> Dict:
        """
        Analyze position-specific matchup factors
        """
        analysis = {
            'defensive_factor': 1.0,
            'defensive_analysis': 'No analysis available',
            'position_advantage': 'neutral',
            'matchup_insights': []
        }
        
        if opponent_team not in self.team_defensive_metrics:
            return analysis
        
        # Get defensive factor
        def_factor, def_analysis = self.get_opponent_defensive_factor(opponent_team, player_position)
        analysis['defensive_factor'] = def_factor
        analysis['defensive_analysis'] = def_analysis
        
        # Analyze position-specific performance
        opp_games = player_data[player_data['opponent'] == opponent_team]
        if len(opp_games) >= 2:
            avg_vs_opp = opp_games['PTS'].mean()
            season_avg = player_data['PTS'].mean()
            
            if avg_vs_opp > season_avg * 1.1:
                analysis['position_advantage'] = 'favorable'
                analysis['matchup_insights'].append(f"Historically strong vs {opponent_team} ({avg_vs_opp:.1f} vs {season_avg:.1f} season avg)")
            elif avg_vs_opp < season_avg * 0.9:
                analysis['position_advantage'] = 'unfavorable'
                analysis['matchup_insights'].append(f"Struggles vs {opponent_team} ({avg_vs_opp:.1f} vs {season_avg:.1f} season avg)")
            else:
                analysis['position_advantage'] = 'neutral'
                analysis['matchup_insights'].append(f"Average performance vs {opponent_team}")
        
        # Add defensive insights
        opp_metrics = self.team_defensive_metrics[opponent_team]
        analysis['matchup_insights'].append(f"{opponent_team} allows {opp_metrics['opp_ppg']:.1f} PPG (Def Rating: {opp_metrics['def_rating']:.1f})")
        
        return analysis
    
    def get_enhanced_opponent_factor(self, player_data: pd.DataFrame, opponent_team: str, player_position: str = 'SF') -> Tuple[float, str, Dict]:
        """
        Get enhanced opponent factor combining historical performance and defensive metrics
        """
        # Get position-specific analysis
        position_analysis = self.analyze_position_specific_matchup(player_data, opponent_team, player_position)
        
        # Get historical performance factor
        opp_games = player_data[player_data['opponent'] == opponent_team]
        if len(opp_games) >= 2:
            avg_vs_opp = opp_games['PTS'].mean()
            season_avg = player_data['PTS'].mean()
            historical_factor = avg_vs_opp / max(season_avg, 1)
        else:
            historical_factor = 1.0
        
        # Get defensive factor
        defensive_factor = position_analysis['defensive_factor']
        
        # Combine factors (70% historical, 30% defensive)
        combined_factor = (historical_factor * 0.7) + (defensive_factor * 0.3)
        combined_factor = max(0.7, min(1.3, combined_factor))
        
        # Generate comprehensive analysis
        if combined_factor > 1.1:
            analysis = f"Strong matchup vs {opponent_team} (historical + defensive advantage)"
        elif combined_factor < 0.9:
            analysis = f"Tough matchup vs {opponent_team} (historical + defensive disadvantage)"
        else:
            analysis = f"Balanced matchup vs {opponent_team}"
        
        return combined_factor, analysis, position_analysis

def test_advanced_analysis():
    """Test the advanced opponent analysis"""
    analyzer = AdvancedOpponentAnalyzer()
    
    print("🏀 Advanced Opponent Analysis Test")
    print("=" * 50)
    
    # Test different matchups
    test_cases = [
        ('LAL', 'SF', 'LeBron James'),
        ('PHX', 'PF', 'Anthony Davis'),
        ('BOS', 'PG', 'Stephen Curry'),
        ('MIA', 'C', 'Nikola Jokic')
    ]
    
    for opponent, position, player in test_cases:
        print(f"\n📊 {player} ({position}) vs {opponent}:")
        def_factor, analysis = analyzer.get_opponent_defensive_factor(opponent, position)
        print(f"   Defensive Factor: {def_factor:.3f}")
        print(f"   Analysis: {analysis}")
        
        # Show defensive metrics
        if opponent in analyzer.team_defensive_metrics:
            metrics = analyzer.team_defensive_metrics[opponent]
            print(f"   Opponent PPG Allowed: {metrics['opp_ppg']:.1f}")
            print(f"   Defensive Rating: {metrics['def_rating']:.1f}")

if __name__ == "__main__":
    test_advanced_analysis()
