#!/usr/bin/env python3
"""
Demo script to showcase the enhanced NBA prediction analysis
"""

import requests
import json
import time

def demo_advanced_analysis():
    print("🏀 NBA Prediction System - Advanced Analysis Demo")
    print("=" * 60)
    
    # Test multiple players to show different analysis scenarios
    test_players = [
        "LeBron James",
        "Stephen Curry", 
        "Luka Doncic",
        "Giannis Antetokounmpo"
    ]
    
    for i, player in enumerate(test_players, 1):
        print(f"\n📊 ANALYSIS {i}: {player}")
        print("-" * 40)
        
        try:
            # Get prediction
            response = requests.post(
                "http://localhost:5004/predict",
                json={"playerName": player},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"🎯 Player: {data['playerName']} ({data.get('team', 'N/A')})")
                print(f"📈 Predicted Points: {data['predictedPoints']}")
                print(f"🆚 Next Opponent: {data.get('nextOpponent', 'Unknown')}")
                print(f"🎲 Confidence: {data['confidence']*100:.0f}%")
                print(f"📊 Trend: {data['trend']}")
                
                # Show analysis breakdown
                print(f"\n🔬 ADVANCED ANALYSIS BREAKDOWN:")
                print(f"   • Recent 5 Games: {data['last5Average']:.1f} pts")
                print(f"   • Season Average: {data['seasonAverage']:.1f} pts")
                print(f"   • Last Game: {data['lastGamePoints']} pts")
                print(f"   • Recent Minutes: {data['recentMinutes']:.1f} min")
                
                if data.get('matchupContext'):
                    print(f"   • Matchup Context: {data['matchupContext']}")
                if data.get('specificMatchup'):
                    print(f"   • Specific Matchup: {data['specificMatchup']}")
                
                # Show performance factors
                print(f"\n⚡ PERFORMANCE FACTORS:")
                if data.get('shootingTrend'):
                    trend_icon = "📈" if data['shootingTrend'] > 1 else "📉" if data['shootingTrend'] < 1 else "➡️"
                    print(f"   • Shooting Trend: {trend_icon} {((data['shootingTrend']-1)*100):+.1f}%")
                if data.get('usageTrend'):
                    trend_icon = "📈" if data['usageTrend'] > 1 else "📉" if data['usageTrend'] < 1 else "➡️"
                    print(f"   • Usage Trend: {trend_icon} {((data['usageTrend']-1)*100):+.1f}%")
                if data.get('momentumFactor'):
                    trend_icon = "📈" if data['momentumFactor'] > 1 else "📉" if data['momentumFactor'] < 1 else "➡️"
                    print(f"   • Momentum: {trend_icon} {((data['momentumFactor']-1)*100):+.1f}%")
                
            else:
                print(f"❌ Error getting prediction: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        if i < len(test_players):
            time.sleep(1)  # Brief pause between players
    
    print(f"\n🎉 DEMO COMPLETE!")
    print(f"💡 The UI now shows:")
    print(f"   • 🧮 Algorithm breakdown with weights and factors")
    print(f"   • 🆚 Detailed opponent matchup analysis")
    print(f"   • 📈 Performance trends and efficiency metrics")
    print(f"   • ✅ Data quality indicators and confidence factors")
    print(f"\n🌐 Open http://localhost:3000 to see the enhanced UI!")

if __name__ == "__main__":
    demo_advanced_analysis()
