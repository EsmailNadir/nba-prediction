#!/usr/bin/env python3
"""
Test script to demonstrate how the system transitions from synthetic to real data
"""

import requests
import json
from datetime import datetime

def test_current_state():
    """Test the current state of the prediction system"""
    print("🏀 NBA Prediction System - Data Transition Test")
    print("=" * 60)
    
    try:
        # Test health endpoint
        response = requests.get('http://localhost:5004/health')
        health_data = response.json()
        
        print("📊 CURRENT STATE:")
        print(f"   Data Source: {health_data['data_source']}")
        print(f"   Data Quality: {health_data['data_quality']}")
        print(f"   NBA Service: {health_data['nba_service_available']}")
        print(f"   Players Available: {health_data['players_available']}")
        print()
        
        print("💡 RECOMMENDATIONS:")
        for rec in health_data['recommendations']:
            print(f"   {rec}")
        print()
        
        # Test a prediction
        print("🔮 TESTING PREDICTION:")
        pred_response = requests.post('http://localhost:5004/predict', 
                                    json={'playerName': 'LeBron James'})
        
        if pred_response.status_code == 200:
            pred_data = pred_response.json()
            print(f"   Player: {pred_data['playerName']}")
            print(f"   Predicted Points: {pred_data['predictedPoints']}")
            print(f"   Confidence: {pred_data['confidence'] * 100:.0f}%")
            print(f"   Next Opponent: {pred_data.get('nextOpponent', 'Unknown')}")
            print(f"   Data Note: {'Real NBA data' if pred_data.get('nextOpponent') else 'Synthetic/fallback data'}")
        else:
            print(f"   ❌ Prediction failed: {pred_response.status_code}")
        
        print()
        print("🔄 TRANSITION SCENARIO:")
        print("   When NBA season starts:")
        print("   1. Run: python3 collect_real_nba_data.py")
        print("   2. System automatically detects real data")
        print("   3. Predictions become accurate")
        print("   4. Opponents are from real NBA schedule")
        print("   5. No code changes needed!")
        
    except Exception as e:
        print(f"❌ Error testing system: {e}")

if __name__ == "__main__":
    test_current_state()
