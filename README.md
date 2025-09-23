

A full-stack machine learning application that predicts NBA player point totals using advanced statistical analysis, opponent defensive metrics, and real-time schedule data.

## 📸 Screenshots

### Main Interface />
<img width="1468" height="771" alt="Screenshot 2025-09-23 at 3 44 38 PM" src="https://github.com/user-attachments/assets/1d546f0e-2d88-47d4-98f9-0d98cba2dd63" />




### Deep Analysis prediction
<img width="1470" height="777" alt="Screenshot 2025-09-23 at 3 45 17 PM" src="https://github.com/user-attachments/assets/fdd9fd4b-c02a-4dc3-a374-6f1eac428502" />


### Deep Analysis Panel - Prediction Algorithm Details
<img width="1470" height="779" alt="Screenshot 2025-09-23 at 3 45 56 PM" src="https://github.com/user-attachments/assets/f5fd9aaf-1136-4856-9b01-e546df9042a9" />



*Detailed view of the prediction algorithm with base weights and combined factors*

## 🚀 Features

### 🎯 **Advanced Prediction Algorithm**
- **Multi-Model Ensemble**: Combines Ridge Regression, Random Forest, and Gradient Boosting
- **30+ Features**: Recent performance, shooting efficiency, usage rate, rest days, momentum
- **Opponent Analysis**: Team defensive ratings, position-specific matchups, historical performance
- **Real-time Data**: Live ESPN API integration for current season schedules

### 🎨 **Modern Web Interface**
- **Next.js 15 + TypeScript**: Modern React framework with type safety
- **Tailwind CSS**: Responsive, mobile-first design with glassmorphism effects
- **Advanced Analysis Panel**: 4-tab breakdown (Algorithm, Matchup, Trends, Data Quality)
- **Accessibility**: ARIA labels, keyboard navigation, screen reader support
- **Real-time Features**: Autocomplete, loading states, error handling

### 🏗️ **Full-Stack Architecture**
- **Frontend**: Next.js + TypeScript + Tailwind CSS
- **Backend**: Spring Boot + Java 17 + PostgreSQL
- **ML Service**: Flask + Python + scikit-learn
- **Data Integration**: ESPN NBA API for live schedule data

## 📊 Technical Highlights

### Machine Learning
- **Ensemble Methods**: Voting Regressor combining multiple models
- **Feature Engineering**: Lag features, opponent analysis, momentum calculations
- **Model Performance**: R² scores, MAE, within-5-points accuracy metrics
- **Advanced Analytics**: Position-specific defensive adjustments, historical matchup analysis

### Data Processing
- **Real NBA Data**: 26,306 games, 569 players from 2024-25 season
- **Advanced Features**: Rest days, back-to-back games, home/away performance
- **Opponent Metrics**: PPG allowed, defensive rating, field goal percentage allowed
- **Historical Analysis**: Player-specific performance against specific teams

### System Architecture
- **Microservices**: Separate ML service for scalability
- **API Integration**: Real-time ESPN data for opponent schedules
- **Database Design**: PostgreSQL with JPA/Hibernate for prediction storage
- **Error Handling**: Comprehensive validation and fallback mechanisms

## 🛠️ Installation & Setup

### Prerequisites
- Java 17+
- Python 3.8+
- Node.js 18+
- PostgreSQL

### Backend Setup (Spring Boot)
```bash
cd backend/nba_backend
./gradlew bootRun
```

### ML Service Setup (Flask)
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start ML service
python3 predict_service.py
```

### Frontend Setup (Next.js)
```bash
cd frontend
npm install
npm run dev
```

## 🎮 Usage

1. **Start all services** (Backend, ML Service, Frontend)
2. **Open browser** to `http://localhost:3001`
3. **Search for a player** using the autocomplete feature
4. **View predictions** with detailed analysis breakdown
5. **Explore advanced analysis** tabs for algorithm insights

## 📈 Model Performance

- **R² Score**: 0.85+ on test data
- **MAE**: <3.5 points average error
- **Within 5 Points**: 75%+ accuracy
- **Real-time Updates**: Live opponent schedule integration

## 🔧 API Endpoints

### Spring Boot Backend (Port 8080)
- `GET /api/predictions/recent` - Recent predictions
- `POST /api/predictions/create` - Create new prediction
- `GET /api/predictions/player/{name}` - Player history

### Flask ML Service (Port 5004)
- `POST /predict` - Get player prediction
- `GET /health` - Service health check
- `GET /data-info` - Data source information
- `GET /players` - Available players list

## 🏆 Key Features

### Advanced Opponent Analysis
- **Defensive Metrics**: Team PPG allowed, defensive rating
- **Position Matchups**: Different factors for PG, SG, SF, PF, C
- **Historical Performance**: Player-specific matchup data
- **Real-time Schedule**: Live ESPN API integration

### Prediction Algorithm
- **Base Prediction**: Weighted average of recent 5, recent 10, season average
- **Adjustment Factors**: Minutes, efficiency, usage, opponent, momentum, rest
- **Confidence Scoring**: Based on data quality and game history
- **Fallback Logic**: Graceful degradation when data is limited

### User Experience
- **Autocomplete Search**: Real-time player suggestions
- **Loading States**: Skeleton screens and progress indicators
- **Error Handling**: User-friendly error messages
- **Mobile Responsive**: Optimized for all device sizes

## 🚀 Deployment

The application is designed for easy deployment:
- **Frontend**: Deploy to Vercel/Netlify
- **Backend**: Deploy to Railway/Heroku
- **ML Service**: Deploy to Railway/Google Cloud Run
- **Database**: Use managed PostgreSQL service

## 📝 License

This project is for educational and portfolio purposes.

## 🤝 Contributing

This is a portfolio project, but suggestions and improvements are welcome!

---

**Built with ❤️ using modern web technologies and machine learning**
