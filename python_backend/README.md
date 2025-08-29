# Stock Portfolio ML API

Advanced Machine Learning and Quantitative Finance API for Portfolio Analysis and Forecasting.

## Features

### 🧠 Machine Learning Models
- **Random Forest Regressor**: Ensemble learning for price prediction
- **Gradient Boosting**: Advanced boosting algorithms
- **XGBoost & LightGBM**: High-performance gradient boosting
- **Neural Networks**: Multi-layer perceptron for complex patterns
- **Support Vector Regression**: Non-linear regression modeling

### 📊 Quantitative Finance Algorithms
- **Monte Carlo Simulation**: 1,000+ path simulations with Geometric Brownian Motion
- **ARIMA Models**: Time series forecasting with automatic parameter selection
- **Risk Metrics**: VaR, Expected Shortfall, Sharpe Ratio, Sortino Ratio
- **Portfolio Optimization**: Sharpe ratio, minimum variance, maximum return
- **Scenario Analysis**: Bull/Bear market scenarios with probability weighting

### 🔬 Advanced Analytics
- **Factor Analysis**: Market, size, value, momentum factors
- **Correlation Analysis**: Dynamic correlation matrices
- **Volatility Forecasting**: GARCH and stochastic volatility models
- **Market Regime Detection**: Bull/Bear/Sideways market classification

## Installation

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
```bash
cd python_backend
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Start the server**
```bash
python start_server.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Forecast Endpoints

#### POST `/api/forecast/portfolio`
Generate comprehensive portfolio forecast using quantitative algorithms.

**Request Body:**
```json
{
  "tickers": ["AAPL", "GOOGL", "MSFT"],
  "portfolio_values": {
    "AAPL": 25000,
    "GOOGL": 30000,
    "MSFT": 20000
  },
  "forecast_period": "1Y",
  "confidence_level": 0.95,
  "num_simulations": 1000
}
```

**Response:**
```json
{
  "monte_carlo": [...],
  "arima_forecast": [...],
  "ml_forecast": [...],
  "risk_metrics": {...},
  "scenarios": [...],
  "stock_forecasts": [...],
  "volatility_forecast": [...],
  "correlation_matrix": [...],
  "market_regime": {...}
}
```

#### GET `/api/forecast/stock/{ticker}`
Forecast individual stock performance.

#### GET `/api/forecast/risk-metrics`
Calculate portfolio risk metrics.

### Portfolio Endpoints

#### POST `/api/portfolio/optimize`
Optimize portfolio weights using various methods.

#### POST `/api/portfolio/analyze`
Analyze portfolio performance and risk.

#### GET `/api/portfolio/rebalance`
Get portfolio rebalancing recommendations.

#### GET `/api/portfolio/diversification`
Analyze portfolio diversification.

### Analytics Endpoints

#### POST `/api/analytics/comprehensive`
Perform comprehensive portfolio analytics.

#### GET `/api/analytics/sector`
Analyze sector allocation.

#### GET `/api/analytics/factor`
Perform factor analysis.

#### GET `/api/analytics/risk`
Perform risk analysis.

### ML Models Endpoints

#### POST `/api/ml/train`
Train machine learning model for financial prediction.

#### POST `/api/ml/predict`
Make predictions using trained ML model.

#### GET `/api/ml/models`
Get list of available ML models.

#### GET `/api/ml/feature-importance`
Get feature importance from trained model.

#### GET `/api/ml/model-evaluation`
Evaluate ML model performance.

## Quantitative Models

### Monte Carlo Simulation
- **Geometric Brownian Motion**: dS = μSdt + σSdW
- **1,000+ Simulations**: Multiple path generation
- **Parameter Estimation**: Drift and volatility from historical data
- **Confidence Intervals**: Statistical bounds on predictions

### ARIMA Models
- **Automatic Parameter Selection**: (p,d,q) optimization
- **Stationarity Testing**: Augmented Dickey-Fuller test
- **Confidence Bands**: Upper and lower prediction bounds
- **Model Validation**: AIC/BIC criteria for model selection

### Machine Learning Models
- **Feature Engineering**: Technical indicators, price patterns, volume analysis
- **Model Selection**: Cross-validation for optimal model choice
- **Hyperparameter Tuning**: Grid search and optimization
- **Ensemble Methods**: Combining multiple models for robustness

### Risk Metrics
- **Value at Risk (VaR)**: 95% and 99% confidence levels
- **Expected Shortfall**: Conditional tail expectation
- **Maximum Drawdown**: Worst historical decline
- **Sharpe/Sortino Ratios**: Risk-adjusted return measures

## Configuration

### Environment Variables
```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=true
LOG_LEVEL=info

# API Keys (optional)
ALPHA_VANTAGE_API_KEY=your_key_here
YAHOO_FINANCE_API_KEY=your_key_here

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost/db
```

### Model Configuration
```python
# Example model configuration
model_config = {
    "model_type": "ensemble",
    "features": ["price", "volume", "technical_indicators"],
    "target_variable": "returns",
    "test_size": 0.2,
    "validation_split": 0.2,
    "hyperparameter_tuning": True
}
```

## Performance

### Optimization Features
- **Async Processing**: Non-blocking API calls
- **Caching**: Redis-based result caching
- **Batch Processing**: Efficient bulk operations
- **Memory Management**: Optimized data structures

### Scalability
- **Horizontal Scaling**: Multiple worker processes
- **Load Balancing**: Request distribution
- **Database Optimization**: Indexed queries
- **CDN Integration**: Static asset delivery

## Monitoring

### Health Checks
```bash
GET /health
```

### Metrics
- Request/response times
- Model accuracy metrics
- API usage statistics
- Error rates and logs

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Linting
flake8 .

# Type checking
mypy .

# Formatting
black .
```

### Docker Support
```bash
# Build image
docker build -t stock-portfolio-ml-api .

# Run container
docker run -p 8000:8000 stock-portfolio-ml-api
```

## Integration

### Frontend Integration
The API is designed to work seamlessly with the Next.js frontend:

```javascript
// Example frontend call
const response = await fetch('http://localhost:8000/api/forecast/portfolio', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(forecastRequest)
});
```

### External APIs
- **Yahoo Finance**: Real-time stock data
- **Alpha Vantage**: Market data and indicators
- **FRED**: Economic indicators
- **News APIs**: Sentiment analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the API examples

---

**Note**: This API is for educational and research purposes. Always validate predictions and consult financial advisors for investment decisions. 