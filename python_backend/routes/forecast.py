from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from models.quantitative_models import (
    MonteCarloSimulation,
    ARIMAModel,
    MachineLearningModel,
    RiskMetrics,
    ScenarioAnalysis,
    VolatilityModel,
    CorrelationModel
)

router = APIRouter()

class ForecastRequest(BaseModel):
    tickers: List[str]
    portfolio_values: Dict[str, float]
    forecast_period: str = "1Y"
    confidence_level: float = 0.95
    num_simulations: int = 1000
    include_risk_metrics: bool = True
    include_scenarios: bool = True

class ForecastResponse(BaseModel):
    monte_carlo: List[Dict[str, Any]]
    arima_forecast: List[Dict[str, Any]]
    ml_forecast: List[Dict[str, Any]]
    risk_metrics: Dict[str, Any]
    scenarios: List[Dict[str, Any]]
    stock_forecasts: List[Dict[str, Any]]
    volatility_forecast: List[Dict[str, Any]]
    correlation_matrix: List[List[float]]
    market_regime: Dict[str, Any]

@router.post("/portfolio", response_model=ForecastResponse)
async def forecast_portfolio(request: ForecastRequest):
    """
    Generate comprehensive portfolio forecast using quantitative algorithms
    """
    try:
        # Initialize models
        mc_model = MonteCarloSimulation()
        arima_model = ARIMAModel()
        ml_model = MachineLearningModel()
        risk_model = RiskMetrics()
        scenario_model = ScenarioAnalysis()
        vol_model = VolatilityModel()
        corr_model = CorrelationModel()
        
        # Get historical data
        historical_data = await get_historical_data(request.tickers)
        
        # Calculate portfolio weights
        total_value = sum(request.portfolio_values.values())
        weights = {ticker: value/total_value for ticker, value in request.portfolio_values.items()}
        
        # Monte Carlo Simulation
        mc_results = mc_model.run_simulation(
            historical_data, 
            weights, 
            request.forecast_period, 
            request.num_simulations
        )
        
        # ARIMA Forecast
        arima_results = arima_model.forecast(
            historical_data, 
            weights, 
            request.forecast_period
        )
        
        # Machine Learning Forecast
        ml_results = ml_model.predict(
            historical_data, 
            weights, 
            request.forecast_period
        )
        
        # Risk Metrics
        risk_results = risk_model.calculate_metrics(
            historical_data, 
            weights, 
            request.confidence_level
        )
        
        # Scenario Analysis
        scenario_results = scenario_model.generate_scenarios(
            total_value, 
            request.forecast_period
        )
        
        # Individual Stock Forecasts
        stock_forecasts = []
        for ticker in request.tickers:
            if ticker in historical_data:
                stock_data = historical_data[ticker]
                stock_forecast = ml_model.predict_stock(stock_data, request.forecast_period)
                stock_forecasts.append({
                    "ticker": ticker,
                    "current_value": request.portfolio_values.get(ticker, 0),
                    "forecast_value": stock_forecast["forecast_value"],
                    "confidence": stock_forecast["confidence"],
                    "volatility": stock_forecast["volatility"]
                })
        
        # Volatility Forecast
        vol_results = vol_model.forecast_volatility(
            historical_data, 
            request.forecast_period
        )
        
        # Correlation Matrix
        corr_matrix = corr_model.calculate_correlation_matrix(historical_data)
        
        # Market Regime Analysis
        regime_analysis = analyze_market_regime(historical_data, weights)
        
        return ForecastResponse(
            monte_carlo=mc_results,
            arima_forecast=arima_results,
            ml_forecast=ml_results,
            risk_metrics=risk_results,
            scenarios=scenario_results,
            stock_forecasts=stock_forecasts,
            volatility_forecast=vol_results,
            correlation_matrix=corr_matrix,
            market_regime=regime_analysis
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@router.get("/stock/{ticker}")
async def forecast_stock(
    ticker: str,
    period: str = Query("1Y", description="Forecast period"),
    confidence: float = Query(0.95, description="Confidence level")
):
    """
    Forecast individual stock performance
    """
    try:
        # Get stock data
        stock_data = await get_stock_data(ticker)
        
        # Initialize models
        ml_model = MachineLearningModel()
        arima_model = ARIMAModel()
        
        # Generate forecasts
        ml_forecast = ml_model.predict_stock(stock_data, period)
        arima_forecast = arima_model.forecast_stock(stock_data, period)
        
        return {
            "ticker": ticker,
            "ml_forecast": ml_forecast,
            "arima_forecast": arima_forecast,
            "confidence_level": confidence
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stock forecast failed: {str(e)}")

@router.get("/risk-metrics")
async def get_risk_metrics(
    tickers: str = Query(..., description="Comma-separated tickers"),
    confidence: float = Query(0.95, description="Confidence level")
):
    """
    Calculate portfolio risk metrics
    """
    try:
        ticker_list = [t.strip() for t in tickers.split(",")]
        historical_data = await get_historical_data(ticker_list)
        
        risk_model = RiskMetrics()
        metrics = risk_model.calculate_metrics(historical_data, {}, confidence)
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk metrics calculation failed: {str(e)}")

async def get_historical_data(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for given tickers"""
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y")
            if not hist.empty:
                data[ticker] = hist
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return data

async def get_stock_data(ticker: str) -> pd.DataFrame:
    """Fetch data for a single stock"""
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period="2y")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not fetch data for {ticker}")

def analyze_market_regime(historical_data: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> Dict[str, Any]:
    """Analyze current market regime"""
    try:
        # Calculate portfolio returns
        portfolio_returns = []
        for ticker, data in historical_data.items():
            if ticker in weights:
                returns = data['Close'].pct_change().dropna()
                portfolio_returns.extend(returns * weights[ticker])
        
        if not portfolio_returns:
            return {"regime": "Unknown", "confidence": 0.0}
        
        # Calculate regime indicators
        returns_array = np.array(portfolio_returns)
        volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
        mean_return = np.mean(returns_array) * 252  # Annualized
        
        # Determine regime
        if mean_return > 0.1 and volatility < 0.2:
            regime = "Bull Market"
            confidence = 0.8
        elif mean_return < -0.05 and volatility > 0.25:
            regime = "Bear Market"
            confidence = 0.7
        elif abs(mean_return) < 0.05 and volatility < 0.15:
            regime = "Sideways"
            confidence = 0.6
        else:
            regime = "Volatile"
            confidence = 0.5
            
        return {
            "regime": regime,
            "confidence": confidence,
            "volatility": volatility,
            "mean_return": mean_return
        }
        
    except Exception as e:
        return {"regime": "Unknown", "confidence": 0.0, "error": str(e)} 