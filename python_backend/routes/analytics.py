from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

router = APIRouter()

class AnalyticsRequest(BaseModel):
    tickers: List[str]
    weights: Dict[str, float]
    analysis_type: str = "comprehensive"  # comprehensive, sector, factor, risk

@router.post("/comprehensive")
async def comprehensive_analytics(request: AnalyticsRequest):
    """
    Perform comprehensive portfolio analytics
    """
    try:
        # Get historical data
        historical_data = await get_historical_data(request.tickers)
        
        if not historical_data:
            raise HTTPException(status_code=400, detail="No valid data found for tickers")
        
        # Perform various analyses
        sector_analysis = analyze_sector_allocation(request.tickers)
        factor_analysis = perform_factor_analysis(historical_data, request.weights)
        risk_analysis = perform_risk_analysis(historical_data, request.weights)
        performance_analysis = perform_performance_analysis(historical_data, request.weights)
        
        return {
            "sector_analysis": sector_analysis,
            "factor_analysis": factor_analysis,
            "risk_analysis": risk_analysis,
            "performance_analysis": performance_analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

@router.get("/sector")
async def sector_analysis(tickers: str = Query(..., description="Comma-separated tickers")):
    """
    Analyze sector allocation
    """
    try:
        ticker_list = [t.strip() for t in tickers.split(",")]
        analysis = analyze_sector_allocation(ticker_list)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sector analysis failed: {str(e)}")

@router.get("/factor")
async def factor_analysis(
    tickers: str = Query(..., description="Comma-separated tickers"),
    weights: str = Query(..., description="Weights as JSON string")
):
    """
    Perform factor analysis
    """
    try:
        import json
        ticker_list = [t.strip() for t in tickers.split(",")]
        weight_dict = json.loads(weights)
        
        historical_data = await get_historical_data(ticker_list)
        analysis = perform_factor_analysis(historical_data, weight_dict)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Factor analysis failed: {str(e)}")

@router.get("/risk")
async def risk_analysis(
    tickers: str = Query(..., description="Comma-separated tickers"),
    weights: str = Query(..., description="Weights as JSON string")
):
    """
    Perform risk analysis
    """
    try:
        import json
        ticker_list = [t.strip() for t in tickers.split(",")]
        weight_dict = json.loads(weights)
        
        historical_data = await get_historical_data(ticker_list)
        analysis = perform_risk_analysis(historical_data, weight_dict)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk analysis failed: {str(e)}")

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

def analyze_sector_allocation(tickers: List[str]) -> Dict[str, Any]:
    """Analyze sector allocation of portfolio"""
    # Simplified sector mapping (in real implementation, would use external API)
    sector_mapping = {
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'GOOGL': 'Technology',
        'AMZN': 'Consumer Discretionary',
        'TSLA': 'Consumer Discretionary',
        'NVDA': 'Technology',
        'META': 'Technology',
        'NFLX': 'Communication Services',
        'JPM': 'Financial Services',
        'JNJ': 'Healthcare',
        'PG': 'Consumer Staples',
        'UNH': 'Healthcare',
        'HD': 'Consumer Discretionary',
        'MA': 'Financial Services',
        'V': 'Financial Services',
        'PYPL': 'Financial Services',
        'ADBE': 'Technology',
        'CRM': 'Technology',
        'NKE': 'Consumer Discretionary',
        'DIS': 'Communication Services'
    }
    
    sector_allocation = {}
    for ticker in tickers:
        sector = sector_mapping.get(ticker, 'Other')
        sector_allocation[sector] = sector_allocation.get(sector, 0) + 1
    
    # Calculate percentages
    total_stocks = len(tickers)
    sector_percentages = {sector: count/total_stocks for sector, count in sector_allocation.items()}
    
    # Calculate concentration metrics
    herfindahl_index = sum(percentage ** 2 for percentage in sector_percentages.values())
    
    return {
        "sector_allocation": sector_allocation,
        "sector_percentages": sector_percentages,
        "herfindahl_index": herfindahl_index,
        "concentration_level": "High" if herfindahl_index > 0.25 else "Medium" if herfindahl_index > 0.1 else "Low",
        "top_sectors": sorted(sector_percentages.items(), key=lambda x: x[1], reverse=True)[:3]
    }

def perform_factor_analysis(historical_data: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> Dict[str, Any]:
    """Perform factor analysis on portfolio"""
    if not historical_data:
        return {"error": "No data available"}
    
    # Calculate portfolio returns
    portfolio_returns = calculate_portfolio_returns(historical_data, weights)
    
    if portfolio_returns.empty:
        return {"error": "No valid returns data"}
    
    # Calculate factor exposures (simplified)
    factor_exposures = {
        "market_beta": calculate_market_beta(portfolio_returns),
        "size_factor": calculate_size_factor(historical_data, weights),
        "value_factor": calculate_value_factor(historical_data, weights),
        "momentum_factor": calculate_momentum_factor(portfolio_returns),
        "volatility_factor": calculate_volatility_factor(portfolio_returns)
    }
    
    # Calculate factor contributions
    factor_contributions = calculate_factor_contributions(factor_exposures, portfolio_returns)
    
    return {
        "factor_exposures": factor_exposures,
        "factor_contributions": factor_contributions,
        "total_factor_contribution": sum(factor_contributions.values())
    }

def perform_risk_analysis(historical_data: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> Dict[str, Any]:
    """Perform comprehensive risk analysis"""
    if not historical_data:
        return {"error": "No data available"}
    
    # Calculate portfolio returns
    portfolio_returns = calculate_portfolio_returns(historical_data, weights)
    
    if portfolio_returns.empty:
        return {"error": "No valid returns data"}
    
    # Calculate various risk metrics
    risk_metrics = {
        "volatility": portfolio_returns.std() * np.sqrt(252),
        "var_95": np.percentile(portfolio_returns, 5),
        "var_99": np.percentile(portfolio_returns, 1),
        "expected_shortfall_95": portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean(),
        "max_drawdown": calculate_max_drawdown(portfolio_returns),
        "downside_deviation": portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252),
        "skewness": portfolio_returns.skew(),
        "kurtosis": portfolio_returns.kurtosis(),
        "calmar_ratio": calculate_calmar_ratio(portfolio_returns),
        "sortino_ratio": calculate_sortino_ratio(portfolio_returns)
    }
    
    # Calculate risk decomposition
    risk_decomposition = calculate_risk_decomposition(historical_data, weights)
    
    # Calculate stress test scenarios
    stress_tests = perform_stress_tests(portfolio_returns)
    
    return {
        "risk_metrics": risk_metrics,
        "risk_decomposition": risk_decomposition,
        "stress_tests": stress_tests
    }

def perform_performance_analysis(historical_data: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> Dict[str, Any]:
    """Perform performance analysis"""
    if not historical_data:
        return {"error": "No data available"}
    
    # Calculate portfolio returns
    portfolio_returns = calculate_portfolio_returns(historical_data, weights)
    
    if portfolio_returns.empty:
        return {"error": "No valid returns data"}
    
    # Calculate performance metrics
    performance_metrics = {
        "total_return": (1 + portfolio_returns).prod() - 1,
        "annualized_return": portfolio_returns.mean() * 252,
        "sharpe_ratio": calculate_sharpe_ratio(portfolio_returns),
        "information_ratio": calculate_information_ratio(portfolio_returns),
        "treynor_ratio": calculate_treynor_ratio(portfolio_returns),
        "jensen_alpha": calculate_jensen_alpha(portfolio_returns)
    }
    
    # Calculate rolling performance
    rolling_metrics = calculate_rolling_metrics(portfolio_returns)
    
    # Calculate attribution analysis
    attribution = calculate_attribution_analysis(historical_data, weights)
    
    return {
        "performance_metrics": performance_metrics,
        "rolling_metrics": rolling_metrics,
        "attribution": attribution
    }

def calculate_portfolio_returns(historical_data: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> pd.Series:
    """Calculate weighted portfolio returns"""
    portfolio_returns = pd.Series(0.0)
    
    for ticker, data in historical_data.items():
        if ticker in weights and not data.empty:
            returns = data['Close'].pct_change().dropna()
            weighted_returns = returns * weights[ticker]
            if portfolio_returns.empty:
                portfolio_returns = weighted_returns
            else:
                portfolio_returns = portfolio_returns.add(weighted_returns, fill_value=0)
    
    return portfolio_returns

def calculate_market_beta(returns: pd.Series) -> float:
    """Calculate market beta (simplified)"""
    # In real implementation, would use market data
    return 1.0

def calculate_size_factor(historical_data: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> float:
    """Calculate size factor exposure"""
    # Simplified size factor calculation
    return 0.5

def calculate_value_factor(historical_data: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> float:
    """Calculate value factor exposure"""
    # Simplified value factor calculation
    return 0.3

def calculate_momentum_factor(returns: pd.Series) -> float:
    """Calculate momentum factor"""
    if len(returns) < 20:
        return 0.0
    return returns.rolling(20).mean().iloc[-1]

def calculate_volatility_factor(returns: pd.Series) -> float:
    """Calculate volatility factor"""
    return returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else returns.std()

def calculate_factor_contributions(factor_exposures: Dict[str, float], returns: pd.Series) -> Dict[str, float]:
    """Calculate factor contributions to returns"""
    # Simplified factor contribution calculation
    contributions = {}
    for factor, exposure in factor_exposures.items():
        contributions[factor] = exposure * returns.mean() * 0.1  # Simplified
    return contributions

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_calmar_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Calmar ratio"""
    max_dd = abs(calculate_max_drawdown(returns))
    annual_return = returns.mean() * 252
    return (annual_return - risk_free_rate) / max_dd if max_dd > 0 else 0

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio"""
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std()
    return excess_returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0

def calculate_risk_decomposition(historical_data: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> Dict[str, float]:
    """Calculate risk decomposition by asset"""
    risk_contributions = {}
    
    for ticker, data in historical_data.items():
        if ticker in weights and not data.empty:
            returns = data['Close'].pct_change().dropna()
            risk_contributions[ticker] = weights[ticker] * returns.std() * np.sqrt(252)
    
    return risk_contributions

def perform_stress_tests(returns: pd.Series) -> Dict[str, float]:
    """Perform stress test scenarios"""
    scenarios = {
        "market_crash": returns.quantile(0.01),
        "high_volatility": returns.std() * 2,
        "extended_drawdown": returns.rolling(30).sum().min(),
        "correlation_breakdown": returns.rolling(20).std().max()
    }
    return scenarios

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

def calculate_information_ratio(returns: pd.Series) -> float:
    """Calculate information ratio (vs zero benchmark)"""
    return returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

def calculate_treynor_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Treynor ratio"""
    beta = 1.0  # Simplified
    excess_return = returns.mean() * 252 - risk_free_rate
    return excess_return / beta if beta > 0 else 0

def calculate_jensen_alpha(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Jensen's alpha"""
    beta = 1.0  # Simplified
    market_return = 0.08  # Simplified market return
    portfolio_return = returns.mean() * 252
    return portfolio_return - (risk_free_rate + beta * (market_return - risk_free_rate))

def calculate_rolling_metrics(returns: pd.Series, window: int = 252) -> Dict[str, List[float]]:
    """Calculate rolling performance metrics"""
    if len(returns) < window:
        return {"error": "Insufficient data"}
    
    rolling_metrics = {
        "rolling_return": returns.rolling(window).mean().tolist(),
        "rolling_volatility": returns.rolling(window).std().tolist(),
        "rolling_sharpe": (returns.rolling(window).mean() / returns.rolling(window).std()).tolist()
    }
    
    return rolling_metrics

def calculate_attribution_analysis(historical_data: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> Dict[str, Any]:
    """Calculate performance attribution"""
    attribution = {
        "asset_contributions": {},
        "sector_contributions": {},
        "factor_contributions": {}
    }
    
    # Calculate asset contributions
    for ticker, data in historical_data.items():
        if ticker in weights and not data.empty:
            returns = data['Close'].pct_change().dropna()
            contribution = weights[ticker] * returns.mean() * 252
            attribution["asset_contributions"][ticker] = contribution
    
    return attribution 