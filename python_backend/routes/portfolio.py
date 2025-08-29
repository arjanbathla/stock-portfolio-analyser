from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from scipy.optimize import minimize
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: cvxpy not available. Portfolio optimization will use simplified methods.")
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

router = APIRouter()

class PortfolioOptimizationRequest(BaseModel):
    tickers: List[str]
    target_return: Optional[float] = None
    risk_tolerance: float = 0.5
    optimization_method: str = "sharpe"  # sharpe, min_variance, max_return

class PortfolioAnalysisRequest(BaseModel):
    tickers: List[str]
    weights: Dict[str, float]
    benchmark: Optional[str] = None

@router.post("/optimize")
async def optimize_portfolio(request: PortfolioOptimizationRequest):
    """
    Optimize portfolio weights using various methods
    """
    try:
        # Get historical data
        historical_data = await get_historical_data(request.tickers)
        
        if not historical_data:
            raise HTTPException(status_code=400, detail="No valid data found for tickers")
        
        # Calculate returns and covariance matrix
        returns_data = calculate_returns_matrix(historical_data)
        mean_returns = returns_data.mean()
        cov_matrix = returns_data.cov()
        
        # Optimize portfolio based on method
        if request.optimization_method == "sharpe":
            optimal_weights = optimize_sharpe_ratio(mean_returns, cov_matrix, request.risk_tolerance)
        elif request.optimization_method == "min_variance":
            optimal_weights = optimize_minimum_variance(cov_matrix)
        elif request.optimization_method == "max_return":
            optimal_weights = optimize_maximum_return(mean_returns, cov_matrix, request.target_return)
        else:
            raise HTTPException(status_code=400, detail="Invalid optimization method")
        
        # Calculate portfolio metrics
        portfolio_metrics = calculate_portfolio_metrics(optimal_weights, mean_returns, cov_matrix)
        
        return {
            "optimal_weights": optimal_weights,
            "portfolio_metrics": portfolio_metrics,
            "optimization_method": request.optimization_method
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio optimization failed: {str(e)}")

@router.post("/analyze")
async def analyze_portfolio(request: PortfolioAnalysisRequest):
    """
    Analyze portfolio performance and risk
    """
    try:
        # Get historical data
        historical_data = await get_historical_data(request.tickers)
        
        if not historical_data:
            raise HTTPException(status_code=400, detail="No valid data found for tickers")
        
        # Calculate portfolio analysis
        analysis = perform_portfolio_analysis(historical_data, request.weights, request.benchmark)
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio analysis failed: {str(e)}")

@router.get("/rebalance")
async def get_rebalancing_recommendations(
    current_weights: str = Query(..., description="Current weights as JSON string"),
    target_weights: str = Query(..., description="Target weights as JSON string"),
    threshold: float = Query(0.05, description="Rebalancing threshold")
):
    """
    Get portfolio rebalancing recommendations
    """
    try:
        import json
        current = json.loads(current_weights)
        target = json.loads(target_weights)
        
        recommendations = calculate_rebalancing_recommendations(current, target, threshold)
        
        return recommendations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rebalancing calculation failed: {str(e)}")

@router.get("/diversification")
async def analyze_diversification(
    tickers: str = Query(..., description="Comma-separated tickers")
):
    """
    Analyze portfolio diversification
    """
    try:
        ticker_list = [t.strip() for t in tickers.split(",")]
        historical_data = await get_historical_data(ticker_list)
        
        diversification_metrics = calculate_diversification_metrics(historical_data)
        
        return diversification_metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diversification analysis failed: {str(e)}")

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

def calculate_returns_matrix(historical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate returns matrix for all stocks"""
    returns_data = {}
    for ticker, data in historical_data.items():
        if not data.empty:
            returns_data[ticker] = data['Close'].pct_change().dropna()
    return pd.DataFrame(returns_data)

def optimize_sharpe_ratio(mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_tolerance: float) -> Dict[str, float]:
    """Optimize portfolio for maximum Sharpe ratio"""
    if not CVXPY_AVAILABLE:
        # Simplified optimization without cvxpy
        return optimize_sharpe_ratio_simple(mean_returns, cov_matrix, risk_tolerance)
    
    n_assets = len(mean_returns)
    
    # Define variables
    weights = cp.Variable(n_assets)
    
    # Define constraints
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0
    ]
    
    # Add risk tolerance constraint
    portfolio_variance = cp.quad_form(weights, cov_matrix.values)
    constraints.append(portfolio_variance <= risk_tolerance)
    
    # Define objective (maximize Sharpe ratio)
    portfolio_return = mean_returns.values @ weights
    risk_free_rate = 0.02  # 2% risk-free rate
    sharpe_ratio = (portfolio_return - risk_free_rate) / cp.sqrt(portfolio_variance)
    
    # Solve optimization problem
    problem = cp.Problem(cp.Maximize(sharpe_ratio), constraints)
    problem.solve()
    
    if problem.status == "optimal":
        optimal_weights = {ticker: weight for ticker, weight in zip(mean_returns.index, weights.value)}
        return optimal_weights
    else:
        # Fallback to equal weights
        equal_weight = 1.0 / n_assets
        return {ticker: equal_weight for ticker in mean_returns.index}

def optimize_sharpe_ratio_simple(mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_tolerance: float) -> Dict[str, float]:
    """Simplified Sharpe ratio optimization without cvxpy"""
    n_assets = len(mean_returns)
    
    # Use scipy.optimize for constrained optimization
    def objective(weights):
        portfolio_return = np.sum(mean_returns.values * weights)
        portfolio_variance = weights.T @ cov_matrix.values @ weights
        sharpe_ratio = (portfolio_return - 0.02) / np.sqrt(portfolio_variance)
        return -sharpe_ratio  # Minimize negative Sharpe ratio
    
    def constraint_sum(weights):
        return np.sum(weights) - 1
    
    def constraint_risk(weights):
        portfolio_variance = weights.T @ cov_matrix.values @ weights
        return risk_tolerance - portfolio_variance
    
    # Initial guess: equal weights
    initial_weights = np.ones(n_assets) / n_assets
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': constraint_sum},
        {'type': 'ineq', 'fun': constraint_risk}
    ]
    
    # Bounds: weights between 0 and 1
    bounds = [(0, 1) for _ in range(n_assets)]
    
    try:
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = {ticker: weight for ticker, weight in zip(mean_returns.index, result.x)}
            return optimal_weights
    except:
        pass
    
    # Fallback to equal weights
    equal_weight = 1.0 / n_assets
    return {ticker: equal_weight for ticker in mean_returns.index}

def optimize_minimum_variance(cov_matrix: pd.DataFrame) -> Dict[str, float]:
    """Optimize portfolio for minimum variance"""
    if not CVXPY_AVAILABLE:
        # Simplified optimization without cvxpy
        return optimize_minimum_variance_simple(cov_matrix)
    
    n_assets = len(cov_matrix)
    
    # Define variables
    weights = cp.Variable(n_assets)
    
    # Define constraints
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0
    ]
    
    # Define objective (minimize variance)
    portfolio_variance = cp.quad_form(weights, cov_matrix.values)
    
    # Solve optimization problem
    problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
    problem.solve()
    
    if problem.status == "optimal":
        optimal_weights = {ticker: weight for ticker, weight in zip(cov_matrix.index, weights.value)}
        return optimal_weights
    else:
        # Fallback to equal weights
        equal_weight = 1.0 / n_assets
        return {ticker: equal_weight for ticker in cov_matrix.index}

def optimize_minimum_variance_simple(cov_matrix: pd.DataFrame) -> Dict[str, float]:
    """Simplified minimum variance optimization without cvxpy"""
    n_assets = len(cov_matrix)
    
    # Use scipy.optimize for constrained optimization
    def objective(weights):
        portfolio_variance = weights.T @ cov_matrix.values @ weights
        return portfolio_variance
    
    def constraint_sum(weights):
        return np.sum(weights) - 1
    
    # Initial guess: equal weights
    initial_weights = np.ones(n_assets) / n_assets
    
    # Constraints
    constraints = {'type': 'eq', 'fun': constraint_sum}
    
    # Bounds: weights between 0 and 1
    bounds = [(0, 1) for _ in range(n_assets)]
    
    try:
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = {ticker: weight for ticker, weight in zip(cov_matrix.index, result.x)}
            return optimal_weights
    except:
        pass
    
    # Fallback to equal weights
    equal_weight = 1.0 / n_assets
    return {ticker: equal_weight for ticker in cov_matrix.index}

def optimize_maximum_return(mean_returns: pd.Series, cov_matrix: pd.DataFrame, target_return: float) -> Dict[str, float]:
    """Optimize portfolio for maximum return given target"""
    if not CVXPY_AVAILABLE:
        # Simplified optimization without cvxpy
        return optimize_maximum_return_simple(mean_returns, cov_matrix, target_return)
    
    n_assets = len(mean_returns)
    
    # Define variables
    weights = cp.Variable(n_assets)
    
    # Define constraints
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0
    ]
    
    if target_return is not None:
        portfolio_return = mean_returns.values @ weights
        constraints.append(portfolio_return >= target_return)
    
    # Define objective (minimize variance while meeting return target)
    portfolio_variance = cp.quad_form(weights, cov_matrix.values)
    
    # Solve optimization problem
    problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
    problem.solve()
    
    if problem.status == "optimal":
        optimal_weights = {ticker: weight for ticker, weight in zip(mean_returns.index, weights.value)}
        return optimal_weights
    else:
        # Fallback to equal weights
        equal_weight = 1.0 / n_assets
        return {ticker: equal_weight for ticker in mean_returns.index}

def optimize_maximum_return_simple(mean_returns: pd.Series, cov_matrix: pd.DataFrame, target_return: float) -> Dict[str, float]:
    """Simplified maximum return optimization without cvxpy"""
    n_assets = len(mean_returns)
    
    # Use scipy.optimize for constrained optimization
    def objective(weights):
        portfolio_variance = weights.T @ cov_matrix.values @ weights
        return portfolio_variance
    
    def constraint_sum(weights):
        return np.sum(weights) - 1
    
    def constraint_return(weights):
        portfolio_return = np.sum(mean_returns.values * weights)
        return portfolio_return - target_return if target_return is not None else 0
    
    # Initial guess: equal weights
    initial_weights = np.ones(n_assets) / n_assets
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    if target_return is not None:
        constraints.append({'type': 'ineq', 'fun': constraint_return})
    
    # Bounds: weights between 0 and 1
    bounds = [(0, 1) for _ in range(n_assets)]
    
    try:
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = {ticker: weight for ticker, weight in zip(mean_returns.index, result.x)}
            return optimal_weights
    except:
        pass
    
    # Fallback to equal weights
    equal_weight = 1.0 / n_assets
    return {ticker: equal_weight for ticker in mean_returns.index}

def calculate_portfolio_metrics(weights: Dict[str, float], mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> Dict[str, Any]:
    """Calculate portfolio performance metrics"""
    # Convert weights to array
    weight_array = np.array([weights.get(ticker, 0) for ticker in mean_returns.index])
    
    # Calculate metrics
    portfolio_return = np.sum(mean_returns.values * weight_array) * 252  # Annualized
    portfolio_variance = weight_array.T @ cov_matrix.values @ weight_array
    portfolio_volatility = np.sqrt(portfolio_variance * 252)  # Annualized
    
    # Calculate Sharpe ratio
    risk_free_rate = 0.02
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
    
    return {
        "expected_return": portfolio_return,
        "volatility": portfolio_volatility,
        "sharpe_ratio": sharpe_ratio,
        "variance": portfolio_variance
    }

def perform_portfolio_analysis(historical_data: Dict[str, pd.DataFrame], 
                             weights: Dict[str, float], 
                             benchmark: Optional[str] = None) -> Dict[str, Any]:
    """Perform comprehensive portfolio analysis"""
    # Calculate portfolio returns
    portfolio_returns = calculate_portfolio_returns(historical_data, weights)
    
    # Calculate benchmark returns if provided
    benchmark_returns = None
    if benchmark:
        try:
            benchmark_data = yf.Ticker(benchmark).history(period="2y")
            if not benchmark_data.empty:
                benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        except:
            pass
    
    # Calculate various metrics
    analysis = {
        "total_return": (1 + portfolio_returns).prod() - 1,
        "annualized_return": portfolio_returns.mean() * 252,
        "volatility": portfolio_returns.std() * np.sqrt(252),
        "sharpe_ratio": calculate_sharpe_ratio(portfolio_returns),
        "max_drawdown": calculate_max_drawdown(portfolio_returns),
        "var_95": np.percentile(portfolio_returns, 5),
        "skewness": portfolio_returns.skew(),
        "kurtosis": portfolio_returns.kurtosis()
    }
    
    # Add benchmark comparison if available
    if benchmark_returns is not None:
        analysis["benchmark_comparison"] = {
            "benchmark_return": (1 + benchmark_returns).prod() - 1,
            "benchmark_volatility": benchmark_returns.std() * np.sqrt(252),
            "excess_return": analysis["total_return"] - analysis["benchmark_comparison"]["benchmark_return"],
            "information_ratio": calculate_information_ratio(portfolio_returns, benchmark_returns),
            "beta": calculate_beta(portfolio_returns, benchmark_returns),
            "alpha": calculate_alpha(portfolio_returns, benchmark_returns)
        }
    
    return analysis

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

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate information ratio"""
    excess_returns = portfolio_returns - benchmark_returns
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

def calculate_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate beta"""
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    benchmark_variance = benchmark_returns.var()
    return covariance / benchmark_variance if benchmark_variance > 0 else 1.0

def calculate_alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate alpha"""
    beta = calculate_beta(portfolio_returns, benchmark_returns)
    portfolio_return = portfolio_returns.mean() * 252
    benchmark_return = benchmark_returns.mean() * 252
    return portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))

def calculate_rebalancing_recommendations(current_weights: Dict[str, float], 
                                        target_weights: Dict[str, float], 
                                        threshold: float) -> Dict[str, Any]:
    """Calculate rebalancing recommendations"""
    recommendations = {
        "trades": [],
        "total_deviation": 0,
        "needs_rebalancing": False
    }
    
    all_tickers = set(current_weights.keys()) | set(target_weights.keys())
    
    for ticker in all_tickers:
        current = current_weights.get(ticker, 0)
        target = target_weights.get(ticker, 0)
        deviation = abs(current - target)
        
        if deviation > threshold:
            recommendations["needs_rebalancing"] = True
            recommendations["trades"].append({
                "ticker": ticker,
                "current_weight": current,
                "target_weight": target,
                "deviation": deviation,
                "action": "buy" if target > current else "sell",
                "amount": abs(target - current)
            })
        
        recommendations["total_deviation"] += deviation
    
    return recommendations

def calculate_diversification_metrics(historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Calculate diversification metrics"""
    if not historical_data:
        return {"error": "No data available"}
    
    # Calculate returns matrix
    returns_data = calculate_returns_matrix(historical_data)
    
    if returns_data.empty:
        return {"error": "No valid returns data"}
    
    # Calculate correlation matrix
    correlation_matrix = returns_data.corr()
    
    # Calculate diversification metrics
    avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
    
    # Calculate effective number of stocks
    equal_weights = np.ones(len(returns_data.columns)) / len(returns_data.columns)
    portfolio_variance = equal_weights.T @ returns_data.cov().values @ equal_weights
    avg_stock_variance = np.mean(np.diag(returns_data.cov().values))
    effective_n = avg_stock_variance / portfolio_variance if portfolio_variance > 0 else len(returns_data.columns)
    
    # Calculate concentration metrics
    herfindahl_index = np.sum(equal_weights ** 2)
    
    return {
        "number_of_stocks": len(returns_data.columns),
        "average_correlation": avg_correlation,
        "effective_number_of_stocks": effective_n,
        "herfindahl_index": herfindahl_index,
        "diversification_score": 1 - avg_correlation,
        "concentration_risk": "High" if herfindahl_index > 0.25 else "Medium" if herfindahl_index > 0.1 else "Low"
    } 