import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

class MonteCarloSimulation:
    """Advanced Monte Carlo simulation for portfolio forecasting"""
    
    def __init__(self):
        self.simulation_results = []
        
    def run_simulation(self, historical_data: Dict[str, pd.DataFrame], 
                      weights: Dict[str, float], 
                      forecast_period: str, 
                      num_simulations: int = 1000) -> List[Dict[str, Any]]:
        """
        Run Monte Carlo simulation using Geometric Brownian Motion
        """
        # Calculate parameters from historical data
        returns_data = self._calculate_returns(historical_data)
        mu, sigma = self._estimate_parameters(returns_data, weights)
        
        # Set simulation parameters
        days = self._get_forecast_days(forecast_period)
        dt = 1/252  # Daily time step
        
        # Initialize results
        simulation_paths = []
        
        for sim in range(num_simulations):
            path = []
            current_value = 1.0  # Start with $1, will scale later
            
            for day in range(days):
                # Geometric Brownian Motion
                z = np.random.normal(0, 1)
                drift = (mu - 0.5 * sigma**2) * dt
                diffusion = sigma * np.sqrt(dt) * z
                current_value *= np.exp(drift + diffusion)
                
                # Add date
                sim_date = datetime.now() + timedelta(days=day)
                
                path.append({
                    "date": sim_date.strftime("%Y-%m-%d"),
                    "value": current_value,
                    "simulation": sim + 1
                })
            
            simulation_paths.extend(path)
        
        return simulation_paths
    
    def _calculate_returns(self, historical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate returns for all stocks"""
        returns_data = {}
        for ticker, data in historical_data.items():
            if not data.empty:
                returns_data[ticker] = data['Close'].pct_change().dropna()
        return pd.DataFrame(returns_data)
    
    def _estimate_parameters(self, returns_data: pd.DataFrame, weights: Dict[str, float]) -> Tuple[float, float]:
        """Estimate drift and volatility parameters"""
        if returns_data.empty:
            return 0.08, 0.15  # Default values
        
        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(0.0, index=returns_data.index)
        for ticker, weight in weights.items():
            if ticker in returns_data.columns:
                portfolio_returns += returns_data[ticker] * weight
        
        # Estimate parameters
        mu = portfolio_returns.mean() * 252  # Annualized drift
        sigma = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility
        
        return mu, sigma
    
    def _get_forecast_days(self, forecast_period: str) -> int:
        """Convert forecast period to number of days"""
        period_map = {
            "6M": 180,
            "1Y": 365,
            "2Y": 730,
            "5Y": 1825
        }
        return period_map.get(forecast_period, 365)

class ARIMAModel:
    """ARIMA time series forecasting model"""
    
    def __init__(self):
        self.models = {}
        
    def forecast(self, historical_data: Dict[str, pd.DataFrame], 
                weights: Dict[str, float], 
                forecast_period: str) -> List[Dict[str, Any]]:
        """
        Generate ARIMA forecast for portfolio
        """
        # Create portfolio time series
        portfolio_series = self._create_portfolio_series(historical_data, weights)
        
        if portfolio_series.empty:
            return self._generate_default_forecast(forecast_period)
        
        # Fit ARIMA model
        try:
            model = self._fit_arima_model(portfolio_series)
            forecast_steps = self._get_forecast_days(forecast_period)
            
            # Generate forecast
            forecast = model.forecast(steps=forecast_steps)
            conf_int = model.get_forecast(steps=forecast_steps).conf_int()
            
            # Format results
            results = []
            for i, (pred, lower, upper) in enumerate(zip(forecast, conf_int.iloc[:, 0], conf_int.iloc[:, 1])):
                date = datetime.now() + timedelta(days=i)
                results.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "value": max(0, pred),
                    "lower_bound": max(0, lower),
                    "upper_bound": max(0, upper),
                    "confidence": 0.95
                })
            
            return results
            
        except Exception as e:
            print(f"ARIMA forecast failed: {e}")
            return self._generate_default_forecast(forecast_period)
    
    def forecast_stock(self, stock_data: pd.DataFrame, forecast_period: str) -> Dict[str, Any]:
        """Forecast individual stock using ARIMA"""
        if stock_data.empty:
            return {"forecast_value": 0, "confidence": 0}
        
        try:
            # Use log returns for stationarity
            log_returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1)).dropna()
            
            # Fit ARIMA model
            model = self._fit_arima_model(log_returns)
            forecast_steps = self._get_forecast_days(forecast_period)
            
            # Generate forecast
            forecast = model.forecast(steps=forecast_steps)
            
            # Convert back to price
            last_price = stock_data['Close'].iloc[-1]
            forecast_price = last_price * np.exp(np.cumsum(forecast))
            
            return {
                "forecast_value": forecast_price.iloc[-1],
                "confidence": 0.85,
                "volatility": log_returns.std() * np.sqrt(252)
            }
            
        except Exception as e:
            print(f"Stock ARIMA forecast failed: {e}")
            return {"forecast_value": stock_data['Close'].iloc[-1], "confidence": 0}
    
    def _create_portfolio_series(self, historical_data: Dict[str, pd.DataFrame], 
                               weights: Dict[str, float]) -> pd.Series:
        """Create portfolio time series from individual stocks"""
        portfolio_values = pd.Series(0.0)
        
        for ticker, data in historical_data.items():
            if ticker in weights and not data.empty:
                weighted_values = data['Close'] * weights[ticker]
                if portfolio_values.empty:
                    portfolio_values = weighted_values
                else:
                    portfolio_values = portfolio_values.add(weighted_values, fill_value=0)
        
        return portfolio_values
    
    def _fit_arima_model(self, series: pd.Series) -> ARIMA:
        """Fit ARIMA model with automatic parameter selection"""
        # Make series stationary
        if not self._is_stationary(series):
            series = series.diff().dropna()
        
        # Try different ARIMA parameters
        best_aic = float('inf')
        best_model = None
        
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_model = fitted_model
                    except:
                        continue
        
        if best_model is None:
            # Fallback to simple model
            model = ARIMA(series, order=(1, 1, 1))
            best_model = model.fit()
        
        return best_model
    
    def _is_stationary(self, series: pd.Series) -> bool:
        """Check if time series is stationary"""
        try:
            result = adfuller(series.dropna())
            return result[1] < 0.05
        except:
            return False
    
    def _get_forecast_days(self, forecast_period: str) -> int:
        """Convert forecast period to number of days"""
        period_map = {
            "6M": 180,
            "1Y": 365,
            "2Y": 730,
            "5Y": 1825
        }
        return period_map.get(forecast_period, 365)
    
    def _generate_default_forecast(self, forecast_period: str) -> List[Dict[str, Any]]:
        """Generate default forecast when model fails"""
        days = self._get_forecast_days(forecast_period)
        results = []
        
        for i in range(days):
            date = datetime.now() + timedelta(days=i)
            results.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": 1.0,
                "lower_bound": 0.9,
                "upper_bound": 1.1,
                "confidence": 0.5
            })
        
        return results

class MachineLearningModel:
    """Advanced machine learning model for financial forecasting"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def predict(self, historical_data: Dict[str, pd.DataFrame], 
               weights: Dict[str, float], 
               forecast_period: str) -> List[Dict[str, Any]]:
        """
        Generate ML-based forecast for portfolio
        """
        # Prepare features
        features, targets = self._prepare_features(historical_data, weights)
        
        if features.empty or targets.empty:
            return self._generate_default_forecast(forecast_period)
        
        # Train model
        model = self._train_model(features, targets)
        
        # Generate future features
        future_features = self._generate_future_features(features, forecast_period)
        
        # Make predictions
        predictions = model.predict(future_features)
        
        # Calculate confidence intervals
        confidence_scores = self._calculate_confidence(model, future_features)
        
        # Format results
        results = []
        for i, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
            date = datetime.now() + timedelta(days=i)
            results.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": max(0, pred),
                "confidence": conf
            })
        
        return results
    
    def predict_stock(self, stock_data: pd.DataFrame, forecast_period: str) -> Dict[str, Any]:
        """Predict individual stock performance"""
        if stock_data.empty:
            return {"forecast_value": 0, "confidence": 0, "volatility": 0}
        
        # Prepare features
        features = self._prepare_stock_features(stock_data)
        targets = stock_data['Close'].pct_change().dropna()
        
        if features.empty or targets.empty:
            return {"forecast_value": stock_data['Close'].iloc[-1], "confidence": 0, "volatility": 0}
        
        # Train model
        model = self._train_model(features, targets)
        
        # Generate future features
        future_features = self._generate_stock_future_features(features, forecast_period)
        
        # Make prediction
        prediction = model.predict(future_features.iloc[-1:])[0]
        
        # Calculate confidence
        confidence = self._calculate_confidence(model, future_features.iloc[-1:])[0]
        
        # Calculate volatility
        volatility = targets.std() * np.sqrt(252)
        
        # Convert to price forecast
        last_price = stock_data['Close'].iloc[-1]
        forecast_price = last_price * (1 + prediction)
        
        return {
            "forecast_value": forecast_price,
            "confidence": confidence,
            "volatility": volatility
        }
    
    def _prepare_features(self, historical_data: Dict[str, pd.DataFrame], 
                         weights: Dict[str, float]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for portfolio prediction"""
        features_list = []
        targets_list = []
        
        for ticker, data in historical_data.items():
            if ticker in weights and not data.empty:
                stock_features = self._prepare_stock_features(data)
                stock_targets = data['Close'].pct_change().dropna()
                
                if not stock_features.empty and not stock_targets.empty:
                    # Weight the features and targets
                    weighted_features = stock_features * weights[ticker]
                    weighted_targets = stock_targets * weights[ticker]
                    
                    features_list.append(weighted_features)
                    targets_list.append(weighted_targets)
        
        if not features_list:
            return pd.DataFrame(), pd.Series()
        
        # Combine features and targets
        combined_features = pd.concat(features_list, axis=1).fillna(0)
        combined_targets = pd.concat(targets_list, axis=1).sum(axis=1)
        
        return combined_features, combined_targets
    
    def _prepare_stock_features(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for individual stock"""
        features = pd.DataFrame(index=stock_data.index)
        
        # Price-based features
        features['price'] = stock_data['Close']
        features['returns'] = stock_data['Close'].pct_change()
        features['log_returns'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
        
        # Technical indicators
        features['sma_5'] = stock_data['Close'].rolling(5).mean()
        features['sma_20'] = stock_data['Close'].rolling(20).mean()
        features['rsi'] = self._calculate_rsi(stock_data['Close'])
        features['volatility'] = stock_data['Close'].rolling(20).std()
        
        # Volume features
        if 'Volume' in stock_data.columns:
            features['volume'] = stock_data['Volume']
            features['volume_sma'] = stock_data['Volume'].rolling(20).mean()
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volume_lag_{lag}'] = features.get('volume', pd.Series(0)).shift(lag)
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _train_model(self, features: pd.DataFrame, targets: pd.Series) -> Any:
        """Train machine learning model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models and select best
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'extra_trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf'),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
        }
        
        best_score = -float('inf')
        best_model = None
        
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                score = model.score(X_test_scaled, y_test)
                if score > best_score:
                    best_score = score
                    best_model = model
            except:
                continue
        
        if best_model is None:
            # Fallback to simple linear regression
            best_model = LinearRegression()
            best_model.fit(X_train_scaled, y_train)
        
        return best_model
    
    def _generate_future_features(self, features: pd.DataFrame, forecast_period: str) -> pd.DataFrame:
        """Generate future features for forecasting"""
        days = self._get_forecast_days(forecast_period)
        future_features = features.copy()
        
        # Extend features for future dates
        last_date = features.index[-1]
        for i in range(1, days + 1):
            future_date = last_date + timedelta(days=i)
            future_features.loc[future_date] = future_features.iloc[-1]
        
        return future_features
    
    def _generate_stock_future_features(self, features: pd.DataFrame, forecast_period: str) -> pd.DataFrame:
        """Generate future features for stock prediction"""
        return self._generate_future_features(features, forecast_period)
    
    def _calculate_confidence(self, model: Any, features: pd.DataFrame) -> List[float]:
        """Calculate confidence scores for predictions"""
        # Simple confidence based on feature variance
        confidence_scores = []
        for i in range(len(features)):
            # Calculate confidence based on feature stability
            feature_variance = features.iloc[i].var()
            confidence = max(0.3, 1 - feature_variance)
            confidence_scores.append(confidence)
        
        return confidence_scores
    
    def _get_forecast_days(self, forecast_period: str) -> int:
        """Convert forecast period to number of days"""
        period_map = {
            "6M": 180,
            "1Y": 365,
            "2Y": 730,
            "5Y": 1825
        }
        return period_map.get(forecast_period, 365)
    
    def _generate_default_forecast(self, forecast_period: str) -> List[Dict[str, Any]]:
        """Generate default forecast when model fails"""
        days = self._get_forecast_days(forecast_period)
        results = []
        
        for i in range(days):
            date = datetime.now() + timedelta(days=i)
            results.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": 1.0,
                "confidence": 0.5
            })
        
        return results

class RiskMetrics:
    """Calculate comprehensive risk metrics"""
    
    def calculate_metrics(self, historical_data: Dict[str, pd.DataFrame], 
                         weights: Dict[str, float], 
                         confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Calculate portfolio risk metrics
        """
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(historical_data, weights)
        
        if portfolio_returns.empty:
            return self._get_default_risk_metrics()
        
        # Calculate various risk metrics
        var = self._calculate_var(portfolio_returns, confidence_level)
        expected_shortfall = self._calculate_expected_shortfall(portfolio_returns, confidence_level)
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        volatility = self._calculate_volatility(portfolio_returns)
        beta = self._calculate_beta(portfolio_returns)
        
        return {
            "var": var,
            "expected_shortfall": expected_shortfall,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "volatility": volatility,
            "beta": beta,
            "confidence_level": confidence_level
        }
    
    def _calculate_portfolio_returns(self, historical_data: Dict[str, pd.DataFrame], 
                                   weights: Dict[str, float]) -> pd.Series:
        """Calculate portfolio returns"""
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
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        var = self._calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate Maximum Drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio"""
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino Ratio"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std()
        
        if downside_deviation == 0:
            return 0
        
        return excess_returns.mean() / downside_deviation * np.sqrt(252)
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(252)
    
    def _calculate_beta(self, returns: pd.Series) -> float:
        """Calculate beta (simplified - would need market data)"""
        # Simplified beta calculation
        return 1.0  # Default to market beta
    
    def _get_default_risk_metrics(self) -> Dict[str, Any]:
        """Return default risk metrics when data is insufficient"""
        return {
            "var": -0.02,
            "expected_shortfall": -0.03,
            "max_drawdown": -0.15,
            "sharpe_ratio": 1.0,
            "sortino_ratio": 1.5,
            "volatility": 0.15,
            "beta": 1.0,
            "confidence_level": 0.95
        }

class ScenarioAnalysis:
    """Generate scenario analysis for portfolio"""
    
    def generate_scenarios(self, current_value: float, forecast_period: str) -> List[Dict[str, Any]]:
        """
        Generate different market scenarios
        """
        scenarios = [
            {
                "scenario": "Bull Market",
                "probability": 0.25,
                "return": 0.15,
                "description": "Strong economic growth, low interest rates",
                "value": current_value * (1 + 0.15) ** (self._get_forecast_years(forecast_period))
            },
            {
                "scenario": "Base Case",
                "probability": 0.50,
                "return": 0.08,
                "description": "Moderate growth, stable conditions",
                "value": current_value * (1 + 0.08) ** (self._get_forecast_years(forecast_period))
            },
            {
                "scenario": "Bear Market",
                "probability": 0.15,
                "return": -0.10,
                "description": "Economic downturn, high volatility",
                "value": current_value * (1 - 0.10) ** (self._get_forecast_years(forecast_period))
            },
            {
                "scenario": "Crisis",
                "probability": 0.10,
                "return": -0.25,
                "description": "Severe market stress, recession",
                "value": current_value * (1 - 0.25) ** (self._get_forecast_years(forecast_period))
            }
        ]
        
        return scenarios
    
    def _get_forecast_years(self, forecast_period: str) -> float:
        """Convert forecast period to years"""
        period_map = {
            "6M": 0.5,
            "1Y": 1.0,
            "2Y": 2.0,
            "5Y": 5.0
        }
        return period_map.get(forecast_period, 1.0)

class VolatilityModel:
    """Volatility forecasting model"""
    
    def forecast_volatility(self, historical_data: Dict[str, pd.DataFrame], 
                          forecast_period: str) -> List[Dict[str, Any]]:
        """
        Forecast volatility for portfolio
        """
        # Calculate portfolio volatility
        portfolio_volatility = self._calculate_portfolio_volatility(historical_data)
        
        if portfolio_volatility.empty:
            return self._generate_default_volatility_forecast(forecast_period)
        
        # Fit volatility model (simplified)
        days = self._get_forecast_days(forecast_period)
        results = []
        
        for i in range(days):
            date = datetime.now() + timedelta(days=i)
            # Simple volatility forecast with some randomness
            volatility = portfolio_volatility.mean() * (1 + np.random.normal(0, 0.1))
            results.append({
                "date": date.strftime("%Y-%m-%d"),
                "volatility": max(0.05, volatility)
            })
        
        return results
    
    def _calculate_portfolio_volatility(self, historical_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """Calculate portfolio volatility"""
        portfolio_returns = pd.Series(0.0)
        
        for ticker, data in historical_data.items():
            if not data.empty:
                returns = data['Close'].pct_change().dropna()
                if portfolio_returns.empty:
                    portfolio_returns = returns
                else:
                    portfolio_returns = portfolio_returns.add(returns, fill_value=0)
        
        # Calculate rolling volatility
        return portfolio_returns.rolling(20).std() * np.sqrt(252)
    
    def _get_forecast_days(self, forecast_period: str) -> int:
        """Convert forecast period to number of days"""
        period_map = {
            "6M": 180,
            "1Y": 365,
            "2Y": 730,
            "5Y": 1825
        }
        return period_map.get(forecast_period, 365)
    
    def _generate_default_volatility_forecast(self, forecast_period: str) -> List[Dict[str, Any]]:
        """Generate default volatility forecast"""
        days = self._get_forecast_days(forecast_period)
        results = []
        
        for i in range(days):
            date = datetime.now() + timedelta(days=i)
            results.append({
                "date": date.strftime("%Y-%m-%d"),
                "volatility": 0.15 + np.random.normal(0, 0.02)
            })
        
        return results

class CorrelationModel:
    """Correlation analysis and forecasting"""
    
    def calculate_correlation_matrix(self, historical_data: Dict[str, pd.DataFrame]) -> List[List[float]]:
        """
        Calculate correlation matrix for stocks
        """
        if not historical_data:
            return []
        
        # Prepare returns data
        returns_data = {}
        for ticker, data in historical_data.items():
            if not data.empty:
                returns_data[ticker] = data['Close'].pct_change().dropna()
        
        if not returns_data:
            return []
        
        # Create DataFrame and calculate correlation
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr().fillna(0)
        
        return correlation_matrix.values.tolist() 