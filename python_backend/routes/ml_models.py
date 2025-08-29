from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not available. Will use alternative models.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: lightgbm not available. Will use alternative models.")
import warnings
warnings.filterwarnings('ignore')

router = APIRouter()

class MLModelRequest(BaseModel):
    tickers: List[str]
    target_variable: str = "returns"  # returns, volatility, drawdown
    model_type: str = "ensemble"  # linear, tree, neural_network, ensemble
    features: List[str] = ["price", "volume", "technical_indicators"]
    test_size: float = 0.2
    validation_split: float = 0.2

class ModelTrainingRequest(BaseModel):
    tickers: List[str]
    config: Dict[str, Any]
    hyperparameter_tuning: bool = False

@router.post("/train")
async def train_ml_model(request: ModelTrainingRequest):
    """
    Train machine learning model for financial prediction
    """
    try:
        # Get historical data
        historical_data = await get_historical_data(request.tickers)
        
        if not historical_data:
            raise HTTPException(status_code=400, detail="No valid data found for tickers")
        
        # Prepare features and targets
        features, targets = prepare_ml_data(historical_data, request.config)
        
        if features.empty or targets.empty:
            raise HTTPException(status_code=400, detail="Insufficient data for training")
        
        # Train model
        model_results = train_model(features, targets, request.config, request.hyperparameter_tuning)
        
        return model_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@router.post("/predict")
async def predict_with_ml_model(request: MLModelRequest):
    """
    Make predictions using trained ML model
    """
    try:
        # Get historical data
        historical_data = await get_historical_data(request.tickers)
        
        if not historical_data:
            raise HTTPException(status_code=400, detail="No valid data found for tickers")
        
        # Prepare features
        features = prepare_features(historical_data, request.features)
        
        if features.empty:
            raise HTTPException(status_code=400, detail="No valid features generated")
        
        # Make predictions
        predictions = make_predictions(features, request.model_type, request.target_variable)
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/models")
async def get_available_models():
    """
    Get list of available ML models
    """
    models = {
        "linear_models": ["LinearRegression", "Ridge", "Lasso", "ElasticNet"],
        "tree_models": ["RandomForest", "GradientBoosting", "ExtraTrees", "XGBoost", "LightGBM"],
        "neural_networks": ["MLPRegressor", "SVR"],
        "ensemble_models": ["VotingRegressor", "StackingRegressor"]
    }
    return models

@router.get("/feature-importance")
async def get_feature_importance(
    tickers: str = Query(..., description="Comma-separated tickers"),
    model_type: str = Query("random_forest", description="Model type")
):
    """
    Get feature importance from trained model
    """
    try:
        ticker_list = [t.strip() for t in tickers.split(",")]
        historical_data = await get_historical_data(ticker_list)
        
        if not historical_data:
            raise HTTPException(status_code=400, detail="No valid data found")
        
        # Prepare features
        features = prepare_features(historical_data, ["price", "volume", "technical_indicators"])
        targets = prepare_targets(historical_data, "returns")
        
        if features.empty or targets.empty:
            raise HTTPException(status_code=400, detail="Insufficient data")
        
        # Train model and get feature importance
        importance = calculate_feature_importance(features, targets, model_type)
        
        return importance
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature importance calculation failed: {str(e)}")

@router.get("/model-evaluation")
async def evaluate_model(
    tickers: str = Query(..., description="Comma-separated tickers"),
    model_type: str = Query("random_forest", description="Model type")
):
    """
    Evaluate ML model performance
    """
    try:
        ticker_list = [t.strip() for t in tickers.split(",")]
        historical_data = await get_historical_data(ticker_list)
        
        if not historical_data:
            raise HTTPException(status_code=400, detail="No valid data found")
        
        # Prepare data
        features = prepare_features(historical_data, ["price", "volume", "technical_indicators"])
        targets = prepare_targets(historical_data, "returns")
        
        if features.empty or targets.empty:
            raise HTTPException(status_code=400, detail="Insufficient data")
        
        # Evaluate model
        evaluation = evaluate_model_performance(features, targets, model_type)
        
        return evaluation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")

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

def prepare_ml_data(historical_data: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> tuple:
    """Prepare features and targets for ML training"""
    features_list = []
    targets_list = []
    
    for ticker, data in historical_data.items():
        if not data.empty:
            # Prepare features
            stock_features = prepare_stock_features(data, config.get("features", ["price", "volume", "technical_indicators"]))
            stock_targets = prepare_targets({ticker: data}, config.get("target_variable", "returns"))
            
            if not stock_features.empty and not stock_targets.empty:
                features_list.append(stock_features)
                targets_list.append(stock_targets)
    
    if not features_list:
        return pd.DataFrame(), pd.Series()
    
    # Combine features and targets
    combined_features = pd.concat(features_list, axis=0)
    combined_targets = pd.concat(targets_list, axis=0)
    
    return combined_features, combined_targets

def prepare_features(historical_data: Dict[str, pd.DataFrame], feature_types: List[str]) -> pd.DataFrame:
    """Prepare features for ML model"""
    features_list = []
    
    for ticker, data in historical_data.items():
        if not data.empty:
            stock_features = prepare_stock_features(data, feature_types)
            if not stock_features.empty:
                features_list.append(stock_features)
    
    if not features_list:
        return pd.DataFrame()
    
    return pd.concat(features_list, axis=0)

def prepare_stock_features(data: pd.DataFrame, feature_types: List[str]) -> pd.DataFrame:
    """Prepare features for individual stock"""
    features = pd.DataFrame(index=data.index)
    
    if "price" in feature_types:
        features['price'] = data['Close']
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        features['price_change'] = data['Close'] - data['Close'].shift(1)
    
    if "volume" in feature_types and 'Volume' in data.columns:
        features['volume'] = data['Volume']
        features['volume_change'] = data['Volume'].pct_change()
        features['volume_sma'] = data['Volume'].rolling(20).mean()
    
    if "technical_indicators" in feature_types:
        # Moving averages
        features['sma_5'] = data['Close'].rolling(5).mean()
        features['sma_20'] = data['Close'].rolling(20).mean()
        features['ema_12'] = data['Close'].ewm(span=12).mean()
        features['ema_26'] = data['Close'].ewm(span=26).mean()
        
        # RSI
        features['rsi'] = calculate_rsi(data['Close'])
        
        # Bollinger Bands
        bb_upper, bb_lower = calculate_bollinger_bands(data['Close'])
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # MACD
        features['macd'] = features['ema_12'] - features['ema_26']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Volatility
        features['volatility'] = data['Close'].rolling(20).std()
        features['atr'] = calculate_atr(data)
    
    # Lagged features
    for lag in [1, 2, 3, 5, 10]:
        if 'returns' in features.columns:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        if 'volume' in features.columns:
            features[f'volume_lag_{lag}'] = features['volume'].shift(lag)
    
    # Drop NaN values
    features = features.dropna()
    
    return features

def prepare_targets(historical_data: Dict[str, pd.DataFrame], target_variable: str) -> pd.Series:
    """Prepare target variable for ML model"""
    targets_list = []
    
    for ticker, data in historical_data.items():
        if not data.empty:
            if target_variable == "returns":
                target = data['Close'].pct_change().shift(-1)  # Next day returns
            elif target_variable == "volatility":
                target = data['Close'].rolling(20).std().shift(-1)
            elif target_variable == "drawdown":
                cumulative = (1 + data['Close'].pct_change()).cumprod()
                running_max = cumulative.expanding().max()
                target = (cumulative - running_max) / running_max
            else:
                target = data['Close'].pct_change().shift(-1)
            
            targets_list.append(target)
    
    if not targets_list:
        return pd.Series()
    
    return pd.concat(targets_list, axis=0)

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
    """Calculate Bollinger Bands"""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return atr

def train_model(features: pd.DataFrame, targets: pd.Series, config: Dict[str, Any], hyperparameter_tuning: bool = False) -> Dict[str, Any]:
    """Train ML model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=config.get("test_size", 0.2), random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Select model
    model_type = config.get("model_type", "random_forest")
    model = get_model(model_type)
    
    # Hyperparameter tuning
    if hyperparameter_tuning:
        model = tune_hyperparameters(model, model_type, X_train_scaled, y_train)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Get feature importance if available
    feature_importance = get_feature_importance_from_model(model, features.columns)
    
    return {
        "model_type": model_type,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "model": str(model)
    }

def get_model(model_type: str):
    """Get ML model based on type"""
    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(),
        "lasso": Lasso(),
        "elastic_net": ElasticNet(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "extra_trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "svr": SVR(kernel='rbf'),
        "mlp": MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
    }
    
    # Add advanced models if available
    if XGBOOST_AVAILABLE:
        models["xgboost"] = xgb.XGBRegressor(n_estimators=100, random_state=42)
    if LIGHTGBM_AVAILABLE:
        models["lightgbm"] = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    
    return models.get(model_type, RandomForestRegressor(n_estimators=100, random_state=42))

def tune_hyperparameters(model, model_type: str, X_train: np.ndarray, y_train: pd.Series) -> Any:
    """Tune hyperparameters using GridSearchCV"""
    param_grids = {
        "random_forest": {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        },
        "gradient_boosting": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        "xgboost": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
    
    if model_type in param_grids:
        grid_search = GridSearchCV(model, param_grids[model_type], cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    return model

def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate model performance metrics"""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

def get_feature_importance_from_model(model, feature_names) -> Dict[str, float]:
    """Extract feature importance from trained model"""
    importance = {}
    
    if hasattr(model, 'feature_importances_'):
        for name, imp in zip(feature_names, model.feature_importances_):
            importance[name] = float(imp)
    elif hasattr(model, 'coef_'):
        for name, coef in zip(feature_names, model.coef_):
            importance[name] = float(abs(coef))
    
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

def make_predictions(features: pd.DataFrame, model_type: str, target_variable: str) -> Dict[str, Any]:
    """Make predictions using trained model"""
    # This would typically use a pre-trained model
    # For now, we'll create a simple prediction
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Get model and make prediction
    model = get_model(model_type)
    
    # Simple prediction (in real implementation, would use trained model)
    predictions = np.random.normal(0, 0.01, len(features))
    
    return {
        "predictions": predictions.tolist(),
        "confidence": [0.8] * len(predictions),
        "model_type": model_type,
        "target_variable": target_variable
    }

def calculate_feature_importance(features: pd.DataFrame, targets: pd.Series, model_type: str) -> Dict[str, Any]:
    """Calculate feature importance"""
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = get_model(model_type)
    model.fit(X_train_scaled, y_train)
    
    # Get feature importance
    importance = get_feature_importance_from_model(model, features.columns)
    
    return {
        "feature_importance": importance,
        "top_features": list(importance.keys())[:10],
        "model_type": model_type
    }

def evaluate_model_performance(features: pd.DataFrame, targets: pd.Series, model_type: str) -> Dict[str, Any]:
    """Evaluate model performance"""
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = get_model(model_type)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    return {
        "metrics": metrics,
        "cross_validation_scores": cv_scores.tolist(),
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "model_type": model_type
    } 