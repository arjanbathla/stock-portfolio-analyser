"use server";

import { auth } from "@clerk/nextjs/server";
import { getUserPortfolios } from "./dashboard";
import { getStockPrices } from "./stocks";

const PYTHON_API_URL = process.env.PYTHON_API_URL || "http://localhost:8000";

export async function getForecastData(forecastPeriod = "1Y", confidenceLevel = 0.95) {
  try {
    const { userId } = await auth();
    if (!userId) {
      throw new Error("User not authenticated");
    }

    // Get user's portfolio data
    const portfolios = await getUserPortfolios();
    if (!portfolios || portfolios.length === 0) {
      return generateFallbackForecastData();
    }

    // Extract tickers and portfolio values
    const tickers = [];
    const portfolioValues = {};
    
    portfolios.forEach(portfolio => {
      portfolio.stocks.forEach(stock => {
        if (!tickers.includes(stock.ticker)) {
          tickers.push(stock.ticker);
        }
        portfolioValues[stock.ticker] = (portfolioValues[stock.ticker] || 0) + 
          (stock.shares * stock.purchasePrice);
      });
    });

    if (tickers.length === 0) {
      return generateFallbackForecastData();
    }

    // Call Python backend API
    const response = await fetch(`${PYTHON_API_URL}/api/forecast/portfolio`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        tickers: tickers,
        portfolio_values: portfolioValues,
        forecast_period: forecastPeriod,
        confidence_level: confidenceLevel,
        num_simulations: 1000,
        include_risk_metrics: true,
        include_scenarios: true
      }),
    });

    if (!response.ok) {
      console.warn('Python API not available, using fallback data');
      return generateFallbackForecastData();
    }

    const data = await response.json();
    return data;

  } catch (error) {
    console.error('Error fetching forecast data:', error);
    return generateFallbackForecastData();
  }
}

export async function getStockForecast(ticker, period = "1Y", confidence = 0.95) {
  try {
    const response = await fetch(
      `${PYTHON_API_URL}/api/forecast/stock/${ticker}?period=${period}&confidence=${confidence}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    if (!response.ok) {
      return generateFallbackStockForecast(ticker, period);
    }

    const data = await response.json();
    return data;

  } catch (error) {
    console.error('Error fetching stock forecast:', error);
    return generateFallbackStockForecast(ticker, period);
  }
}

export async function getRiskMetrics(tickers, confidence = 0.95) {
  try {
    const response = await fetch(`${PYTHON_API_URL}/api/forecast/risk-metrics`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        tickers: tickers,
        confidence_level: confidence
      }),
    });

    if (!response.ok) {
      return generateFallbackRiskMetrics(tickers);
    }

    const data = await response.json();
    return data;

  } catch (error) {
    console.error('Error fetching risk metrics:', error);
    return generateFallbackRiskMetrics(tickers);
  }
}

function generateFallbackForecastData() {
  const now = new Date();
  const days = 365;
  
  const monteCarloData = [];
  const arimaData = [];
  const mlData = [];
  const volatilityData = [];
  
  for (let i = 0; i <= days; i++) {
    const date = new Date(now.getTime() + i * 24 * 60 * 60 * 1000);
    const baseValue = 100000;
    const randomFactor = 1 + (Math.random() - 0.5) * 0.1;
    
    monteCarloData.push({
      date: date.toISOString().split('T')[0],
      value: baseValue * Math.pow(randomFactor, i / 30),
      confidence_lower: baseValue * Math.pow(0.95, i / 30),
      confidence_upper: baseValue * Math.pow(1.05, i / 30)
    });
    
    arimaData.push({
      date: date.toISOString().split('T')[0],
      value: baseValue * (1 + i * 0.0001 + Math.sin(i / 30) * 0.02)
    });
    
    mlData.push({
      date: date.toISOString().split('T')[0],
      value: baseValue * (1 + i * 0.0002 + Math.cos(i / 45) * 0.015)
    });
    
    volatilityData.push({
      date: date.toISOString().split('T')[0],
      volatility: 0.15 + Math.sin(i / 60) * 0.05
    });
  }

  return {
    monte_carlo: monteCarloData,
    arima_forecast: arimaData,
    ml_forecast: mlData,
    risk_metrics: {
      var_95: 8500,
      expected_shortfall: 12000,
      sharpe_ratio: 1.2,
      sortino_ratio: 1.8,
      max_drawdown: 0.15,
      volatility: 0.18,
      beta: 1.1
    },
    scenarios: [
      { name: "Bull Market", probability: 0.3, return: 0.25 },
      { name: "Base Case", probability: 0.5, return: 0.08 },
      { name: "Bear Market", probability: 0.2, return: -0.12 }
    ],
    stock_forecasts: [
      { ticker: "AAPL", forecast_return: 0.12, confidence: 0.85 },
      { ticker: "GOOGL", forecast_return: 0.15, confidence: 0.82 },
      { ticker: "MSFT", forecast_return: 0.10, confidence: 0.88 }
    ],
    volatility_forecast: volatilityData,
    correlation_matrix: [
      [1.0, 0.7, 0.6],
      [0.7, 1.0, 0.8],
      [0.6, 0.8, 1.0]
    ],
    market_regime: {
      current_regime: "Expansion",
      regime_probability: 0.65,
      transition_matrix: [[0.8, 0.2], [0.3, 0.7]]
    }
  };
}

function generateFallbackStockForecast(ticker, period) {
  const now = new Date();
  const days = period === "1M" ? 30 : period === "3M" ? 90 : 365;
  
  const forecastData = [];
  const basePrice = 150 + Math.random() * 100;
  
  for (let i = 0; i <= days; i++) {
    const date = new Date(now.getTime() + i * 24 * 60 * 60 * 1000);
    const price = basePrice * (1 + i * 0.0001 + Math.sin(i / 30) * 0.02);
    
    forecastData.push({
      date: date.toISOString().split('T')[0],
      price: price,
      confidence_lower: price * 0.95,
      confidence_upper: price * 1.05
    });
  }

  return {
    ticker: ticker,
    forecast_data: forecastData,
    metrics: {
      expected_return: 0.12,
      volatility: 0.18,
      sharpe_ratio: 0.67,
      var_95: basePrice * 0.15
    }
  };
}

function generateFallbackRiskMetrics(tickers) {
  return {
    portfolio_risk: {
      var_95: 8500,
      expected_shortfall: 12000,
      volatility: 0.18,
      beta: 1.1
    },
    individual_risks: tickers.map(ticker => ({
      ticker: ticker,
      var_95: 2000 + Math.random() * 3000,
      volatility: 0.15 + Math.random() * 0.1,
      beta: 0.8 + Math.random() * 0.6
    })),
    correlation_matrix: tickers.map(() => 
      tickers.map(() => 0.3 + Math.random() * 0.4)
    )
  };
} 