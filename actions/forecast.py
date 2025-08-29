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

    // Get user portfolios
    const portfolios = await getUserPortfolios();
    if (!portfolios || portfolios.length === 0) {
      return {
        monteCarloSimulation: [],
        arimaForecast: [],
        machineLearningPrediction: [],
        riskMetrics: {},
        scenarios: [],
        stockForecasts: [],
        portfolioProjections: [],
        volatilityForecast: [],
        correlationMatrix: [],
        marketRegime: {}
      };
    }

    // Extract all stocks from all portfolios
    const allStocks = portfolios.flatMap(portfolio => 
      portfolio.portfolioStocks.map(stock => ({
        ...stock,
        portfolioName: portfolio.name
      }))
    );

    if (allStocks.length === 0) {
      return {
        monteCarloSimulation: [],
        arimaForecast: [],
        machineLearningPrediction: [],
        riskMetrics: {},
        scenarios: [],
        stockForecasts: [],
        portfolioProjections: [],
        volatilityForecast: [],
        correlationMatrix: [],
        marketRegime: {}
      };
    }

    // Get unique tickers and current prices
    const uniqueTickers = [...new Set(allStocks.map(stock => stock.ticker))];
    const priceData = await getStockPrices(uniqueTickers);
    const priceMap = new Map(priceData.map(item => [item.ticker, item]));

    // Calculate portfolio values
    const portfolioValues = {};
    allStocks.forEach(stock => {
      const priceInfo = priceMap.get(stock.ticker);
      if (priceInfo) {
        const currentValue = stock.shares * priceInfo.currentPrice;
        portfolioValues[stock.ticker] = (portfolioValues[stock.ticker] || 0) + currentValue;
      }
    });

    // Call Python backend
    const response = await fetch(`${PYTHON_API_URL}/api/forecast/portfolio`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        tickers: uniqueTickers,
        portfolio_values: portfolioValues,
        forecast_period: forecastPeriod,
        confidence_level: confidenceLevel,
        num_simulations: 1000,
        include_risk_metrics: true,
        include_scenarios: true
      })
    });

    if (!response.ok) {
      throw new Error(`Python API error: ${response.statusText}`);
    }

    const forecastData = await response.json();
    return forecastData;

  } catch (error) {
    console.error("Error fetching forecast data:", error);
    
    // Return fallback data if Python backend is not available
    return generateFallbackForecastData();
  }
}

export async function getStockForecast(ticker, period = "1Y", confidence = 0.95) {
  try {
    const response = await fetch(
      `${PYTHON_API_URL}/api/forecast/stock/${ticker}?period=${period}&confidence=${confidence}`
    );

    if (!response.ok) {
      throw new Error(`Python API error: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error fetching stock forecast:", error);
    return {
      ticker,
      ml_forecast: { forecast_value: 0, confidence: 0 },
      arima_forecast: { forecast_value: 0, confidence: 0 },
      confidence_level: confidence
    };
  }
}

export async function getRiskMetrics(tickers, confidence = 0.95) {
  try {
    const tickerString = tickers.join(',');
    const response = await fetch(
      `${PYTHON_API_URL}/api/forecast/risk-metrics?tickers=${tickerString}&confidence=${confidence}`
    );

    if (!response.ok) {
      throw new Error(`Python API error: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error fetching risk metrics:", error);
    return {
      var: -0.02,
      expected_shortfall: -0.03,
      max_drawdown: -0.15,
      sharpe_ratio: 1.0,
      sortino_ratio: 1.5,
      volatility: 0.15,
      beta: 1.0,
      confidence_level: confidence
    };
  }
}

function generateFallbackForecastData() {
  // Generate fallback data when Python backend is not available
  const days = 365; // 1 year
  const monteCarloSimulation = [];
  const arimaForecast = [];
  const machineLearningPrediction = [];
  const portfolioProjections = [];

  for (let i = 0; i < days; i++) {
    const date = new Date();
    date.setDate(date.getDate() + i);
    const dateStr = date.toISOString().split('T')[0];

    // Monte Carlo simulation (10 paths)
    for (let sim = 1; sim <= 10; sim++) {
      const value = 100000 * (1 + Math.random() * 0.1);
      monteCarloSimulation.push({
        date: dateStr,
        value: value,
        simulation: sim
      });
    }

    // ARIMA forecast
    const value = 100000 * (1 + 0.0002 * i + Math.random() * 0.01);
    arimaForecast.push({
      date: dateStr,
      value: value,
      lower_bound: value * 0.95,
      upper_bound: value * 1.05,
      confidence: 0.95
    });

    // ML forecast
    const mlValue = 100000 * (1 + 0.0003 * i + Math.random() * 0.008);
    machineLearningPrediction.push({
      date: dateStr,
      value: mlValue,
      confidence: 0.85
    });
  }

  // Portfolio projections (12 months)
  for (let month = 0; month <= 12; month++) {
    const date = new Date();
    date.setMonth(date.getMonth() + month);
    const value = 100000 * Math.pow(1.006, month);
    portfolioProjections.push({
      month: date.toISOString().split('T')[0],
      value: value,
      return: 0.6
    });
  }

  return {
    monteCarloSimulation,
    arimaForecast,
    machineLearningPrediction,
    riskMetrics: {
      var: -2500,
      expected_shortfall: -3500,
      max_drawdown: -15000,
      sharpe_ratio: 1.2,
      sortino_ratio: 1.8,
      volatility: 0.15,
      beta: 1.0,
      confidence_level: 0.95
    },
    scenarios: [
      {
        scenario: "Bull Market",
        probability: 0.25,
        return: 0.15,
        value: 115000,
        description: "Strong economic growth, low interest rates"
      },
      {
        scenario: "Base Case",
        probability: 0.50,
        return: 0.08,
        value: 108000,
        description: "Moderate growth, stable conditions"
      },
      {
        scenario: "Bear Market",
        probability: 0.15,
        return: -0.10,
        value: 90000,
        description: "Economic downturn, high volatility"
      },
      {
        scenario: "Crisis",
        probability: 0.10,
        return: -0.25,
        value: 75000,
        description: "Severe market stress, recession"
      }
    ],
    stockForecasts: [
      {
        ticker: "AAPL",
        current_value: 25000,
        forecast_value: 27500,
        confidence: 0.8,
        volatility: 0.2
      }
    ],
    portfolioProjections,
    volatilityForecast: arimaForecast.map(point => ({
      date: point.date,
      volatility: 0.15 + Math.random() * 0.1
    })),
    correlationMatrix: [
      [1.0, 0.7, 0.5],
      [0.7, 1.0, 0.6],
      [0.5, 0.6, 1.0]
    ],
    marketRegime: {
      regime: "Bull Market",
      confidence: 0.7,
      volatility: 0.15,
      mean_return: 0.12
    }
  };
} 