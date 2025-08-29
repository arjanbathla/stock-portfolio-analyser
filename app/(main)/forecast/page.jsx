"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  ComposedChart,
} from 'recharts';
import { getForecastData } from '@/actions/forecast';
import { formatCurrency } from '@/lib/portfolioHistory';
import { toast } from 'sonner';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Target, 
  BarChart3, 
  Activity,
  Award,
  AlertTriangle,
  ArrowUpRight,
  ArrowDownRight,
  Zap,
  Target as TargetIcon,
  Users,
  Globe,
  Clock,
  Calendar,
  Filter,
  Download,
  RefreshCw,
  Brain,
  Cpu,
  Database,
  BarChart4,
  PieChart,
  LineChart as LineChartIcon,
  Target as TargetIcon2
} from 'lucide-react';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FFC658', '#FF6B6B', '#4ECDC4', '#45B7D1'];

export default function ForecastPage() {
  const [portfolioData, setPortfolioData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [forecastPeriod, setForecastPeriod] = useState('1Y');
  const [confidenceLevel, setConfidenceLevel] = useState([80]);
  const [forecastData, setForecastData] = useState({
    monte_carlo: [],
    arima_forecast: [],
    ml_forecast: [],
    stock_forecasts: [],
    risk_metrics: {},
    scenarios: [],
    volatility_forecast: [],
    correlation_matrix: [],
    market_regime: {}
  });

  useEffect(() => {
    fetchForecastData();
  }, [forecastPeriod, confidenceLevel]);

  const fetchForecastData = async () => {
    try {
      setLoading(true);
      
      // Import the forecast action
      const { getForecastData } = await import('@/actions/forecast');
      
      // Get forecast data from Python backend
      const forecasts = await getForecastData(forecastPeriod, confidenceLevel[0] / 100);
      setForecastData(forecasts);

    } catch (error) {
      console.error('Error fetching forecast data:', error);
      toast.error('Failed to load forecast data');
    } finally {
      setLoading(false);
    }
  };



  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric' 
    });
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-4xl font-bold text-gray-900">Forecast</h1>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-gray-600">Running quantitative analysis...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="pt-4 pb-2">
        <div className="flex items-center justify-between">
          <h1 className="text-4xl font-bold text-gray-900">Forecast</h1>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">Confidence Level:</span>
              <Slider
                value={confidenceLevel}
                onValueChange={setConfidenceLevel}
                max={95}
                min={50}
                step={5}
                className="w-24"
              />
              <span className="text-sm font-medium">{confidenceLevel[0]}%</span>
            </div>
            <Select value={forecastPeriod} onValueChange={setForecastPeriod}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="6M">6 Months</SelectItem>
                <SelectItem value="1Y">1 Year</SelectItem>
                <SelectItem value="2Y">2 Years</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="outline" size="sm">
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
          </div>
        </div>
      </div>

      {/* Forecast Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-purple-500" />
              <h3 className="text-sm font-medium">ML Prediction</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-gray-900">
              {formatCurrency(forecastData.ml_forecast?.[forecastData.ml_forecast.length - 1]?.value || 0)}
            </p>
            <p className="text-sm text-gray-500">AI-powered forecast</p>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Cpu className="h-5 w-5 text-blue-500" />
              <h3 className="text-sm font-medium">ARIMA Forecast</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-gray-900">
              {formatCurrency(forecastData.arima_forecast?.[forecastData.arima_forecast.length - 1]?.value || 0)}
            </p>
            <p className="text-sm text-gray-500">Statistical model</p>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Database className="h-5 w-5 text-green-500" />
              <h3 className="text-sm font-medium">VaR (95%)</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-red-600">
              -{formatCurrency(forecastData.risk_metrics?.var_95 || 0)}
            </p>
            <p className="text-sm text-gray-500">Value at Risk</p>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <TargetIcon2 className="h-5 w-5 text-orange-500" />
              <h3 className="text-sm font-medium">Expected Return</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-600">
              +8.5%
            </p>
            <p className="text-sm text-gray-500">Annualized</p>
          </CardContent>
        </Card>
      </div>

      {/* Monte Carlo Simulation */}
      <Card className="bg-white">
        <CardHeader>
          <div className="flex items-center gap-2">
            <BarChart4 className="h-5 w-5 text-indigo-500" />
            <h3 className="text-lg font-medium">Monte Carlo Simulation (1,000 Paths)</h3>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={forecastData.monte_carlo}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={formatDate}
                />
                <YAxis tickFormatter={(value) => formatCurrency(value)} />
                <Tooltip 
                  formatter={(value) => [formatCurrency(value), 'Portfolio Value']}
                  labelFormatter={formatDate}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="confidence_upper"
                  stroke="#10B981"
                  strokeWidth={1}
                  dot={false}
                  strokeDasharray="5 5"
                />
                <Line
                  type="monotone"
                  dataKey="confidence_lower"
                  stroke="#EF4444"
                  strokeWidth={1}
                  dot={false}
                  strokeDasharray="5 5"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* ARIMA vs ML Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">ARIMA Forecast with Confidence Bands</h3>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={forecastData.arima_forecast}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="date" 
                    tickFormatter={formatDate}
                  />
                  <YAxis tickFormatter={(value) => formatCurrency(value)} />
                  <Tooltip 
                    formatter={(value) => [formatCurrency(value), 'Value']}
                    labelFormatter={formatDate}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#3B82F6" 
                    strokeWidth={3}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Machine Learning Prediction</h3>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={forecastData.ml_forecast}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="date" 
                    tickFormatter={formatDate}
                  />
                  <YAxis tickFormatter={(value) => formatCurrency(value)} />
                  <Tooltip 
                    formatter={(value) => [formatCurrency(value), 'Value']}
                    labelFormatter={formatDate}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#8B5CF6" 
                    fill="#8B5CF6" 
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Scenario Analysis and Risk Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Scenario Analysis</h3>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {forecastData.scenarios.map((scenario, index) => (
                <div key={scenario.name} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{scenario.name}</span>
                    <span className={`font-medium ${scenario.return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {(scenario.return * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full" 
                      style={{ width: `${scenario.probability * 100}%` }}
                    ></div>
                  </div>
                  <p className="text-xs text-gray-500">
                    {(scenario.probability * 100).toFixed(0)}% probability
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Risk Metrics Forecast</h3>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Expected Shortfall</span>
                <span className="font-medium text-red-600">
                  -{formatCurrency(forecastData.risk_metrics?.expected_shortfall || 0)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Max Drawdown</span>
                <span className="font-medium text-red-600">
                  -{formatCurrency(forecastData.risk_metrics?.max_drawdown || 0)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Sharpe Ratio</span>
                <span className="font-medium text-green-600">
                  {forecastData.risk_metrics?.sharpe_ratio?.toFixed(2) || '0.00'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Sortino Ratio</span>
                <span className="font-medium text-green-600">
                  {forecastData.risk_metrics?.sortino_ratio?.toFixed(2) || '0.00'}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Market Regime Prediction</h3>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(forecastData.market_regime).map(([regime, probability], index) => (
                <div key={regime} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: COLORS[index % COLORS.length] }}
                    ></div>
                    <div>
                      <p className="font-medium">{regime}</p>
                      <p className="text-xs text-gray-500">{probability.duration}</p>
                    </div>
                  </div>
                  <span className="font-medium">
                    {(probability.probability * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Stock Forecasts and Portfolio Projections */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Individual Stock Forecasts</h3>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {forecastData.stock_forecasts.slice(0, 5).map((forecast, index) => (
                <div key={`forecast-${forecast.ticker}-${index}`} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                      <span className="text-xs font-medium text-blue-600">{forecast.ticker}</span>
                    </div>
                    <div>
                      <p className="font-medium">{forecast.ticker}</p>
                      <p className="text-xs text-gray-500">Forecast Return</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className={`font-medium ${forecast.forecast_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {(forecast.forecast_return * 100).toFixed(1)}%
                    </span>
                    <p className="text-xs text-gray-500">
                      {(forecast.confidence * 100).toFixed(0)}% confidence
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Portfolio Projections (12 Months)</h3>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={forecastData.portfolioProjections}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="month" 
                    tickFormatter={formatDate}
                  />
                  <YAxis yAxisId="left" tickFormatter={(value) => formatCurrency(value)} />
                  <YAxis yAxisId="right" orientation="right" tickFormatter={(value) => `${value.toFixed(1)}%`} />
                  <Tooltip 
                    formatter={(value, name) => [
                      name === 'value' ? formatCurrency(value) : `${value.toFixed(2)}%`,
                      name === 'value' ? 'Portfolio Value' : 'Monthly Return'
                    ]}
                    labelFormatter={formatDate}
                  />
                  <Area 
                    yAxisId="left"
                    type="monotone" 
                    dataKey="value" 
                    stroke="#10B981" 
                    fill="#10B981" 
                    fillOpacity={0.3} 
                  />
                  <Bar 
                    yAxisId="right"
                    dataKey="return" 
                    fill="#3B82F6" 
                    opacity={0.7}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
} 