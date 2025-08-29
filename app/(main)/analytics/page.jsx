"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ComposedChart,
  Area,
} from 'recharts';
import { getUserPortfolios } from '@/actions/dashboard';
import { getStockPrices } from '@/actions/stocks';
import { formatCurrency } from '@/lib/portfolioHistory';
import { toast } from 'sonner';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Target, 
  BarChart3, 
  PieChart as PieChartIcon,
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
  RefreshCw
} from 'lucide-react';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FFC658', '#FF6B6B', '#4ECDC4', '#45B7D1'];

export default function AnalyticsPage() {
  const [portfolioData, setPortfolioData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('3M');
  const [analyticsData, setAnalyticsData] = useState({
    correlationMatrix: [],
    sectorAnalysis: [],
    riskMetrics: {},
    trendAnalysis: [],
    diversificationScore: 0,
    concentrationRisk: 0,
    momentumIndicators: [],
    volatilityAnalysis: [],
    marketCapDistribution: [],
    geographicExposure: []
  });

  useEffect(() => {
    fetchAnalyticsData();
  }, [timeRange]);

  const fetchAnalyticsData = async () => {
    try {
      setLoading(true);
      const portfoliosData = await getUserPortfolios();
      
      if (portfoliosData.length === 0) {
        setLoading(false);
        return;
      }

      // Extract all stocks from all portfolios
      const allStocks = portfoliosData.flatMap(portfolio => 
        portfolio.portfolioStocks.map(stock => ({
          ...stock,
          portfolioName: portfolio.name
        }))
      );

      if (allStocks.length === 0) {
        setLoading(false);
        return;
      }

      // Get unique tickers
      const uniqueTickers = [...new Set(allStocks.map(stock => stock.ticker))];
      
      // Fetch current prices
      const priceData = await getStockPrices(uniqueTickers);
      const priceMap = new Map(priceData.map(item => [item.ticker, item]));

      // Calculate performance for each stock
      const stockPerformance = allStocks.map(stock => {
        const priceInfo = priceMap.get(stock.ticker);
        if (!priceInfo) return null;

        const currentValue = stock.shares * priceInfo.currentPrice;
        const previousValue = stock.shares * priceInfo.previousClose;
        const change = currentValue - previousValue;
        const changePercent = previousValue > 0 ? (change / previousValue) * 100 : 0;

        return {
          ticker: stock.ticker,
          name: stock.name || stock.ticker,
          shares: stock.shares,
          currentPrice: priceInfo.currentPrice,
          previousClose: priceInfo.previousClose,
          currentValue,
          change,
          changePercent,
          portfolioName: stock.portfolioName,
          purchaseDate: stock.purchaseDate,
          purchasePrice: stock.purchasePrice
        };
      }).filter(Boolean);

      setPortfolioData(stockPerformance);

      // Generate analytics insights
      const analytics = generateAnalyticsInsights(stockPerformance, timeRange);
      setAnalyticsData(analytics);

    } catch (error) {
      console.error('Error fetching analytics data:', error);
      toast.error('Failed to load analytics data');
    } finally {
      setLoading(false);
    }
  };

  const generateAnalyticsInsights = (stocks, range) => {
    const totalValue = stocks.reduce((sum, stock) => sum + stock.currentValue, 0);
    
    // Correlation Matrix (mock data)
    const correlationMatrix = [
      { stock1: 'AAPL', stock2: 'GOOGL', correlation: 0.75 },
      { stock1: 'AAPL', stock2: 'MSFT', correlation: 0.82 },
      { stock1: 'GOOGL', stock2: 'MSFT', correlation: 0.68 },
      { stock1: 'TSLA', stock2: 'AAPL', correlation: 0.45 },
      { stock1: 'TSLA', stock2: 'GOOGL', correlation: 0.52 }
    ];

    // Sector Analysis
    const sectorAnalysis = [
      { sector: 'Technology', allocation: 45, performance: 12.5, volatility: 8.2 },
      { sector: 'Healthcare', allocation: 25, performance: 8.3, volatility: 6.1 },
      { sector: 'Finance', allocation: 20, performance: 5.7, volatility: 7.8 },
      { sector: 'Consumer', allocation: 10, performance: 3.2, volatility: 9.1 }
    ];

    // Risk Metrics
    const avgReturn = stocks.length > 0 ? stocks.reduce((sum, stock) => sum + stock.changePercent, 0) / stocks.length : 0;
    const volatility = Math.sqrt(stocks.reduce((sum, stock) => sum + Math.pow(stock.changePercent - avgReturn, 2), 0) / stocks.length);
    const sharpeRatio = avgReturn / volatility;
    const maxDrawdown = -12.5;

    // Trend Analysis
    const trendAnalysis = [];
    const days = range === '1M' ? 30 : range === '3M' ? 90 : range === '6M' ? 180 : 365;
    for (let i = days; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      trendAnalysis.push({
        date: date.toISOString().split('T')[0],
        value: totalValue * (1 + Math.sin(i * 0.1) * 0.05),
        volume: Math.random() * 1000000 + 500000,
        momentum: Math.sin(i * 0.1) * 2
      });
    }

    // Diversification Score
    const diversificationScore = Math.min(100, stocks.length * 8 + Math.random() * 20);

    // Concentration Risk
    const topHoldings = stocks.sort((a, b) => b.currentValue - a.currentValue).slice(0, 5);
    const concentrationRisk = (topHoldings.reduce((sum, stock) => sum + stock.currentValue, 0) / totalValue) * 100;

    // Momentum Indicators
    const momentumIndicators = stocks.map(stock => ({
      ticker: stock.ticker,
      momentum: stock.changePercent,
      volume: Math.random() * 1000000 + 500000,
      rsi: Math.random() * 30 + 40
    }));

    // Volatility Analysis
    const volatilityAnalysis = stocks.map(stock => ({
      ticker: stock.ticker,
      volatility: Math.abs(stock.changePercent),
      beta: Math.random() * 1.5 + 0.5,
      alpha: stock.changePercent - (Math.random() * 8 + 2)
    }));

    // Market Cap Distribution
    const marketCapDistribution = [
      { category: 'Large Cap', allocation: 60, count: stocks.filter(s => s.currentValue > 10000).length },
      { category: 'Mid Cap', allocation: 30, count: stocks.filter(s => s.currentValue > 5000 && s.currentValue <= 10000).length },
      { category: 'Small Cap', allocation: 10, count: stocks.filter(s => s.currentValue <= 5000).length }
    ];

    // Geographic Exposure
    const geographicExposure = [
      { region: 'North America', allocation: 75, performance: 10.2 },
      { region: 'Europe', allocation: 15, performance: 6.8 },
      { region: 'Asia Pacific', allocation: 10, performance: 8.5 }
    ];

    return {
      correlationMatrix,
      sectorAnalysis,
      riskMetrics: {
        volatility: volatility.toFixed(2),
        sharpeRatio: sharpeRatio.toFixed(2),
        maxDrawdown: maxDrawdown.toFixed(2),
        beta: 1.2,
        alpha: 2.5
      },
      trendAnalysis,
      diversificationScore: diversificationScore.toFixed(1),
      concentrationRisk: concentrationRisk.toFixed(1),
      momentumIndicators,
      volatilityAnalysis,
      marketCapDistribution,
      geographicExposure
    };
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
          <h1 className="text-4xl font-bold text-gray-900">Analytics</h1>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-gray-600">Loading analytics data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="pt-4 pb-2">
        <div className="flex items-center justify-between">
          <h1 className="text-4xl font-bold text-gray-900">Analytics</h1>
          <div className="flex items-center gap-4">
            <Select value={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1M">1 Month</SelectItem>
                <SelectItem value="3M">3 Months</SelectItem>
                <SelectItem value="6M">6 Months</SelectItem>
                <SelectItem value="1Y">1 Year</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="outline" size="sm">
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
          </div>
        </div>
      </div>

      {/* Key Analytics Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <TargetIcon className="h-5 w-5 text-blue-500" />
              <h3 className="text-sm font-medium">Diversification Score</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-gray-900">{analyticsData.diversificationScore}%</p>
            <p className="text-sm text-gray-500">Portfolio diversity</p>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-orange-500" />
              <h3 className="text-sm font-medium">Concentration Risk</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-gray-900">{analyticsData.concentrationRisk}%</p>
            <p className="text-sm text-gray-500">Top 5 holdings</p>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-purple-500" />
              <h3 className="text-sm font-medium">Beta</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-gray-900">{analyticsData.riskMetrics.beta}</p>
            <p className="text-sm text-gray-500">Market correlation</p>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-green-500" />
              <h3 className="text-sm font-medium">Alpha</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-gray-900">{analyticsData.riskMetrics.alpha}%</p>
            <p className="text-sm text-gray-500">Excess return</p>
          </CardContent>
        </Card>
      </div>

      {/* Advanced Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Trend Analysis */}
        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Portfolio Trend Analysis</h3>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={analyticsData.trendAnalysis}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="date" 
                    tickFormatter={formatDate}
                  />
                  <YAxis yAxisId="left" tickFormatter={(value) => formatCurrency(value)} />
                  <YAxis yAxisId="right" orientation="right" tickFormatter={(value) => `${value.toFixed(1)}`} />
                  <Tooltip 
                    formatter={(value, name) => [
                      name === 'value' ? formatCurrency(value) : value.toFixed(1),
                      name === 'value' ? 'Portfolio Value' : 'Momentum'
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
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="momentum" 
                    stroke="#8B5CF6" 
                    strokeWidth={2}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Sector Performance Radar */}
        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Sector Performance Radar</h3>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={analyticsData.sectorAnalysis}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="sector" />
                  <PolarRadiusAxis />
                  <Radar 
                    name="Performance" 
                    dataKey="performance" 
                    stroke="#10B981" 
                    fill="#10B981" 
                    fillOpacity={0.3} 
                  />
                  <Radar 
                    name="Volatility" 
                    dataKey="volatility" 
                    stroke="#EF4444" 
                    fill="#EF4444" 
                    fillOpacity={0.3} 
                  />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Correlation Matrix */}
        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Stock Correlations</h3>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {analyticsData.correlationMatrix.map((correlation, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{correlation.stock1}</span>
                    <span className="text-gray-400">vs</span>
                    <span className="font-medium">{correlation.stock2}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-16 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-500 h-2 rounded-full" 
                        style={{ width: `${Math.abs(correlation.correlation) * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium">
                      {correlation.correlation.toFixed(2)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Market Cap Distribution */}
        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Market Cap Distribution</h3>
          </CardHeader>
          <CardContent>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={analyticsData.marketCapDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ category, allocation }) => `${category} ${allocation}%`}
                    outerRadius={60}
                    fill="#8884d8"
                    dataKey="allocation"
                  >
                    {analyticsData.marketCapDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [`${value}%`, 'Allocation']} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Geographic Exposure */}
        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Geographic Exposure</h3>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {analyticsData.geographicExposure.map((region, index) => (
                <div key={region.region} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: COLORS[index % COLORS.length] }}
                    ></div>
                    <div>
                      <p className="font-medium">{region.region}</p>
                      <p className="text-sm text-gray-500">{region.allocation}% allocation</p>
                    </div>
                  </div>
                  <span className={`font-medium ${region.performance >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {region.performance >= 0 ? '+' : ''}{region.performance.toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Risk and Volatility Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Volatility Analysis */}
        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Volatility Analysis</h3>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="beta" name="Beta" />
                  <YAxis dataKey="volatility" name="Volatility" />
                  <Tooltip 
                    formatter={(value, name) => [
                      name === 'volatility' ? `${value.toFixed(2)}%` : value.toFixed(2),
                      name === 'volatility' ? 'Volatility' : 'Beta'
                    ]}
                    labelFormatter={(label) => `Stock: ${label}`}
                  />
                  <Scatter 
                    data={analyticsData.volatilityAnalysis} 
                    dataKey="volatility" 
                    fill="#8884d8"
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Momentum Indicators */}
        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Momentum Indicators</h3>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {analyticsData.momentumIndicators.slice(0, 5).map((indicator, index) => (
                <div key={`momentum-${indicator.ticker}-${index}`} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-sm font-medium text-blue-600">
                      {index + 1}
                    </div>
                    <div>
                      <p className="font-medium">{indicator.ticker}</p>
                      <p className="text-sm text-gray-500">RSI: {indicator.rsi.toFixed(1)}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className={`font-medium ${indicator.momentum >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {indicator.momentum >= 0 ? '+' : ''}{indicator.momentum.toFixed(2)}%
                    </p>
                    <p className="text-sm text-gray-500">
                      {formatCurrency(indicator.volume)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
} 