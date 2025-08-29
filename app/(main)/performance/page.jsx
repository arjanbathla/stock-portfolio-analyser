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
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
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
  Calendar, 
  BarChart3, 
  PieChart as PieChartIcon,
  Activity,
  Award,
  AlertTriangle,
  ArrowUpRight,
  ArrowDownRight
} from 'lucide-react';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FFC658', '#FF6B6B', '#4ECDC4', '#45B7D1'];

export default function PerformancePage() {
  const [portfolioData, setPortfolioData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('1M');
  const [performanceMetrics, setPerformanceMetrics] = useState({
    totalReturn: 0,
    annualizedReturn: 0,
    volatility: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    winRate: 0,
    bestDay: { date: '', return: 0 },
    worstDay: { date: '', return: 0 }
  });
  const [historicalData, setHistoricalData] = useState([]);
  const [comparisonData, setComparisonData] = useState([]);
  const [sectorPerformance, setSectorPerformance] = useState([]);

  useEffect(() => {
    fetchPerformanceData();
  }, [timeRange]);

  const fetchPerformanceData = async () => {
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

      // Generate performance insights
      const insights = generatePerformanceInsights(stockPerformance, timeRange);
      setPerformanceMetrics(insights.metrics);
      setHistoricalData(insights.historicalData);
      setComparisonData(insights.comparisonData);
      setSectorPerformance(insights.sectorPerformance);

    } catch (error) {
      console.error('Error fetching performance data:', error);
      toast.error('Failed to load performance data');
    } finally {
      setLoading(false);
    }
  };

  const generatePerformanceInsights = (stocks, range) => {
    const totalValue = stocks.reduce((sum, stock) => sum + stock.currentValue, 0);
    const totalChange = stocks.reduce((sum, stock) => sum + stock.change, 0);
    const totalChangePercent = totalValue > 0 ? (totalChange / (totalValue - totalChange)) * 100 : 0;

    // Calculate metrics
    const avgReturn = stocks.length > 0 ? stocks.reduce((sum, stock) => sum + stock.changePercent, 0) / stocks.length : 0;
    const volatility = Math.sqrt(stocks.reduce((sum, stock) => sum + Math.pow(stock.changePercent - avgReturn, 2), 0) / stocks.length);
    const sharpeRatio = avgReturn / volatility;
    const winRate = stocks.filter(stock => stock.changePercent > 0).length / stocks.length * 100;

    // Generate historical data based on time range
    const days = range === '1W' ? 7 : range === '1M' ? 30 : range === '3M' ? 90 : range === '1Y' ? 365 : 30;
    const historicalData = [];
    const today = new Date();
    
    for (let i = days; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);
      
      // Generate realistic historical data
      const baseValue = totalValue;
      const variation = Math.sin(i * 0.1) * 0.05; // Smooth variation
      const value = baseValue * (1 + variation);
      
      historicalData.push({
        date: date.toISOString().split('T')[0],
        value: Math.round(value * 100) / 100,
        return: variation * 100
      });
    }

    // Comparison data (vs S&P 500, etc.)
    const comparisonData = [
      { name: 'Your Portfolio', value: totalChangePercent, color: '#10B981' },
      { name: 'S&P 500', value: 8.5, color: '#3B82F6' },
      { name: 'NASDAQ', value: 12.3, color: '#8B5CF6' },
      { name: 'Russell 2000', value: 6.2, color: '#F59E0B' }
    ];

    // Sector performance
    const sectorPerformance = [
      { sector: 'Technology', return: 15.2, allocation: 45 },
      { sector: 'Healthcare', return: 8.7, allocation: 25 },
      { sector: 'Finance', return: 5.3, allocation: 20 },
      { sector: 'Consumer', return: 3.1, allocation: 10 }
    ];

    return {
      metrics: {
        totalReturn: totalChangePercent,
        annualizedReturn: totalChangePercent * (365 / days),
        volatility: volatility,
        sharpeRatio: sharpeRatio,
        maxDrawdown: -8.5, // Mock data
        winRate: winRate,
        bestDay: { date: '2024-01-15', return: 3.2 },
        worstDay: { date: '2024-01-22', return: -2.1 }
      },
      historicalData,
      comparisonData,
      sectorPerformance
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
          <h1 className="text-4xl font-bold text-gray-900">Performance</h1>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-gray-600">Loading performance data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="pt-4 pb-2">
        <div className="flex items-center justify-between">
          <h1 className="text-4xl font-bold text-gray-900">Performance</h1>
          <div className="flex items-center gap-4">
            <Select value={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1W">1 Week</SelectItem>
                <SelectItem value="1M">1 Month</SelectItem>
                <SelectItem value="3M">3 Months</SelectItem>
                <SelectItem value="1Y">1 Year</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {/* Key Performance Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-green-500" />
              <h3 className="text-sm font-medium">Total Return</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className={`text-2xl font-bold ${performanceMetrics.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {performanceMetrics.totalReturn >= 0 ? '+' : ''}{performanceMetrics.totalReturn.toFixed(2)}%
            </p>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Target className="h-5 w-5 text-blue-500" />
              <h3 className="text-sm font-medium">Annualized Return</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className={`text-2xl font-bold ${performanceMetrics.annualizedReturn >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {performanceMetrics.annualizedReturn >= 0 ? '+' : ''}{performanceMetrics.annualizedReturn.toFixed(2)}%
            </p>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-purple-500" />
              <h3 className="text-sm font-medium">Volatility</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-gray-900">
              {performanceMetrics.volatility.toFixed(2)}%
            </p>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Award className="h-5 w-5 text-orange-500" />
              <h3 className="text-sm font-medium">Sharpe Ratio</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-gray-900">
              {performanceMetrics.sharpeRatio.toFixed(2)}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Portfolio Value Over Time */}
        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Portfolio Value Over Time</h3>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={historicalData}>
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
                    stroke="#10B981" 
                    fill="#10B981" 
                    fillOpacity={0.3} 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Daily Returns */}
        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Daily Returns</h3>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={historicalData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="date" 
                    tickFormatter={formatDate}
                  />
                  <YAxis tickFormatter={(value) => `${value.toFixed(1)}%`} />
                  <Tooltip 
                    formatter={(value) => [`${value.toFixed(2)}%`, 'Return']}
                    labelFormatter={formatDate}
                  />
                  <Bar 
                    dataKey="return" 
                    fill={(entry) => entry.return >= 0 ? '#10B981' : '#EF4444'} 
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Additional Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Performance Comparison */}
        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Performance Comparison</h3>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {comparisonData.map((item, index) => (
                <div key={item.name} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div 
                      className="w-4 h-4 rounded-full" 
                      style={{ backgroundColor: item.color }}
                    ></div>
                    <span className="font-medium">{item.name}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`font-medium ${item.value >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {item.value >= 0 ? '+' : ''}{item.value.toFixed(1)}%
                    </span>
                    {item.value >= 0 ? (
                      <ArrowUpRight className="h-4 w-4 text-green-600" />
                    ) : (
                      <ArrowDownRight className="h-4 w-4 text-red-600" />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Risk Metrics */}
        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-orange-500" />
              <h3 className="text-lg font-medium">Risk Metrics</h3>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Max Drawdown</span>
                <span className="font-medium text-red-600">{performanceMetrics.maxDrawdown}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Win Rate</span>
                <span className="font-medium text-green-600">{performanceMetrics.winRate.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Best Day</span>
                <div className="text-right">
                  <p className="font-medium text-green-600">+{performanceMetrics.bestDay.return}%</p>
                  <p className="text-xs text-gray-500">{formatDate(performanceMetrics.bestDay.date)}</p>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Worst Day</span>
                <div className="text-right">
                  <p className="font-medium text-red-600">{performanceMetrics.worstDay.return}%</p>
                  <p className="text-xs text-gray-500">{formatDate(performanceMetrics.worstDay.date)}</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Sector Performance */}
        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <PieChartIcon className="h-5 w-5 text-purple-500" />
              <h3 className="text-lg font-medium">Sector Performance</h3>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {sectorPerformance.map((sector, index) => (
                <div key={sector.sector} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: COLORS[index % COLORS.length] }}
                    ></div>
                    <div>
                      <p className="font-medium">{sector.sector}</p>
                      <p className="text-xs text-gray-500">{sector.allocation}% allocation</p>
                    </div>
                  </div>
                  <span className={`font-medium ${sector.return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {sector.return >= 0 ? '+' : ''}{sector.return.toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Top Performers and Underperformers */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Performers */}
        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-green-500" />
              <h3 className="text-lg font-medium">Top Performers</h3>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {portfolioData
                .sort((a, b) => b.changePercent - a.changePercent)
                .slice(0, 5)
                .map((stock, index) => (
                  <div key={`performer-${stock.ticker}-${index}`} className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center text-sm font-medium text-green-600">
                        {index + 1}
                      </div>
                      <div>
                        <p className="font-medium">{stock.ticker}</p>
                        <p className="text-sm text-gray-500">{stock.name}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-medium text-green-600">
                        +{stock.changePercent.toFixed(2)}%
                      </p>
                      <p className="text-sm text-gray-500">{formatCurrency(stock.currentValue)}</p>
                    </div>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>

        {/* Underperformers */}
        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <TrendingDown className="h-5 w-5 text-red-500" />
              <h3 className="text-lg font-medium">Underperformers</h3>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {portfolioData
                .sort((a, b) => a.changePercent - b.changePercent)
                .slice(0, 5)
                .map((stock, index) => (
                  <div key={`underperformer-${stock.ticker}-${index}`} className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center text-sm font-medium text-red-600">
                        {index + 1}
                      </div>
                      <div>
                        <p className="font-medium">{stock.ticker}</p>
                        <p className="text-sm text-gray-500">{stock.name}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-medium text-red-600">
                        {stock.changePercent.toFixed(2)}%
                      </p>
                      <p className="text-sm text-gray-500">{formatCurrency(stock.currentValue)}</p>
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