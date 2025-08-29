"use client";

import React, { useState, useCallback, useMemo } from 'react';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarImage } from '@/components/ui/avatar';
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
} from 'recharts';
import PortfolioStocksTable from '@/components/PortfolioStocksTable';
import { generatePortfolioHistory, formatCurrency } from '@/lib/portfolioHistory';
import { TrendingUp, TrendingDown, DollarSign, Target, Clock, Star } from 'lucide-react';

// Initial metrics state
const initialMetrics = [
  { title: 'Total Value', value: '$0' },
  { title: '1D Change', value: '$0 (0.00%)' },
  { title: 'Number of Stocks', value: '0' },
];

// Initial chart data state
const initialChartData = [];

export default function DashboardPage() {
  // Portfolio metrics state
  const [metrics, setMetrics] = useState(initialMetrics);
  const [chartData, setChartData] = useState(initialChartData);
  const [portfolioInsights, setPortfolioInsights] = useState({
    topPerformers: [],
    recentActivity: [],
    allocationBreakdown: [],
    riskMetrics: {
      volatility: 0,
      beta: 0,
      sharpeRatio: 0
    }
  });

  const handlePortfolioValueUpdate = useCallback(async (portfolioValue, portfolios) => {
    const formattedValue = new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(portfolioValue.totalValue);

    const changeSign = portfolioValue.totalChange >= 0 ? '+' : '';
    const formattedChange = new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(Math.abs(portfolioValue.totalChange));

    const formattedChangePercent = portfolioValue.totalChangePercent.toFixed(2);
    const changeColor = portfolioValue.totalChange >= 0 ? 'text-green-600' : 'text-red-600';

    // Calculate additional metrics
    const totalStocks = portfolioValue.stockCount;
    const avgReturn = totalStocks > 0 ? portfolioValue.totalChangePercent / totalStocks : 0;
    const formattedAvgReturn = avgReturn.toFixed(2);

    setMetrics([
      { title: 'Total Value', value: formattedValue },
      { 
        title: '1D Change', 
        value: `${changeSign}${formattedChange} (${changeSign}${formattedChangePercent}%)`,
        className: changeColor
      },
      { title: 'Number of Stocks', value: totalStocks.toString() },
      { 
        title: 'Avg Return', 
        value: `${changeSign}${formattedAvgReturn}%`,
        className: changeColor
      },
    ]);

    // Generate chart data for portfolio history
    if (portfolios && portfolios.length > 0) {
      const history = await generatePortfolioHistory(portfolios, 30);
      setChartData(history);

      // Generate portfolio insights
      const insights = generatePortfolioInsights(portfolios, portfolioValue);
      setPortfolioInsights(insights);
    }
  }, []);

  const generatePortfolioInsights = (portfolios, portfolioValue) => {
    // Extract all stocks
    const allStocks = portfolios.flatMap(portfolio => 
      portfolio.portfolioStocks.map(stock => ({
        ...stock,
        portfolioName: portfolio.name
      }))
    );

    // Calculate top performers (mock data for now)
    const topPerformers = allStocks.slice(0, 3).map((stock, index) => ({
      ticker: stock.ticker,
      name: stock.name || stock.ticker,
      change: Math.random() * 20 - 10, // Mock performance
      changePercent: Math.random() * 15 - 7.5,
      value: stock.shares * 150 // Mock current value
    }));

    // Recent activity (mock data)
    const recentActivity = [
      { action: 'Added', stock: 'AAPL', shares: 10, date: '2 hours ago' },
      { action: 'Added', stock: 'GOOGL', shares: 5, date: '1 day ago' },
      { action: 'Added', stock: 'TSLA', shares: 8, date: '3 days ago' }
    ];

    // Allocation breakdown
    const allocationBreakdown = [
      { sector: 'Technology', percentage: 45, color: '#3B82F6' },
      { sector: 'Healthcare', percentage: 25, color: '#10B981' },
      { sector: 'Finance', percentage: 20, color: '#F59E0B' },
      { sector: 'Consumer', percentage: 10, color: '#EF4444' }
    ];

    return {
      topPerformers,
      recentActivity,
      allocationBreakdown,
      riskMetrics: {
        volatility: 12.5,
        beta: 1.2,
        sharpeRatio: 0.85
      }
    };
  };

  return (
    <div className="min-h-screen bg-gray-50 text-black p-6">
      {/* Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {metrics.map((m) => (
          <Card key={m.title} className="bg-white">
            <CardHeader>
              <h3 className="text-sm text-gray-800">{m.title}</h3>
            </CardHeader>
            <CardContent>
              <p className={`text-3xl font-semibold ${m.className || 'text-green-600'}`}>{m.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Main Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Line Chart - spans 2/3 width */}
        <Card className="bg-white lg:col-span-2">
          <CardHeader>
            <h3 className="text-lg font-medium">Portfolio Value</h3>
          </CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="date" 
                  stroke="#9ca3af" 
                  tickFormatter={(value) => {
                    const date = new Date(value);
                    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                  }}
                />
                <YAxis 
                  stroke="#9ca3af" 
                  tickFormatter={(value) => formatCurrency(value)}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
                  formatter={(value) => [formatCurrency(value), 'Portfolio Value']}
                  labelFormatter={(label) => {
                    const date = new Date(label);
                    return date.toLocaleDateString('en-US', { 
                      weekday: 'long', 
                      year: 'numeric', 
                      month: 'long', 
                      day: 'numeric' 
                    });
                  }}
                />
                <Line type="monotone" dataKey="value" stroke="#10b981" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Portfolio Stocks Table - spans 1/3 width */}
        <PortfolioStocksTable onPortfolioValueUpdate={handlePortfolioValueUpdate} />
      </div>

      {/* Additional Dashboard Boxes */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-6">
        {/* Top Performers */}
        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Star className="h-5 w-5 text-yellow-500" />
              <h3 className="text-lg font-medium">Top Performers</h3>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {portfolioInsights.topPerformers.map((stock, index) => (
                <div key={stock.ticker} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center text-sm font-medium">
                      {index + 1}
                    </div>
                    <div>
                      <p className="font-medium">{stock.ticker}</p>
                      <p className="text-sm text-gray-500">{stock.name}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className={`font-medium ${stock.changePercent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {stock.changePercent >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                    </p>
                    <p className="text-sm text-gray-500">{formatCurrency(stock.value)}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Clock className="h-5 w-5 text-blue-500" />
              <h3 className="text-lg font-medium">Recent Activity</h3>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {portfolioInsights.recentActivity.map((activity, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                      <TrendingUp className="h-4 w-4 text-green-600" />
                    </div>
                    <div>
                      <p className="font-medium">{activity.action} {activity.stock}</p>
                      <p className="text-sm text-gray-500">{activity.shares} shares</p>
                    </div>
                  </div>
                  <p className="text-sm text-gray-500">{activity.date}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Risk Metrics */}
        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Target className="h-5 w-5 text-purple-500" />
              <h3 className="text-lg font-medium">Risk Metrics</h3>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Volatility</span>
                <span className="font-medium">{portfolioInsights.riskMetrics.volatility}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Beta</span>
                <span className="font-medium">{portfolioInsights.riskMetrics.beta}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Sharpe Ratio</span>
                <span className="font-medium">{portfolioInsights.riskMetrics.sharpeRatio}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Sector Allocation Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Sector Allocation</h3>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={portfolioInsights.allocationBreakdown}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="sector" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${value}%`, 'Allocation']} />
                  <Bar dataKey="percentage" fill="#3B82F6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Quick Actions</h3>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <button className="w-full p-3 text-left border rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                    <TrendingUp className="h-5 w-5 text-blue-600" />
                  </div>
                  <div>
                    <p className="font-medium">Add New Stock</p>
                    <p className="text-sm text-gray-500">Purchase additional shares</p>
                  </div>
                </div>
              </button>
              <button className="w-full p-3 text-left border rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                    <DollarSign className="h-5 w-5 text-green-600" />
                  </div>
                  <div>
                    <p className="font-medium">View Portfolio</p>
                    <p className="text-sm text-gray-500">Detailed portfolio analysis</p>
                  </div>
                </div>
              </button>
              <button className="w-full p-3 text-left border rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                    <Target className="h-5 w-5 text-purple-600" />
                  </div>
                  <div>
                    <p className="font-medium">Performance</p>
                    <p className="text-sm text-gray-500">Track your returns</p>
                  </div>
                </div>
              </button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

