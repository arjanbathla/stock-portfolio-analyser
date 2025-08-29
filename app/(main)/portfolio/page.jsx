"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Calendar } from '@/components/ui/calendar';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid } from 'recharts';
import { getUserPortfolios } from '@/actions/dashboard';
import { getStockPrices } from '@/actions/stocks';
import { addStock } from '@/actions/stocks';
import { formatCurrency } from '@/lib/portfolioHistory';
import { toast } from 'sonner';
import { Loader2, Plus, CalendarIcon, TrendingUp, TrendingDown, DollarSign, Target, Clock, Star, BarChart3, AlertTriangle } from 'lucide-react';
import { format } from 'date-fns';
import { cn } from '@/lib/utils';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FFC658', '#FF6B6B', '#4ECDC4', '#45B7D1'];

export default function PortfolioPage() {
  const [portfolioData, setPortfolioData] = useState([]);
  const [allocationData, setAllocationData] = useState([]);
  const [portfolios, setPortfolios] = useState([]);
  const [loading, setLoading] = useState(true);
  const [addingStock, setAddingStock] = useState(false);
  const [showAddForm, setShowAddForm] = useState(false);
  const [portfolioInsights, setPortfolioInsights] = useState({
    performanceMetrics: {},
    sectorBreakdown: [],
    riskIndicators: {},
    topGainers: [],
    topLosers: [],
    recentTransactions: []
  });
  
  // Form state
  const [formData, setFormData] = useState({
    ticker: '',
    shares: '',
    purchasePrice: '',
    purchaseDate: new Date(),
    portfolioId: ''
  });

  useEffect(() => {
    fetchPortfolioData();
  }, []);

  const fetchPortfolioData = async () => {
    try {
      setLoading(true);
      const portfoliosData = await getUserPortfolios();
      setPortfolios(portfoliosData);
      
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

      // Calculate allocation data for pie chart
      const totalValue = stockPerformance.reduce((sum, stock) => sum + stock.currentValue, 0);
      const allocation = stockPerformance.map((stock, index) => ({
        name: stock.ticker,
        value: stock.currentValue,
        percentage: totalValue > 0 ? (stock.currentValue / totalValue) * 100 : 0,
        color: COLORS[index % COLORS.length]
      }));

      setAllocationData(allocation);

      // Generate portfolio insights
      const insights = generatePortfolioInsights(stockPerformance, totalValue);
      setPortfolioInsights(insights);
    } catch (error) {
      console.error('Error fetching portfolio data:', error);
      toast.error('Failed to load portfolio data');
    } finally {
      setLoading(false);
    }
  };

  const generatePortfolioInsights = (stocks, totalValue) => {
    // Performance metrics
    const totalChange = stocks.reduce((sum, stock) => sum + stock.change, 0);
    const totalChangePercent = totalValue > 0 ? (totalChange / (totalValue - totalChange)) * 100 : 0;
    const avgReturn = stocks.length > 0 ? stocks.reduce((sum, stock) => sum + stock.changePercent, 0) / stocks.length : 0;

    // Top gainers and losers
    const sortedByPerformance = [...stocks].sort((a, b) => b.changePercent - a.changePercent);
    const topGainers = sortedByPerformance.slice(0, 3);
    const topLosers = sortedByPerformance.slice(-3).reverse();

    // Sector breakdown (mock data for now)
    const sectorBreakdown = [
      { sector: 'Technology', percentage: 45, value: totalValue * 0.45 },
      { sector: 'Healthcare', percentage: 25, value: totalValue * 0.25 },
      { sector: 'Finance', percentage: 20, value: totalValue * 0.20 },
      { sector: 'Consumer', percentage: 10, value: totalValue * 0.10 }
    ];

    // Risk indicators
    const volatility = Math.sqrt(stocks.reduce((sum, stock) => sum + Math.pow(stock.changePercent - avgReturn, 2), 0) / stocks.length);
    const beta = 1.2; // Mock beta calculation
    const sharpeRatio = avgReturn / volatility;

    // Recent transactions (mock data)
    const recentTransactions = [
      { action: 'Added', stock: 'AAPL', shares: 10, date: '2 hours ago', value: 1500 },
      { action: 'Added', stock: 'GOOGL', shares: 5, date: '1 day ago', value: 2500 },
      { action: 'Added', stock: 'TSLA', shares: 8, date: '3 days ago', value: 1200 }
    ];

    return {
      performanceMetrics: {
        totalValue,
        totalChange,
        totalChangePercent,
        avgReturn,
        totalStocks: stocks.length
      },
      sectorBreakdown,
      riskIndicators: {
        volatility: volatility.toFixed(2),
        beta: beta.toFixed(2),
        sharpeRatio: sharpeRatio.toFixed(2)
      },
      topGainers,
      topLosers,
      recentTransactions
    };
  };

  const handleAddStock = async (e) => {
    e.preventDefault();
    
    if (!formData.ticker || !formData.shares || !formData.purchasePrice || !formData.portfolioId) {
      toast.error('Please fill in all required fields');
      return;
    }

    try {
      setAddingStock(true);
      
      const stockData = {
        ticker: formData.ticker.toUpperCase(),
        shares: parseFloat(formData.shares),
        purchaseDate: formData.purchaseDate,
        purchasePrice: parseFloat(formData.purchasePrice),
        reinvestDividends: false,
        targetWeight: null
      };

      await addStock(formData.portfolioId, stockData);
      
      toast.success('Stock added successfully!');
      setFormData({
        ticker: '',
        shares: '',
        purchasePrice: '',
        purchaseDate: new Date(),
        portfolioId: ''
      });
      setShowAddForm(false);
      
      // Refresh portfolio data
      await fetchPortfolioData();
    } catch (error) {
      console.error('Error adding stock:', error);
      toast.error('Failed to add stock');
    } finally {
      setAddingStock(false);
    }
  };

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    });
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold text-gray-900">Portfolio</h1>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-gray-600">Loading portfolio data...</p>
        </div>
      </div>
    );
  }

  const totalValue = portfolioData.reduce((sum, stock) => sum + stock.currentValue, 0);
  const totalChange = portfolioData.reduce((sum, stock) => sum + stock.change, 0);
  const totalChangePercent = totalValue > 0 ? (totalChange / (totalValue - totalChange)) * 100 : 0;

  return (
    <div className="space-y-6">
      <div className="pt-4 pb-2">
        <div className="flex items-center justify-between">
          <h1 className="text-4xl font-bold text-gray-900">Portfolio</h1>
          <div className="text-right">
            <p className="text-2xl font-bold text-gray-900">{formatCurrency(totalValue)}</p>
            <p className={`text-sm ${totalChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {totalChange >= 0 ? '+' : ''}{formatCurrency(totalChange)} ({totalChangePercent >= 0 ? '+' : ''}{totalChangePercent.toFixed(2)}%)
            </p>
          </div>
        </div>
      </div>

      {/* Performance Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <DollarSign className="h-5 w-5 text-green-500" />
              <h3 className="text-sm font-medium">Total Value</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-gray-900">{formatCurrency(portfolioInsights.performanceMetrics.totalValue || 0)}</p>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-blue-500" />
              <h3 className="text-sm font-medium">Total Return</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className={`text-2xl font-bold ${totalChangePercent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {totalChangePercent >= 0 ? '+' : ''}{totalChangePercent.toFixed(2)}%
            </p>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-purple-500" />
              <h3 className="text-sm font-medium">Total Stocks</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-gray-900">{portfolioInsights.performanceMetrics.totalStocks || 0}</p>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Target className="h-5 w-5 text-orange-500" />
              <h3 className="text-sm font-medium">Avg Return</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className={`text-2xl font-bold ${(portfolioInsights.performanceMetrics.avgReturn || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {(portfolioInsights.performanceMetrics.avgReturn || 0).toFixed(2)}%
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Stock Performance Table */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium">Stock Performance</h3>
              <Button 
                onClick={() => setShowAddForm(!showAddForm)}
                size="sm"
                className="flex items-center gap-2"
              >
                <Plus size={16} />
                Add Stock
              </Button>
            </div>
          </CardHeader>
          
          {/* Add Stock Form */}
          {showAddForm && (
            <div className="px-6 pb-4 border-b">
              <form onSubmit={handleAddStock} className="grid grid-cols-1 md:grid-cols-5 gap-4">
                <div>
                  <label className="text-sm font-medium text-gray-700">Ticker</label>
                  <Input
                    placeholder="AAPL"
                    value={formData.ticker}
                    onChange={(e) => handleInputChange('ticker', e.target.value)}
                    className="mt-1"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700">Shares</label>
                  <Input
                    type="number"
                    step="0.01"
                    placeholder="10"
                    value={formData.shares}
                    onChange={(e) => handleInputChange('shares', e.target.value)}
                    className="mt-1"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700">Purchase Price</label>
                  <Input
                    type="number"
                    step="0.01"
                    placeholder="150.25"
                    value={formData.purchasePrice}
                    onChange={(e) => handleInputChange('purchasePrice', e.target.value)}
                    className="mt-1"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700">Purchase Date</label>
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        className={cn(
                          "w-full mt-1 justify-start text-left font-normal",
                          !formData.purchaseDate && "text-muted-foreground"
                        )}
                      >
                        <CalendarIcon className="mr-2 h-4 w-4" />
                        {formData.purchaseDate ? format(formData.purchaseDate, "PPP") : "Pick a date"}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0">
                      <Calendar
                        mode="single"
                        selected={formData.purchaseDate}
                        onSelect={(date) => handleInputChange('purchaseDate', date)}
                        disabled={(date) => date > new Date()}
                        initialFocus
                      />
                    </PopoverContent>
                  </Popover>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700">Portfolio</label>
                  <Select 
                    value={formData.portfolioId} 
                    onValueChange={(value) => handleInputChange('portfolioId', value)}
                  >
                    <SelectTrigger className="mt-1">
                      <SelectValue placeholder="Select portfolio" />
                    </SelectTrigger>
                    <SelectContent>
                      {portfolios.map((portfolio) => (
                        <SelectItem key={portfolio.id} value={portfolio.id}>
                          {portfolio.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="md:col-span-5 flex gap-2">
                  <Button 
                    type="submit" 
                    disabled={addingStock}
                    className="flex items-center gap-2"
                  >
                    {addingStock ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Adding...
                      </>
                    ) : (
                      <>
                        <Plus size={16} />
                        Add Stock
                      </>
                    )}
                  </Button>
                  <Button 
                    type="button" 
                    variant="outline"
                    onClick={() => setShowAddForm(false)}
                  >
                    Cancel
                  </Button>
                </div>
              </form>
            </div>
          )}

          <CardContent>
            {portfolioData.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-gray-600">No stocks found in your portfolio.</p>
                <p className="text-sm text-gray-500 mt-2">Add your first stock to get started!</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-2">Stock</th>
                      <th className="text-right py-2 px-2">Shares</th>
                      <th className="text-right py-2 px-2">Price</th>
                      <th className="text-right py-2 px-2">Value</th>
                      <th className="text-right py-2 px-2">Change</th>
                      <th className="text-right py-2 px-2">Purchase Date</th>
                      <th className="text-right py-2 px-2">Portfolio</th>
                    </tr>
                  </thead>
                  <tbody>
                    {portfolioData.map((stock, index) => (
                      <tr key={`${stock.ticker}-${stock.portfolioName}-${index}`} className="border-b hover:bg-gray-50">
                        <td className="py-3 px-2">
                          <div>
                            <p className="font-medium">{stock.ticker}</p>
                            <p className="text-sm text-gray-500">{stock.name}</p>
                          </div>
                        </td>
                        <td className="text-right py-3 px-2">{stock.shares.toLocaleString()}</td>
                        <td className="text-right py-3 px-2">{formatCurrency(stock.currentPrice)}</td>
                        <td className="text-right py-3 px-2 font-medium">{formatCurrency(stock.currentValue)}</td>
                        <td className="text-right py-3 px-2">
                          <div>
                            <p className={`font-medium ${stock.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {stock.change >= 0 ? '+' : ''}{formatCurrency(stock.change)}
                            </p>
                            <p className={`text-sm ${stock.changePercent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {stock.changePercent >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                            </p>
                          </div>
                        </td>
                        <td className="text-right py-3 px-2 text-sm text-gray-600">
                          {formatDate(stock.purchaseDate)}
                        </td>
                        <td className="text-right py-3 px-2">
                          <Badge variant="outline">{stock.portfolioName}</Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Allocation Pie Chart */}
        <Card>
          <CardHeader>
            <h3 className="text-lg font-medium">Portfolio Allocation</h3>
          </CardHeader>
          <CardContent>
            {allocationData.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-gray-600">No allocation data available</p>
              </div>
            ) : (
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={allocationData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percentage }) => `${name} ${percentage.toFixed(1)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {allocationData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      formatter={(value) => [formatCurrency(value), 'Value']}
                      labelFormatter={(label) => `${label} (${allocationData.find(item => item.name === label)?.percentage.toFixed(1)}%)`}
                    />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Additional Portfolio Insights */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-6">
        {/* Top Gainers */}
        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-green-500" />
              <h3 className="text-lg font-medium">Top Gainers</h3>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {portfolioInsights.topGainers.map((stock, index) => (
                <div key={`gainer-${stock.ticker}-${index}`} className="flex items-center justify-between">
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

        {/* Top Losers */}
        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <TrendingDown className="h-5 w-5 text-red-500" />
              <h3 className="text-lg font-medium">Top Losers</h3>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {portfolioInsights.topLosers.map((stock, index) => (
                <div key={`loser-${stock.ticker}-${index}`} className="flex items-center justify-between">
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

        {/* Risk Indicators */}
        <Card className="bg-white">
          <CardHeader>
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-orange-500" />
              <h3 className="text-lg font-medium">Risk Indicators</h3>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Volatility</span>
                <span className="font-medium">{portfolioInsights.riskIndicators.volatility}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Beta</span>
                <span className="font-medium">{portfolioInsights.riskIndicators.beta}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Sharpe Ratio</span>
                <span className="font-medium">{portfolioInsights.riskIndicators.sharpeRatio}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Sector Analysis and Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sector Breakdown */}
        <Card className="bg-white">
          <CardHeader>
            <h3 className="text-lg font-medium">Sector Breakdown</h3>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={portfolioInsights.sectorBreakdown}>
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
              {portfolioInsights.recentTransactions.map((transaction, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                      <Plus className="h-4 w-4 text-green-600" />
                    </div>
                    <div>
                      <p className="font-medium">{transaction.action} {transaction.stock}</p>
                      <p className="text-sm text-gray-500">{transaction.shares} shares</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-500">{transaction.date}</p>
                    <p className="text-sm font-medium">{formatCurrency(transaction.value)}</p>
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