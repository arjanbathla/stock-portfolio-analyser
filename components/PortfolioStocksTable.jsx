"use client";

import React, { useEffect, useState } from 'react';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { getUserPortfolios } from '@/actions/dashboard';
import { calculatePortfolioValue } from '@/lib/portfolioCalculator';

export default function PortfolioStocksTable({ onPortfolioValueUpdate }) {
  const [portfolios, setPortfolios] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPortfolios = async () => {
      try {
        const data = await getUserPortfolios();
        setPortfolios(data);
        
        // Calculate portfolio value and notify parent component
        if (data.length > 0) {
          const portfolioValue = await calculatePortfolioValue(data);
          if (onPortfolioValueUpdate) {
            onPortfolioValueUpdate(portfolioValue, data);
          }
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchPortfolios();
  }, [onPortfolioValueUpdate]);

  if (loading) {
    return (
      <Card className="bg-white lg:col-span-1">
        <CardHeader>
          <h3 className="text-lg font-medium">Portfolio Stocks</h3>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64">
            <p className="text-gray-600">Loading stocks...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="bg-white lg:col-span-1">
        <CardHeader>
          <h3 className="text-lg font-medium">Portfolio Stocks</h3>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64">
            <p className="text-red-600">Error: {error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Flatten all stocks from all portfolios
  const allStocks = portfolios.flatMap(portfolio => 
    portfolio.portfolioStocks.map(stock => ({
      ...stock,
      portfolioName: portfolio.name
    }))
  );

  if (allStocks.length === 0) {
    return (
      <Card className="bg-white lg:col-span-1">
        <CardHeader>
          <h3 className="text-lg font-medium">Portfolio Stocks</h3>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64">
            <p className="text-gray-600">No stocks found in your portfolio</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-white lg:col-span-1">
      <CardHeader>
        <h3 className="text-lg font-medium">Portfolio Stocks</h3>
      </CardHeader>
      <CardContent>
        <div className="max-h-64 overflow-y-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-xs">Ticker</TableHead>
                <TableHead className="text-xs">Shares</TableHead>
                <TableHead className="text-xs">Purchase Price</TableHead>
                <TableHead className="text-xs">Portfolio</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {allStocks.map((stock) => (
                <TableRow key={stock.id}>
                  <TableCell className="text-xs font-medium">{stock.ticker}</TableCell>
                  <TableCell className="text-xs">{stock.shares}</TableCell>
                  <TableCell className="text-xs">
                    {stock.purchasePrice ? `$${stock.purchasePrice.toFixed(2)}` : 'N/A'}
                  </TableCell>
                  <TableCell className="text-xs">{stock.portfolioName}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
} 