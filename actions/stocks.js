"use server";

import { db } from "@/lib/prisma";
import { auth } from "@clerk/nextjs/server";
import yahooFinance from 'yahoo-finance2';

export async function addStock(portfolioId, stockData) {
  const { userId } = await auth();
  if (!userId) throw new Error("Unauthorized");

  const user = await db.user.findUnique({
    where: { clerkUserId: userId },
  });

  if (!user) {
    throw new Error("User not found");
  }

  // Convert portfolioId to integer
  const portfolioIdInt = parseInt(portfolioId, 10);
  if (isNaN(portfolioIdInt)) {
    throw new Error("Invalid portfolio ID");
  }

  // Verify the portfolio belongs to the user
  const portfolio = await db.portfolio.findFirst({
    where: { id: portfolioIdInt, userId: user.id },
  });

  if (!portfolio) {
    throw new Error("Portfolio not found");
  }

  const stock = await db.portfolioStock.create({
    data: {
      portfolioId: portfolioIdInt,
      ticker: stockData.ticker,
      shares: stockData.shares,
      purchaseDate: stockData.purchaseDate,
      purchasePrice: stockData.purchasePrice,
      reinvestDividends: stockData.reinvestDividends,
      targetWeight: stockData.targetWeight,
    },
  });

  return stock;
}

export async function getStockPrices(tickers) {
  try {
    if (!tickers || tickers.length === 0) {
      return [];
    }

    // Fetch current prices for all tickers
    const pricePromises = tickers.map(async (ticker) => {
      try {
        const quote = await yahooFinance.quote(ticker);
        return {
          ticker,
          currentPrice: quote.regularMarketPrice,
          previousClose: quote.regularMarketPreviousClose,
          change: quote.regularMarketChange,
          changePercent: quote.regularMarketChangePercent
        };
      } catch (error) {
        console.error(`Error fetching price for ${ticker}:`, error);
        return {
          ticker,
          currentPrice: 0,
          previousClose: 0,
          change: 0,
          changePercent: 0
        };
      }
    });

    const priceData = await Promise.all(pricePromises);
    return priceData;
  } catch (error) {
    console.error('Error fetching stock prices:', error);
    return [];
  }
}


