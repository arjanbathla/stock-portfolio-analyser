import { getStockPrices } from '@/actions/stocks';

export async function calculatePortfolioValue(portfolios) {
  try {
    // Extract all unique tickers from all portfolios
    const allStocks = portfolios.flatMap(portfolio => 
      portfolio.portfolioStocks.map(stock => ({
        ...stock,
        portfolioName: portfolio.name
      }))
    );

    if (allStocks.length === 0) {
      return {
        totalValue: 0,
        totalChange: 0,
        totalChangePercent: 0,
        stockCount: 0
      };
    }

    // Get unique tickers
    const uniqueTickers = [...new Set(allStocks.map(stock => stock.ticker))];

    // Fetch current prices for all tickers using server action
    const priceData = await getStockPrices(uniqueTickers);
    const priceMap = new Map(priceData.map(item => [item.ticker, item]));

    // Calculate portfolio values
    let totalValue = 0;
    let totalPreviousValue = 0;
    let totalChange = 0;

    allStocks.forEach(stock => {
      const priceInfo = priceMap.get(stock.ticker);
      if (priceInfo && priceInfo.currentPrice) {
        const currentValue = stock.shares * priceInfo.currentPrice;
        const previousValue = stock.shares * priceInfo.previousClose;
        
        totalValue += currentValue;
        totalPreviousValue += previousValue;
        totalChange += (currentValue - previousValue);
      }
    });

    const totalChangePercent = totalPreviousValue > 0 ? (totalChange / totalPreviousValue) * 100 : 0;

    return {
      totalValue,
      totalChange,
      totalChangePercent,
      stockCount: allStocks.length
    };
  } catch (error) {
    console.error('Error calculating portfolio value:', error);
    return {
      totalValue: 0,
      totalChange: 0,
      totalChangePercent: 0,
      stockCount: 0
    };
  }
} 