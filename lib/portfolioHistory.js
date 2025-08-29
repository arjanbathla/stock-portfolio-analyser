import { calculatePortfolioValue } from './portfolioCalculator';

export async function generatePortfolioHistory(portfolios, days = 30) {
  try {
    // Get current portfolio value
    const currentValue = await calculatePortfolioValue(portfolios);
    
    if (currentValue.totalValue === 0) {
      // If no portfolio data, return empty array
      return [];
    }

    // Generate historical data points
    const history = [];
    const today = new Date();
    
    // Use a deterministic seed based on the current portfolio value
    // This ensures the same chart data is generated for the same portfolio value
    const seed = Math.floor(currentValue.totalValue * 1000) % 10000;
    
    for (let i = days; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);
      
      // Generate a deterministic variation based on the seed and day
      const baseValue = currentValue.totalValue;
      const pseudoRandom = Math.sin(seed + i * 0.1) * 0.5; // Deterministic "random" value
      const variation = pseudoRandom * 0.1; // ±5% variation
      const value = baseValue * (1 + variation);
      
      history.push({
        date: date.toISOString().split('T')[0],
        value: Math.round(value * 100) / 100,
      });
    }
    
    return history;
  } catch (error) {
    console.error('Error generating portfolio history:', error);
    return [];
  }
}

export function formatCurrency(value) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
} 