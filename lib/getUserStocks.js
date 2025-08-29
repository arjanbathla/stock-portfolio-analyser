import { prisma } from '@/lib/prisma';
import { auth }    from '@clerk/nextjs/server';

export async function getUserStocks() {
  const { userId } = auth();
  if (!userId) throw new Error('Not authenticated');

  return prisma.stock.findMany({
    where: { userId },
    orderBy: { symbol: 'asc' },
  });
}
