import { PrismaClient } from '../lib/generated/prisma/index.js';
import { faker } from '@faker-js/faker';

const prisma = new PrismaClient();

const sampleTickers = [
  'AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX'
];

async function main() {
  const userId = '5010a418-b441-494e-a56b-f4001ab28caf';

  // Verify the user exists
  const user = await prisma.user.findUnique({ where: { id: userId } });
  if (!user) throw new Error('User not found: ' + userId);

  // Create a new portfolio with 4 random stocks
  const portfolio = await prisma.portfolio.create({
    data: {
      name: 'Random Seed Portfolio',
      userId,
      portfolioStocks: {
        create: Array.from({ length: 4 }).map(() => {
          return {
            ticker: faker.helpers.arrayElement(sampleTickers),
            shares: faker.number.int({ min: 1, max: 100 }),
            purchaseDate: faker.date.past({ years: 1 }),
            purchasePrice: faker.number.float({ min: 10, max: 500, precision: 0.01 }),
            reinvestDividends: faker.datatype.boolean(),
            targetWeight: faker.number.float({ min: 5, max: 50, precision: 0.1 }),
          };
        }),
      },
    },
  });

  console.log('Created portfolio:', portfolio);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
