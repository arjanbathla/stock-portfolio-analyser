# 📊 Stock Portfolio Dashboard

[![Next.js](https://img.shields.io/badge/Next.js-14-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python)](https://python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue?style=for-the-badge&logo=typescript)](https://typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.0+-38B2AC?style=for-the-badge&logo=tailwind-css)](https://tailwindcss.com/)
[![Code Coverage](https://img.shields.io/badge/Coverage-85%25-brightgreen?style=for-the-badge)](https://github.com/arjanbathla/stock-portfolio-analyser)

> **Advanced quantitative portfolio management platform with institutional-grade risk analytics and machine learning forecasting**

## ✨ Features

### 📈 **Portfolio Management**
- **Multi-portfolio support** with real-time tracking
- **Stock position management** (buy/sell/split tracking)
- **Transaction history** with detailed analytics
- **Portfolio rebalancing** recommendations

### 🧮 **Quantitative Analytics**
- **Monte Carlo simulations** for value projections
- **ARIMA time series forecasting** with confidence intervals
- **Machine learning models** for stock price predictions
- **Risk-adjusted return metrics** (Sharpe, Sortino, Treynor ratios)

### ⚠️ **Risk Management**
- **Value at Risk (VaR)** calculations at multiple confidence levels
- **Expected Shortfall** and maximum drawdown analysis
- **Portfolio stress testing** with scenario analysis
- **Correlation matrix** and diversification metrics

### 📊 **Advanced Visualizations**
- **Interactive charts** using Chart.js and D3.js
- **Real-time data streaming** with WebSocket support
- **Customizable dashboards** with drag-and-drop widgets
- **Mobile-responsive design** for all devices

### 🔐 **Security & Authentication**
- **Clerk authentication** with role-based access
- **JWT token management** for API security
- **Rate limiting** and request validation
- **Data encryption** at rest and in transit

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API    │    │   Python ML     │
│   (Next.js 14)  │◄──►│   (Next.js API)  │◄──►│   (FastAPI)     │
│                 │    │                 │    │                 │
│ • React 18      │    │ • Server Actions │    │ • Monte Carlo   │
│ • TypeScript    │    │ • Edge Runtime   │    │ • ARIMA Models  │
│ • Tailwind CSS  │    │ • Clerk Auth     │    │ • ML Algorithms │
│ • Chart.js      │    │ • Prisma ORM     │    │ • Risk Metrics  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🛠️ Tech Stack

### **Frontend**
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe JavaScript development
- **Tailwind CSS** - Utility-first CSS framework
- **Chart.js** - Interactive data visualizations
- **Clerk** - Authentication and user management

### **Backend**
- **Next.js API Routes** - Server-side API endpoints
- **Prisma** - Database ORM and migrations
- **PostgreSQL** - Primary database
- **Redis** - Caching and session storage

### **Machine Learning**
- **Python 3.12+** - Core ML algorithms
- **FastAPI** - High-performance Python web framework
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning models
- **yfinance** - Real-time financial data

### **DevOps & Tools**
- **GitHub Actions** - CI/CD pipeline
- **Docker** - Containerization
- **ESLint & Prettier** - Code quality
- **Jest** - Testing framework

## 📸 Screenshots

### Dashboard Overview
![Dashboard](docs/images/dashboard.png)
*Main portfolio dashboard with key metrics and charts*

### Risk Analytics
![Risk Analytics](docs/images/risk-analytics.png)
*Advanced risk metrics and stress testing interface*

### Forecasting Models
![Forecasting](docs/images/forecasting.png)
*Machine learning predictions with confidence intervals*

### Portfolio Management
![Portfolio](docs/images/portfolio.png)
*Stock position management and rebalancing tools*

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18.17+ 
- **Python** 3.12+
- **PostgreSQL** 14+
- **Redis** 6+

### 1. Clone Repository
```bash
git clone https://github.com/arjanbathla/stock-portfolio-analyser.git
cd stock-portfolio-analyser
```

### 2. Install Dependencies
```bash
# Frontend dependencies
npm install

# Python backend dependencies
cd python_backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

### 3. Environment Setup
```bash
# Copy environment files
cp .env.example .env
cp python_backend/.env.example python_backend/.env

# Configure your environment variables
nano .env
```

**Required Environment Variables:**
```env
# Database
DATABASE_URL="postgresql://user:password@localhost:5432/portfolio_db"
REDIS_URL="redis://localhost:6379"

# Authentication
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_clerk_key
CLERK_SECRET_KEY=your_clerk_secret

# Python API
PYTHON_API_URL="http://localhost:8000"

# Financial APIs
ALPHA_VANTAGE_API_KEY=your_api_key
```

### 4. Database Setup
```bash
# Run database migrations
npx prisma migrate dev
npx prisma generate

# Seed initial data (optional)
npx prisma db seed
```

### 5. Start Development Servers
```bash
# Terminal 1: Frontend
npm run dev

# Terminal 2: Python Backend
cd python_backend
uvicorn main:app --reload --port 8000

# Terminal 3: Redis (if not running)
redis-server
```

### 6. Access Application
- **Frontend**: http://localhost:3000
- **Python API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🧪 Testing

```bash
# Frontend tests
npm run test
npm run test:watch

# Backend tests
cd python_backend
pytest

# E2E tests
npm run test:e2e
```


### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Manual Deployment
```bash
# Build production bundle
npm run build

# Start production server
npm start
```

### Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards
- **TypeScript** for all new JavaScript code
- **ESLint** and **Prettier** for code formatting
- **Jest** for unit testing
- **Conventional Commits** for commit messages

## 📊 Performance Metrics

- **Lighthouse Score**: 95+ (Performance, Accessibility, Best Practices, SEO)
- **Bundle Size**: < 250KB (gzipped)
- **API Response Time**: < 200ms (95th percentile)
- **Database Query Time**: < 50ms (average)

## 🙏 Acknowledgments

- **Financial data providers**: Alpha Vantage, Yahoo Finance
- **Open source libraries**: Next.js, FastAPI, Chart.js


**Made by Arjan Bathla**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/arjanbathla)
(www.linkedin.com/in/arjan-bathla-b59484236)




</div>
