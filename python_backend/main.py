from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from dotenv import load_dotenv

# Import routers
from routes import forecast, portfolio, analytics, ml_models

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Stock Portfolio ML API",
    description="Advanced Machine Learning and Quantitative Finance API for Portfolio Analysis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(forecast.router, prefix="/api/forecast", tags=["Forecast"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(ml_models.router, prefix="/api/ml", tags=["Machine Learning"])

@app.get("/")
async def root():
    return {"message": "Stock Portfolio ML API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Stock Portfolio ML API"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 