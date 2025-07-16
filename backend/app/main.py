"""
GCT SaaS FastAPI Application
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import coherence
from app.core.coherence_calculator import CoherenceCalculator
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="GCT SaaS API",
    description="Grounded Coherence Theory as a Service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(coherence.router, prefix="/api/v1/coherence", tags=["coherence"])

# Initialize core calculator (later move to dependency injection)
coherence_calculator = CoherenceCalculator()

@app.get("/")
async def root():
    return {"message": "GCT SaaS API v1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "gct-saas-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
