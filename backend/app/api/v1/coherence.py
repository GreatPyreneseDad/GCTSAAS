"""
Coherence calculation and tracking API endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from app.core.coherence_calculator import CoherenceCalculator, CoherenceComponents
from app.services.grok_service import GrokService

router = APIRouter()

class CoherenceMeasurementRequest(BaseModel):
    user_id: str
    psi: float = Field(..., ge=0.0, le=1.0, description="Internal consistency")
    rho: float = Field(..., ge=0.0, le=1.0, description="Accumulated wisdom")
    q: float = Field(..., ge=0.0, le=1.0, description="Moral activation energy")
    f: float = Field(..., ge=0.0, le=1.0, description="Social belonging")
    context: Optional[Dict] = None

class CoherenceResponse(BaseModel):
    coherence_score: float
    q_optimal: float
    components: Dict[str, float]
    breakdown: Dict[str, float]
    timestamp: datetime

# Dependency injection (simplified for now)
def get_coherence_calculator():
    return CoherenceCalculator()

def get_grok_service():
    return GrokService()

@router.post("/calculate", response_model=CoherenceResponse)
async def calculate_coherence(
    request: CoherenceMeasurementRequest,
    calculator: CoherenceCalculator = Depends(get_coherence_calculator)
):
    """Calculate coherence score for given measurements"""
    try:
        components = CoherenceComponents(
            psi=request.psi,
            rho=request.rho,
            q=request.q,
            f=request.f,
            timestamp=datetime.now(),
            user_id=request.user_id,
            context=request.context
        )
        
        result = calculator.calculate_coherence(components)
        
        return CoherenceResponse(
            coherence_score=result.coherence_score,
            q_optimal=result.q_optimal,
            components={
                "psi": components.psi,
                "rho": components.rho,
                "q": components.q,
                "f": components.f
            },
            breakdown=result.breakdown,
            timestamp=components.timestamp
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/scenarios/{variable_type}")
async def get_assessment_scenarios(
    variable_type: str,
    cultural_context: str = "western_corporate",
    count: int = 5,
    grok_service: GrokService = Depends(get_grok_service)
):
    """Generate assessment scenarios for specific coherence variable"""
    if variable_type not in ["psi", "rho", "q", "f"]:
        raise HTTPException(status_code=400, detail="Invalid variable type")
    
    try:
        scenarios = await grok_service.generate_assessment_scenarios(
            variable_type, cultural_context, count
        )
        return {"scenarios": scenarios, "variable_type": variable_type}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scenario generation failed: {e}")

@router.get("/health")
async def coherence_health():
    return {"status": "healthy", "module": "coherence"}
