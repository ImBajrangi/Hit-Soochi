from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from optimizer import QueryOptimizer
import uvicorn

app = FastAPI(
    title="Hit Soochi: Universal Search Optimization",
    description="Semantic search optimization API for Vrindopnishad ecosystem",
    version="2.0.0"
)

# Enable CORS for cross-platform access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize optimizer (loads model once at startup)
optimizer = QueryOptimizer()

# ============ Request/Response Models ============

class SearchQuery(BaseModel):
    query: str

class SuggestionQuery(BaseModel):
    partial: str
    limit: Optional[int] = 5

class RankRequest(BaseModel):
    query: str
    items: List[Dict[str, str]]

class OptimizationResponse(BaseModel):
    original: str
    optimized: str
    intent: str
    confidence: str
    seo_keywords: List[str]

class ServiceRecommendation(BaseModel):
    service: str
    description: str
    icon: str
    url: str
    cta: str

class RecommendationResponse(BaseModel):
    query: str
    detected_intent: str
    confidence: str
    primary_recommendation: ServiceRecommendation
    other_services: List[ServiceRecommendation]

class Suggestion(BaseModel):
    text: str
    score: float

class SuggestionResponse(BaseModel):
    partial: str
    suggestions: List[Suggestion]

class RankedItem(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    relevance_score: float

class RankResponse(BaseModel):
    query: str
    ranked_items: List[Dict[str, Any]]

# ============ API Endpoints ============

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_query(request: SearchQuery):
    """Optimize a search query with domain-specific context and SEO keywords."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    result = optimizer.optimize_query(request.query)
    seo = optimizer.generate_seo_keywords(request.query)
    
    return {
        **result,
        "seo_keywords": seo
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: SearchQuery):
    """Get service recommendations based on query intent."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    return optimizer.get_recommendations(request.query)

@app.post("/suggest", response_model=SuggestionResponse)
async def get_suggestions(request: SuggestionQuery):
    """Get autocomplete suggestions for a partial query."""
    suggestions = optimizer.get_suggestions(request.partial, request.limit)
    return {
        "partial": request.partial,
        "suggestions": suggestions
    }

@app.post("/rank", response_model=RankResponse)
async def rank_results(request: RankRequest):
    """Rank a list of items by semantic similarity to the query."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    ranked = optimizer.rank_results(request.query, request.items)
    return {
        "query": request.query,
        "ranked_items": ranked
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "Hit Soochi Optimizer",
        "version": "2.0.0"
    }

@app.get("/")
async def root():
    """API documentation redirect."""
    return {
        "message": "Hit Soochi Universal Search Optimization API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "optimize": "POST /optimize - Optimize query with context",
            "recommend": "POST /recommend - Get intent-based recommendations",
            "suggest": "POST /suggest - Get autocomplete suggestions",
            "rank": "POST /rank - Rank items by relevance"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
