from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from optimizer import QueryOptimizer
import uvicorn

app = FastAPI(title="Hit Soochi: Universal Search Optimization")
optimizer = QueryOptimizer()

class SearchQuery(BaseModel):
    query: str

class OptimizationResponse(BaseModel):
    original: str
    optimized: str
    intent: str
    confidence: str
    seo_keywords: list[str]

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_query(request: SearchQuery):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    result = optimizer.optimize_query(request.query)
    seo = optimizer.generate_seo_keywords(request.query)
    
    return {
        **result,
        "seo_keywords": seo
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Hit Soochi Optimizer"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
