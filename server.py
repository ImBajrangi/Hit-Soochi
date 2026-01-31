from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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

# ============ Test Interface ============

TEST_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HitSoochi - Search Optimizer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 2rem;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #ff6b35, #f7931e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { text-align: center; color: #888; margin-bottom: 2rem; }
        
        .tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }
        .tab {
            padding: 0.75rem 1.5rem;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            color: #aaa;
        }
        .tab:hover { background: rgba(255,255,255,0.1); }
        .tab.active {
            background: linear-gradient(135deg, #ff6b35, #f7931e);
            color: #000;
            border-color: transparent;
        }
        
        .panel { display: none; }
        .panel.active { display: block; }
        
        .card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        input, textarea {
            width: 100%;
            padding: 1rem;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 8px;
            color: #fff;
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        input:focus, textarea:focus {
            outline: none;
            border-color: #ff6b35;
        }
        
        button {
            background: linear-gradient(135deg, #ff6b35, #f7931e);
            color: #000;
            border: none;
            padding: 1rem 2rem;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            font-size: 1rem;
            transition: transform 0.2s;
        }
        button:hover { transform: translateY(-2px); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        
        .result {
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 0.9rem;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background: rgba(255,107,53,0.2);
            border: 1px solid rgba(255,107,53,0.4);
            border-radius: 999px;
            font-size: 0.8rem;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .rec-card {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            background: rgba(255,107,53,0.1);
            border-radius: 12px;
            margin-top: 1rem;
        }
        .rec-icon { font-size: 2rem; }
        .rec-content { flex: 1; }
        .rec-content strong { display: block; margin-bottom: 0.25rem; }
        .rec-content small { color: #888; }
        
        .loading { opacity: 0.6; pointer-events: none; }
        
        @media (max-width: 600px) {
            body { padding: 1rem; }
            h1 { font-size: 1.8rem; }
            .tab { padding: 0.5rem 1rem; font-size: 0.9rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 HitSoochi</h1>
        <p class="subtitle">Universal Search Optimization for Vrindopnishad</p>
        
        <div class="tabs">
            <div class="tab active" data-tab="optimize">Optimize Query</div>
            <div class="tab" data-tab="recommend">Recommendations</div>
            <div class="tab" data-tab="suggest">Suggestions</div>
            <div class="tab" data-tab="rank">Rank Items</div>
        </div>
        
        <!-- Optimize Panel -->
        <div class="panel active" id="optimize">
            <div class="card">
                <h3 style="margin-bottom:1rem">🎯 Query Optimizer</h3>
                <input type="text" id="opt-query" placeholder="Enter search query (e.g., 'order sattvic lunch')">
                <button onclick="optimize()">Optimize Query</button>
                <div class="result" id="opt-result" style="display:none"></div>
            </div>
        </div>
        
        <!-- Recommend Panel -->
        <div class="panel" id="recommend">
            <div class="card">
                <h3 style="margin-bottom:1rem">💡 Service Recommendations</h3>
                <input type="text" id="rec-query" placeholder="What are you looking for? (e.g., 'temple visit')">
                <button onclick="recommend()">Get Recommendations</button>
                <div id="rec-result"></div>
            </div>
        </div>
        
        <!-- Suggest Panel -->
        <div class="panel" id="suggest">
            <div class="card">
                <h3 style="margin-bottom:1rem">✨ Auto-Suggestions</h3>
                <input type="text" id="sug-query" placeholder="Start typing... (e.g., 'ban')">
                <button onclick="suggest()">Get Suggestions</button>
                <div class="result" id="sug-result" style="display:none"></div>
            </div>
        </div>
        
        <!-- Rank Panel -->
        <div class="panel" id="rank">
            <div class="card">
                <h3 style="margin-bottom:1rem">📊 Semantic Ranking</h3>
                <input type="text" id="rank-query" placeholder="Search query">
                <textarea id="rank-items" rows="5" placeholder='Items JSON (e.g., [{"title":"Sattvic Thali","description":"Pure vegetarian meal"}])'></textarea>
                <button onclick="rankItems()">Rank Items</button>
                <div class="result" id="rank-result" style="display:none"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab).classList.add('active');
            });
        });
        
        async function apiCall(endpoint, body) {
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            return await res.json();
        }
        
        async function optimize() {
            const query = document.getElementById('opt-query').value;
            if (!query) return;
            
            const result = document.getElementById('opt-result');
            result.style.display = 'block';
            result.textContent = 'Processing...';
            
            const data = await apiCall('/optimize', { query });
            
            let html = `<strong>Original:</strong> ${data.original}\\n`;
            html += `<strong>Optimized:</strong> ${data.optimized}\\n`;
            html += `<strong>Intent:</strong> ${data.intent} (${data.confidence})\\n`;
            html += `<strong>SEO Keywords:</strong> ${data.seo_keywords?.join(', ') || 'N/A'}`;
            result.innerHTML = html;
        }
        
        async function recommend() {
            const query = document.getElementById('rec-query').value;
            if (!query) return;
            
            const result = document.getElementById('rec-result');
            result.innerHTML = '<p style="color:#888">Loading...</p>';
            
            const data = await apiCall('/recommend', { query });
            const p = data.primary_recommendation;
            
            let html = `<div class="rec-card">
                <span class="rec-icon">${p.icon}</span>
                <div class="rec-content">
                    <strong>${p.service}</strong>
                    <small>${p.description}</small>
                </div>
                <span class="badge">${p.cta}</span>
            </div>`;
            html += `<p style="margin-top:1rem;color:#888">Intent: ${data.detected_intent} (${data.confidence})</p>`;
            
            if (data.other_services?.length) {
                html += '<p style="margin-top:1rem;margin-bottom:0.5rem">Other services:</p>';
                data.other_services.forEach(s => {
                    html += `<span class="badge">${s.icon} ${s.service}</span>`;
                });
            }
            result.innerHTML = html;
        }
        
        async function suggest() {
            const partial = document.getElementById('sug-query').value;
            if (!partial) return;
            
            const result = document.getElementById('sug-result');
            result.style.display = 'block';
            result.textContent = 'Loading...';
            
            const data = await apiCall('/suggest', { partial, limit: 8 });
            
            let html = '<strong>Suggestions:</strong>\\n\\n';
            data.suggestions?.forEach((s, i) => {
                html += `${i+1}. ${s.text} (score: ${s.score.toFixed(2)})\\n`;
            });
            result.innerHTML = html;
        }
        
        async function rankItems() {
            const query = document.getElementById('rank-query').value;
            const itemsText = document.getElementById('rank-items').value;
            if (!query || !itemsText) return;
            
            const result = document.getElementById('rank-result');
            result.style.display = 'block';
            result.textContent = 'Ranking...';
            
            try {
                const items = JSON.parse(itemsText);
                const data = await apiCall('/rank', { query, items });
                
                let html = '<strong>Ranked Results:</strong>\\n\\n';
                data.ranked_items?.forEach((item, i) => {
                    html += `${i+1}. ${item.title || 'Untitled'} - Score: ${item.relevance_score?.toFixed(3)}\\n`;
                });
                result.innerHTML = html;
            } catch (e) {
                result.textContent = 'Error: Invalid JSON format for items';
            }
        }
    </script>
</body>
</html>
"""

# ============ API Endpoints ============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the test interface."""
    return TEST_UI_HTML

@app.get("/api")
async def api_info():
    """API documentation info."""
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
