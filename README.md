---
title: HitSoochi
emoji: 🔍
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
license: mit
---

# HitSoochi: Universal Search Optimization API

Semantic search optimization for the Vrindopnishad ecosystem. Powers intelligent search, recommendations, and auto-suggestions across all platforms.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/optimize` | POST | Optimize query with domain-specific context |
| `/recommend` | POST | Get intent-based service recommendations |
| `/suggest` | POST | Get autocomplete suggestions |
| `/rank` | POST | Rank items by semantic similarity |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |

## Example Usage

```python
import requests

response = requests.post(
    "https://YOUR-SPACE.hf.space/optimize",
    json={"query": "how to reach banke bihari"}
)
print(response.json())
```

## Tech Stack
- **FastAPI** for high-performance API
- **Sentence Transformers** (all-MiniLM-L6-v2) for semantic understanding
- **PyTorch** for model inference
