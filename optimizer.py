import re
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from typing import List, Dict, Any

class QueryOptimizer:
    def __init__(self):
        # Stage 1: Bi-Encoder for fast initial retrieval
        print("Loading Bi-Encoder (Stage 1: Fast Retrieval)...")
        self.bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Stage 2: Cross-Encoder for accurate re-ranking
        print("Loading Cross-Encoder (Stage 2: Precision Re-Ranking)...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Keep backward compatibility
        self.model = self.bi_encoder
        
        # Define intent categories and their prototype phrases
        self.intents = {
            "FOOD": ["order food", "sattvic meal", "menu", "foody vrinda", "restaurant", "lunch", "dinner", "breakfast", "prasad", "bhog", "thali", "khichdi"],
            "TOUR": ["plan trip", "temple visit", "yatra guide", "vrinda tours", "84 kos", "parikrama", "darshan", "pilgrimage", "banke bihari", "prem mandir", "iskcon"],
            "ART": ["buy paintings", "spiritual art", "chitra vrinda", "gallery", "wallpapers", "krishna art", "radha krishna", "vedic art", "canvas", "poster"],
            "SCRIPTURE": ["read bhagavad gita", "pdf library", "shlokas", "sant vaani", "books", "hanuman chalisa", "mantra", "stotra", "ved", "upanishad", "puran"],
            "COMMUNITY": ["connect with devotees", "vrinda chat", "qr code", "messenger", "satsang", "devotee", "sangat", "community"]
        }
        
        # Service recommendations for each intent
        self.service_recommendations = {
            "FOOD": {
                "service": "Foody Vrinda",
                "description": "Order pure Sattvic Bhog prepared with love and devotion",
                "icon": "🍲",
                "url": "/foody-vrinda",
                "cta": "Order Sattvic Meal"
            },
            "TOUR": {
                "service": "Vrinda Tours",
                "description": "Expert guided temple visits and 84 Kos Vrindavan Parikrama",
                "icon": "🛕",
                "url": "/vrinda-tours",
                "cta": "Plan Your Yatra"
            },
            "ART": {
                "service": "Chitra Vrinda",
                "description": "Beautiful spiritual art, Krishna paintings and wallpapers",
                "icon": "🎨",
                "url": "/chitra-vrinda",
                "cta": "Explore Art Gallery"
            },
            "SCRIPTURE": {
                "service": "VrindaVaani",
                "description": "Sacred texts, shlokas, mantras and spiritual literature",
                "icon": "📜",
                "url": "/vrindavaani",
                "cta": "Read Sacred Texts"
            },
            "COMMUNITY": {
                "service": "Vrinda Chat",
                "description": "Connect with fellow devotees and spiritual seekers",
                "icon": "💬",
                "url": "/vrinda-chat",
                "cta": "Join Community"
            },
            "GENERAL": {
                "service": "Vrindopnishad",
                "description": "Your complete spiritual companion for Vrindavan",
                "icon": "🙏",
                "url": "/",
                "cta": "Explore All Services"
            }
        }
        
        # Pre-calculate intent embeddings
        self.intent_labels = list(self.intents.keys())
        self.intent_prototypes = [". ".join(v) for v in self.intents.values()]
        self.intent_embeddings = self.model.encode(self.intent_prototypes, convert_to_tensor=True)
        
        # Popular search suggestions cache
        self.popular_suggestions = [
            "Banke Bihari darshan",
            "Sattvic thali order",
            "Krishna paintings",
            "Hanuman Chalisa",
            "84 kos parikrama",
            "Prem Mandir visit",
            "Bhagavad Gita",
            "Temple tour guide",
            "Radha Krishna art",
            "Vrindavan yatra"
        ]
        self.suggestion_embeddings = self.model.encode(self.popular_suggestions, convert_to_tensor=True)

    def classify_intent(self, query: str) -> tuple:
        """Classifies the user query into a core domain."""
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.intent_embeddings)[0]
        best_idx = torch.argmax(cos_scores).item()
        score = cos_scores[best_idx].item()
        
        # Return intent if confidence is reasonable, else GENERAL
        if score > 0.35:
            return self.intent_labels[best_idx], score
        return "GENERAL", score

    def optimize_query(self, query: str) -> Dict[str, Any]:
        """Expands the query with domain-specific context for better SEO/Search."""
        clean_q = re.sub(r'[^\w\s]', '', query).strip()
        intent, score = self.classify_intent(clean_q)
        
        optimized = clean_q
        
        # Add context based on intent
        if intent == "FOOD" and "foody" not in clean_q.lower():
            optimized += " with Foody Vrinda Sattvic"
        elif intent == "TOUR" and "tour" not in clean_q.lower():
            optimized += " Vrindavan temple guide"
        elif intent == "ART" and "art" not in clean_q.lower():
            optimized += " Vedic spiritual art gallery"
            
        # Ensure Vrindavan context if missing
        if "vrindavan" not in optimized.lower() and "brij" not in optimized.lower():
            optimized += " in Vrindavan"
            
        return {
            "original": query,
            "optimized": optimized,
            "intent": intent,
            "confidence": f"{score:.2f}"
        }

    def get_recommendations(self, query: str) -> Dict[str, Any]:
        """Returns service recommendations based on query intent."""
        intent, score = self.classify_intent(query)
        primary = self.service_recommendations.get(intent, self.service_recommendations["GENERAL"])
        
        # Get secondary recommendations (other related services)
        secondary = []
        for key, rec in self.service_recommendations.items():
            if key != intent and key != "GENERAL":
                secondary.append(rec)
        
        return {
            "query": query,
            "detected_intent": intent,
            "confidence": f"{score:.2f}",
            "primary_recommendation": primary,
            "other_services": secondary[:3]  # Top 3 other services
        }

    def rank_results(self, query: str, items: List[Dict[str, str]], top_k: int = 50, final_k: int = 10) -> List[Dict[str, Any]]:
        """
        Two-Stage Semantic Ranking:
        
        Stage 1 (Bi-Encoder): Fast vector similarity to get Top-K candidates
        Stage 2 (Cross-Encoder): Precise re-ranking for final results
        
        Args:
            query: Search query
            items: List of items to rank
            top_k: Number of candidates from Stage 1 (default: 50)
            final_k: Number of final results after Stage 2 (default: 10)
        """
        if not items:
            return []
        
        # Create text representations of items
        item_texts = []
        for item in items:
            text = f"{item.get('title', '')} {item.get('description', '')} {item.get('category', '')}"
            item_texts.append(text.strip())
        
        # ============ STAGE 1: Bi-Encoder (Fast Retrieval) ============
        query_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        item_embeddings = self.bi_encoder.encode(item_texts, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, item_embeddings)[0]
        
        # Get top-K candidates from Stage 1
        actual_top_k = min(top_k, len(items))
        top_k_indices = torch.topk(similarities, actual_top_k).indices.tolist()
        
        # Prepare candidates with bi-encoder scores
        candidates = []
        for idx in top_k_indices:
            item_copy = items[idx].copy()
            item_copy['bi_encoder_score'] = float(similarities[idx])
            item_copy['item_text'] = item_texts[idx]
            candidates.append(item_copy)
        
        # ============ STAGE 2: Cross-Encoder (Precision Re-Ranking) ============
        if len(candidates) > 0:
            # Create query-document pairs for cross-encoder
            cross_encoder_pairs = [[query, c['item_text']] for c in candidates]
            
            # Get cross-encoder scores
            cross_encoder_scores = self.cross_encoder.predict(cross_encoder_pairs)
            
            # Add cross-encoder scores to candidates
            for i, candidate in enumerate(candidates):
                candidate['cross_encoder_score'] = float(cross_encoder_scores[i])
                # Normalize to 0-1 range (cross-encoder outputs raw logits)
                candidate['relevance_score'] = float(1 / (1 + torch.exp(torch.tensor(-cross_encoder_scores[i]))))
                # Clean up internal field
                del candidate['item_text']
            
            # Sort by cross-encoder score (final precision ranking)
            candidates.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
        
        # Return top final_k results
        return candidates[:final_k]
    
    def rank_results_fast(self, query: str, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Fast ranking using only Bi-Encoder (single-stage).
        Use this for real-time applications where speed is critical.
        """
        if not items:
            return []
        
        query_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        
        item_texts = []
        for item in items:
            text = f"{item.get('title', '')} {item.get('description', '')} {item.get('category', '')}"
            item_texts.append(text.strip())
        
        item_embeddings = self.bi_encoder.encode(item_texts, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, item_embeddings)[0]
        
        ranked_items = []
        for idx, item in enumerate(items):
            item_copy = item.copy()
            item_copy['relevance_score'] = float(similarities[idx])
            ranked_items.append(item_copy)
        
        ranked_items.sort(key=lambda x: x['relevance_score'], reverse=True)
        return ranked_items

    def get_suggestions(self, partial_query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Returns autocomplete suggestions based on partial query."""
        if not partial_query or len(partial_query) < 2:
            return [{"text": s, "score": 1.0} for s in self.popular_suggestions[:limit]]
        
        query_embedding = self.model.encode(partial_query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, self.suggestion_embeddings)[0]
        
        # Get top suggestions by similarity
        suggestions = []
        for idx, suggestion in enumerate(self.popular_suggestions):
            # Also check if partial query is substring match
            is_prefix = suggestion.lower().startswith(partial_query.lower())
            contains = partial_query.lower() in suggestion.lower()
            score = float(similarities[idx])
            
            # Boost score for prefix/contains matches
            if is_prefix:
                score += 0.5
            elif contains:
                score += 0.3
            
            suggestions.append({
                "text": suggestion,
                "score": min(score, 1.0)
            })
        
        # Sort by score and return top suggestions
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        return suggestions[:limit]

    def generate_seo_keywords(self, query: str) -> List[str]:
        """Generates high-value keywords for web meta tags."""
        result = self.optimize_query(query)
        base_keywords = result['optimized'].split()
        
        # Add standard high-value tokens
        seo_set = set(base_keywords + ["Vrindavan", "Brij", "Vrindopnishad", "Spiritual"])
        return list(seo_set)

if __name__ == "__main__":
    opt = QueryOptimizer()
    
    print("=== Query Optimization ===")
    print(opt.optimize_query("how to reach banke bihari"))
    print(opt.optimize_query("order lunch"))
    
    print("\n=== Recommendations ===")
    print(opt.get_recommendations("order sattvic food"))
    
    print("\n=== Suggestions ===")
    print(opt.get_suggestions("tem"))
    
    print("\n=== Ranking ===")
    test_items = [
        {"title": "Pizza", "description": "Italian food"},
        {"title": "Khichdi", "description": "Sattvic rice dish"},
        {"title": "Prasad Thali", "description": "Temple food offering"}
    ]
    print(opt.rank_results("sattvic meal", test_items))
