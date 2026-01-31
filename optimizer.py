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
        
        # Domain-to-category mapping for platform-specific filtering
        # Each domain has allowed categories and keywords that items MUST match
        self.domain_filters = {
            "vrindavaani": {
                "categories": ["scripture", "spiritual", "mantra", "stotra", "gita", "puran", "ved", "upanishad", "chalisa", "aarti", "bhajan", "kirtan", "discourse", "teaching"],
                "keywords": ["gita", "chalisa", "mantra", "stotra", "ved", "puran", "scripture", "spiritual", "bhajan", "kirtan", "aarti", "discourse", "sant", "teaching", "shloka"],
                "exclude": ["food", "restaurant", "thali", "meal", "order", "delivery", "tour", "trip", "painting", "wallpaper", "poster"]
            },
            "foody_vrinda": {
                "categories": ["food", "meal", "prasad", "bhog", "thali", "restaurant", "kitchen", "sattvic", "recipe", "menu"],
                "keywords": ["food", "meal", "thali", "prasad", "bhog", "sattvic", "khichdi", "recipe", "menu", "order", "lunch", "dinner", "breakfast"],
                "exclude": ["scripture", "gita", "chalisa", "mantra", "painting", "art", "tour", "trip"]
            },
            "chitra_vrinda": {
                "categories": ["art", "painting", "wallpaper", "poster", "canvas", "digital art", "gallery", "artwork"],
                "keywords": ["art", "painting", "wallpaper", "poster", "canvas", "gallery", "image", "photo", "picture", "artwork", "digital"],
                "exclude": ["food", "meal", "scripture", "gita", "tour", "trip"]
            },
            "vrinda_tours": {
                "categories": ["tour", "travel", "yatra", "pilgrimage", "darshan", "temple", "parikrama", "guide"],
                "keywords": ["tour", "trip", "travel", "yatra", "darshan", "temple", "parikrama", "guide", "visit", "pilgrimage", "84 kos"],
                "exclude": ["food", "meal", "scripture", "gita", "painting", "art"]
            },
            "vrindopnishad": {
                # Main website - allows all content but prioritizes based on section
                "categories": ["all"],
                "keywords": [],
                "exclude": []
            }
        }
        
        # Source configuration - maps source to content domain
        # This tells the model WHERE the search is happening and WHAT content to focus on
        self.source_config = {
            # Web sources (vrindopnishad.in sections)
            "vrindopnishad": {"domain": "vrindopnishad", "platform": "web", "content_type": "all", "description": "Main Vrindopnishad website"},
            "vrinda_tours": {"domain": "vrinda_tours", "platform": "web", "content_type": "tours", "description": "Vrinda Tours section"},
            "foody_vrinda_web": {"domain": "foody_vrinda", "platform": "web", "content_type": "food", "description": "Foody Vrinda web section"},
            "chitra_vrinda": {"domain": "chitra_vrinda", "platform": "web", "content_type": "art", "description": "Chitra Vrinda gallery"},
            "vrindavaani_web": {"domain": "vrindavaani", "platform": "web", "content_type": "scripture", "description": "VrindaVaani web section"},
            
            # App sources
            "foody_vrinda_app": {"domain": "foody_vrinda", "platform": "app", "content_type": "food", "description": "Foody Vrinda mobile app"},
            "vrindavaani_app": {"domain": "vrindavaani", "platform": "app", "content_type": "scripture", "description": "VrindaVaani/Sant Vaani mobile app"},
            "vrinda_tours_app": {"domain": "vrinda_tours", "platform": "app", "content_type": "tours", "description": "Vrinda Tours mobile app"},
        }
        
        print("✅ Source configuration loaded:")
        for source, config in self.source_config.items():
            print(f"   - {source} ({config['platform']}): {config['description']}")

    def get_context_info(self, platform: str = None, source: str = None, domain: str = None) -> Dict[str, Any]:
        """
        STEP 1: Identify WHERE the search is happening.
        
        This method resolves the search context and returns:
        - platform: 'web' or 'app'
        - source: The specific section/app making the request
        - domain: The content domain to filter by
        - content_type: What kind of content to prioritize
        """
        context_info = {
            "platform": platform or "unknown",
            "source": source or "unknown",
            "domain": domain,
            "content_type": "all",
            "description": "General search - no filtering"
        }
        
        # If source is provided, get full config from source_config
        if source and source.lower() in self.source_config:
            config = self.source_config[source.lower()]
            context_info["platform"] = config["platform"]
            context_info["domain"] = config["domain"]
            context_info["content_type"] = config["content_type"]
            context_info["description"] = config["description"]
        
        # Legacy: If only domain provided, infer from domain
        elif domain and domain.lower() in self.domain_filters:
            context_info["domain"] = domain.lower()
            context_info["content_type"] = domain.lower().replace("_", " ")
        
        # Log the context clearly
        print(f"\n{'='*60}")
        print(f"🔍 SEARCH CONTEXT IDENTIFIED")
        print(f"{'='*60}")
        print(f"   Platform: {context_info['platform'].upper()}")
        print(f"   Source:   {context_info['source']}")
        print(f"   Domain:   {context_info['domain']}")
        print(f"   Content:  {context_info['content_type']}")
        print(f"   Info:     {context_info['description']}")
        print(f"{'='*60}\n")
        
        return context_info

    def filter_by_domain(self, items: List[Dict[str, str]], domain: str) -> List[Dict[str, Any]]:
        """
        Pre-filter items based on the calling platform's domain.
        This ensures VrindaVaani gets only spiritual content, Foody Vrinda gets only food, etc.
        """
        if not domain or domain.lower() == "general" or domain not in self.domain_filters:
            return items  # No filtering for general or unknown domains
        
        filters = self.domain_filters[domain.lower()]
        allowed_keywords = filters["categories"] + filters["keywords"]
        excluded_keywords = filters.get("exclude", [])
        
        filtered_items = []
        for item in items:
            # Create searchable text from all item fields
            item_text = " ".join([
                str(item.get("title", "")),
                str(item.get("description", "")),
                str(item.get("category", "")),
                str(item.get("type", ""))
            ]).lower()
            
            # Check if item contains any excluded keywords
            has_excluded = any(kw in item_text for kw in excluded_keywords)
            if has_excluded:
                continue  # Skip items with excluded keywords
            
            # Check if item matches any allowed keywords/categories
            has_allowed = any(kw in item_text for kw in allowed_keywords)
            
            # For domain filtering, item must match at least one allowed keyword
            # OR if the item's category explicitly matches the domain
            item_category = str(item.get("category", "")).lower()
            category_match = any(cat in item_category for cat in filters["categories"])
            
            if has_allowed or category_match:
                item_copy = item.copy()
                item_copy["domain_match"] = True
                filtered_items.append(item_copy)
        
        # If no items match, return original items with penalty scores
        if not filtered_items:
            return items
        
        return filtered_items
    
    def apply_facet_filters(self, items: List[Dict[str, str]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply faceted filters like Amazon/Flipkart.
        Startup-friendly: Works with item metadata, no user data needed.
        
        Filters:
            - category: Filter by category (e.g., "scripture", "food")
            - type: Filter by type (e.g., "pdf", "video", "article")
            - featured_only: Show only featured items
            - new_only: Show only new items (created in last 7 days)
        """
        if not filters or not items:
            return items
        
        filtered = items
        
        # Category filter
        if filters.get('category'):
            cat = filters['category'].lower()
            filtered = [i for i in filtered if cat in str(i.get('category', '')).lower()]
        
        # Type filter
        if filters.get('type'):
            item_type = filters['type'].lower()
            filtered = [i for i in filtered if item_type in str(i.get('type', '')).lower()]
        
        # Featured only
        if filters.get('featured_only'):
            filtered = [i for i in filtered if i.get('featured') or i.get('is_featured')]
        
        # New/Recent only
        if filters.get('new_only'):
            filtered = [i for i in filtered if i.get('is_new') or i.get('new')]
        
        print(f"📌 Facet filters: {len(items)} → {len(filtered)} items")
        return filtered if filtered else items

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

    def rank_results(self, query: str, items: List[Dict[str, str]], domain: str = None, 
                      platform: str = None, source: str = None, filters: Dict[str, Any] = None,
                      top_k: int = 50, final_k: int = 10) -> List[Dict[str, Any]]:
        """
        Context-Aware Three-Stage Semantic Ranking:
        
        Step 0 (Context): Identify WHERE the search is happening
        Stage 1 (Pre-filter): Filter items by platform's content domain
        Stage 2 (Bi-Encoder): Fast vector similarity for Top-K candidates
        Stage 3 (Cross-Encoder): Precise re-ranking for final results
        
        Args:
            query: Search query
            items: List of items to rank
            domain: Legacy domain filter (vrindavaani, foody_vrinda, etc.)
            platform: 'web' or 'app'
            source: Specific source (vrindavaani_app, foody_vrinda_web, etc.)
            filters: Faceted filters (category, type, featured_only, new_only)
            top_k: Number of candidates from Stage 2 (default: 50)
            final_k: Number of final results after Stage 3 (default: 10)
        """
        if not items:
            return []
        
        # ============ STEP 0: IDENTIFY SEARCH CONTEXT ============
        context = self.get_context_info(platform=platform, source=source, domain=domain)
        effective_domain = context["domain"]
        
        # Stage 1a: Pre-filter by domain (only show relevant content)
        filtered_items = self.filter_by_domain(items, effective_domain) if effective_domain else items
        print(f"📊 Domain filter: {len(items)} → {len(filtered_items)} items")
        
        # Stage 1b: Apply faceted filters (category, type, featured_only, etc.)
        if filters:
            filtered_items = self.apply_facet_filters(filtered_items, filters)
        
        # Create text representations of items
        item_texts = []
        for item in filtered_items:
            text = f"{item.get('title', '')} {item.get('description', '')} {item.get('category', '')}"
            item_texts.append(text.strip())
        
        # ============ STAGE 1: Bi-Encoder (Fast Retrieval) ============
        query_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        item_embeddings = self.bi_encoder.encode(item_texts, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, item_embeddings)[0]
        
        # Get top-K candidates from Stage 1
        actual_top_k = min(top_k, len(filtered_items))
        top_k_indices = torch.topk(similarities, actual_top_k).indices.tolist()
        
        # Prepare candidates with bi-encoder scores
        candidates = []
        for idx in top_k_indices:
            item_copy = filtered_items[idx].copy()
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
            
            # ============ STARTUP-FRIENDLY ENHANCEMENTS ============
            
            # 1. EDITORIAL BOOST: Boost items marked as featured/new (no user data needed)
            for candidate in candidates:
                boost = 0
                
                # Featured items get a significant boost
                if candidate.get('featured') or candidate.get('is_featured'):
                    boost += 0.3
                    candidate['boost_reason'] = 'featured'
                
                # New items get a small boost (helps surface fresh content)
                if candidate.get('is_new') or candidate.get('new'):
                    boost += 0.15
                    candidate['boost_reason'] = candidate.get('boost_reason', '') + ' new'
                
                # Trending/popular items (can be set by admin)
                if candidate.get('trending') or candidate.get('popular'):
                    boost += 0.2
                    candidate['boost_reason'] = candidate.get('boost_reason', '') + ' trending'
                
                # Apply boost to final score
                candidate['editorial_boost'] = boost
                candidate['final_score'] = candidate['relevance_score'] + boost
            
            # Sort by final score (relevance + editorial boost)
            candidates.sort(key=lambda x: x.get('final_score', x['relevance_score']), reverse=True)
            
            # 2. DIVERSITY: For main search, mix categories (prevents all same type)
            if effective_domain == "vrindopnishad" or not effective_domain:
                candidates = self._apply_diversity(candidates, final_k)
        
        # Return top final_k results
        return candidates[:final_k]
    
    def _apply_diversity(self, candidates: List[Dict], limit: int) -> List[Dict]:
        """
        Ensures result diversity by interleaving categories.
        Startup-friendly: No user data needed, just category mixing.
        """
        if not candidates or len(candidates) <= 3:
            return candidates
        
        # Group by category
        by_category = {}
        for item in candidates:
            cat = item.get('category', 'general').lower()
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item)
        
        # If only one category, no diversity needed
        if len(by_category) <= 1:
            return candidates
        
        # Round-robin pick from each category
        diverse_results = []
        categories = list(by_category.keys())
        idx = 0
        
        while len(diverse_results) < limit and any(by_category.values()):
            cat = categories[idx % len(categories)]
            if by_category[cat]:
                diverse_results.append(by_category[cat].pop(0))
            idx += 1
            
            # Remove empty categories
            categories = [c for c in categories if by_category.get(c)]
        
        return diverse_results
    
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
