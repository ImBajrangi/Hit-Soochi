import re
import torch
from sentence_transformers import SentenceTransformer, util

class QueryOptimizer:
    def __init__(self):
        # Using the same semantic model as the chatbot for consistency
        print("Loading Optimizer Semantic Model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define intent categories and their prototype phrases
        self.intents = {
            "FOOD": ["order food", "sattvic meal", "menu", "foody vrinda", "restaurant"],
            "TOUR": ["plan trip", "temple visit", "yatra guide", "vrinda tours", "84 kos"],
            "ART": ["buy paintings", "spiritual art", "chitra vrinda", "gallery", "wallpapers"],
            "SCRIPTURE": ["read bhagavad gita", "pdf library", "shlokas", "sant vaani", "books"],
            "COMMUNITY": ["connect with devotees", "vrinda chat", "qr code", "messenger"]
        }
        
        # Pre-calculate intent embeddings
        self.intent_labels = list(self.intents.keys())
        self.intent_prototypes = [". ".join(v) for v in self.intents.values()]
        self.intent_embeddings = self.model.encode(self.intent_prototypes, convert_to_tensor=True)

    def classify_intent(self, query):
        """Classifies the user query into a core domain."""
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.intent_embeddings)[0]
        best_idx = torch.argmax(cos_scores).item()
        score = cos_scores[best_idx].item()
        
        # Return intent if confidence is reasonable, else GENERAL
        if score > 0.4:
            return self.intent_labels[best_idx], score
        return "GENERAL", score

    def optimize_query(self, query):
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

    def generate_seo_keywords(self, query):
        """Generates high-value keywords for web meta tags."""
        result = self.optimize_query(query)
        base_keywords = result['optimized'].split()
        
        # Add standard high-value tokens
        seo_set = set(base_keywords + ["Vrindavan", "Brij", "Vrindopnishad", "Spiritual"])
        return list(seo_set)

if __name__ == "__main__":
    opt = QueryOptimizer()
    print(opt.optimize_query("how to reach banke bihari"))
    print(opt.optimize_query("order lunch"))
