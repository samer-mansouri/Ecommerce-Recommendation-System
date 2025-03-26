from app.models.collaborative import CollaborativeRecommender
from app.models.content_based import ContentBasedRecommender
import pandas as pd

class HybridRecommender:
    def __init__(self, feedback, products, views):
        self.collab = CollaborativeRecommender(feedback)
        self.content = ContentBasedRecommender(products)
        self.feedback = feedback
        self.views = views
        self.products = products

    def train(self):
        self.collab.train()
        self.content.train()

    def recommend(self, user_id, top_k=5):
        try:
            viewed_products = self.views[self.views['user_id'] == user_id]['product_id'].tolist()
        except Exception:
            viewed_products = []

        collab_recs = self.collab.recommend(user_id, top_k * 2)

        # CASE 1: Collaborative + Content-based
        if collab_recs:
            content_recs = []
            for pid in viewed_products:
                content_recs.extend(self.content.recommend(pid, top_k=2))
            combined = list(set(collab_recs + content_recs))
            return combined[:top_k]

        # CASE 2: Only views available → Content-based
        if viewed_products:
            content_recs = []
            for pid in viewed_products:
                content_recs.extend(self.content.recommend(pid, top_k=2))
            return list(set(content_recs))[:top_k]

        # CASE 3: Cold-start fallback (Top-rated products)
        if not self.feedback.empty:
            top_products = (
                self.feedback.groupby("product_id")["rate"]
                .mean()
                .sort_values(ascending=False)
                .head(top_k)
            )
            return list(top_products.index)

        # CASE 4: If no feedback or ratings → return random products (if available)
        if not self.products.empty:
            return list(self.products['product_id'].sample(n=top_k, replace=True))

        # CASE 5: No data at all
        return []
