from app.models.hybrid import HybridRecommender
from app.data.fetcher import fetch_data
import joblib

class RecommendationEngine:
    def __init__(self):
        self.train()

    def train(self):
        feedback, products, views = fetch_data()
        self.model = HybridRecommender(feedback, products, views)
        self.model.train()
        joblib.dump(self.model, "model.pkl")

    def recommend(self, user_id, top_k=5):
        return self.model.recommend(user_id, top_k)

    def benchmark(self):
        return self.model.benchmark()