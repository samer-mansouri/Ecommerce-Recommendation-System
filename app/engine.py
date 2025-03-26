from app.models.hybrid import HybridRecommender
from app.data.fetcher import fetch_data
import joblib
import os

class RecommendationEngine:
    def __init__(self):
        self.train()

    def train(self):
        if os.path.exists("model.pkl"):
            print("Model already exists. Skipping training.")
            self.model = joblib.load("model.pkl")
        else:
            feedback, products, views = fetch_data()
            self.model = HybridRecommender(feedback, products, views)
            self.model.train()
            joblib.dump(self.model, "model.pkl")

    def recommend(self, user_id, top_k=5):
        return self.model.recommend(user_id, top_k)