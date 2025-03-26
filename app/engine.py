from app.models.hybrid import HybridRecommender
from app.data.fetcher import fetch_data
import joblib
import os

class RecommendationEngine:
    def __init__(self):
        self.train()

    def train(self, source = "server"):
    
        if source == "retrain" or not os.path.exists("model.pkl"):
            feedback, products, views = fetch_data()
            print("Training model from fresh data...")
            self.model = HybridRecommender(feedback, products, views)
            self.model.train()
            joblib.dump(self.model, "model.pkl")
            print("Model trained and saved.")
        
        else:
            print("Model already exists. Loading from disk.")
            self.model = joblib.load("model.pkl")
            

    def recommend(self, user_id, top_k=5):
        return self.model.recommend(user_id, top_k)