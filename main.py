from fastapi import FastAPI
from app.engine import RecommendationEngine

app = FastAPI()
engine = RecommendationEngine()

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int, top_k: int = 5):
    return {"recommendations": engine.recommend(user_id, top_k)}

@app.get("/benchmark")
def benchmark():
    return engine.benchmark()

@app.post("/retrain")
def retrain():
    engine.train()
    return {"message": "Retraining triggered"}