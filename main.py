from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.engine import RecommendationEngine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RecommendationEngine()

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int, top_k: int = 5):
    return {"recommendations": engine.recommend(user_id, top_k)}

@app.post("/retrain")
def retrain():
    engine.train("retrain")
    return {"message": "Retraining triggered"}
