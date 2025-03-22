from app.models.collaborative import CollaborativeRecommender
from app.models.content_based import ContentBasedRecommender
from app.evaluator.benchmark import BenchmarkEvaluator

class HybridRecommender:
    def __init__(self, feedback, products, views):
        self.collab = CollaborativeRecommender(feedback)
        self.content = ContentBasedRecommender(products)
        self.feedback = feedback
        self.views = views

    def train(self):
        self.collab.train()
        self.content.train()

    def recommend(self, user_id, top_k=5):
        viewed_products = self.views[self.views['user_id'] == user_id]['product_id'].tolist()
        collab_recs = self.collab.recommend(user_id, top_k * 2)
        content_recs = []
        for pid in viewed_products:
            content_recs.extend(self.content.recommend(pid, top_k=2))
        content_recs = list(set(content_recs))
        combined = list(set(collab_recs + content_recs))
        return combined[:top_k]

    def benchmark(self):
        evaluator = BenchmarkEvaluator(self.feedback)
        return evaluator.evaluate()