from sklearn.neighbors import NearestNeighbors
import pandas as pd

class CollaborativeRecommender:
    def __init__(self, df_feedback):
        self.df = df_feedback
        self.model = None
        self.user_item_matrix = None

    def train(self):
        if self.df.empty:
            print("CollaborativeRecommender: No feedback available, skipping training.")
            return
        self.user_item_matrix = self.df.pivot_table(index='user_id', columns='product_id', values='rate', fill_value=0)
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(self.user_item_matrix)

    def recommend(self, user_id, top_k=5):
        if self.model is None or self.user_item_matrix is None:
            return []
        if user_id not in self.user_item_matrix.index:
            return []
        distances, indices = self.model.kneighbors([self.user_item_matrix.loc[user_id]], n_neighbors=top_k + 1)
        neighbors = self.user_item_matrix.iloc[indices.flatten()[1:]]
        mean_ratings = neighbors.mean().sort_values(ascending=False).head(top_k)
        return list(mean_ratings.index)
