from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, products_df):
        self.df = products_df
        self.vectorizer = TfidfVectorizer()
        self.sim_matrix = None
        self.trained = False

    def train(self):
        if self.df.empty:
            print("ContentBasedRecommender: No products available, skipping training.")
            return
        self.df['text'] = self.df['Name'] + ' ' + self.df['Description'].fillna('')
        tfidf_matrix = self.vectorizer.fit_transform(self.df['text'])
        self.sim_matrix = cosine_similarity(tfidf_matrix)
        self.trained = True

    def recommend(self, product_id, top_k=5):
        if not self.trained or self.sim_matrix is None:
            return []
        index = self.df.index[self.df['product_id'] == product_id].tolist()
        if not index:
            return []
        idx = index[0]
        sim_scores = list(enumerate(self.sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in sim_scores[1:top_k + 1]]
        return list(self.df.iloc[top_indices]['product_id'])
