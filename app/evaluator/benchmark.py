from sklearn.metrics import mean_squared_error
import numpy as np

class BenchmarkEvaluator:
    def __init__(self, feedback_df):
        self.df = feedback_df

    def evaluate(self):
        actual = self.df['rate']
        predicted = self.df.groupby('product_id')['rate'].transform('mean')
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        return {"RMSE": rmse}