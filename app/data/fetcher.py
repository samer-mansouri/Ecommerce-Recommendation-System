import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

def fetch_data():
    engine = create_engine(os.getenv("DB_URL"))
    feedback = pd.read_sql("SELECT user_id, product_id, rate FROM Feedback", con=engine)
    products = pd.read_sql("SELECT id AS product_id, Name, Description, ProductBrandId, ProductTypeId FROM Product", con=engine)
    views = pd.read_sql("SELECT user_id, product_id FROM product_views", con=engine)
    return feedback, products, views