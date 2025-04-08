from pymongo import MongoClient
import pandas as pd

def get_db():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["healthcare_db"]
    return db

def insert_records_to_mongo(data):
    db = get_db()
    db.admissions.insert_many(data)

def parse_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_dict(orient='records')
