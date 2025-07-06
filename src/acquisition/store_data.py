import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from pathlib import Path

def upload_csv_to_mongodb(csv_path, symbol):
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Connect to MongoDB
    client = MongoClient("mongodb+srv://admin:odgMvc682LYgK1KY@cluster0.4x6am.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client["stock_database"]
    collection = db["stocks"]
    
    # Transform data to JSON
    records = df.to_dict(orient="records")
    transformed_data = {
        "symbol": symbol,
        "data": [{
            "date": datetime.strptime(str(record["Date"]), "%Y-%m-%d %H:%M:%S%z"),
            "open": record["Open"],
            "high": record["High"],
            "low": record["Low"],
            "close": record["Close"],
            "volume": record["Volume"],
            "dividends": record["Dividends"],
            "stockSplits": record["Stock Splits"]
        } for record in records]
    }
    
    # Upload to MongoDB, replacing existing data if it exists
    collection.replace_one(
        {"symbol": symbol},
        transformed_data,
        upsert=True
    )
    
    # Create indexes
    collection.create_index("symbol")
    collection.create_index("data.date")
    
    print(f"Uploaded {len(records)} records for {symbol}")