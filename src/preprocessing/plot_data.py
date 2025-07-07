from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt

def fetch_data_from_mongo(symbol, collection_name):
    client = MongoClient("***")
    db = client["stock_database"]
    collection = db[collection_name]

    # Fetch data for the given symbol
    document = collection.find_one({"symbol": symbol})

    # Extract the 'data' array into a DataFrame
    if document and "data" in document:
        df = pd.DataFrame(document["data"])
    else:
        raise ValueError(f"No data found for symbol: {symbol}")
    
    return df


def plot_separate(data, symbol):
    # Plot the closing prices
    plt.figure(figsize=(10, 6))
    plt.plot(data['date'], data['close'], label=symbol)
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.title(f'Closing Price of {symbol} over time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save the plot to the 'figures' folder
    plt.savefig(f'figures/{symbol.lower()}_closing_price.png')
    print(f"Figure for {symbol} saved in the 'figures' folder.")


def plot_together(data_dict, title="Closing Prices of all stocks", output_file = None):
    # Plot the closing prices of all stocks together
    plt.figure(figsize=(12, 6))
    
    for label, data in data_dict.items():
        plt.plot(data['date'], data['close'], label=label)
    
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'figures/{output_file}.png')
    print(f"Figure saved as: {output_file}")


def normalize_prices(data_dict):
    
    normalized_data_dict = {}
    
    for label, data in data_dict.items():
        normalized_data = data.copy()  # Create a copy to avoid modifying the original data
        min_price = normalized_data['close'].min()
        max_price = normalized_data['close'].max()
        
        # Normalize the 'close' prices using Min-Max normalization
        normalized_data['close'] = (normalized_data['close'] - min_price) / (max_price - min_price)
        normalized_data_dict[label] = normalized_data
    
    return normalized_data_dict


def analyse_seasonality(data, symbol):
    # Ensure 'date' is in datetime format and extract year and month
    data['date'] = pd.to_datetime(data['date'])
    data['Year'] = data['date'].dt.year
    data['Month'] = data['date'].dt.month

    # Plot 1: Overall seasonality (all years)
    monthly_avg = data.groupby('Month')['close'].mean()
    plt.figure(figsize=(10, 6))
    monthly_avg.plot(kind='bar')
    plt.xlabel('Month')
    plt.ylabel('Average Closing Price (USD)')
    plt.title(f'Seasonality: Average {symbol} Closing Price by Month (All Years)')
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
    plt.tight_layout()
    plt.savefig(f'figures/{symbol.lower()}_seasonality_all_years.png')
    plt.close()
    print(f"Seasonality analysis for {symbol} (all years) saved in the 'figures' folder.")
