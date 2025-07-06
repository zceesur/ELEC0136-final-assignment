from pathlib import Path
from src.acquisition.fetch_stock import get_stock_data
from src.acquisition.store_data import upload_csv_to_mongodb
from src.preprocessing.plot_data import fetch_data_from_mongo, plot_separate, plot_together, normalize_prices, analyse_seasonality
from src.preprocessing.features import preprocess_data
from src.training.model import train_and_evaluate_model
from src.training.ablation import ablation_study
import pandas as pd

def main():
    
    """DATA ACQUISITION AND STORAGE.

    Program will fetch S&P500, AAPL, and MSFT stock data from Yahoo Finance API, save it to CSV files, and store it in MongoDB.
    """
    # Define date range
    start_date = "2017-04-01"
    end_date = "2024-04-01"

    # Define output paths
    sp500_path = "data/raw/sp500.csv"
    aapl_path = "data/raw/aapl.csv"
    msft_path = "data/raw/msft.csv"

    # Fetch and save stock data
    print("Fetching S&P 500 data...")
    get_stock_data("^GSPC", start_date, end_date, sp500_path)
    print(f"S&P 500 data saved to {sp500_path}")

    print("Fetching AAPL data...")
    get_stock_data("AAPL", start_date, end_date, aapl_path)
    print(f"AAPL data saved to {aapl_path}")

    print("Fetching MSFT data...")
    get_stock_data("MSFT", start_date, end_date, msft_path)
    print(f"MSFT data saved to {msft_path}")
    
    # Store stock data in MongoDB
    upload_csv_to_mongodb(sp500_path, "SPX")
    upload_csv_to_mongodb(aapl_path, "AAPL")
    upload_csv_to_mongodb(msft_path, "MSFT")


    """DATA PREPROCESSING AND VISUALISATION.

    Program will now fetch data from database and convert the date columns to datetime.
    3 separate plots are made for each stock for the entire collected time period.
    Then, all three stocks are plotted together with absolute and normalised prices. A seasonality analysis of the S&P500 is also performed.
    Finally, data for all 3 stocks are preprocessed to prepare for model training and evaluation. This is done by merging all the stocks,
    dropping unnecessary columns, creating lagged and moving average features, and splitting the data into training and testing sets.
    """
    #Fetching each stock from MongoDB:
    sp500_data = fetch_data_from_mongo("SPX", "stocks")
    print(sp500_data.head())

    aapl_data = fetch_data_from_mongo("AAPL", "stocks")
    print(aapl_data.head())
    
    msft_data = fetch_data_from_mongo("MSFT", "stocks")
    print(msft_data.head())


    # Convert the 'date' column to datetime for proper plotting
    sp500_data['date'] = pd.to_datetime(sp500_data['date'])
    aapl_data['date'] = pd.to_datetime(aapl_data['date'])
    msft_data['date'] = pd.to_datetime(msft_data['date'])

    
    # Plot and save the figures for each stock
    plot_separate(sp500_data, "SPX")
    plot_separate(aapl_data, "AAPL")
    plot_separate(msft_data, "MSFT")

    # Prepare the data dictionary (to plot AAPL and MSFT together)
    data_dict = {
    "AAPL": aapl_data,
    "MSFT": msft_data,
    "SPX": sp500_data
    }
    
    #Plot all three stocks together
    print("Now plotting all three stocks together")
    plot_together(data_dict, title="Closing Prices of SPX, AAPL and MSFT over time", output_file='all_stocks')

    #Normalize prices because S&P 500 is significantly higher in value
    print("Now normalizing the stock prices")
    normalized_data_dict = normalize_prices(data_dict)

    #Plot the nornmalized stocks together
    print("Now plotting the normalized stock prices")
    plot_together(normalized_data_dict, title="Normalized Closing Prices of SPX, AAPL and MSFT over time", output_file='all_stock_norm')

    #Seasonality analysis for SP500 stock
    print("Performing seasonality analysis and plotting the results")
    analyse_seasonality(sp500_data, "SPX")

    #Preprocess data to prepare for model training, as well as correlation matrix
    print("Preprocessing data to prepare for model training")
    x_train, y_train, x_test, y_test, features = preprocess_data(sp500_data, aapl_data, msft_data)



    """MODEL TRAINING AND EVALUATION.
    
    Program will now train a linear regression model, plot the results, and evaluate the model through an ablation study.
    The ablation study simply removes the other supplemtary stocks AAPL and MSFT from the features to see how the model performs with just S&P500 data.
    The results of the training are summarised using an MAE and R2 score.
    """
    #Train and evaluate the model
    print("Training model, plotting results...")
    train_and_evaluate_model(x_train, y_train, x_test, y_test, model_name="Linear Regression")

    # Perform ablation study
    print("Performing ablation study...")
    ablation_study(x_train, x_test, y_train, y_test, features)

if __name__ == "__main__":
    main()