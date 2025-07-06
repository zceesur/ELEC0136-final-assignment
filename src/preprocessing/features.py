import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(sp500_data, aapl_data, msft_data):
    # Merge all datasets by date
    merged = sp500_data.merge(aapl_data[['date', 'close']], on='date', suffixes=('_SP500', '_AAPL'))
    merged = merged.merge(msft_data[['date', 'close']], on='date', suffixes=('', '_MSFT'))
    merged.rename(columns={'close': 'close_MSFT'}, inplace=True)  # Rename because suffixes does not work if there are no conflicting columns
    
    #Drop unnecessary columns
    columns_to_drop = ['open', 'high', 'low', 'volume', 'dividends', 'stockSplits', 'Year', 'Month']
    merged = merged.drop(columns=columns_to_drop)
    print(merged.head())

    #Export merged dataframe to CSV to manually verify
    merged.to_csv("data/merged_data.csv", index=False)

    # Feature 1: Lagged closing prices, 1 day and 7 day
    merged['SP500_Lag1'] = merged['close_SP500'].shift(1)
    merged['AAPL_Lag1'] = merged['close_AAPL'].shift(1)
    merged['MSFT_Lag1'] = merged['close_MSFT'].shift(1)    
    merged['SP500_Lag7'] = merged['close_SP500'].shift(7)
    merged['AAPL_Lag7'] = merged['close_AAPL'].shift(7)
    merged['MSFT_Lag7'] = merged['close_MSFT'].shift(7)

    # Feature 2: Moving averages
    merged['SP500_MA7'] = merged['close_SP500'].rolling(7).mean()
    merged['SP500_MA30'] = merged['close_SP500'].rolling(30).mean()


    # Drop rows with missing values (from lag/rolling windows)
    merged.dropna(inplace=True)

    # Split into train/test (training = before Feb 2024)
    train = merged[merged['date'] < '2024-02-01']
    test = merged[merged['date'] >= '2024-02-01']

    # Define features (X) and target (y)
    features = ['SP500_Lag1', 'SP500_Lag7', 'SP500_MA7', 'SP500_MA30', 'AAPL_Lag1', 'AAPL_Lag7', 'MSFT_Lag1', 'MSFT_Lag7']
    x_train = train[features]  # Features for training
    y_train = train['close_SP500']  # Target (S&P 500 price)
    x_test = test[features]  # Features for testing
    y_test = test['close_SP500']  # Target (S&P 500 price)

    train.to_csv("data/train_data.csv", index=False)
    test.to_csv("data/test_data.csv", index=False)


    '''CORRELATION MATRIX FUNCTION

    '''
    # Combine features and target
    columns_to_analyze = features + ['close_SP500']

    # Compute the correlation matrix
    correlation_matrix = merged[columns_to_analyze].corr()

    # Visualize the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    
    # Save the matrix to figures folder
    plt.savefig('figures/correlation_matrix.png')
    print("Correlation matrix saved in the 'figures' folder.")

    return x_train, y_train, x_test, y_test, features