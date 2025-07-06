from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

def ablation_study(x_train, x_test, y_train, y_test, features):
    results = {}
    # Remove one feature
    reduced_features = [f for f in features if f not in ['AAPL_Lag1', 'AAPL_Lag7', 'MSFT_Lag1', 'MSFT_Lag7']]
    x_train_reduced = x_train[reduced_features]
    x_test_reduced = x_test[reduced_features]
    
    # Train and evaluate
    model = LinearRegression()
    model.fit(x_train_reduced, y_train)
    predictions = model.predict(x_test_reduced)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Print results
    results['without_AAPL_MSFT'] = {'MAE': mae, 'R²': r2}
    print(f"Without AAPL and MSFT: MAE={mae:.2f}, R²={r2:.2f}")

    # Plot actual vs. predicted prices without AAPL and MSFT
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Actual Price')
    plt.plot(predictions, label='Predicted Price', color='red', linestyle='--')
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    plt.title(f'Ablation study: Actual vs. Predicted S&P 500 Prices without AAPL and MSFT')
    plt.legend()
    plt.savefig(f'figures/ablation_predictions.png')
    plt.close()

    print('Ablation study results saved in "figures" folder.')

    return results