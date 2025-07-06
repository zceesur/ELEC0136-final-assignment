from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def train_and_evaluate_model(x_train, y_train, x_test, y_test, model_name="Linear Regression"):
    # Initialize and train the model
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    # Make predictions
    predictions = model.predict(x_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Plot actual vs. predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Actual Price')
    plt.plot(predictions, label='Predicted Price', color='orange', linestyle='--')
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    plt.title(f'{model_name}: Actual vs. Predicted S&P 500 Prices')
    plt.legend()
    plt.savefig(f'figures/linearregression_predictions.png')
    plt.close()

    print("Model trained, results saved in 'figures' folder. \nMAE: {:.2f}, R²: {:.2f}".format(mae, r2))

    
    return model, mae, r2

