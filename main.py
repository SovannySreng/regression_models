
import pandas as pd
from src.data_preprocessing import load_data, split_data
from src.eda import plot_distributions, plot_tree
from src.evaluation import evaluate_model
from src.model_training import train_linear_regression, train_decision_tree, train_random_forest, save_model, load_model

def main():
    # Load data
    df = load_data('H:/My Drive/BISI II/Data Science/Term Assignments/Regression_Models_Solution/data/final.csv')

    # EDA
    #plot_distributions(df, 'LoanAmount')

    # Split data
    x_train, x_test, y_train, y_test = split_data(df, 'price')

    # Train models
    lr_model = train_linear_regression(x_train, y_train)
    dt_model = train_decision_tree(x_train, y_train)
    rf_model = train_random_forest(x_train, y_train)

    # Evaluate models
    lr_train_mae, lr_test_mae = evaluate_model(lr_model, x_train, y_train, x_test, y_test)
    dt_train_mae, dt_test_mae = evaluate_model(dt_model, x_train, y_train, x_test, y_test)
    rf_train_mae, rf_test_mae = evaluate_model(rf_model, x_train, y_train, x_test, y_test)

    print(f'Linear Regression - Train MAE: {lr_train_mae}, Test MAE: {lr_test_mae}')
    print(f'Decision Tree - Train MAE: {dt_train_mae}, Test MAE: {dt_test_mae}')
    print(f'Random Forest - Train MAE: {rf_train_mae}, Test MAE: {rf_test_mae}')

    # Plot decision tree
    plot_tree(dt_model, dt_model.feature_names_in_)

    # Save model
    save_model(rf_model, 'RE_Model.pkl')

    # Load model and make a prediction
    loaded_model = load_model('RE_Model.pkl')
    prediction = loaded_model.predict([[2012, 216, 74, 1, 1, 618, 2000, 600, 1, 0, 0, 6, 0]])
    print(f'Prediction: {prediction}')

if __name__ == '__main__':
    main()