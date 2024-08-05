

from sklearn.metrics import mean_absolute_error

def evaluate_model(model, x_train, y_train, x_test, y_test):
    ytrain_pred = model.predict(x_train)
    ytest_pred = model.predict(x_test)

    train_mae = mean_absolute_error(y_train, ytrain_pred)
    test_mae = mean_absolute_error(y_test, ytest_pred)

    return train_mae, test_mae