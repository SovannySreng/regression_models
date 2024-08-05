
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def train_linear_regression(x_train, y_train):
    model = LinearRegression().fit(x_train, y_train)
    return model

def train_decision_tree(x_train, y_train, max_depth=3, max_features=10, random_state=567):
    model = DecisionTreeRegressor(max_depth=max_depth, max_features=max_features, random_state=random_state).fit(x_train, y_train)
    return model

def train_random_forest(x_train, y_train, n_estimators=200, criterion='absolute_error'):
    model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion).fit(x_train, y_train)
    return model

def save_model(model, file_name):
    import pickle
    pickle.dump(model, open(file_name, 'wb'))

def load_model(file_name):
    import pickle
    return pickle.load(open(file_name, 'rb'))