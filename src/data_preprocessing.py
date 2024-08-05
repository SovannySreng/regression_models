

import pandas as pd

def load_data(file_path='H:/My Drive/BISI II/Data Science/Term Assignments/Regression_Models_Solution/data/final.csv'):
    df = pd.read_csv(file_path)
    return df

def split_data(df, target_col, test_size=0.2, random_state=1234):
    from sklearn.model_selection import train_test_split
    x = df.drop(target_col, axis=1)
    y = df[target_col]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test