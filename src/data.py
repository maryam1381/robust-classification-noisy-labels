# read data from dataset 
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_mnist_arff(path):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    # target column
    y = df["class"].astype(int)

    # features
    X = df.drop("class", axis=1).astype(float)

    return X, y



def split_data(X, y, test_size):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test