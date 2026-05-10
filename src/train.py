from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC


def train_model_LR(X_train, y_train):

    model = LogisticRegression(
        max_iter=1200,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def train_model_SVC(X_train, y_train):

    model = LinearSVC()

    model.fit(X_train, y_train)

    return model


def train_model_RF(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_model_MLP(X_train, y_train):

    model = MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=50,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model