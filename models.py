from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with feature scaling."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model
