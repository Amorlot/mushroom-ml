import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


class Logreg:

    def __init__(self):
        self.model = None
        self.best_params = None

    def train(self, X_train, y_train, cv=5, scoring='f1_weighted'):
        param_grid = [
            {
                'penalty': ['l1'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear']
            },
            {
                'penalty': ['l2'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'liblinear']
            },
            {
                'penalty': ['elasticnet'],
                'C': [0.001, 0.01, 0.1, 1, 10],
                'solver': ['saga'],
                'l1_ratio': [0.2, 0.5, 0.8]
            }
        ]

        lr = LogisticRegression(max_iter=5000, random_state=42)

        grid = GridSearchCV(
            estimator=lr,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        self.model = grid.best_estimator_
        self.best_params = grid.best_params_

        return self.model

    def evaluate(self, X_test, y_test, average='weighted'):
        if self.model is None:
            raise ValueError("Eseguire prima train().")

        predictions = self.model.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average=average),
            "recall": recall_score(y_test, predictions, average=average),
            "f1_score": f1_score(y_test, predictions, average=average)
        }

    def classification_report(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Eseguire prima train().")

        predictions = self.model.predict(X_test)
        return classification_report(y_test, predictions)

    def coefficients(self, feature_names: list = None) -> pd.DataFrame:
        """Restituisce coefficienti e odds ratio ordinati per |coeff| decrescente."""
        if self.model is None:
            raise ValueError("Eseguire prima train().")

        coef = self.model.coef_[0]
        names = feature_names if feature_names is not None else [f"x{i}" for i in range(len(coef))]

        df = pd.DataFrame({
            "feature":     names,
            "coefficient": coef,
            "odds_ratio":  np.exp(coef),
        })
        return df.reindex(df["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)
