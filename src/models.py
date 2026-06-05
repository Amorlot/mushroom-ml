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

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.model = None
        self.best_params = None
        self.predictions = None

    def logistic_regression(self, cv=5, scoring='f1'):
        """
        Addestra una Logistic Regression utilizzando GridSearchCV.
        """

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

        lr = LogisticRegression(
            max_iter=5000,
            random_state=42
        )

        grid = GridSearchCV(
            estimator=lr,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        grid.fit(self.X_train, self.y_train)

        self.model = grid.best_estimator_
        self.best_params = grid.best_params_

        self.predictions = self.model.predict(self.X_test)

        return self.model

    def evaluate(self):

        if self.predictions is None:
            raise ValueError("Eseguire prima logistic_regression().")

        metrics = {
            "accuracy": accuracy_score(self.y_test, self.predictions),
            "precision": precision_score(
                self.y_test,
                self.predictions,
                average='weighted'
            ),
            "recall": recall_score(
                self.y_test,
                self.predictions,
                average='weighted'
            ),
            "f1_score": f1_score(
                self.y_test,
                self.predictions,
                average='weighted'
            )
        }

        return metrics

    def classification_report(self):

        if self.predictions is None:
            raise ValueError("Eseguire prima logistic_regression().")

        return classification_report(
            self.y_test,
            self.predictions
        )