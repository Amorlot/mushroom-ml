# src/model_xgboost.py
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


class ModelXGBoost:
    def __init__(self):
        self.model = None
        self.best_params = None

    def train(self, X_train, y_train):
        """Addestra con GridSearchCV per trovare i migliori parametri."""
        param_grid = {
            'n_estimators':  [100, 200, 500],
            'max_depth':     [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample':     [0.8, 1.0],
            'min_child_weight': [1, 5]
        }
        xgb = XGBClassifier(random_state=42, eval_metric='logloss')

        grid = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=5,           # 5-fold cross validation
            scoring='accuracy',
            n_jobs=-1,      # usa tutti i core disponibili
            verbose=1
        )

        grid.fit(X_train, y_train)

        self.model = grid.best_estimator_
        self.best_params = grid.best_params_

        print(f"\n--- MIGLIORI PARAMETRI ---")
        for k, v in self.best_params.items():
            print(f"  {k}: {v}")

    def evaluate(self, X_test, y_test):
        """Valuta il modello sul test set."""
        y_pred = self.model.predict(X_test)

        print(f"\n--- RISULTATI ---")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

        return y_pred