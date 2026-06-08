from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src import DataLoader, Logreg


def test_data_loader():
    dl = DataLoader(
        dataset_id=73,
        target_col='poisonous',
        drop_missing_thresh=0.5,
    )
    dl.load()
    dl.info()
    dl.report_missing()

    assert dl.df is not None
    assert len(dl.df) == 8124
    assert 'poisonous' in dl.df.columns

    print("\nTest DataLoader passati!")


def test_logreg():
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Logreg()
    model.train(X_train, y_train)
    assert model.model is not None
    assert model.best_params is not None

    metrics = model.evaluate(X_test, y_test)
    assert set(metrics.keys()) == {'accuracy', 'precision', 'recall', 'f1_score'}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())

    report = model.classification_report(X_test, y_test)
    assert isinstance(report, str)
    assert 'precision' in report

    print("\nTest Logreg passati!")


if __name__ == "__main__":
    test_data_loader()
    test_logreg()
