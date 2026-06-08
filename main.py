# main.py
from src import DataLoader, DataCleaner, Split, Eda, Encoder, ModelXGBoost, Logreg


SEP = "=" * 60


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def main():
    # ── CARICAMENTO ──────────────────────────────────────────────
    section("CARICAMENTO DATASET")
    dl = DataLoader(dataset_id=73, target_col='poisonous')
    dl.load()
    dl.info()
    dl.report_missing()

    # ── EDA ──────────────────────────────────────────────────────
    section("EDA")
    eda = Eda(dl.df)

    info = eda.info()
    print(f"Shape:     {info['n_righe, n_colonne']}")
    print(f"Variabili: {info['variabili']}\n")

    print("-- Frequency table: cap-shape --")
    print(eda.frequency_table('cap-shape').to_string(index=False))

    print("\n-- Frequency table: gill-color --")
    print(eda.frequency_table('gill-color').to_string(index=False))

    print("\n-- Chi-quadro: cap-shape vs poisonous --")
    chi = eda.chi_square('cap-shape', 'poisonous')
    print(f"  chi2={chi['chi2']}  p={chi['p_value']}  dof={chi['dof']}  "
          f"Cramér's V={chi['cramers_v']}  associati={chi['associati']}")

    print("\n-- Chi-quadro: gill-color vs poisonous --")
    chi2 = eda.chi_square('gill-color', 'poisonous')
    print(f"  chi2={chi2['chi2']}  p={chi2['p_value']}  dof={chi2['dof']}  "
          f"Cramér's V={chi2['cramers_v']}  associati={chi2['associati']}")

    # ── SPLIT ────────────────────────────────────────────────────
    section("SPLIT DATASET")
    X = dl.df.drop(columns=['poisonous'])
    y = dl.df['poisonous']

    splitter = Split()
    X_train, X_test, y_train, y_test = splitter.split(X, y)
    print(f"Train: {X_train.shape[0]} righe   Test: {X_test.shape[0]} righe")

    # ── PULIZIA ──────────────────────────────────────────────────
    section("PULIZIA DATI")
    cleaner = DataCleaner()
    cleaner.drop_cols(['veil-type']).fix_missing_categorical_unknown(cols=['stalk-root'])

    X_train = cleaner.fit_transform(X_train)
    X_test  = cleaner.transform(X_test)

    cleaner.report(X_train)
    print(f"Train pulito: {X_train.shape[0]} righe, {X_train.shape[1]} colonne")
    print(f"Test pulito:  {X_test.shape[0]} righe, {X_test.shape[1]} colonne")

    # ── ENCODING ─────────────────────────────────────────────────
    section("ENCODING")
    cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()

    encoder = Encoder()
    X_train = encoder.fit_transform_features(X_train, cat_cols)
    X_test  = encoder.transform_features(X_test)

    y_train = encoder.encode_target(y_train, fit=True)
    y_test  = encoder.encode_target(y_test,  fit=False)

    print(f"Train encoded: {X_train.shape[0]} righe, {X_train.shape[1]} colonne")
    print(f"Test encoded:  {X_test.shape[0]} righe, {X_test.shape[1]} colonne")

    # ── XGBOOST ──────────────────────────────────────────────────
    section("TRAINING XGBOOST")
    xgb_model = ModelXGBoost()
    xgb_model.train(X_train, y_train)
    xgb_model.evaluate(X_test, y_test)

    # ── LOGISTIC REGRESSION ──────────────────────────────────────
    section("TRAINING LOGISTIC REGRESSION")
    logreg = Logreg()
    logreg.train(X_train, y_train)

    print("\n-- Metriche --")
    metrics = logreg.evaluate(X_test, y_test)
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.4f}")

    print("\n-- Classification report --")
    print(logreg.classification_report(X_test, y_test))

    print("-- Coefficienti (top 20) --")
    feature_names = X_train.columns.tolist()
    coef_df = logreg.coefficients(feature_names)
    print(coef_df.head(20).to_string(index=False))

    print(f"\n{SEP}\n  FINE\n{SEP}\n")


if __name__ == "__main__":
    main()