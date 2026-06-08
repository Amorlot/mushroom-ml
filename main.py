# main.py
from src import DataLoader, DataCleaner, Split, Eda, Encoder, ModelXGBoost, Logreg


def main():
    # --- CARICAMENTO ---
    print("\n>>> CARICAMENTO DATASET")
    dl = DataLoader(
        dataset_id=73,
        target_col='poisonous',
        # drop_missing_thresh non passato → None → saltato
    )
    dl.load()
    dl.info()
    dl.report_missing()

    # --- EDA (sul dataset completo, solo analisi, non modifica i dati) ---
    print("\n>>> EDA")
    eda = Eda(dl.df)
    print(eda.info())
    # plot_histograms e corrplot skippate: dataset solo categoriale, nessuna variabile numerica

    # --- SPLIT (prima della pulizia per evitare data leakage) ---
    print("\n>>> SPLIT DATASET")
    X = dl.df.drop(columns=['poisonous'])
    y = dl.df['poisonous']

    splitter = Split()
    X_train, X_test, y_train, y_test = splitter.split(X, y)

    # --- PULIZIA (fit SOLO su train, transform su entrambi) ---
    print("\n>>> PULIZIA DATI")
    cleaner = DataCleaner()
    cleaner.drop_cols(['veil-type']).fix_missing_categorical_unknown(cols=['stalk-root'])

    X_train = cleaner.fit_transform(X_train)
    X_test  = cleaner.transform(X_test)

    cleaner.report(X_train)

    print(f"Train pulito: {X_train.shape[0]} righe, {X_train.shape[1]} colonne")
    print(f"Test pulito:  {X_test.shape[0]} righe, {X_test.shape[1]} colonne")

    # --- ENCODING (fit SOLO su train, transform su entrambi) ---
    print("\n>>> ENCODING")
    cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()

    encoder = Encoder()
    X_train = encoder.fit_transform_features(X_train, cat_cols)  # fit + transform
    X_test  = encoder.transform_features(X_test)                 # solo transform

    y_train = encoder.encode_target(y_train, fit=True)   # fit + transform
    y_test  = encoder.encode_target(y_test,  fit=False)  # solo transform

    print(f"Train encoded: {X_train.shape[0]} righe, {X_train.shape[1]} colonne")
    print(f"Test encoded:  {X_test.shape[0]} righe, {X_test.shape[1]} colonne")

    # --- MODELLO XGBOOST ---
    print("\n>>> TRAINING XGBOOST")
    xgb_model = ModelXGBoost()
    xgb_model.train(X_train, y_train)
    xgb_model.evaluate(X_test, y_test)

    # --- MODELLO LOGISTIC REGRESSION ---
    print("\n>>> TRAINING LOGISTIC REGRESSION")
    logreg = Logreg()
    logreg.train(X_train, y_train)
    print(logreg.evaluate(X_test, y_test))
    print(logreg.classification_report(X_test, y_test))


if __name__ == "__main__":
    main()