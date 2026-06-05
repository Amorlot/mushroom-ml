# main.py
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.split import Split
from src.eda import Eda
from src.encoder import Encoder
from src.model_xgboost import ModelXGBoost


def main():
    # --- CARICAMENTO ---
    print("\n>>> CARICAMENTO DATASET")
    dl = DataLoader(
        dataset_id=73,
        target_col='poisonous',
        drop_cols=['veil-type']
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
    cleaner.fix_missing_categorical_unknown(cols=['stalk-root'])

    X_train = cleaner.fit_transform(X_train)
    X_test  = cleaner.transform(X_test)

    cleaner.report(X_train)

    print(f"Train pulito: {X_train.shape[0]} righe, {X_train.shape[1]} colonne")
    print(f"Test pulito:  {X_test.shape[0]} righe, {X_test.shape[1]} colonne")

    # --- ENCODING FEATURES ---
    print("\n>>> ENCODING FEATURES")
    cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()

    encoder = Encoder()
    X_train = encoder.fit_transform_features(X_train, cat_cols)
    X_test  = encoder.transform_features(X_test)

    print(f"Train encoded: {X_train.shape[0]} righe, {X_train.shape[1]} colonne")
    print(f"Test encoded:  {X_test.shape[0]} righe, {X_test.shape[1]} colonne")

    # --- ENCODING TARGET ---
    print("\n>>> ENCODING TARGET")
    encoder_y = Encoder()
    y_train = encoder_y.encode_target(y_train, fit=True)
    y_test  = encoder_y.encode_target(y_test, fit=False)
    print(f"  y_train: {y_train[:5]}  (esempio)")
    print(f"  y_test:  {y_test[:5]}   (esempio)")

    # --- MODELLO ---
    print("\n>>> TRAINING XGBOOST")
    model = ModelXGBoost()
    model.train(X_train, y_train)
    model.evaluate(X_test, y_test)


if __name__ == "__main__":
    main()