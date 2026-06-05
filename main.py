from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.split import Split
from src.eda import Eda
from src.encoder import Encoder
 
 
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
 
    # --- ENCODING (separato su train e test per evitare data leakage) ---
    print("\n>>> ENCODING")
    cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()
 
    encoder_train = Encoder(X_train)
    X_train = encoder_train.one_hot_encode(cols=cat_cols)
 
    encoder_test = Encoder(X_test)
    X_test = encoder_test.one_hot_encode(cols=cat_cols)
 
    # Allinea le colonne del test a quelle del train
    # (riempie con 0 le eventuali categorie mancanti nel test)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
 
    print(f"Train encoded: {X_train.shape[0]} righe, {X_train.shape[1]} colonne")
    print(f"Test encoded:  {X_test.shape[0]} righe, {X_test.shape[1]} colonne")
 
 
if __name__ == "__main__":
    main()
