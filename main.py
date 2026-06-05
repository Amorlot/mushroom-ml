# main.py

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner


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

    # --- PULIZIA ---
    print("\n>>> PULIZIA DATI")
    cleaner = DataCleaner(dl.df)
    cleaner.fix_missing_categorical_unknown(cols=['stalk-root'])
    cleaner.report()

    df_clean = cleaner.get_clean_df()
    print(f"Dataset pulito: {df_clean.shape[0]} righe, {df_clean.shape[1]} colonne")


if __name__ == "__main__":
    main()