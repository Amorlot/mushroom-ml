# src/data_cleaner.py

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


class DataCleaner:
    """
    DataCleaner modulare — chiama solo le funzioni che servono.
    Aggiungi nuove funzioni man mano che incontri nuovi casi.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ------------------------------------------------------------------ #
    #  Numerici                                                            #
    # ------------------------------------------------------------------ #

    def fix_missing_numerical_median(self) -> "DataCleaner":
        """Imputa i numerici con la mediana."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        missing = [c for c in num_cols if self.df[c].isnull().any()]

        if not missing:
            print("[DataCleaner] Nessun mancante numerico.")
            return self

        imputer = SimpleImputer(strategy='median')
        before = self.df[missing].isnull().sum()
        self.df[missing] = imputer.fit_transform(self.df[missing])

        print("\n--- NUMERICI → MEDIANA ---")
        for i, col in enumerate(missing):
            print(f"  {col:<30} {before[col]:>5} mancanti → mediana ({imputer.statistics_[i]:.2f})")
        return self

    # ------------------------------------------------------------------ #
    #  Categoriali                                                         #
    # ------------------------------------------------------------------ #

    def fix_missing_categorical_mode(self) -> "DataCleaner":
        """Imputa i categoriali con la moda (best per <5% mancanti)."""
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        missing = [c for c in cat_cols if self.df[c].isnull().any()]

        if not missing:
            print("[DataCleaner] Nessun mancante categoriale.")
            return self

        imputer = SimpleImputer(strategy='most_frequent')
        before = self.df[missing].isnull().sum()
        self.df[missing] = imputer.fit_transform(self.df[missing])

        print("\n--- CATEGORIALI → MODA ---")
        for i, col in enumerate(missing):
            print(f"  {col:<30} {before[col]:>5} mancanti → moda ('{imputer.statistics_[i]}')")
        return self

    def fix_missing_categorical_unknown(self, cols: list = None) -> "DataCleaner":
        """
        Sostituisce i NaN con 'unknown' (best per >5% mancanti).
        Preserva l'informazione che il valore mancava — può essere predittivo.
        Se cols=None lo applica a tutte le categoriali con mancanti.
        """
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        target_cols = cols if cols else [c for c in cat_cols if self.df[c].isnull().any()]

        print("\n--- CATEGORIALI → 'unknown' ---")
        for col in target_cols:
            count = self.df[col].isnull().sum()
            pct = count / len(self.df) * 100
            self.df[col] = self.df[col].fillna('unknown')
            print(f"  {col:<30} {count:>5} mancanti ({pct:.1f}%) → 'unknown'")
        return self

    # ------------------------------------------------------------------ #
    #  Report                                                              #
    # ------------------------------------------------------------------ #

    def report(self) -> None:
        print(f"\n{'='*45}")
        print("  STATO VALORI MANCANTI")
        print(f"{'='*45}")
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("  Nessun valore mancante rimasto.")
        else:
            for col, count in missing.items():
                pct = count / len(self.df) * 100
                print(f"  {col:<30} {count:>5} ({pct:.1f}%)")
        print(f"{'='*45}\n")

    def get_clean_df(self) -> pd.DataFrame:
        return self.df