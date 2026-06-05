import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
 
 
class DataCleaner:
    """
    DataCleaner con pattern fit / transform.
    Casistiche supportate:
      - fix_missing_numerical_median   → imputa numerici con la mediana del train
      - fix_missing_categorical_unknown → sostituisce NaN con 'unknown'
    """
 
    def __init__(self):
        self._use_median = False
        self._median_imputer = None   # SimpleImputer fittato sul train
        self._median_cols = []        # colonne numeriche con mancanti
        self._unknown_cols = []       # colonne categoriali da riempire con 'unknown'
 
    # ------------------------------------------------------------------ #
    #  Configurazione (chainable)                                          #
    # ------------------------------------------------------------------ #
 
    def fix_missing_numerical_median(self) -> "DataCleaner":
        """Registra: imputa i numerici con la mediana del train."""
        self._use_median = True
        return self
 
    def fix_missing_categorical_unknown(self, cols: list = None) -> "DataCleaner":
        """
        Registra: sostituisce NaN con 'unknown'.
        Se cols=None lo applica a tutte le categoriali con mancanti.
        """
        self._unknown_cols = cols if cols is not None else []
        return self
 
    # ------------------------------------------------------------------ #
    #  Fit / Transform                                                     #
    # ------------------------------------------------------------------ #
 
    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """Impara le statistiche SOLO dal train set."""
 
        # Mediana numerica
        if self._use_median:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self._median_cols = [c for c in num_cols if df[c].isnull().any()]
            if self._median_cols:
                self._median_imputer = SimpleImputer(strategy='median')
                self._median_imputer.fit(df[self._median_cols])
                print("\n--- FIT NUMERICI → MEDIANA ---")
                for i, col in enumerate(self._median_cols):
                    print(f"  {col:<30} mediana = {self._median_imputer.statistics_[i]:.2f}")
 
        # 'unknown' è costante, nessun fit necessario
        # se cols=None, ricaviamo le colonne dal train
        if not self._unknown_cols:
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            self._unknown_cols = [c for c in cat_cols if df[c].isnull().any()]
 
        return self
 
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica le trasformazioni apprese. Usabile su train e test."""
        df = df.copy()
 
        if self._median_imputer and self._median_cols:
            df[self._median_cols] = self._median_imputer.transform(df[self._median_cols])
 
        for col in self._unknown_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
 
        return df
 
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit + transform in un colpo solo — usa solo su X_train."""
        return self.fit(df).transform(df)
 
    # ------------------------------------------------------------------ #
    #  Report                                                              #
    # ------------------------------------------------------------------ #
 
    def report(self, df: pd.DataFrame) -> None:
        print(f"\n{'='*45}")
        print("  STATO VALORI MANCANTI")
        print(f"{'='*45}")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("  Nessun valore mancante rimasto.")
        else:
            for col, count in missing.items():
                pct = count / len(df) * 100
                print(f"  {col:<30} {count:>5} ({pct:.1f}%)")
        print(f"{'='*45}\n")
