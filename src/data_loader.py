# src/data_loader.py

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


class DataLoader:
    """
    DataLoader per dataset UCI ML Repository.

    Parametri
    ---------
    dataset_id        : ID numerico del dataset UCI
    target_col        : nome della colonna target (es. 'poisonous')
    drop_cols         : colonne da rimuovere (es. colonne costanti o ID)
    drop_missing_thresh: soglia % NaN oltre cui droppare la colonna (0.0-1.0)
    """

    def __init__(
        self,
        dataset_id: int,
        target_col: str,
        drop_cols: list = None,
        drop_missing_thresh: float = None,
    ):
        self.dataset_id = dataset_id
        self.target_col = target_col
        self.drop_cols = drop_cols or []
        self.drop_missing_thresh = drop_missing_thresh

        self.df = None

    def load(self) -> pd.DataFrame:
        dataset = fetch_ucirepo(id=self.dataset_id)

        self.df = pd.concat([
            dataset.data.features.copy(),
            dataset.data.targets.copy()
        ], axis=1)

        self.df.replace(
            to_replace=r'^\s*(\?|nan|NaN|N/A|NA|none|None|--)\s*$',
            value=np.nan,
            regex=True,
            inplace=True
        )

        if self.drop_cols:
            cols = [c for c in self.drop_cols if c in self.df.columns]
            self.df.drop(columns=cols, inplace=True)

        if self.drop_missing_thresh is not None:
            thresh = int((1 - self.drop_missing_thresh) * len(self.df))
            before = set(self.df.columns)
            self.df.dropna(thresh=thresh, axis=1, inplace=True)
            dropped = before - set(self.df.columns)
            if dropped:
                print(f"[DataLoader] Colonne droppate (>{self.drop_missing_thresh*100:.0f}% NaN): {dropped}")

        return self.df

    def info(self) -> None:
        print(f"\n{'='*45}")
        print(f"  DATASET UCI  —  id={self.dataset_id}")
        print(f"{'='*45}")
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        print(f"  Righe      : {len(self.df)}")
        print(f"  Colonne    : {len(self.df.columns)}")
        print(f"  Features   : {len(X.columns)}")

        num = X.select_dtypes(include='number').columns.tolist()
        cat = X.select_dtypes(exclude='number').columns.tolist()
        print(f"  Numeriche  : {len(num)}")
        print(f"  Categoriali: {len(cat)}")
        print(f"\n  Target: '{self.target_col}'")
        print(y.value_counts().to_string())
        print(f"{'='*45}\n")

    def report_missing(self) -> None:
        print(f"\n{'='*45}")
        print("  VALORI MANCANTI")
        print(f"{'='*45}")
        total = len(self.df)
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("  Nessun valore mancante.")
        else:
            for col, count in missing.items():
                pct = count / total * 100
                bar = '█' * int(pct / 5)
                print(f"  {col:<30} {count:>5} ({pct:5.1f}%)  {bar}")
        print(f"{'='*45}\n")
