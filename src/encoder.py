import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class Encoder:
    def __init__(self):
        self.le = LabelEncoder()
        self._ohe = None
        self._cat_cols = None

    def fit_transform_features(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Fit OHE sul train e restituisce il DataFrame encoded."""
        self._cat_cols = cols
        self._ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
        encoded = self._ohe.fit_transform(df[cols])
        feature_names = self._ohe.get_feature_names_out(cols)
        num_df = df.drop(columns=cols)
        return pd.concat(
            [num_df, pd.DataFrame(encoded, columns=feature_names, index=df.index)],
            axis=1
        )

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trasforma il test usando l'OHE già fittato sul train."""
        encoded = self._ohe.transform(df[self._cat_cols])
        feature_names = self._ohe.get_feature_names_out(self._cat_cols)
        num_df = df.drop(columns=self._cat_cols)
        return pd.concat(
            [num_df, pd.DataFrame(encoded, columns=feature_names, index=df.index)],
            axis=1
        )

    def encode_target(self, y, fit: bool = False):
        """Label encoding: fit=True su y_train, fit=False su y_test."""
        if fit:
            return self.le.fit_transform(y)
        return self.le.transform(y)
