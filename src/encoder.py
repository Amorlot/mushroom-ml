import pandas as pd


class Encoder:
    def __init__(self , df):
        self.df = df
        pass

    def one_hot_encode(self, cols):
        self.df = pd.get_dummies(self.df, columns=cols, drop_first=True)
        return self.df