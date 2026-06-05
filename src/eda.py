import math
import os
import matplotlib.pyplot as plt
import numpy as np  
from sklearn.linear_model import LinearRegression  
import pandas as pd  
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, normaltest


class Eda:
    def __init__(self, df, output_dir='output_plots'):
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def info(self):
        describe = self.df.describe()
        shape = self.df.shape
        colonne = self.df.columns
        return {
            'describe': describe.to_dict(),
            'n_righe, n_colonne': shape,
            'variabili': colonne.to_list()
        }

    def qqplot(self, colonna):
        plt.figure(figsize=(8, 6))
        stats.probplot(self.df[colonna].dropna(), dist="norm", plot=plt)
        plt.title(f'QQ Plot - {colonna}')
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/qqplot_{colonna}.png')
        plt.close()

    def boxplot(self, colonna):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df[colonna])
        plt.title('Boxplot')
        plt.xticks(rotation=45)
        plt.savefig(f'{self.output_dir}/boxplot_{colonna}.png')
        plt.close()

    def jarque_bera_test(self, colonna):
        stat, pvalue = jarque_bera(self.df[colonna].dropna())
        return {
            "test": "Jarque-Bera",
            "colonna": colonna,
            "statistica": float(round(stat, 4)),
            "pvalue": float(round(pvalue, 10)),
            "normale": bool(pvalue > 0.05)
        }

    def normal_test(self, colonna):
        data = self.df[colonna].dropna()
        stat, pvalue = normaltest(data)
        return {
            "test": "D'Agostino-Pearson",
            "colonna": colonna,
            "statistica": float(stat),
            "pvalue": float(pvalue),
            "normale": bool(pvalue > 0.05)
        }

    def plot_histograms(self, bins=30, figsize=(15, 10)):
        numeric_df = self.df.select_dtypes(include='number')
        n_cols = numeric_df.shape[1]
        if n_cols == 0:
            print("Nessuna variabile numerica trovata.")
            return
        n_cols_grid = math.ceil(math.sqrt(n_cols))
        n_rows_grid = math.ceil(n_cols / n_cols_grid)
        fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        for i, col in enumerate(numeric_df.columns):
            axes[i].hist(numeric_df[col].dropna(), bins=bins, edgecolor='black')
            axes[i].set_title(col)
        for j in range(i + 1, n_rows_grid * n_cols_grid):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/histograms.png')
        plt.close()

    def find_outliers(self, colonna):
        q1 = self.df[colonna].quantile(0.25)
        q3 = self.df[colonna].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = self.df[
            (self.df[colonna] < lower_bound) |
            (self.df[colonna] > upper_bound)
            ]
        return outliers.index.tolist()

    def corrplot(self, figsize=(12, 10)):

        corr_matrix = self.df.corr(numeric_only=True)

        plt.figure(figsize=figsize)

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )

        plt.title("Correlation Matrix", fontsize=16)
        plt.tight_layout()
        plt.show()

    def tolerance(self):

        numeric_df = self.df.select_dtypes(include=np.number)

        results = []

        for col in numeric_df.columns:
            X = numeric_df.drop(columns=[col])
            y = numeric_df[col]

            model = LinearRegression()
            model.fit(X, y)

            r2 = model.score(X, y)

            tol = 1 - r2

            vif = np.inf if tol == 0 else 1 / tol

            results.append({
                "Variable": col,
                "Tolerance": round(tol, 4),
                "VIF": round(vif, 2)
            })

        return pd.DataFrame(results).sort_values("VIF", ascending=False)

