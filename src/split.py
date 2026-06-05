
from sklearn.model_selection import train_test_split
 
 
class Split:
    def __init__(self):
        pass
 
    def split(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        print(f"\n--- SPLIT 80/20 ---")
        print(f"Train: {X_train.shape[0]} righe")
        print(f"Test:  {X_test.shape[0]} righe")
        return X_train, X_test, y_train, y_test
 
