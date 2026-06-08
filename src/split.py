from sklearn.model_selection import train_test_split


class Split:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        print(f"\n--- SPLIT {int((1-self.test_size)*100)}/{int(self.test_size*100)} ---")
        print(f"Train: {X_train.shape[0]} righe")
        print(f"Test:  {X_test.shape[0]} righe")
        return X_train, X_test, y_train, y_test
