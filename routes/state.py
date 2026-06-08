import threading

pipeline = {
    'loader':   None,
    'cleaner':  None,
    'encoder':  None,
    'logreg':   None,
    'xgboost':  None,
    'X_train':  None,
    'X_test':   None,
    'y_train':  None,
    'y_test':   None,
}

pipeline_lock = threading.Lock()
