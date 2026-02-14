class BasePredictor:
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        raise NotImplementedError
    def predict_proba(self, X, lengths=None, **kwargs):
        raise NotImplementedError
