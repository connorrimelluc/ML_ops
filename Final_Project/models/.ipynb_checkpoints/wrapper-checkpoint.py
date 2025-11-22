from sklearn.base import BaseEstimator

class PreprocessAndAutoML(BaseEstimator):
    """
    Shim class to make joblib unpickling work.
    The *real* class will be patched in after loading.
    """
    def __init__(self, preprocessor=None, automl=None):
        self.preprocessor = preprocessor
        self.automl = automl

    def predict(self, X):
        X_proc = self.preprocessor.transform(X)
        return self.automl.predict(X_proc)