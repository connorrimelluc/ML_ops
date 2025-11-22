class PreprocessAndAutoML:
    """Wrapper that applies sklearn preprocessor + FLAML AutoML."""
    def __init__(self, preprocessor, automl):
        self.preprocessor = preprocessor
        self.automl = automl

    def predict(self, X):
        X_proc = self.preprocessor.transform(X)
        return self.automl.predict(X_proc)