from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

class Model:
    def __init__(self, model_type='random_forest', **kwargs):
        self.model_type = model_type
        self.model = self._initialize_model(**kwargs)

    def _initialize_model(self, **kwargs):
        if self.model_type == 'random_forest':
            return RandomForestClassifier(**kwargs)
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(**kwargs)
        elif self.model_type == 'svm':
            return SVC(**kwargs)
        elif self.model_type == 'decision_tree':
            return DecisionTreeClassifier(**kwargs)
        else:
            raise ValueError(f"Model type '{self.model_type}' is not supported.")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)