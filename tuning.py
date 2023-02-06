from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

# map column indices to col name
def feature_num_to_name(indicies, df):
    cols = df.columns
    return list(map(lambda i: cols[i], indicies))

class Tuner: 
    def __init__(self, estimator, hyperparams, cv):
        self.estimator = estimator
        self.hyperparams = hyperparams
        self.estimator_name = type(estimator).__name__
        self.cv = cv

        self.results = None
        self.X = None
        self.y = None

        self._best_params = None
        self._best_estimator = None


    def tune(self, X, y):
        self.X = X                               
        self.y = y                               
        
        self.grid = GridSearchCV(self.estimator, param_grid=self.hyperparams, cv=self.cv, n_jobs=-1, verbose=1)
        self.grid.fit(X, y)
        self._best_params = {self.estimator_name: self.grid.best_params_}

    def get_results(self):

        return pd.DataFrame(self.grid.cv_results_)

    def get_best_params(self):
        return self._best_params

    def get_best_estimator(self):
        return self.grid.best_estimator_

    def get_best_score(self):
        return self.grid.best_score_
        