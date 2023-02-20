from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import pickle
import uuid
import ast
from sklearn.preprocessing import StandardScaler
# map column indices to col name
def feature_num_to_name(indicies, df):
    cols = df.columns
    return list(map(lambda i: cols[i], indicies))



class Tuner: 
    def __init__(self, estimator, hyperparams, cv, best_features=None):
        self.estimator = estimator
        self.hyperparams = hyperparams
        self.estimator_name = type(estimator).__name__
        self.cv = cv
        self.best_features = best_features

        self.results = None
        self.X = None
        self.y = None


        self._best_params = None
        self._best_estimator = None




    def get_best_features(self):
        aggregate_best_features_df = pd.read_excel("aggregate_results/aggregate_best_features.xlsx", engine="openpyxl")
        best_features = ast.literal_eval(aggregate_best_features_df.loc[aggregate_best_features_df["model"] == self.estimator_name, "features"].values[0])

        return best_features

    def tune(self, X, y):
        
        # best_features = self.get_best_features()
        # print(best_features)
        # X = X[self.best_features]

        self.X = X                               
        self.y = y         
        scaler = StandardScaler()
        X = scaler.fit_transform(X)                      
        self.grid = GridSearchCV(self.estimator, param_grid=self.hyperparams, cv=self.cv, verbose=1, refit=True)
        self.grid.fit(X, y)
        self._best_params = {self.estimator_name: self.grid.best_params_}
        with open(f"models/{self.estimator_name}_{round(self.get_best_score(), 4)}_{y.name}_{uuid.uuid4()}.pkl", 'wb') as f:
            pickle.dump(self.get_best_estimator(), f)


        

    def get_results(self):

        return pd.DataFrame(self.grid.cv_results_)

    def get_best_params(self):
        return self._best_params

    def get_best_estimator(self):
        return self.grid.best_estimator_

    def get_best_score(self):
        return self.grid.best_score_
        