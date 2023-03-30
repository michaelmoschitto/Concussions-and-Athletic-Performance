from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import pandas as pd
import numpy as np
# map column indices to col name
def feature_num_to_name(indicies, cols):
    return list(map(lambda i: cols[i], indicies))

class FeatureSelector:
    def __init__(self, estimator, selection_type, floating, scoring, k_features, cv) -> None:
        self.estimator = estimator
        self.selection_type = selection_type
        self.selection_type = selection_type
        self.floating = floating
        self.scoring = scoring
        self.k_features = k_features
        self.cv = cv

        self.selector = None
        self.results = None
        self.estimator_name = type(estimator).__name__
       
        self.X = None
        self.y = None



    def fit(self, X, y):
        self.X = X                               
        self.y = y                               

        sfs1 = SequentialFeatureSelector(estimator=self.estimator, 
                k_features=self.k_features,
                forward=self.selection_type, 
                floating=self.floating, 
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=-1,
                verbose=-1)

        # pipe = make_pipeline(StandardScaler(), sfs1)

        # sfs1 = pipe.fit(X, y)[1]
        pipe = make_pipeline(sfs1)

        sfs1 = sfs1.fit(X, y)

        try:
            self.subsets = sfs1.subsets_
        except:
            self.subsets = sfs1[0].subsets_

    
        self.selector = sfs1

        # print('best combination (ACC: %.3f): %s\n' % (sfs1.k_score_, feature_num_to_name(sfs1.k_feature_idx_, X_train)), end="\n\n")
        # print('all subsets:\n', sfs1.subsets_)
        # plot_sfs(sfs1.get_metric_dict(), kind='std_err')

        return self

    def get_results(self, X_cols):
        scores = []
        features = []
        confidences = []
        indicies = []
        for run_num in self.subsets.keys():
            run = self.subsets[run_num]
            feature_indicies = run["feature_idx"]
            feature_names = feature_num_to_name(feature_indicies, X_cols)
            score = run["avg_score"]

            indicies.append(feature_indicies)
            confidences.append(self.selector.get_metric_dict()[int(run_num)]["ci_bound"])
            scores.append(score)
            features.append(tuple(list(map(lambda x : "'" + str(x) + "'", feature_names))))

        # print(self.selector.get_metric_dict())
        # print("features: ", features)
            
        result = pd.DataFrame({"features" : features, "score" : scores,"ci bound" : confidences ,"feature idx" : indicies,"score times ci" : np.multiply(scores, confidences),"model" : [self.estimator_name] * len(features)}).sort_values(by="score", ascending=False)
        self.feature_idx = list(result["feature idx"].iloc[0])
        return result

    def plot_results(self):
        plot_sfs(self.selector.get_metric_dict(), kind='std_err')

    def get_metric_dict(self):
        return pd.DataFrame.from_dict(self.selector.get_metric_dict()).T.sort_values(by="avg_score", ascending=False)

    def get_params(self):
        return self.selector.get_params(deep=True)

    def get_named_estimators(self):
        return self.selector.named_estimators

    def get_feature_idx(self):
        return self.feature_idx




