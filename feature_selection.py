from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import pandas as pd

# map column indices to col name
def feature_num_to_name(indicies, df):
    cols = df.columns
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
                n_jobs=-1)

        pipe = make_pipeline(StandardScaler(), sfs1)

        pipe.fit(X, y)


        try:
            self.subsets = sfs1.subsets_
        except:
            self.subsets = sfs1[0].subsets_

        self.selector = sfs1

        # print('best combination (ACC: %.3f): %s\n' % (sfs1.k_score_, feature_num_to_name(sfs1.k_feature_idx_, X_train)), end="\n\n")
        # print('all subsets:\n', sfs1.subsets_)
        # plot_sfs(sfs1.get_metric_dict(), kind='std_err')

        return self

    def get_results(self):
        scores = []
        features = []
        for run_num in self.subsets.keys():
            run = self.subsets[run_num]
            feature_indicies = run["feature_idx"]
            feature_names = feature_num_to_name(feature_indicies, self.X)
            score = run["avg_score"]

            scores.append(score)
            features.append(tuple(feature_names))
            
        return pd.DataFrame({"features" : features, "score" : scores, "model" : [self.estimator_name] * len(features)}).sort_values(by="score", ascending=False)

    def plot_results(self):
        plot_sfs(self.selector.get_metric_dict(), kind='std_err')



