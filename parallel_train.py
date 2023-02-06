import uuid
import ray 
import pandas as pd
from feature_selection import FeatureSelector
import json
from time import sleep
import numpy as np

@ray.remote
def parrallell_feature_selection(X, y, estimator, kwargs):
    """
    This fuction does forward and backward feature selection on the given estimator.
    """

    # self.training_args = dict(kwargs)
    ftsl = FeatureSelector(estimator, **kwargs)
    ftsl.fit(X, y)
    results = ftsl.get_results()
    return results


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)    

class Trainer:
    def __init__(self, X, y, models):
        self.X = X
        self.y = y
        self.models = models
        self.result_ids = []
        self.output_df = pd.DataFrame()


    def train(self, how="feature_selection", kwargs=None):
        result_ids = []
        self.training_args = kwargs

        for model in self.models:
            result_ids.append(parrallell_feature_selection.remote(self.X, self.y, model, kwargs))

        self.result_ids = result_ids


    def write_logs(self, id):
        dir = "./training_logs"

        with open(f'{dir}/{id}.json', 'w') as fp:
            json.dump(self.training_args, cls=NumpyEncoder, fp=fp)



    def get_results(self, filename=""):
        """
        This function collects the results from training and returns them as a pandas dataframe.
        """

        all_parallel_results = pd.DataFrame()
        for r in ray.get(self.result_ids):
            res_df = r
            all_parallel_results = pd.concat([all_parallel_results, res_df])

        self.output_df  = all_parallel_results.sort_values(by="score", ascending=False)

        training_id = uuid.uuid4()
        self.write_logs(str(training_id))

        if filename != "":
            self.output_df.to_excel(f"./training_output/{filename}_{training_id}.xlsx", index=False)
        

        return self.output_df


