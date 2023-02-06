import pandas as pd
import numpy as np

# utility
from data_cleaning import clean_raw_data, create_dataset

# parallel
import ray
try:
    ray.init()
except:
    print("ray already started")

# feature selection / preprocessing
from sklearn.model_selection import StratifiedShuffleSplit


# models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from parallel_train import Trainer



df = clean_raw_data(filename="./Implementation/Brdi_db_march.xlsx")
X, y = create_dataset(df, target_col="NHL")
kf = list(StratifiedShuffleSplit(test_size=.2, n_splits=5, random_state=0).split(X, y))
models = [LogisticRegression(), MLPClassifier(), XGBClassifier()]
kwargs = {"selection_type":"forward", "floating":True, "scoring":"f1", "k_features":len(X.iloc[0]), "cv":kf}

trainer = Trainer(X, y, models)
trainer.train(how="feature_selection", kwargs=kwargs)

output = trainer.get_results(filename="LR_MLP_XGB_All")