# The Effects of Concussion and Visuomotor Metrics on NHL Performance: an Explainable AI Approach - RQ2

This repository contains code to answer the first research question of this thesis, are visuomotor metrics and concussion history indicators of athletic performance?

## Installation

To create a virtual environment, install all needed packages (requirements.txt) via pip, and create a jupyter kernel from the virtual environment use the following shell script `install_kernel.sh`. 


Running `install_kernel.sh`
```bash
chmod +x install_kernel.sh
./install_kernel.sh
```
## Models Implemented
Tuning using Random Search and Bayesian Optimization has been implemented for the following models:

* SVC
* LGBM
* Elastic Net
* Decision Tree
* Random Forest
* PyTorch DNN
* Logistic Regression 
* XGB
* Linear Tree
* Linear Boosting Tree

### Demo
A simplified demo of the entire workflow can be seen in ./src/Demo.ipynb. This notebook contains all the code to import packages, parallel / sequential training, train 1 model (Decision Tree), and collect results. 

## Model Tuning

### Bayesian Optimization
```python
# Tune a Decision Tree using Optuna Bayesian Optimization 

# model 
model = DecisionTreeClassifier(random_state=SEED)

# feature scaler, feature selector, and hyperparameters
param_dist = {
        'scaler' : CustomDistribution([myStandardScaler]),
        "selector" : CustomDistribution([RandomUnderSampler(random_state=42)]),
        "model__criterion" : optuna.distributions.CategoricalDistribution(["gini"]),
        "model__max_depth" : optuna.distributions.IntDistribution(2, 8),
        "model__min_samples_split" : optuna.distributions.IntDistribution(8, 15),
        "model__min_samples_leaf" : optuna.distributions.IntDistribution(8, 15),
        "model__max_features" : optuna.distributions.CategoricalDistribution(["sqrt"]),
        "model__class_weight" : optuna.distributions.CategoricalDistribution(["balanced"]),
    }

# metadata, train, validation results
dt_est_tpe, dt_train_tpe, dt_val_tpe = outer_tune_loop(model=model, param_dist=param_dist,scoring="F1_weighted" ,n_iter=100, target_col="previous_concussions", parallel=False, search_type="tpe")
```

### Random Search
```python
Tune a Decision Tree using Sklearn RandomSearch 

# model
model = DecisionTreeClassifier(random_state=SEED)

# feature scaler, feature selector, and hyperparameters
param_dist = {
        'scaler' : [myStandardScaler],
        "selector" : [ 
            NamedFeatureSelector(["Delta_MT", "Delta_AE", "cvRT_HR", "Corrective_HR", "RT_V", "age", "NHL"]),
        ],
        "model__criterion" : ["gini"],
        "model__max_depth" : [4, 5],
        "model__min_samples_split" : [13],
        "model__min_samples_leaf" : [11],
        "model__max_features" : ["sqrt"],
        "model__class_weight" : ["balanced"],
    }

# metadata, train, validation results
dt_est, dt_train, dt_val = outer_tune_loop(model=model, param_dist=param_dist,scoring="F1_weighted" ,n_iter=100,target_col="previous_concussions", parallel=True)
```
### Adjustable Training Attributes
* Model (PyTorch / Sklearn)
* Hyperparameter Search Algorithm (Random Search / Bayesian Optimization using TPE)
* Pipeline steps (scaler, selector, model)
* Parallel vs Sequential Training
* Maximum Training iterations
* Scoring (F1, Precision, Recall, Accuracy, Weighted F1, ...)
* Model Hyperparameters 

## Results and Latex Table Creation
1. Individual Model Training/Validation Results: ./results
2. Aggregated Results for All Models: ./mean_results

All training and validation results for all models are saved to /results. In order to streamline reporting all Python DataFrames and Latex Tables for training and validation can be automatically generated using code at the end of the notebook. 

## Visuals 
As this research prioritizes explainability multiple types of visuals are supported. 

* Permutation Feature Importance
* Feature Correlation Heatmap (/scratch/NHL.ipynb)
* Class Balance
* SVC Decision Boundary 
* Graphic Decision Tree 
* One-way ANOVA for Validation Distributions
* Confusion Matrices 

## Scratch Work 
A lot of code considered as "scratch" that wasn't used in the final paper may remain as valuable for future work. This is contained in the /scratch directory and includes features balancing techniques (Adasyn, SMOTE, KNNOR, RandomOversamping), feature correlations, sequential feature selection, other models etc..
