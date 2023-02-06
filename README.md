# Concussions-and-Athletic-Performance
Investigating the effect of concussions on athletic performance in hockey players as part of Cal Poly Masters Thesis


## Implemented Thus Far

### Models Tested
    * Support Vector Machine
    * Multilayer Perceptron
    * Logistic Regression
    * Decision Tree
    * Linear Tree 
    * XGBoost 
    * CatBoost
    * Gaussian Naive Bayes

____
### Feature Scaling
* Standard scaling

____
### Dataset Balancing
Eval: *Stratified KFold, n=5, test_size=.2*

* SMOTE
* ADASYN
* Random Over Sampling
* KNNOR

Results:
____
### Feature Selection
Eval: *Stratified KFold, n=5, test_size=.2*
* Drop correlated features w/ Pearson Correlation
* Importance Ranking via ExtraTreesClassifier
* Sequential Feature Selection
    * Brute force floating forward selection
    * 5-fold CV
    * k_features 1-15
    * Feature Selector Class
    * Parallelized


Play around with SVC, and Logistic regression
Regularization <- c. Box parameter

Parameter Tuning: 
Focus on Boosting, Regression, Logistic, Decision, SVC