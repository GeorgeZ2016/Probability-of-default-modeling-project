# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 19:20:11 2016

@author: George
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

# Data
data=pd.read_csv('E:/project.csv')
df=data.dropna()
df=df.as_matrix()
y=df[:, -1]
x=df[:,:-1]
# Get number of principle component:5
stdsc = StandardScaler()
x1= stdsc.fit_transform(x)
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(x1)
pca.explained_variance_ratio_

## first pipeline
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
pipe_lr = Pipeline([('scl', StandardScaler()),
('pca', PCA(n_components=1)),
('clf', LogisticRegression(random_state=1))])
# K-fold CV
kfold = StratifiedKFold(y=y_train,n_folds=10,random_state=1)
scores=[]
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
# another way
scores = cross_val_score(estimator=pipe_lr,X=X_train,y=y_train,cv=10,n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.8f +/- %.8f' % (np.mean(scores),np.std(scores)))
# modeling
pipe_lr.fit(X_train, y_train)
print('Test Accuracy: %.6f' % pipe_lr.score(X_test, y_test))

## Second pipeline
x1=x[:,[0,2,3,4,5]]
X_train1, X_test1, y_train1, y_test1 = train_test_split(x1, y, test_size=0.3, random_state=0)
pipe_lr1 = Pipeline([('scl', StandardScaler()),
('clf', LogisticRegression(random_state=0))])

# Tuning hyperparameter.The solution is 0.0001
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range}]
gs = GridSearchCV(estimator=pipe_lr1,param_grid=param_grid,scoring='accuracy',
cv=10,n_jobs=-1)
gs = gs.fit(X_train1, y_train1)
print(gs.best_score_)
print(gs.best_params_)

# single step, accuracy score
m=LogisticRegression(random_state=0)
m.fit(X_train1, y_train1)
y_pred = m.predict(X_test1)
print('Accuracy: %.8f' % accuracy_score(y_test1, y_pred))

# K-fold CV
kfold1 = StratifiedKFold(y=y_train1,n_folds=10,random_state=1)
scores1=[]
pipe_lr1 = Pipeline([('scl', StandardScaler()),
('clf', LogisticRegression(C=10**(-5),random_state=0))])
for k, (train, test) in enumerate(kfold1):
    pipe_lr1.fit(X_train1[train], y_train1[train])
    score1 = pipe_lr1.score(X_train1[test], y_train1[test])
    scores1.append(score1)
print('CV accuracy: %.8f +/- %.8f' % (np.mean(scores1), np.std(scores1)))
# another way
scores1 = cross_val_score(estimator=pipe_lr1,X=X_train1,y=y_train1,cv=10,n_jobs=1)
print('CV accuracy scores: %s' % scores1)
print('CV accuracy: %.8f +/- %.8f' % (np.mean(scores),np.std(scores1)))
# modeling
pipe_lr1.fit(X_train1, y_train1)
print('Test Accuracy: %.6f' % pipe_lr1.score(X_test1, y_test1))



