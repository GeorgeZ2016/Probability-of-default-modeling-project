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
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import random
from sklearn.utils import shuffle
# EDA
data=pd.read_csv('E:/project.csv')
sum(data['loss_flag']==1)
df=data.dropna()
sum(df['loss_flag']==1)
sum(df['loss_flag']==0)

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
sum(y_train==1)
sum(y_test==1)

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
#为什么说模型不好(Why the model is bad)
pipe_lr.score(X_test[y_test==1], y_test[y_test==1])
pipe_lr.score(X_test[y_test==0], y_test[y_test==0])


# confusion matrix
ypre=pipe_lr.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=ypre)

## Second pipeline
x1=x[:,[0,2,3,4,5]]
X_train1, X_test1, y_train1, y_test1 = train_test_split(x1, y, test_size=0.3, random_state=0)
pipe_lr1 = Pipeline([('scl', StandardScaler()),
('clf', LogisticRegression(random_state=0))])

# single step
m=LogisticRegression(random_state=0)
m.fit(X_train1, y_train1)
y_pred = m.predict(X_test1)
print('Accuracy: %.8f' % accuracy_score(y_test1, y_pred))

# K-fold CV
kfold1 = StratifiedKFold(y=y_train1,n_folds=10,random_state=1)
scores1=[]
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

# Adaboost
x_pca=X_train_pca[:,0:5]
xtrada,xtestada,ytrada,ytestada=train_test_split(x_pca, y, test_size=0.3, random_state=0)
m_ori=DecisionTreeClassifier(criterion='entropy',max_depth=5)
ada = AdaBoostClassifier(base_estimator=m_ori,n_estimators=200,learning_rate=1,
                         random_state=0)
ada = ada.fit(xtrada, ytrada)
ada_predict=ada.predict(xtestada)
print('Test Accuracy: %.6f' % accuracy_score(ytestada, ada_predict))
print('Test Accuracy: %.6f' % accuracy_score(ytestada[ytestada==0], ada_predict[ytestada==0]))
print('Test Accuracy: %.6f' % accuracy_score(ytestada[ytestada==1], ada_predict[ytestada==1]))

# Oversampling
x_pca=X_train_pca[:,0:5]
xtrpca,xtestpca,ytrpca,ytespca=train_test_split(x_pca, y, test_size=0.3, random_state=0)
length=0.7*len(x)
x_maj=xtrpca[ytrpca==0]
x_min=xtrpca[ytrpca==1]
y_maj,y_min=ytrpca[ytrpca==0],ytrpca[ytrpca==1]
lx=[]
ly=[]
for i in range(int(length*0.7)):
    r=random.randint(0,len(x_maj-1))
    lx.append(list(x_maj[r]))
    ly.append([y_maj[r]])
for i in range(int(length*0.3)):
    r=random.randint(0,len(x_min)-1)
    lx.append(list(x_min[r]))
    ly.append([y_min[r]])
x3=np.array(lx)
y3=np.array(ly)
x3,y3=shuffle(x3,y3,random_state=0)
# Model1:Logistic
m3=LogisticRegression(random_state=0)
m3.fit(x3, y3)
y3_pred = m3.predict(xtestpca)
print('Test Accuracy: %.6f' % accuracy_score(ytespca, y3_pred))
print('Test Accuracy: %.6f' % accuracy_score(ytespca[ytespca==0], y3_pred[ytespca==0]))
print('Test Accuracy: %.6f' % accuracy_score(ytespca[ytespca==1], y3_pred[ytespca==1]))

# Model2:Adaboost
m_ori=DecisionTreeClassifier(criterion='entropy',max_depth=5)
ada = AdaBoostClassifier(base_estimator=m_ori,n_estimators=50,learning_rate=0.01,
                         random_state=0)
ada = ada.fit(x3, y3)
ada_pred=ada.predict(xtestpca)
print('Ada Test Accuracy: %.6f' % accuracy_score(ytespca, ada_pred))
print('Ada Test Accuracy: %.6f' % accuracy_score(ytespca[ytespca==0], ada_pred[ytespca==0]))
print('Ada Test Accuracy: %.6f' % accuracy_score(ytespca[ytespca==1], ada_pred[ytespca==1]))
