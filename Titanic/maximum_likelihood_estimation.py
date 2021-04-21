import numpy as np
import pandas as pd
# Data Loading
dataset = pd.read_csv('./data/Titanic.csv')
dataset['Sex'].replace(['female', 'male'], [0, 1], inplace=True)
dataset['Q'] = dataset.Embarked.apply(lambda x: 1 if x == 'Q' else 0)
dataset['S'] = dataset.Embarked.apply(lambda x: 1 if x == 'S' else 0)
dataset['C'] = dataset.Embarked.apply(lambda x: 1 if x == 'C' else 0)
Y_ = dataset['Survived']
X_ = dataset[[
'Pclass',
'Sex',
'Age',
'SibSp',
'Parch',
'Fare',
'Q',
'S',
# 'C'
]]
Y_ = Y_.to_numpy()
X_ = X_.to_numpy()
Y = Y_.astype(np.float32).reshape([-1, 1])
X = np.concatenate( [np.ones_like(Y).reshape([-1,1]), X_], axis=1
).astype(np.float32)
W = np.ones([ X.shape[1], 1 ]).astype(np.float32)
print(Y.shape, X.shape, W.shape)
# MLE
pFpW = lambda w: ( ( Y - 1./(1.+np.exp(-X.dot(w))) ) * X ).sum(axis=0)
max_iter = 100000
count_iter = 0
th = 1e-5
W_t = W.copy()
W_t1 = W_t.copy()
eta = 1e-5
dist = np.inf
while dist > th:
    W_t1 = W_t + eta * pFpW(W_t).reshape([9,1])
    dist = np.linalg.norm(W_t1-W_t)
    count_iter += 1
    if count_iter > max_iter:
        break
    W_t = W_t1
print(W_t1)

# Confidence Interval
p=1./( 1+np.exp( -X.dot(W_t1) ) )
p=p.ravel()
I=np.diag(p*(1-p))
C=np.linalg.inv(X.T.dot(I).dot(X))
CI=1.96*np.sqrt(np.diagonal(C))
print(CI)