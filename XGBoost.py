# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:04:47 2020

@author: Xiangqi
"""

data_path = 'C:/Users/Xiangqi/Desktop/Singapore Modules Folders/is5152/group project/preprocessing/'
import time
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.decomposition import PCA

k=12 # k-fold cross validation
prefix = 'dropna_'
if_pca = True

error = [0.]*k
totaltime = [0.]*k

param={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':7,
    'lambda':1.3,
    'colsample_bytree':0.5,
    'eta': 0.1,
    'seed':719,
    'nthread':7,
     'silent':1}

if if_pca:
    pca = joblib.load(data_path + 'pca90.pkl') 
start = time.time()
eval_dict = {'auc':[], 'acc':[], 'recall':[], 'f1_score':[], 'precision':[]}
"""1. start kfold evaludation """
for fold in range(k):
    df_train = pd.read_csv(data_path + prefix + 'train_' + str(fold + 1) + '.csv')
    df_val = pd.read_csv(data_path + prefix + 'validation_' + str(fold + 1) + '.csv')
    """ 2.1 reset labels"""
    df_train['label'].replace(-1, 0, inplace = True)
    df_val['label'].replace(-1, 0, inplace = True)
    labels_train = np.array(df_train['label'])
    labels_val = np.array(df_val['label'])
    """ 2.2 transform dataframe to matrix """
    df_train = df_train.iloc[:, :-1]
    df_val = df_val.iloc[:, :-1]
    if if_pca:
        df_train=pd.DataFrame(pca.fit_transform(df_train))
        df_val=pd.DataFrame(pca.fit_transform(df_val))
    dtrain = df_train.as_matrix()
    dtrain = xgb.DMatrix(dtrain, label=labels_train)
    dval = df_val.as_matrix()
    dval = xgb.DMatrix(dval, label=labels_val)
    """ 2.3 train xgboost model """
    time_start=time.time()
    evallist = [(dtrain,'train')]
    num_round = 30
    bst = xgb.train(param, dtrain, num_round, evallist)
    """ 2.4 predict test labels """
    val_pred = bst.predict(dval)
    val_pred_binary = (val_pred >= 0.5)*1
    eval_dict['auc'].append(metrics.roc_auc_score(labels_val,val_pred))
    eval_dict['acc'].append(metrics.accuracy_score(labels_val,val_pred_binary))
    eval_dict['recall'].append(metrics.recall_score(labels_val,val_pred_binary))
    eval_dict['f1_score'].append(metrics.f1_score(labels_val,val_pred_binary))
    eval_dict['precision'].append(metrics.precision_score(labels_val,val_pred_binary))
    print('AUC: ' + str(eval_dict['auc'][-1]))
    print('ACC: ' + str(eval_dict['acc'][-1]))
    print('Recall: ' + str(eval_dict['recall'][-1]))
    print('F1-score: ' + str(eval_dict['f1_score'][-1]))
    print('Precesion: ' + str(eval_dict['precision'][-1]))

    del dtrain, df_train
    del dval, df_val

print("validation auc mean:",np.mean(eval_dict['auc']))
print("validation acc mean:",np.mean(eval_dict['acc']))
print("validation acc std:",np.std(eval_dict['acc']))
print("validation recall mean:",np.mean(eval_dict['recall']))
print("validation f1_score mean:",np.mean(eval_dict['f1_score']))
print("validation precision mean:",np.mean(eval_dict['precision']))
print("training time mean:",np.mean(totaltime))
print("training time std:",np.std(totaltime))

"""2. train whole data """

df_train = pd.read_csv(data_path + prefix + 'train' + '.csv')
df_train['label'].replace(-1, 0, inplace = True)
labels_train = np.array(df_train['label'])
df_train = df_train.iloc[:, :-1]
if if_pca:
    df_train=pd.DataFrame(pca.fit_transform(df_train))
dtrain = df_train.as_matrix()
dtrain = xgb.DMatrix(dtrain, label=labels_train)
evallist = [(dtrain,'train')]
num_round = 30
bst = xgb.train(param, dtrain, num_round, evallist)
del dtrain, df_train

"""3. test data """
df_test = pd.read_csv(data_path + prefix + 'test_data' + '.csv')
df_test['label'].replace(-1, 0, inplace = True)
labels_test = np.array(df_test['label'])
df_test = df_test.iloc[:, :-1]
if if_pca:
    df_test=pd.DataFrame(pca.fit_transform(df_test))
test_amount = len(df_test)
dtest = df_test.as_matrix()
dtest = xgb.DMatrix(dtest, label=labels_test)
del df_test
    
"""4. predict test data """
ypred = bst.predict(dtest)
y_pred_binary = (ypred >= 0.5)*1
print('test AUC: ' + str(metrics.roc_auc_score(labels_test,ypred)))
print('test ACC: ' + str(metrics.accuracy_score(labels_test,y_pred_binary)))
print('test Recall: ' + str(metrics.recall_score(labels_test,y_pred_binary)))
print('test F1-score: ' + str(metrics.f1_score(labels_test,y_pred_binary)))
print('test Precesion: ' + str(metrics.precision_score(labels_test,y_pred_binary)))


import winsound
winsound.Beep(600,1000)
