# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 23:32:50 2020

@author: Xiangqi
"""
data_path = 'C:/Users/Xiangqi/Desktop/Singapore Modules Folders/is5152/group project/preprocessing/'

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.datasets import make_gaussian_quantiles
#from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn import metrics


k=12 # k-fold cross validation
prefix = 'dropna_'

error = [0.]*k
totaltime = [0.]*k


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
    dtrain = df_train.iloc[:, :-1].as_matrix()  
    dval = df_val.iloc[:, :-1].as_matrix()
    
    """ 2.3 train xgboost model """
    time_start=time.time()
    evallist = [(dtrain,'train')]
    num_round = 30
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=9, min_samples_split=200, min_samples_leaf=5),
                             algorithm="SAMME",
                             n_estimators=10, learning_rate=0.3)
    bdt.fit(dtrain, labels_train)
    """ 2.4 predict test labels """
    val_pred = bdt.predict(dval)
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
dtrain = df_train.iloc[:, :-1].as_matrix()
evallist = [(dtrain,'train')]
num_round = 30
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, min_samples_split=100, min_samples_leaf=5),
                             algorithm="SAMME",
                             n_estimators=10, learning_rate=0.1)
bdt.fit(dtrain, labels_train)
del dtrain, df_train

"""3. test data """
df_test = pd.read_csv(data_path + prefix + 'test_data' + '.csv')
df_test['label'].replace(-1, 0, inplace = True)
test_amount = len(df_test)
labels_test = np.array(df_test['label'])
dtest = df_test.iloc[:, :-1].as_matrix()
del df_test
    
"""4. predict test data """
ypred = bdt.predict(dtest)
y_pred_binary = (ypred >= 0.5)*1
print('test AUC: ' + str(metrics.roc_auc_score(labels_test,ypred)))
print('test ACC: ' + str(metrics.accuracy_score(labels_test,y_pred_binary)))
print('test Recall: ' + str(metrics.recall_score(labels_test,y_pred_binary)))
print('test F1-score: ' + str(metrics.f1_score(labels_test,y_pred_binary)))
print('test Precesion: ' + str(metrics.precision_score(labels_test,y_pred_binary)))


import winsound
winsound.Beep(600,1000)
