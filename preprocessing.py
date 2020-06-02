# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:41:12 2020

@author: Xiangqi
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

remove_missing_value = True

"""1. load data set"""
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
attributes = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 
              'marital_status', 'occupation', 'relationship', 'race', 'sex', 
              'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'labels']
categorical_attributes = ['workclass', 'education', 'marital_status', 
                          'occupation', 'relationship', 'race', 'sex', 
                          'native_country']
adult_data = pd.read_csv(url + 'adult.data', names = attributes, sep = ', ')
adult_test = pd.read_csv(url + 'adult.test', skiprows = 1, names = attributes, sep = ', ')

"""2. filter missing values"""
#print(adult_data.find('?'))
def missing_filter(df, attributes, remove):
    missing_dict = dict()
    if remove:
        drop_index = list()
        for attr in attributes:
            missing_dict[attr] = df[df[attr] == '?'].index.tolist()
            drop_index = list(set(drop_index) | set(missing_dict[attr]))
        df = df.drop(drop_index, axis = 0)
    else:
        for attr in attributes:
            missing_dict[attr] = df[df[attr] == '?'].index.tolist()
            if len(missing_dict[attr]) > 0:
                grouped_attr = df.groupby(attr)
                attr_count_dict = dict()
                max_freq = 0
                for key, group in grouped_attr:
                    attr_count_dict[key] = len(group)
                    if len(group) > max_freq:
                        max_key = key
                        max_freq = len(group)
                df.loc[missing_dict[attr],attr] = max_key   
    print('missing value filtered')                       
    return missing_dict, df

"""3. categorical data to one-hot encoding"""
def one_hot_encoding(df, attributes):
    for attr in attributes:
#        print(attr)
        df_attr = pd.DataFrame(df[attr])
        df_attr = pd.get_dummies(df_attr, prefix = attr)
        df_drop = df.drop(attr, axis = 1)
        df = pd.concat([df_drop, df_attr], axis = 1)
    print('one hot encoding done')
    return df

#full_columns = one_hot_data.columns.tolist()
"""4. seperate labels"""
def seperate_labels(df, lab):
    labels = df.labels.tolist()
    labels = [-1 if lab[0] in item else 1 for item in labels]
    df = df.drop('labels', axis = 1)
    print('labels seperated')
    return df, labels

"""5. normalization: map all column values to [0, 1]"""
from sklearn.preprocessing import StandardScaler

def normalization_train(df):
    df_dict = dict()
    param_dict = dict()
    attributes = df.columns.tolist()
    for attr in attributes:
#        print(attr)
        df_attr = np.array(df[attr].tolist())
        param_dict[attr] = [max(df_attr), min(df_attr)]
        df_dict[attr] = (df_attr - min(df_attr)) / (max(df_attr) - min(df_attr))
    
    df = pd.DataFrame(df_dict)
    
    
    print('train data normalization')
    return df, param_dict

def scale(df):
    df = StandardScaler().fit_transform(df)
    df = pd.DataFrame(df)
    return df

def normalization_test(df, param):
    df_dict = dict()
    for attr in param:
        df_attr = np.array(df[attr].tolist())
        df_dict[attr] = (df_attr - param[attr][1])/(param[attr][0] - param[attr][1])
    df = pd.DataFrame(df_dict)
    print('test data normalization')
    return df

"""6. 12-fold cross_validation"""
def k_fold_cv(k, df, remove):
    if remove:
        prefix = 'dropna_'
    else:
        prefix = ''
    kf = KFold(n_splits = k, shuffle = True, random_state = 5152)
    fold = 1
    for train, validation in kf.split(df):
        df_train = df.loc[train]
        df_train.to_csv(prefix + 'train_' + str(fold) + '.csv')
        df_validation = df.loc[validation]
        df_validation.to_csv(prefix + 'validation_' + str(fold) + '.csv')
        fold += 1
    print(str(k) + ' fold cv divided')

missing_data, filtered_data = missing_filter(adult_data, attributes, remove_missing_value)
missing_test, filtered_test = missing_filter(adult_test, attributes, remove_missing_value)
one_hot_data = one_hot_encoding(filtered_data, categorical_attributes)
one_hot_test = one_hot_encoding(filtered_test, categorical_attributes)
zero_columns = list(set(one_hot_data.columns.tolist()) - set(one_hot_test.columns.tolist()))
drop_columns = list(set(one_hot_test.columns.tolist()) - set(one_hot_data.columns.tolist()))
one_hot_test = one_hot_test.drop(drop_columns, axis = 1)
for col in zero_columns:
    one_hot_test[col] = [0] * len(one_hot_test)
X_train, Y_train = seperate_labels(one_hot_data, ['<=', '>'])
X_test, Y_test = seperate_labels(one_hot_test, ['<=', '>'])
X_train, param_dict = normalization_train(X_train)
X_test = normalization_test(X_test, param_dict)
#X_train = scale(X_train)
#X_test = scale(X_test)
data_train = pd.concat([X_train, pd.DataFrame({'label': Y_train})], axis = 1)
if remove_missing_value:
    prefix = 'dropna_'
else:
    prefix = ''
data_train.to_csv(prefix + 'train' + '.csv')
data_test = pd.concat([X_test, pd.DataFrame({'label': Y_test})], axis = 1)
if remove_missing_value:
    data_test.to_csv('dropna_test_data.csv')
else:
    data_test.to_csv('test_data.csv')


k_fold_cv(12, data_train, remove_missing_value)

















