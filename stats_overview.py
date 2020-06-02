# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:18:45 2020

@author: Xiangqi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

remove_missing_value = True

"""1. load data set"""
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
attributes = ['age', 'workclass', 'fnlwgt', 'education', 'ed num', 
              'marital', 'occupation', 'relationship', 'race', 'sex', 
              'gain', 'loss', 'hours', 'nation', 'labels']
categorical_attributes = ['workclass', 'education', 'marital', 
                          'occupation', 'relationship', 'race', 'sex', 
                          'nation']
adult_data = pd.read_csv(url + 'adult.data', names = attributes, sep = ', ')
adult_test = pd.read_csv(url + 'adult.test', skiprows = 1, names = attributes, sep = ', ')

""" 2. variation for categorical data"""
var_dict = dict()
port_dict = dict()
percentile_dict = dict()
for attr in categorical_attributes:
    set_attr = list(set(adult_data[attr].values.tolist()))
    attr_port_dict = dict()
    for cat in set_attr:
        attr_port_dict[cat] = len(adult_data[adult_data[attr] == cat])/len(adult_data) 
    port_dict[attr] = attr_port_dict

    var_dict[attr] = len(set_attr)
    
""" 3. na count """
na_dict = dict()
for attr in attributes:
    na_dict[attr] = len(adult_data[adult_data[attr] == '?'].index.tolist())
    
    

""" 4. boxplot for numeric data """

df = adult_data[attributes]
df.boxplot(sym='r*',vert=False,notch = True,patch_artist=True,meanline=False,showmeans=True)
plt.xscale('log')
plt.show()
for attr in list(set(attributes[:-1]) - set(categorical_attributes)):
    var_dict[attr] = [np.max((adult_data[attr].values.tolist())), np.min((adult_data[attr].values.tolist())), 
            np.mean((adult_data[attr].values.tolist())), np.std((adult_data[attr].values.tolist()))]


""" 5. heat map """
import seaborn as sns
cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
pt = df.corr()   # pt为数据框或者是协方差矩阵
plt.figure()
sns.heatmap(pt, linewidths = 0.05,  vmax=1, vmin=0, cmap=cmap)
plt.xlabel('')
plt.show()

