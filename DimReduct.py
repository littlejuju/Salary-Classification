# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 23:50:24 2020

@author: Xiangqi
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
prefix = 'dropna_'
df_train = pd.read_csv(prefix + 'train' + '.csv')
X = df_train.iloc[:, :-1]

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

""" 1. unsupervised: PCA """
from sklearn.decomposition import PCA
import numpy as np
pca = PCA(n_components=None, copy=True, whiten=False)
pca1 = PCA(n_components=1, copy=True, whiten=False)
pca2 = PCA(n_components=2, copy=True, whiten=False)
pca10 = PCA(n_components=20, copy=True, whiten=False)
pca20 = PCA(n_components=20, copy=True, whiten=False)
pca40 = PCA(n_components=40, copy=True, whiten=False)
pca90 = PCA(n_components=90, copy=True, whiten=False)
pca.fit(X)
pca1.fit(X)
pca2.fit(X)
pca10.fit(X)
pca20.fit(X)
pca40.fit(X)
pca90.fit(X)
x_transform = pca2.transform(X)
plt.figure()
X_low = x_transform[df_train['label'] == -1]
X_high = x_transform[df_train['label'] == 1]
plt.plot(np.array(X_low[:,0]), np.array(X_low[:,1]), 'o', color = 'steelblue', label = '<=50K')
plt.plot(np.array(X_high[:,0]), np.array(X_high[:,1]), 'o', color = 'purple', label = '>=50K')
plt.legend()
plt.show()
variance_ratio = pca.explained_variance_ratio_
feature_num  = [index for index in range(len(variance_ratio)+1)]
baseline = [0.95 for index in range(len(variance_ratio)+1)]
variance_cum = [sum(variance_ratio[:index+1]) for index in range(len(variance_ratio))]
variance_cum = [0] + variance_cum
plt.figure()
plt.plot(feature_num, variance_cum, color = 'b', label = 'cumulative variance ratio', linewidth=2, linestyle="-")
plt.plot(feature_num, baseline, color = 'r', label = '95% baseline', linewidth=1, linestyle="--")
plt.legend()
plt.xlabel('feature number')
plt.ylabel('cumulative variance ratio')
plt.show()

# save models
joblib.dump(pca1, 'pca1.pkl') 
joblib.dump(pca10, 'pca10.pkl') 
joblib.dump(pca20, 'pca20.pkl') 
joblib.dump(pca40, 'pca40.pkl') 
joblib.dump(pca90, 'pca90.pkl') 



""" 2. supervised: LDA """

""" Step 1: Computing the d-dimensional mean vectors"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {
        'red': [(0, 1, 1), (1, 0.7, 0.7)],
        'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
        'blue': [(0, 0.7, 0.7), (1, 1, 1)]
    }
)

plt.cm.register_cmap(cmap=cmap)

def plot_data(lda, X, y, y_pred, fig_index):
#    if fig_index == 1:
#        plt.title('Linear Discriminant Analysis')
#    elif fig_index == 2:
#        plt.title('Quadratic Discriminant Analysis')
#    elif fig_index == 3:
#        plt.ylabel('Data with varying covariances')
    tp = (y == y_pred) #正样本中，分类正确的数目
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0 , X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]
    plt.plot(np.array(X0_tp[:, 0]), np.array(X0_tp[:, 1]), 'o', color='red', label = '>50K TP')
    plt.plot(np.array(X0_fp[:, 0]), np.array(X0_fp[:, 1]), '.', color='#990000')
    plt.plot(np.array(X1_tp[:, 0]), np.array(X1_tp[:, 1]), 'o', color='blue', label = '<=50K TP')
    plt.plot(np.array(X1_fp[:, 0]), np.array(X1_fp[:, 1]), '.', color='#000099')
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()    
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                             np.linspace(y_min, y_max, ny))
    
    plt.plot(lda.means_[0][0], lda.means_[0][1],
                 'o', color='k', markersize=10)
    plt.plot(lda.means_[1][0], lda.means_[1][2],
                 'o', color='k', markersize=10)
    plt.legend()
    plt.axis('tight')
    


df_train['label'].replace(-1, 0, inplace = True)
y = df_train['label']

lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
y_pred = lda.fit(X, y).predict(X)
print(lda.explained_variance_ratio_)
xx = range(1, 106)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.bar(np.array(xx), np.array(lda.coef_)[0])
#plt.show()
ax1.set_ylabel('LDA Coefficients')
ax1.set_xlabel('features')
ax2 = ax1.twinx()
ax2.set_ylabel('transformed X')
target_names = ['>50K', '<=50K']
X_r2 = lda.fit(X, y).transform(X)

X_Zreo = np.zeros(X_r2.shape)
for c ,shape, i , target_names in zip('rb', 'o.', [0, 1], target_names):
    plt.plot((np.array(X_r2[y == i])-min(X_r2))*105/(max(X_r2) - min(X_r2)), np.array(X_Zreo[y == i]), shape, c=c, label=target_names)
#splot = plot_data(lda, X, y, y_pred, fig_index= 1)
plt.legend()
plt.show()


qda = QuadraticDiscriminantAnalysis()
y_pred = qda.fit(X, y).predict(X)

plt.figure()
splot = plot_data(qda, X, y, y_pred, fig_index= 2)
plt.show()
#import numpy as np
#np.set_printoptions(precision=4)
#df_train['label'].replace(-1, 0, inplace = True)
#labels = df_train['label'].values.tolist()
#mean_vectors = [np.array(np.mean(X[df_train.label == 0])), np.array(np.mean(X[df_train.label == 0]))]
#
#""" Step 2: Computing the Scatter Matrices"""
## 2.1 Within-class scatter matrix SW
#n = X.shape[1]
#S_W = np.zeros((n,n))
#for cl,mv in enumerate(mean_vectors):
#    class_sc_mat = np.zeros((n,n))                  # scatter matrix for every class
#    for row in np.array(X[df_train.label == cl]):
#        row, mv = row.reshape(n,1), mv.reshape(n,1) # make column vectors
#        class_sc_mat += (row-mv).dot((row-mv).T)
#    S_W += class_sc_mat                             # sum class scatter matrices
#
## 2.2 Between-class scatter matrix SB
#overall_mean = np.array(np.mean(X, axis=0))
#
#S_B = np.zeros((n,n))
#for i,mean_vec in enumerate(mean_vectors):  
#    num = X[df_train.label==i].shape[0]
#    mean_vec = mean_vec.reshape(n,1) # make column vector
#    overall_mean = overall_mean.reshape(n,1) # make column vector
#    S_B += num * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
#
#""" Step 3: Solving the generalized eigenvalue problem for the matrix S−1WSB"""
#eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
#
#""" Step 4: Selecting linear discriminants for the new feature subspace"""
#eig_pairs = [(eig_vals[i], eig_vecs[:,i]) for i in range(len(eig_vals))]
#
## Sort the (eigenvalue, eigenvector) tuples from high to low
#eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
#
## Visually confirm that the list is correctly sorted by decreasing eigenvalues
## 4.1. Sorting the eigenvectors by decreasing eigenvalues
#print('Eigenvalues in decreasing order:\n')
#for i in eig_pairs:
#    print(i[0])
#
#print('Variance explained:\n')
#eigv_sum = sum(eig_vals)
#for i,j in enumerate(eig_pairs):
#    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
#
## 4.2. Choosing k eigenvectors with the largest eigenvalues
#W = np.hstack((eig_pairs[0][1].reshape(n,1), eig_pairs[1][1].reshape(n,1)))
#print('Matrix W:\n', W.real)
#
#""" Step 5: Transforming the samples onto the new subspace"""
#X_lda = X.dot(W)
#from matplotlib import pyplot as plt
#label_list = ['<=50K', '>=50K']
#test = X_lda[df_train.label == 0][0].real
#min_test = min(test)
#max_test = max(test)
#testy = X_lda[df_train.label == 0][1].real
#min_testy = min(testy)
#max_testy = max(testy)
#test1 = X_lda[df_train.label == 1][0].real
#min_test1 = min(test1)
#max_test1 = max(test1)
#testy1 = X_lda[df_train.label == 1][1].real
#min_testy1 = min(testy1)
#max_testy1 = max(testy1)
#x_left = min(min_test, min_test1)
#x_right = max(max_test, max_test1)
#y_left = min(min_testy, min_testy1)
#y_right = max(max_testy, max_testy1)
#def plot_step_lda():
#
#    ax = plt.subplot(111)
#    for label,marker,color in zip(
#        range(2),('s', 'o'),('blue', 'red')):
#
#        plt.scatter(x=X_lda[df_train.label == label][0].real,
#                y=X_lda[df_train.label == label][1].real,
#                marker=marker,
#                color=color,
#                alpha=0.5,
#                label=label_list[label]
#                )
#
#    plt.xlabel('LD1')
#    plt.ylabel('LD2')
#
#    leg = plt.legend(loc='upper right', fancybox=True)
#    leg.get_frame().set_alpha(0.5)
#    plt.title('LDA: Iris projection onto the first 2 linear discriminants')
#
#    # hide axis ticks
#    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
#            labelbottom="on", left="off", right="off", labelleft="on")
#
#    # remove axis spines
#    ax.spines["top"].set_visible(False)  
#    ax.spines["right"].set_visible(False)
#    ax.spines["bottom"].set_visible(False)
#    ax.spines["left"].set_visible(False)    
#    plt.xlim(x_left, x_right)
#    plt.ylim(y_left, y_right)
#
#    plt.grid()
#    plt.tight_layout
#    plt.show()
#plt.figure()
#plot_step_lda()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
