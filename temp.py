# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 20:18:46 2019

@author: user
"""

from bert_serving.client import BertClient
bc = BertClient()

import pandas as pd
data = pd.read_excel('test.xlsx', header = None)
data = data.values.tolist()
msg = sum(data, [])
vector = bc.encode(msg)

import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

def tuning_clustering(n_components, n_clusters, input_data):
    transformer = KernelPCA(n_components = n_components, kernel = 'rbf', n_jobs = -1)
    vecter_trans = transformer.fit_transform(input_data)
    model = SpectralClustering(n_clusters = n_clusters, n_jobs = -1)
    model.fit(vecter_trans)
    labels = model.labels_
    score1 = calinski_harabasz_score(input_data, labels)
    score2 = davies_bouldin_score(input_data, labels)
    score3 = silhouette_score(input_data, labels)
    return [n_components, n_clusters, score1, score2, score3, labels, vecter_trans]

results = [tuning_clustering(i, j, vector) for i in range(3, 8) for j in range(2, 7)]
score = pd.DataFrame([results[i][2:5] for i in range(len(results))])

model1, model2, model3 = score.iloc[:, 0].argmin(), score.iloc[:, 1].argmin(), score.iloc[:, 2].argmax()

def classification(x, y):
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt import gp_minimize
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    
    model = GradientBoostingClassifier()
    
    space = [Real(10**-2, 0.5, name = 'learning_rate'),
         Integer(10, 500, name = 'n_estimators'),
         Categorical(['sqrt', 'log2'], name = 'max_features'),
         Real(0.5, 1.0, name = 'subsample'),
         Real(0.1, 0.4, name = 'validation_fraction')]
    
    @use_named_args(space)
    def objective(**params):
        import numpy as np
        model.set_params(**params)
        return -np.mean(cross_val_score(model, x, y, cv = 10, n_jobs = -1, scoring = 'f1_macro'))
    u0 = gp_minimize(objective, space, n_calls = 10, random_state = 0, n_jobs = -1)
    fun, values, func_vals = u0.fun, u0.x, u0.func_vals
    return [fun, values]

test1, test2, tes3 = classification(x = results[model1][-1], y = results[model1][-2]),  classification(x = results[model2][-1], y = results[model2][-2]),  classification(x = results[model3][-1], y = results[model3][-2])
