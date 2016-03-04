import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

t0 = time()

def cal_NNSet(r, k, length, distances, indices):
    NNSet = []
    for t in range(length):
        NN = []
        for tt in range(k+1):
            if distances[t][tt] <= r:
                NN.append(indices[t][tt])
        NNSet.append(NN)
    return NNSet

def cal_SiSet(NNSet, length):
    si_list = []
    for i in range(length):
        s_set = []
        ni = len(NNSet[i]) - 1 
        for j in NNSet[i]:
            nj = len(NNSet[j]) - 1 
            s = ni - nj
            s_set.append(s)
        s_max = max(s_set)
        si_list.append(s_max)
    return si_list
k = 15
rare_class = 3

col_names = [
    "Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight",
    "Viscera weight", "Shell weight", "Rings"     
]
num_names = [
    "Length", "Diameter", "Height", "Whole weight", "Shucked weight",
    "Viscera weight", "Shell weight"
]
#read data
data = pd.read_csv('../abalone/abalone.data', header=None, names=col_names)

#dupicate remove
#data_nd = data.drop_duplicates(cols=col_names, take_last=True)

#choose data normal. and back.
data_nd_b = data.loc[data['Rings'].isin([9,rare_class])]

#features, labels
features = data_nd_b[num_names].astype(float)
labels = data_nd_b['Rings']
features.index = range(len(features))
labels.index = range(len(labels))
#scaler
features_sc = features.apply(lambda x: MinMaxScaler().fit_transform(x))
#step 1 find neighbor
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(features_sc)
distances, indices = nbrs.kneighbors(features_sc)

#cal r 
dist_max = []
for t in range(len(features_sc)):
    dist_max.append(max(distances[t]))

r = min(dist_max)

#features_sc['label'] = labels
#features_sc.to_csv('snmpguess.csv', sep='\t', encoding='utf-8')
#step 2
n = len(features_sc)
print "size : ", n
print "radius : ", r

selected = []
for t in range(n):
    SiSet =[]
    NNSet =[]
    NNSet =cal_NNSet((t+1)*r, k, n, distances, indices)
    SiSet = cal_SiSet(NNSet, n)
    for i in range(len(selected)): 
        SiSet[selected[i]] = -99999
    while True :
        xi = SiSet.index(max(SiSet))
        print (xi, SiSet[xi], t+1, labels[xi])
        if xi not in selected:
            selected.append(xi)
            break
    #xi is query point
    if labels[xi] == rare_class:
        num_query = t+1
        break


t1 = time() - t0
print "radius : ", r
print "No. of Query", num_query, ". Index : ", xi
print "Spend time : ", t1, " s"

