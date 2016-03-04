import pandas as pd
import numpy as np
import random
from time import time
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing

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
k = 386
rare_class = 'back.'
c = 7

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

num_names = [
    "duration","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]
#read data
data = pd.read_csv('../kdd99/corrected', header=None, names=col_names)

#dupicate remove
data_nd = data.drop_duplicates(cols=col_names, take_last=True)

#choose data normal. and back.
data_nd_b = data_nd.loc[data_nd['label'].isin(['normal.', rare_class])]

#features, labels
features = data_nd_b[num_names].astype(float)
labels = data_nd_b['label']
features.index = range(len(features))
labels.index = range(len(labels))



#features selection, delete zero varience data, and choose best c features
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
features_nz = sel.fit_transform(features)

#scaler
features_nz_scaled = preprocessing.scale(features_nz)
df_sc = pd.DataFrame(features_nz_scaled)


#step 1 find neighbor
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(df_sc)
distances, indices = nbrs.kneighbors(df_sc)

n = len(df_sc)
#cal r 
dist_max = []
for t in range(n):
    dist_max.append(max(distances[t]))

r = min(dist_max)

df_sc['label'] = labels
df_sc.to_csv( rare_class +'csv', sep='\t', encoding='utf-8')
#step 2

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

