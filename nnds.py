import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

def density(x, r, features):
    count = 0
    set = []
    for t in range(len(features)):
        dst = distance.euclidean(features.iloc[x], features.iloc[t])
        if dst <= r:
            count = count+1
            set.append(t)
    return set

k = 386
rare_class = 'back.'
dist_sum = []
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
data = pd.read_csv('../corrected', header=None, names=col_names)

#dupicate remove
data_nd = data.drop_duplicates(cols=col_names, take_last=True)

#choose data normal. and back.
data_nd_b = data_nd.loc[data_nd['label'].isin(['normal.', rare_class])]

#features, labels
features = data_nd_b[num_names].astype(float)
labels = data_nd_b['label']
features.index = range(len(features))
labels.index = range(len(labels))
#scaler
features_sc = features.apply(lambda x: MinMaxScaler().fit_transform(x))
#step 1 find neighbor
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(features_sc)
distances, indices = nbrs.kneighbors(features_sc)

#sum distance
for i in range(len(features_sc)):
    dist_sum.append(distances[i].sum())

min_dist = sorted(dist_sum)[0]
for i in range(len(dist_sum)):
    if min_dist == dist_sum[i]:
        min_dist_index = i

r = max(distances[min_dist_index])
#r = min_dist / k

density_sets = []

#step 2
for t in range(len(features_sc)):
    de_set=[]
    for tt in range(k+1):
        if distances[t][tt] <= r:
            de_set.append(indices[t][tt])
    density_sets.append(de_set)

#cal si

si_list = []
for i in range(len(features_sc)):
    s_set = []
    ni = len(density_sets[i]) - 1 
    for j in density_sets[i]:
        nj = len(density_sets[j]) - 1 
        s = ni - nj
        s_set.append(s)
    s_max = max(s_set)
    si_list.append(s_max)

selected = []
for t in range(len(features_sc)):
    #find max si to query
    while True :
        xi = si_list.index(max(si_list))
        print (xi, si_list[xi])
        if xi not in selected:
            selected.append(xi)
            si_list[xi] = -999999
            break
    #xi is query point
    if labels[xi] == rare_class:
        num_query = t+1
        break
        
print num_query
#features_sc['label'] = labels
#features_sc.to_csv('apache2R.csv', sep='\t', encoding='utf-8')

