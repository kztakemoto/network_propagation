import warnings
warnings.simplefilter('ignore')

import argparse
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from network_propagation_methods import minprop_2
from sklearn.metrics import roc_auc_score, auc
import matplotlib.pyplot as plt

#### Parameters #############
parser = argparse.ArgumentParser(description='Runs MINProp')
parser.add_argument('--alphaP', type=float, default=0.25, help='diffusion parameter for the protein-protein interaction network')
parser.add_argument('--alphaD', type=float, default=0.25, help='diffusion parameter for the disease similarity network')
parser.add_argument('--max_iter', type=int, default=1000, help='maximum number of iterations')
parser.add_argument('--eps', type=float, default=1.0e-6, help='convergence threshold')
parser.add_argument('--dir_data', type=str, default='./data/', help='directory of pickled network data')
args = parser.parse_args()

#### load data ############
### protein-protein interaction network
with open(args.dir_data + 'norm_adj_networkP.pickle', mode='rb') as f:
    norm_adj_networkP = pickle.load(f)
nb_proteins = norm_adj_networkP.shape[0]

### disease similarity network
with open(args.dir_data + 'adj_networkD.pickle', mode='rb') as f:
    adj_networkD = pickle.load(f)
nb_diseases = adj_networkD.shape[0]

# normalized adjacency matrix
deg_networkD = np.sum(adj_networkD, axis=0)
norm_adj_networkD = sp.csr_matrix(adj_networkD / np.sqrt(np.dot(deg_networkD.T, deg_networkD)), dtype=np.float64)
del(adj_networkD)
del(deg_networkD)

### protein-disease network (data used in PRINCE study)
with open(args.dir_data + 'biadj_networkPD.pickle', mode='rb') as f:
    biadj_networkPD = pickle.load(f)

# get the list of protein-disease pairs
PD_pairs = biadj_networkPD.nonzero()
# number of protein-disease pairs
nb_PD_pairs = len(PD_pairs[0])

#### Network propagation MINProp ###########################
roc_value_set = np.array([], dtype=np.float64)
rankings = np.array([], dtype=np.int64)
for i in range(nb_PD_pairs):
    # leave-one-out validation
    # remove a protein-disease association
    idx_P = PD_pairs[0][i]
    idx_D = PD_pairs[1][i]
    biadj_networkPD[idx_P, idx_D] = 0.0
    biadj_networkPD.eliminate_zeros()
    # normalized biadjacency matrix (ToDo: faster implementation)
    degP = np.sum(biadj_networkPD, axis=1)
    degD = np.sum(biadj_networkPD, axis=0)
    norm_biadj_networkPD = sp.csr_matrix(biadj_networkPD / np.sqrt(np.dot(degP, degD)), dtype=np.float64)
    norm_biadj_networkPD.data[np.isnan(norm_biadj_networkPD.data)] = 0.0
    norm_biadj_networkPD.eliminate_zeros()
    # set initial label
    yP = np.zeros(nb_proteins, dtype=np.float64)
    yD = np.zeros(nb_diseases, dtype=np.float64)
    yD[idx_D] = 1.0
    # propagation
    fP, fD, convergent = minprop_2(norm_adj_networkP, norm_adj_networkD, norm_biadj_networkPD, yP, yD, args.alphaP, args.alphaD, args.eps, args.max_iter)
    # ranking
    labels_real = np.zeros(nb_proteins)
    labels_real[idx_P] = 1
    rank = int(np.where(labels_real[np.argsort(-fP)]==1)[0]) + 1
    rankings = np.append(rankings, rank)
    # get AUC value
    roc_value = roc_auc_score(labels_real, fP)
    print(i, "AUC:", roc_value, convergent)
    roc_value_set = np.append(roc_value_set, roc_value)
    # reassign the protein-disease association
    biadj_networkPD[idx_P, idx_D] = 1.0

print("Average AUC", np.mean(roc_value_set))

# compute sensitivity and top rate (ROC-like curve)
# ToDo: faster implementation
sen_set = np.array([], dtype=np.float64)
top_rate_set = np.array([], dtype=np.float64)
for k in range(nb_proteins):
    # sensitibity
    sen = (rankings <= (k+1)).sum() / nb_PD_pairs
    # top rate
    top_rate = (k + 1) / nb_proteins
    
    sen_set = np.append(sen_set, sen)
    top_rate_set = np.append(top_rate_set, top_rate)

# get AUC value
print("Summarized AUC", auc(top_rate_set, sen_set))

# plot ROC-like curve
plt.scatter(top_rate_set, sen_set)
plt.show()
