import warnings
warnings.simplefilter('ignore')
import sys
sys.dont_write_bytecode = True

import argparse
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from network_propagation_methods import minprop_2
from sklearn.metrics import roc_auc_score, auc
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

#### Parameters #############
parser = argparse.ArgumentParser(description='Runs MINProp')
parser.add_argument('--alphaP', type=float, default=0.25, help='diffusion parameter for the protein-protein interaction network')
parser.add_argument('--alphaD', type=float, default=0.25, help='diffusion parameter for the disease similarity network')
parser.add_argument('--max_iter', type=int, default=1000, help='maximum number of iterations')
parser.add_argument('--eps', type=float, default=1.0e-6, help='convergence threshold')
parser.add_argument('--dir_data', type=str, default='./', help='directory of network data')
args = parser.parse_args()

#### load data ############
### protein-protein interaction network
data = pd.read_csv(args.dir_data + "hippie_current.txt", delimiter='\t', header=None)
edgelist = data[[1,3,4]]
del(data)
edgelist = edgelist.rename(columns={1: 'source', 3: 'target', 4: 'weight'})
edgelist = edgelist[edgelist['weight'] > 0.0]
edgelist = edgelist.drop_duplicates()
# network object
g = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr='weight')
# extract the largest connected component
obj_networkP = max(nx.connected_component_subgraphs(g), key=len)
del(g)

# get adjacency matrix
adj_networkP = nx.adjacency_matrix(obj_networkP)
# remove selfloop
sp.csr_matrix.setdiag(adj_networkP, 0)
adj_networkP.eliminate_zeros()
# get node list
nodelist_networkP = list(obj_networkP.nodes)
del(obj_networkP)
# normalized adjacency matrix
deg_networkP = np.sum(adj_networkP, axis=0)
norm_adj_networkP = sp.csr_matrix(adj_networkP / np.sqrt(np.dot(deg_networkP.T, deg_networkP)), dtype=np.float64)
# #proteins
nb_proteins = adj_networkP.shape[0]
del(adj_networkP)

### disease similarity network
adj_networkD = pd.read_table(args.dir_data + "PhenSim.tsv", delimiter='\t', header=None, index_col=0)
# get node list
nodelist_networkD = list(adj_networkD.index.values)
# conversion using logstic function
PheSim = np.array(adj_networkD)
PheSim = 1 / (1 + np.exp(-15 * PheSim + np.log(9999)))
np.fill_diagonal(PheSim, 1.0)
PheSim = sp.csr_matrix(PheSim)
PheSim.eliminate_zeros()
# normalized adjacency matrix
deg_networkD = np.sum(PheSim, axis=0)
norm_adj_networkD = sp.csr_matrix(PheSim / np.sqrt(np.dot(deg_networkD.T, deg_networkD)), dtype=np.float64)
del(adj_networkD)
del(PheSim)

### protein-disease network (data used in PRINCE study)
biadj_networkPD = sp.lil_matrix((len(nodelist_networkP), len(nodelist_networkD)), dtype=np.float64)
data = pd.read_csv(args.dir_data + "associations.txt", header=None, delimiter='\t', comment='#')
data_sub = data[[0,1]]
del(data)
data_sub = data_sub.drop_duplicates()
# extract the pairs of nodes appering network P and network D, respectively
data_sub = data_sub[data_sub[1].isin(nodelist_networkP) & data_sub[0].isin(nodelist_networkD)]

# ToDo: faster implementation
for index, row in data_sub.iterrows():
    idxP = nodelist_networkP.index(row[1])
    idxD = nodelist_networkD.index(row[0])
    biadj_networkPD[idxP,idxD] = 1

biadj_networkPD = sp.csr_matrix(biadj_networkPD)
del(data_sub)

# get the list of protein-disease pairs
PD_pairs = biadj_networkPD.nonzero()
# number of protein-disease pairs
nb_PD_pairs = len(PD_pairs[0])

#### Network propagation MINProp ###########################
def proc_minprop(i):
    # leave-one-out validation
    # remove a protein-disease association
    idx_P = PD_pairs[0][i]
    idx_D = PD_pairs[1][i]
    mod_biadj_networkPD = biadj_networkPD.copy()
    mod_biadj_networkPD[idx_P, idx_D] = 0.0
    mod_biadj_networkPD.eliminate_zeros()
    # normalized biadjacency matrix (ToDo: faster implementation)
    degP = np.sum(mod_biadj_networkPD, axis=1)
    degD = np.sum(mod_biadj_networkPD, axis=0)
    norm_biadj_networkPD = sp.csr_matrix(mod_biadj_networkPD / np.sqrt(np.dot(degP, degD)), dtype=np.float64)
    norm_biadj_networkPD.data[np.isnan(norm_biadj_networkPD.data)] = 0.0
    norm_biadj_networkPD.eliminate_zeros()
    # set initial label
    yP = np.zeros(len(nodelist_networkP), dtype=np.float64)
    yD = np.zeros(len(nodelist_networkD), dtype=np.float64)
    yD[idx_D] = 1.0
    # propagation
    fP, fD, convergent = minprop_2(norm_adj_networkP, norm_adj_networkD, norm_biadj_networkPD, yP, yD, args.alphaP, args.alphaD, args.eps, args.max_iter)
    # ranking
    labels_real = np.zeros(nb_proteins)
    labels_real[idx_P] = 1
    rank = int(np.where(labels_real[np.argsort(-fP)]==1)[0]) + 1
    # get AUC value
    roc_value = roc_auc_score(labels_real, fP)
    print(i, "AUC:", roc_value, convergent)
    return convergent, roc_value, rank

# Parallel computation using Joblib. n_jobs: #cores (n_jobs = -1 means that all available CPUs are used)
sub = Parallel(n_jobs=-1, backend="threading")( [delayed(proc_minprop)(i) for i in range(nb_PD_pairs)] )
sub = pd.DataFrame(sub)

roc_value_set = sub[1]
print("Average AUC", np.mean(roc_value_set))

# compute sensitivity and top rate (ROC-like curve)
# ToDo: faster implementation
rankings = sub[2]
sen_set = np.array([])
top_rate_set = np.array([])
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