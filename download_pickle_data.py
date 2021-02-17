import argparse
import urllib.request
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import pickle

#### Parameters #############
parser = argparse.ArgumentParser(description='download and pickle network data')
parser.add_argument('--dir_data', type=str, default='./data/', help='directory of network data')
args = parser.parse_args()

#### load data ############
### protein-protein interaction network
print("## Protein-protein interaction network")
print("download the data")
urllib.request.urlretrieve("http://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/hippie_current.txt", args.dir_data + "networkP.txt")
print("generate the normalized adjacency matrix")
data = pd.read_csv(args.dir_data + "networkP.txt", delimiter='\t', header=None)
edgelist = data[[1,3,4]]
edgelist = edgelist.rename(columns={1: 'source', 3: 'target', 4: 'weight'})
edgelist = edgelist[edgelist['weight'] > 0.0]
edgelist = edgelist.drop_duplicates()
# network object
g = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr='weight')
# extract the largest connected component
gcc = sorted(nx.connected_components(g), key=len, reverse=True)
g = g.subgraph(gcc[0])
del(data)
# get adjacency matrix
adj_networkP = nx.adjacency_matrix(g)
# remove selfloop
sp.csr_matrix.setdiag(adj_networkP, 0)
adj_networkP.eliminate_zeros()
# get node list
nodelist_networkP = list(g.nodes)
del(g)
# normalized adjacency matrix
deg_networkP = np.sum(adj_networkP, axis=0)
norm_adj_networkP = sp.csr_matrix(adj_networkP / np.sqrt(np.dot(deg_networkP.T, deg_networkP)), dtype=np.float64)
del(adj_networkP)

print("pickle the matrix")
with open(args.dir_data + 'norm_adj_networkP.pickle', mode='wb') as f:
    pickle.dump(norm_adj_networkP, f)


### disease similarity network
print("\n## Disease similarity network")
print("download the data")
urllib.request.urlretrieve("http://www.cs.tau.ac.il/%7Ebnet/software/PrincePlugin/PhenSim.tsv", args.dir_data + "networkD.txt")
print("generate the adjacency matrix")
adj_networkD = pd.read_table(args.dir_data + "networkD.txt", delimiter='\t', header=None, index_col=0)
# get node list
nodelist_networkD = list(adj_networkD.index.values)
# conversion using logistic function
PheSim = np.array(adj_networkD)
PheSim = 1 / (1 + np.exp(-15 * PheSim + np.log(9999)))
np.fill_diagonal(PheSim, 1.0)
PheSim = sp.csr_matrix(PheSim)
PheSim.eliminate_zeros()

print("pickle the matrix")
with open(args.dir_data + 'adj_networkD.pickle', mode='wb') as f:
    pickle.dump(PheSim, f)


### protein-disease network (data used in PRINCE study)
print("\n## Protein-disease association network")
print("download the data")
urllib.request.urlretrieve("http://www.cs.tau.ac.il/%7Ebnet/software/PrincePlugin/associations.txt", args.dir_data + "binetworkPD.txt")
print("generate the biadjacency matrix")
data = pd.read_csv(args.dir_data + "binetworkPD.txt", header=None, delimiter='\t', comment='#')
data_sub = data[[0,1]]
data_sub = data_sub.drop_duplicates()
# extract the pairs of nodes appering network P and network D, respectively
data_sub = data_sub[data_sub[1].isin(nodelist_networkP) & data_sub[0].isin(nodelist_networkD)]

biadj_networkPD = sp.lil_matrix((len(nodelist_networkP), len(nodelist_networkD)), dtype=np.float64)
# ToDo: faster implementation
for index, row in data_sub.iterrows():
    idxP = nodelist_networkP.index(row[1])
    idxD = nodelist_networkD.index(row[0])
    biadj_networkPD[idxP,idxD] = 1

biadj_networkPD = sp.csr_matrix(biadj_networkPD)

print("pickle the matrix")
with open(args.dir_data + 'biadj_networkPD.pickle', mode='wb') as f:
    pickle.dump(biadj_networkPD, f)

