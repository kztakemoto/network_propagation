import warnings
warnings.simplefilter('ignore')

import argparse
import numpy as np
import networkx as nx
import scipy.sparse as sp
from network_propagation_methods import minprop_2
from utils import *
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from tqdm import tqdm

#### Parameters #############
parser = argparse.ArgumentParser(description='Runs MINProp')
parser.add_argument('--alphaP', type=float, default=0.25, help='diffusion parameter for the protein-protein interaction network')
parser.add_argument('--alphaD', type=float, default=0.25, help='diffusion parameter for the disease similarity network')
parser.add_argument('--max_iter', type=int, default=100, help='maximum number of iterations')
parser.add_argument('--eps', type=float, default=0.001, help='convergence threshold')
parser.add_argument('--dir_data', type=str, default='./data/', help='directory of pickled network data')
args = parser.parse_args()

#### load data ############
# protein-protein interaction network
norm_adj_networkP = load_pickle(args.dir_data + 'norm_adj_networkP.pickle')
nb_proteins = norm_adj_networkP.shape[0]
# disease similarity network
adj_networkD = load_pickle(args.dir_data + 'adj_networkD.pickle')
nb_diseases = adj_networkD.shape[0]
# protein-disease network (data used in PRINCE study)
biadj_networkPD = load_pickle(args.dir_data + 'biadj_networkPD.pickle')

# normalize adjacency matrix
deg_networkD = np.sum(adj_networkD, axis=0)
norm_adj_networkD = sp.csr_matrix(adj_networkD / np.sqrt(np.dot(deg_networkD.T, deg_networkD)), dtype=np.float64)
del(adj_networkD)
del(deg_networkD)

# get the list of protein-disease pairs
PD_pairs = biadj_networkPD.nonzero()
# number of protein-disease pairs
nb_PD_pairs = len(PD_pairs[0])


#### Network propagation MINProp (leave-one-out validation) #######################
def compute_ranking_and_roc(i, PD_pairs, norm_adj_networkD, norm_adj_networkP, biadj_networkPD, args):
    # remove a protein-disease association
    idx_P = PD_pairs[0][i]
    idx_D = PD_pairs[1][i]
    biadj_networkPD_copy = biadj_networkPD.copy()
    biadj_networkPD_copy[idx_P, idx_D] = 0.0
    biadj_networkPD_copy.eliminate_zeros()

    # normalize biadjacency matrix
    degP = np.sum(biadj_networkPD_copy, axis=1)
    degD = np.sum(biadj_networkPD_copy, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_biadj_networkPD = sp.csr_matrix(biadj_networkPD_copy / np.sqrt(np.dot(degP, degD)), dtype=np.float64)
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
    # get AUC value
    roc_value = roc_auc_score(labels_real, fP)

    return rank, roc_value

# run in parallel
results = Parallel(n_jobs=-1)(delayed(compute_ranking_and_roc)(i, PD_pairs, norm_adj_networkD, norm_adj_networkP, biadj_networkPD, args) for i in tqdm(range(nb_PD_pairs)))

#### Save the results and plot ROC-like curve
outputs("minprop2", results, nb_proteins, nb_PD_pairs, args)
