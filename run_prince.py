import warnings
warnings.simplefilter('ignore')

import argparse
import numpy as np
import networkx as nx
import scipy.sparse as sp
from network_propagation_methods import netprop
from utils import *
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from tqdm import tqdm

#### Parameters #############
parser = argparse.ArgumentParser(description='Runs PRINCE')
parser.add_argument('--alpha', type=float, default=0.25, help='diffusion parameter')
parser.add_argument('--max_iter', type=int, default=1000, help='maximum number of iterations')
parser.add_argument('--eps', type=float, default=1.0e-6, help='convergence threshold')
parser.add_argument('--dir_data', type=str, default='./data/', help='directory of pickled network data')
args = parser.parse_args()

#### load data ############
# protein-protein interaction network
norm_adj_networkP = load_pickle(args.dir_data + 'norm_adj_networkP.pickle')
nb_proteins = norm_adj_networkP.shape[0]
# disease similarity network
adj_networkD = load_pickle(args.dir_data + 'adj_networkD.pickle')
# protein-disease network (data used in PRINCE study)
biadj_networkPD = load_pickle(args.dir_data + 'biadj_networkPD.pickle')

# get the list of protein-disease pairs
PD_pairs = biadj_networkPD.nonzero()
# number of protein-disease pairs
nb_PD_pairs = len(PD_pairs[0])

#### Network propagation PRINCE (leave-one-out validation) ########################
def compute_ranking_and_roc(i, PD_pairs, adj_networkD, norm_adj_networkP, biadj_networkPD, args):
    idx_P = PD_pairs[0][i]
    idx_D = PD_pairs[1][i]
    # remove a protein-disease association
    biadj_networkPD_copy = biadj_networkPD.copy()
    biadj_networkPD_copy[idx_P, idx_D] = 0.0
    biadj_networkPD_copy.eliminate_zeros()
    # set initial label
    yP = np.ravel(sp.csr_matrix.max(biadj_networkPD_copy.multiply(adj_networkD[idx_D]), axis=1).todense())
    del(biadj_networkPD_copy)
    # propagation
    fP, convergent = netprop(norm_adj_networkP, yP, args.alpha, args.eps, args.max_iter)
    # ranking
    labels_real = np.zeros(nb_proteins)
    labels_real[idx_P] = 1
    rank = int(np.where(labels_real[np.argsort(-fP)]==1)[0]) + 1
    # get AUC value
    roc_value = roc_auc_score(labels_real, fP)
    
    return rank, roc_value

# run in parallel
results = Parallel(n_jobs=-1)(delayed(compute_ranking_and_roc)(i, PD_pairs, adj_networkD, norm_adj_networkP, biadj_networkPD, args) for i in tqdm(range(nb_PD_pairs)))

#### Save the results and plot ROC-like curve
outputs("prince", results, nb_proteins, nb_PD_pairs, args)
