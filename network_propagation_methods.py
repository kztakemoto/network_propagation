import numpy as np
import scipy.sparse as sp
import networkx as nx
from networkx.algorithms import bipartite

def netprop(adj_X, yX, alpha, eps, max_iter):
    y = yX

    nb_iter_inner = 0
    f = yX
    f_prev = np.full(len(yX), np.inf, dtype=np.float64)
    while np.linalg.norm(f - f_prev, ord=1) > eps and nb_iter_inner < max_iter:
        f_prev = f
        f = (1 - alpha) * y + alpha * adj_X.dot(f)
        nb_iter_inner = nb_iter_inner + 1

    convergent = nb_iter_inner < max_iter

    return f, convergent

###########################################

def single_network_propagation_2(adj_X, biadj_XY, fY, yX, alpha=0.3, eps=1e-6, max_iter=1000):
    sum_sij_fj = biadj_XY.dot(fY)
    y = (1 - 2 * alpha) * yX / (1 - alpha) + alpha * sum_sij_fj / (1 - alpha)

    nb_iter_inner = 0
    f = np.zeros(len(yX), dtype=np.float64)
    f_prev = np.full(len(yX), np.inf, dtype=np.float64)
    while np.linalg.norm(f - f_prev, ord=1) > eps and nb_iter_inner < max_iter:
        f_prev = f
        f = (1 - alpha) * y + alpha * adj_X.dot(f)
        nb_iter_inner = nb_iter_inner + 1

    convergent = nb_iter_inner < max_iter

    return f, convergent


def minprop_2(adj_X, adj_Y, biadj_XY, yX, yY, alphaX=0.3, alphaY=0.3, eps=1e-6, max_iter=1000):
    # vectors of final label values (initialization)
    fX = np.zeros(len(yX), dtype=np.float64)
    fY = np.zeros(len(yY), dtype=np.float64)
    # temporal vectors for numerical computation (initialization)
    fX_old = np.full(len(yX), np.inf, dtype=np.float64)
    fY_old = np.full(len(yY), np.inf, dtype=np.float64)

    nb_iter_outer = 0
    while (np.linalg.norm(fX - fX_old, ord=1) > eps or np.linalg.norm(fY - fY_old, ord=1) > eps) and nb_iter_outer < max_iter:
        fX_old = fX
        fY_old = fY

        # for network P
        fX, convergent = single_network_propagation_2(adj_X, biadj_XY, fY, yX, alphaX, eps, max_iter)
        # for network D
        fY, convergent = single_network_propagation_2(adj_Y, biadj_XY.T, fX, yY, alphaY, eps, max_iter)
        # iteration
        nb_iter_outer = nb_iter_outer + 1

    convergent = nb_iter_outer < max_iter
    
    return fX, fY, convergent

##################################################
def single_network_propagation_3(adj_X, biadj_XY, biadj_XZ, fY, fZ, yX, alpha=0.3, eps=1e-6, max_iter=1000):
    sum_sij_fj = biadj_XY.dot(fY) + biadj_XZ.dot(fZ)
    y = (1 - 3 * alpha) * yX / (1 - alpha) + alpha * sum_sij_fj / (1 - alpha)

    nb_iter_inner = 0
    f = np.zeros(len(yX), dtype=np.float64)
    f_prev = np.full(len(yX), np.inf, dtype=np.float64)
    while np.linalg.norm(f - f_prev, ord=1) > eps and nb_iter_inner < max_iter:
        f_prev = f
        f = (1 - alpha) * y + alpha * adj_X.dot(f)
        nb_iter_inner = nb_iter_inner + 1

    convergent = nb_iter_inner < max_iter

    return f, convergent

def minprop_3(adj_X, adj_Y, adj_Z, biadj_XY, biadj_XZ, biadj_YZ, yX, yY, yZ, alphaX=0.3, alphaY=0.3, alphaZ=0.3, eps=1e-6, max_iter=1000):
    # vectors of final label values (initialization)
    fX = np.zeros(len(yX), dtype=np.float64)
    fY = np.zeros(len(yY), dtype=np.float64)
    fZ = np.zeros(len(yZ), dtype=np.float64)
    # temporal vectors for numerical computation (initialization)
    fX_old = np.full(len(yX), np.inf, dtype=np.float64)
    fY_old = np.full(len(yY), np.inf, dtype=np.float64)
    fZ_old = np.full(len(yZ), np.inf, dtype=np.float64)

    nb_iter_outer = 0
    while (np.linalg.norm(fX - fX_old, ord=1) > eps or np.linalg.norm(fY - fY_old, ord=1) > eps or np.linalg.norm(fZ - fZ_old, ord=1) > eps) and nb_iter_outer < max_iter:
        fX_old = fX
        fY_old = fY
        fZ_old = fZ

        # for network P
        fX, convergent = single_network_propagation_3(adj_X, biadj_XY, biadj_XZ, fY, fZ, yX, alphaX, eps, max_iter)
        # for network D
        fY, convergent = single_network_propagation_3(adj_Y, biadj_XY.T, biadj_YZ, fX, fZ, yY, alphaY, eps, max_iter)
        # for network C
        fZ, convergent = single_network_propagation_3(adj_Z, biadj_XZ.T, biadj_YZ.T, fX, fY, yZ, alphaZ, eps, max_iter)
        # iteration
        nb_iter_outer = nb_iter_outer + 1

    convergent = nb_iter_outer < max_iter
    
    return fX, fY, fZ, convergent

def sample_data(seed=123):
    #### load networks (hypothetical) #######
    nb_nodes_networkP = 4
    nb_nodes_networkD = 6
    nb_nodes_networkC = 8
    # Homo networks
    obj_networkP = nx.gnm_random_graph(nb_nodes_networkP, nb_nodes_networkP * 2, seed=seed)
    obj_networkD = nx.gnm_random_graph(nb_nodes_networkD, nb_nodes_networkD * 2, seed=seed)
    obj_networkC = nx.gnm_random_graph(nb_nodes_networkC, nb_nodes_networkC * 2, seed=seed)
    # Hetero networks
    obj_networkPD = bipartite.random_graph(nb_nodes_networkP, nb_nodes_networkD, 0.8, seed=seed)
    obj_networkPC = bipartite.random_graph(nb_nodes_networkP, nb_nodes_networkC, 0.8, seed=seed)
    obj_networkDC = bipartite.random_graph(nb_nodes_networkD, nb_nodes_networkC, 0.8, seed=seed)

    ### Weight matrices for network propagation (Normalized weighted adjacency matrices)
    # Homo networks (Assuming no node with degree 0)
    adj_networkP = nx.adjacency_matrix(obj_networkP)
    deg_networkP = np.sum(adj_networkP, axis=0)
    norm_adj_networkP = sp.csr_matrix(adj_networkP / np.sqrt(np.dot(deg_networkP.T, deg_networkP)), dtype=np.float64)

    adj_networkD = nx.adjacency_matrix(obj_networkD)
    deg_networkD = np.sum(adj_networkD, axis=0)
    norm_adj_networkD = sp.csr_matrix(adj_networkD / np.sqrt(np.dot(deg_networkD.T, deg_networkD)), dtype=np.float64)

    adj_networkC = nx.adjacency_matrix(obj_networkC)
    deg_networkC = np.sum(adj_networkC, axis=0)
    norm_adj_networkC = sp.csr_matrix(adj_networkC / np.sqrt(np.dot(deg_networkC.T, deg_networkC)), dtype=np.float64)

    # Hetero networks
    biadj_networkPD = bipartite.biadjacency_matrix(obj_networkPD, row_order=range(nb_nodes_networkP))
    degP = np.sum(biadj_networkPD, axis=1)
    degD = np.sum(biadj_networkPD, axis=0)
    norm_biadj_networkPD = sp.csr_matrix(biadj_networkPD / np.sqrt(np.dot(degP, degD)), dtype=np.float64)
    norm_biadj_networkPD.data[np.isnan(norm_biadj_networkPD.data)] = 0.0
    norm_biadj_networkPD.eliminate_zeros()

    biadj_networkPC = bipartite.biadjacency_matrix(obj_networkPC, row_order=range(nb_nodes_networkP))
    degP = np.sum(biadj_networkPC, axis=1)
    degC = np.sum(biadj_networkPC, axis=0)
    norm_biadj_networkPC = sp.csr_matrix(biadj_networkPC / np.sqrt(np.dot(degP, degC)), dtype=np.float64)
    norm_biadj_networkPC.data[np.isnan(norm_biadj_networkPC.data)] = 0.0
    norm_biadj_networkPC.eliminate_zeros()

    biadj_networkDC = bipartite.biadjacency_matrix(obj_networkDC, row_order=range(nb_nodes_networkD))
    degD = np.sum(biadj_networkDC, axis=1)
    degC = np.sum(biadj_networkDC, axis=0)
    norm_biadj_networkDC = sp.csr_matrix(biadj_networkDC / np.sqrt(np.dot(degD, degC)), dtype=np.float64)
    norm_biadj_networkDC.data[np.isnan(norm_biadj_networkDC.data)] = 0.0
    norm_biadj_networkDC.eliminate_zeros()

    return norm_adj_networkP, norm_adj_networkD, norm_adj_networkC, norm_biadj_networkPD, norm_biadj_networkPC, norm_biadj_networkDC
