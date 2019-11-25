import sys
sys.dont_write_bytecode = True

import numpy as np
import scipy.sparse as sp
from network_propagation_methods import sample_data, netprop, minprop_2, minprop_3

#### Parameters #############
# convergence threshold
eps = 1e-6
# maximum number of iterations
max_iter = 1000
# diffusion parameters
alphaP, alphaD, alphaC = 0.33, 0.33, 0.33
# random seed
seed = 123

#### load networks (hypothetical data) #######
norm_adj_networkP, norm_adj_networkD, norm_adj_networkC, norm_biadj_networkPD, norm_biadj_networkPC, norm_biadj_networkDC = sample_data(seed)

#### Network propagation ###########################
# Initial labels (hypothetical data)
np.random.seed(seed=seed)
yP = np.array(np.random.rand(norm_adj_networkP.shape[0]), dtype=np.float64)
yD = np.array(np.random.rand(norm_adj_networkD.shape[0]), dtype=np.float64)
yC = np.array(np.random.rand(norm_adj_networkC.shape[0]), dtype=np.float64)

## network propagation with single network
fP, convergent = netprop(norm_adj_networkP, yP, alphaP, eps, max_iter)

print(convergent)
print(fP)

## MINProp with 2 homo subnetworks
fP, fD, convergent = minprop_2(norm_adj_networkP, norm_adj_networkD, norm_biadj_networkPD, yP, yD, alphaP, alphaD, eps, max_iter)

print(convergent)
print(fP)
print(fD)

## MINProp with 3 homo subnetworks
fP, fD, fC, convergent = minprop_3(norm_adj_networkP, norm_adj_networkD, norm_adj_networkC, norm_biadj_networkPD, norm_biadj_networkPC, norm_biadj_networkDC, yP, yD, yC, alphaP, alphaD, alphaC, eps, max_iter)

print(convergent)
print(fP)
print(fD)
print(fC)
