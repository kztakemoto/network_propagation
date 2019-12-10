# Network propagation
Network propagation-based link prediction methods implemented in Python.

PRINCE and MINProp are available.

## Network data
* [Human protein-protein interaction network](http://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/hippie_current.txt) form [HIPPIE (Human Integrated Protein-Protein Interaction rEference)](http://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/)
* [Phenotypic disease similarity (zipped)](http://www.cs.tau.ac.il/%7Ebnet/software/PrincePlugin/PhenSim.zip) from [PRINCE Plugin](http://www.cs.tau.ac.il/~bnet/software/PrincePlugin/)
* [OMIM disease-gene associations](http://www.cs.tau.ac.il/%7Ebnet/software/PrincePlugin/associations.txt) from [PRINCE Plugin](http://www.cs.tau.ac.il/~bnet/software/PrincePlugin/)

### Usage
Download and pickle the network data.
```
python download_pickle_data.py
```

## PRINCE (PRIoritizatioN and Complex Elucidation)
Vanunu O, Magger O, Ruppin E, Shlomi T, Sharan R (2010) Associating Genes and Protein Complexes with Disease via Network Propagation. PLoS Comput Biol 6(1): e1000641. doi:[10.1371/journal.pcbi.1000641](https://doi.org/10.1371/journal.pcbi.1000641)

### Usage
Protein-disease association prediction.
```
python run_prince.py --alpha 0.5
```

## MINProp (Mutual Interaction-based Network Propagation)
Hwang TH, Kuang R (2010) A Heterogeneous Label Propagation Algorithm for Disease Gene Discovery. in Proceedings of the 2010 SIAM International Conference on Data Mining, 583-594. doi:[10.1137/1.9781611972801.51](https://doi.org/10.1137/1.9781611972801.51)

### Usage
#### General
See ``example_minprop.py`` for the usage of the MINProp-related functions.

#### Example
Protein-disease association prediction using 2 homo subnetworks: the protein-protein interaction network and disease similarity network.
```
python run_minprop_PD.py --alphaP 0.15 --alphaD 0.05 --eps 0.01
```
