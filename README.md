# Network propagation
Python-implemented link prediction algorithms based on network propagation are available.

## PRINCE (PRIoritizatioN and Complex Elucidation)
Vanunu O, Magger O, Ruppin E, Shlomi T, Sharan R (2010) Associating Genes and Protein Complexes with Disease via Network Propagation. PLoS Comput Biol 6(1): e1000641. doi:[10.1371/journal.pcbi.1000641](https://doi.org/10.1371/journal.pcbi.1000641)

### Data
* [Human protein-protein interaction network](http://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/hippie_current.txt)form [HIPPIE (Human Integrated Protein-Protein Interaction rEference)](http://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/)
* [Phenotypic disease similarity (zipped)](http://www.cs.tau.ac.il/%7Ebnet/software/PrincePlugin/PhenSim.zip) from [PRINCE Plugin](http://www.cs.tau.ac.il/~bnet/software/PrincePlugin/)
* [OMIM disease-gene associations](http://www.cs.tau.ac.il/%7Ebnet/software/PrincePlugin/associations.txt) from [PRINCE Plugin](http://www.cs.tau.ac.il/~bnet/software/PrincePlugin/)

### Usage
```
python run_prince.py --alpha 0.5
```

## MINPROP
Hwang TH, Kuang R (2010) A Heterogeneous Label Propagation Algorithm for Disease Gene Discovery. in Proceedings of the 2010 SIAM International Conference on Data Mining, 583-594. doi:[10.1137/1.9781611972801.51](https://doi.org/10.1137/1.9781611972801.51)

*in preparation.*
