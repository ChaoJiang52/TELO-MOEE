# Multi-Objectivising Acquisition Functions in Bayesian Optimisation

This repository contains the Python3 code for the MOEE method.

## Reproduction of experiments

The python file `optimizer.py` provides a convenient way to reproduce all 
experimental evaluations carried out the paper. 

```bash
> python optimizer.py
```

## Training data

The initial training locations for each of the 30 sets of
[Latin hypercube](https://www.jstor.org/stable/1268522) samples are located in
the `training_data` directory in this repository with the filename structure
`ProblemName_number`, e.g. the first set of training locations for the Branin
problem is stored in `Branin_1.npz`. 
To load and inspect these values use the following instructions:

```python
> cd /egreedy
> python
>>> import numpy as np
>>> with np.load('training_data/Branin_1.npz') as data:
        Xtr = data['arr_0']
        Ytr = data['arr_1']
>>> Xtr.shape, Ytr.shape
((4, 2), (4, 1))
```

## Citation

If you use this code in your work, please consider cite our
[paper](https://doi.org/10.1145/3716504):

```bibtex
@article{jiang2025multi,
author = {Jiang, Chao and Li, Miqing},
title = {Multi-objectivising acquisition functions in Bayesian optimisation},
journal = {ACM Transactions on Evolutionary Learning and Optimization},
year = {2025},
volume = {5},
number = {2},
articleno = {15},
numpages = {33},
publisher = {Association for Computing Machinery}
}
```

## Acknowledgement

This repository has been developed based on the work from [egreedy](https://github.com/georgedeath/egreedy) [De Ath et al., 2021]. We would like to express our gratitude to the original authors and contributors for their pioneering efforts and for making their code available. Their work has been instrumental in the development of this project.

[De Ath et al., 2021] George De Ath, Richard M. Everson, Alma A. M. Rahat, and Jonathan E. Fieldsend. 2021. Greed Is Good: Exploration and Exploitation Trade-offs in Bayesian Optimisation. ACM Trans. Evol. Learn. Optim. 1, 1, Article 1 (May 2021), 22 pages.
