import numpy as np
from .nsga2_pareto_front import NSGA2_pygmo_MACE
from .acq_func_optimisers import BaseOptimiser


class MACE(BaseOptimiser):

    def __call__(self, model):
        X_front, musigma_front = NSGA2_pygmo_MACE(model, self.acq_budget,
                                                  self.lb, self.ub, self.cf)
        Xnew = X_front[np.random.choice(X_front.shape[0], 1), :]

        return Xnew
