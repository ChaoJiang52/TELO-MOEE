import numpy as np
import scipy
from numpy import exp, sqrt
from scipy.stats import norm

from .acq_func_optimisers import BaseOptimiser


def cal_mgfi(y_hat, sd, y_best, t):
    y_hat = np.reshape(y_hat, (-1, 1))
    sd = np.reshape(sd, (-1, 1))

    y_hat_p = y_hat - t * sd ** 2.0
    # mgfi = 0 if sd = 0, so mask the non-zero sigma elements
    mgfi = np.zeros_like(y_hat_p)
    mask = (sd != 0).ravel()

    beta_p = (y_best - y_hat_p) / sd[mask]
    term = t * (y_best - y_hat - 1)

    mgfi[mask] = norm.cdf(beta_p) * exp(term + t ** 2.0 * sd[mask] ** 2.0 / 2.0)

    # minimisation
    return -mgfi


class MGFI(BaseOptimiser):
    """Moment Generating Function of the Improvement (MGFI)

    M(x; t) = \Phi((plugin - m(x) + s(x)^2 * t - 1) / s(x)) * exp((plugin - m(x) - 1) * t + (s(x) * t)^2 / 2)

    References:

        Wang, Hao, Bas van Stein, Michael Emmerich, and Thomas Back. "A new acquisition function for Bayesian
        optimization based on the moment-generating function." In 2017 IEEE International Conference on
        Systems, Man, and Cybernetics (SMC), pp. 507-512. IEEE, 2017.
    """

    def __call__(self, model):
        D = self.lb.shape[0]
        # t = 2
        t = self.acquisition_args['t']

        def min_obj(x):
            # if we have a constraint function and it is violated,
            # return a bad UCB value
            if (self.cf is not None) and (not self.cf(x)):
                return np.inf

            mu, sigmaSQR = model.predict(np.atleast_2d(x), full_cov=False)
            sigma = np.sqrt(sigmaSQR)

            points = cal_mgfi(
                mu,
                sigma,
                y_best=np.min(model.Y),
                t=t
            ).ravel()

            return points

        # number of optimisation runs and *estimated* number of L-BFGS-B
        # function evaluations per run; note this was calculate empirically and
        # may not be true for all functions.
        N_opt_runs = 10
        fevals_assumed_per_run = 100

        N_samples = int(self.acq_budget - (N_opt_runs * fevals_assumed_per_run))
        if N_samples <= N_opt_runs:
            N_samples = N_opt_runs

        # initially perform a grid search for N_samples
        x0_points = np.random.uniform(self.lb, self.ub, size=(N_samples, D))
        fx0 = min_obj(x0_points).ravel()

        # select the top N_opt_runs to evaluate with L-BFGS-B
        x0_points = x0_points[np.argsort(fx0)[:N_opt_runs - 1], :]
        x0_points = np.concatenate((x0_points, np.atleast_2d(model.X[np.argmin(model.Y)])))
        # Find the best optimum by starting from n_restart different random points.
        # below is equivilent to: [(l, b) for (l, b) in zip(self.lb, self.ub)]
        bounds = [*zip(self.lb, self.ub)]

        # storage for the best found location (xb) and its function value (fx)
        xb = np.zeros((N_opt_runs, D))
        fx = np.zeros((N_opt_runs, 1))

        # ensure we're using a good stopping criterion
        # ftol = factr * numpy.finfo(float).eps
        factr = 1e-15 / np.finfo(float).eps

        # run L-BFGS-B on each of the 'N_opt_runs' starting locations
        for i, x0 in enumerate(x0_points):
            xb[i, :], fx[i, :], _ = scipy.optimize.fmin_l_bfgs_b(
                min_obj, x0=x0, bounds=bounds, approx_grad=True, factr=factr
            )

        # return the best location
        best_idx = np.argmin(fx.flat)
        return xb[best_idx, :]
