import numpy as np
import pandas as pd
import scipy
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import trim_mean
from scipy.stats import norm

from .acq_func_optimisers import BaseOptimiser


def LCB(mu, sigma, t, d):
    beta = 2 * np.log(d * t ** 2 / 1)
    return mu - np.sqrt(beta) * sigma


def UCB(mu, sigma, t, d):
    beta = 2 * np.log(d * t ** 2 / 1)
    return mu + np.sqrt(beta) * sigma


def WEI(mu, sigma, y_best, alpha, PHI=norm.cdf, phi=norm.pdf):
    # ensure mu and sigma are column vectors
    mu = np.reshape(mu, (-1, 1))
    sigma = np.reshape(sigma, (-1, 1))

    # minimisation version of EI
    improvement = y_best - mu

    # EI = 0 if sigma = 0, so mask the non-zero sigma elements
    ei = np.zeros_like(improvement)
    mask = (sigma != 0).ravel()

    s = improvement[mask] / sigma[mask]

    ei[mask] = alpha * improvement[mask] * PHI(s) + (1 - alpha) * sigma[mask] * phi(s)

    return ei


def detect_switch(UBR, window_size=10, atol_rel=0.1):
    miqm = apply_moving_iqm(U=UBR, window_size=window_size)
    miqm_gradient = np.gradient(miqm)

    # max_grad = np.maximum.accumulate(miqm_gradient)
    # switch = np.array([np.isclose(miqm_gradient[i], 0, atol=atol_rel*max_grad[i]) for i in range(len(miqm_gradient))])
    # switch[0] = 0  # misleading signal bc of iqm

    G_abs = np.abs(miqm_gradient)
    max_grad = [np.nanmax(G_abs[:i + 1]) for i in range(len(G_abs))]
    switch = np.array([np.isclose(miqm_gradient[i], 0, atol=atol_rel * max_grad[i]) for i in range(len(miqm_gradient))])
    # switch = np.isclose(miqm_gradient, 0, atol=1e-5)
    switch[:window_size] = 0  # misleading signal bc of iqm

    return switch


# Moving IQM
def apply_moving_iqm(U: np.array, window_size: int = 5) -> np.array:
    def moving_iqm(X: np.array) -> float:
        return trim_mean(X, 0.25)

    U_padded = np.concatenate((np.array([U[0]] * (window_size - 1)), U))
    # slices = sliding_window_view(U_padded, window_size)
    slices = sliding_window_view(U_padded.reshape(-1), window_size)
    miqm = np.array([moving_iqm(s) for s in slices])
    return miqm


class AWEI(BaseOptimiser):
    """Self-Adjusting Weighted Expected Improvement (SAWEI)

    References:

        Carolin Benjamins, Elena Raponi, Anja Jankovic, Carola Doerr, and Marius Lindauer. 2023.
        Self-Adjusting Weighted Expected Improvement for Bayesian Optimization.
        In International Conference on Automated Machine Learning (AutoML 2023). Potsdam, Germany.
    """

    def __call__(self, model):
        # alpha = 0.5
        delta = 0.1
        window_size = 7
        atol_rel = 0.1
        track_attitude = "last"
        # Check if it is time to switch
        switch = False
        D = self.lb.shape[0]
        incumbent = model.Y.min()

        def min_UCB(x):
            # if we have a constraint function and it is violated,
            # return a bad UCB value
            if (self.cf is not None) and (not self.cf(x)):
                return np.inf

            mu, sigmaSQR = model.predict(np.atleast_2d(x), full_cov=False)

            points = UCB(
                mu,
                np.sqrt(sigmaSQR),
                t=model.X.shape[0] + 1,
                d=model.X.shape[1]
            ).ravel()

            return points

        # objective function wrapper for L-BFGS-B
        def min_LCB(x):
            # if we have a constraint function and it is violated,
            # return a bad UCB value
            if (self.cf is not None) and (not self.cf(x)):
                return np.inf

            mu, sigmaSQR = model.predict(np.atleast_2d(x), full_cov=False)

            points = LCB(
                mu,
                np.sqrt(sigmaSQR),
                t=model.X.shape[0] + 1,
                d=model.X.shape[1]
            ).ravel()
            return points

        # minimal ucb

        min_ucb = np.min(min_UCB(model.X))

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
        fx0 = min_LCB(x0_points).ravel()

        # select the top N_opt_runs to evaluate with L-BFGS-B
        x0_points = x0_points[np.argsort(fx0)[:N_opt_runs-1], :]
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
                min_LCB, x0=x0, bounds=bounds, approx_grad=True, factr=factr
            )

        # return the best location
        best_idx = np.argmin(fx.flat)

        min_lcb = xb[best_idx, :]

        ubr = min_ucb - min_lcb
        self.acquisition_args['ubr'].append(ubr)

        UBR = self.acquisition_args['ubr']

        # and we need at least 2 UBRs to compute the gradient
        if len(UBR) > 2:
            switch = detect_switch(UBR=UBR[1:], window_size=window_size, atol_rel=atol_rel)[-1]

        print(switch)
        if switch:
            if track_attitude == "last":
                # Calculate attitude: Exploring or exploiting?
                # Exploring = when ei term is bigger
                # Exploiting = when pi term is bigger
                exploring = self.acquisition_args["wei_pi_pure_term"] <= self.acquisition_args["wei_ei_term"]

            # If attitude is
            # - exploring (exploring==True): increase alpha, change to exploiting
            # - exploiting (exploring==False): decrease alpha, change to exploring
            sign = 1 if exploring else -1
            alpha = self.acquisition_args['alpha'] + sign * delta

            # Bound alpha
            self.acquisition_args['alpha'] = max(0., min(1., alpha))

        # objective function wrapper for L-BFGS-B
        def min_wei(x):
            # if we have a constraint function and it is violated,
            # return a bad EI value
            if (self.cf is not None) and (not self.cf(x)):
                return np.inf

            mu, sigmaSQR = model.predict(np.atleast_2d(x), full_cov=False)

            # negate EI because we're using a minimiser
            ei = -WEI(
                mu, np.sqrt(sigmaSQR), incumbent, self.acquisition_args['alpha']
            ).ravel()
            return ei

        # initially perform a grid search for N_samples
        x0_points = np.random.uniform(self.lb, self.ub, size=(N_samples, D))
        fx0 = min_wei(x0_points).ravel()

        # select the top N_opt_runs to evaluate with L-BFGS-B
        x0_points = x0_points[np.argsort(fx0)[:N_opt_runs], :]

        # Find the best optimum by starting from n_restart different random points.
        # below is equivilent to: [(l, b) for (l, b) in zip(self.lb, self.ub)]
        bounds = [*zip(self.lb, self.ub)]

        # storage for the best found location (xb) and its function value (fx)
        xb = np.zeros((N_opt_runs, D))
        fx = np.zeros((N_opt_runs, 1))

        # run L-BFGS-B on each of the 'N_opt_runs' starting locations
        for i, x0 in enumerate(x0_points):
            xb[i, :], fx[i, :], _ = scipy.optimize.fmin_l_bfgs_b(
                min_wei, x0=x0, bounds=bounds, approx_grad=True, factr=factr
            )

        # return the best location
        best_idx = np.argmin(fx.flat)

        min_wei = xb[best_idx, :]

        mu, sigmaSQR = model.predict(np.atleast_2d(min_wei), full_cov=False)
        sigma = np.sqrt(sigmaSQR)

        # minimisation version of EI
        improvement = incumbent - mu

        s = improvement / sigma
        self.acquisition_args["wei_pi_pure_term"] = norm.cdf(s)
        self.acquisition_args["wei_ei_term"] = sigma * norm.pdf(s)

        print(self.acquisition_args["alpha"])
        return min_wei
