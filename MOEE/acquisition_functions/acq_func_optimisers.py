"""A set of classes that take in a GPy model and optimise their respective
acquisition functions over the model's decision space.

Each class can be used as follows:
>> acq_class = EI
>> acq_optimiser = acq_class(lb, ub, acq_budget, cf=None, args)
>> acq_optimiser(gpy_model)

The basic usage is that an optimiser is instantiated with the problem bounds,
``lb`` and ``ub``, a budget of calls to the GPy model (used for predicting the
mean and variance of locations in decision space), a constraint function that
returns True or False depending on if the decision vector it is given violates
any problem constraints, and additional arguments in the form of a dictionary
containing key: value pairs that are passed into the acquisition function used
by the optimiser; e.g. for the UCB acquisition function the value of beta is
needed and can be specified: args = {'beta': 2.5}.

Note that all acquisition optimisers use the NSGA-II algorithm apart from PI
which uses a multi-restart strategy, seeded by the best locations found from
uniformly sampling decision space.
"""
import scipy
import numpy as np

from . import standard_acq_funcs_minimize
from . import egreedy_acq_funcs_minimize
from .nsga2_pareto_front import NSGA2_pygmo, two_ac_pf
import matplotlib.pyplot as plt
# from kneebow.rotor import Rotor
from topsis import topsis
import re


def normalized_f(f):
    return (f - np.min(f)) / (np.max(f) - np.min(f))


def get_L(point1, point2):
    # get extreme line L: ax+by+c=0
    # a=y1-y2,
    # b=x2-x1,
    # c=x1y2-x2y1,
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]

    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    return a, b, c


def get_distance(point, a, b, c):
    # get the distance to extreme line L
    numerator = a * point[0] + b * point[1] + c
    denominator = np.sqrt(a ** 2 + b ** 2)
    if numerator > 0:
        return np.abs(numerator) / denominator
    else:
        return -np.abs(numerator) / denominator


def get_knee(X_front, acf1_acf2_front, dim, acquisition_args=None):
    # normalizing
    # data = np.array([normalized_f(acf1_acf2_front[:, 0]), normalized_f(acf1_acf2_front[:, 1])]).T
    # a, b, c = 1, 1, -1

    data = np.array([acf1_acf2_front[:, 0], acf1_acf2_front[:, 1]]).T
    a, b, c = get_L(data[0], data[-1])
    if a < 0:
        a *= -1
        b *= -1
        c *= -1
    all_dis = np.zeros(len(acf1_acf2_front))
    for idx, value in enumerate(data):
        # get distance
        numerator = a * value[0] + b * value[1] + c
        denominator = np.sqrt(a ** 2 + b ** 2)
        if numerator > 0:
            dis = np.abs(numerator) / denominator
        else:
            dis = -np.abs(numerator) / denominator

        all_dis[idx] = dis

    max_idx = np.argmax(all_dis)

    # plot front
    plt.plot(data[:, 0], data[:, 1])
    plt.scatter(data[max_idx][0], data[max_idx][1], color="r")
    plt.scatter(data[0][0], data[0][1], color="yellow")
    plt.show()

    # if trap into local optimal, execute explore
    if acquisition_args['explore'] >= dim:
        acquisition_args['explore'] = 0
        return X_front[max_idx]
        # return X_front[-1]
    else:
        return X_front[0]
    # if max_idx != 0:
    #     return X_front[max_idx]
    # else:
    #     if np.random.random() < 0.1:
    #         # return X_front[np.random.choice(X_front.shape[0]), :]
    #         return X_front[-1]
    #     else:
    #         return X_front[0]


# def get_knee_from_kneebow(X_front, acf1_acf2_front):
#     rotor = Rotor()
#     rotor.fit_rotate(acf1_acf2_front)
#     elbow_idx = rotor.get_elbow_index()
#     # rotor.plot_elbow()
#     # plt.show()
#     knee_x = X_front[elbow_idx]
#     return knee_x


def get_topsis(X_front, acf1_acf2_front, acquisition_args=None):
    data = np.array([acf1_acf2_front[:, 0], acf1_acf2_front[:, 1]]).T
    # if acquisition_args['explore'] >= np.sqrt(dim):
    acquisition_args['explore'] = 0
    # a = [[7, 9, 9, 8], [8, 7, 8, 7], [9, 6, 8, 9], [6, 7, 8, 6]]
    # if acquisition_args["type"] == "EI":
    #     w = [0.4, 0.6]
    # elif acquisition_args["type"] == "UCB":
    #     w = [0.4, 0.6]
    if "weights" in acquisition_args:
        w = acquisition_args["weights"]
    else:
        w = [0.4, 0.6]
    I = [1, 1]  # benefit (1) cost (0)
    decision = topsis(data, w, I)
    topsis_idx = int(re.findall(r'\d+', str(decision))[0])
    # topsis_idx = decision.optimum_choice

    if topsis_idx == 0:
        print("change point")
        topsis_idx = -1

    # plt.plot(acf1_acf2_front[:, 0], acf1_acf2_front[:, 1])
    # plt.scatter(data[topsis_idx][0], data[topsis_idx][1], linewidths=5, label='Elbow point', edgecolors='purple')
    # plt.show()
    point = X_front[topsis_idx]
    print("topsis result", point)
    return point
    # else:
    #     print("exploit")
    #     return X_front[0]


class BaseOptimiser:
    """Class of methods that maximise an acquisition function over a GPy model.

    Parameters
    ----------
        lb : (D, ) numpy.ndarray
            Lower bound box constraint on D
        ub : (D, ) numpy.ndarray
            Upper bound box constraint on D
        acq_budget : int
            Maximum number of calls to the GPy model
        cf : callable, optional
            Constraint function that returns True if it is called with a
            valid decision vector, else False.
        acquisition_args : dict, optional
            A dictionary containing key: value pairs that will be passed to the
            corresponding acquisition function, see the classes below for
            further details.
    """

    # add dim
    def __init__(self, lb, ub, acq_budget, cf=None, acquisition_args={}):
        self.lb = lb
        self.ub = ub
        self.cf = cf
        self.acquisition_args = acquisition_args
        self.acq_budget = acq_budget

    def __call__(self, model):
        raise NotImplementedError()


class ParetoFrontOptimiser(BaseOptimiser):
    """Class of acquisition function optimisers that use Pareto fronts.

    The (estimated) Pareto front is calculated using NSGA-II [1]_, for full
    details of the method see: nsga2_pareto_front.NSGA2_pygmo

    References
    ----------
    .. [1] Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan.
       A fast and elitist multiobjective genetic algorithm: NSGA-II.
       IEEE Transactions on Evolutionary Computation 6, 2 (2001), 182–197.
    """

    def get_front(self, model):
        """Gets the (estimated) Pareto front of the predicted mean and
        standard deviation of a GPy.models.GPRegression model.
        """
        X_front, musigma_front = NSGA2_pygmo(
            model, self.acq_budget, self.lb, self.ub, self.cf
        )

        return X_front, musigma_front[:, 0], musigma_front[:, 1]

    def __call__(self, model):
        raise NotImplementedError()


class EI(ParetoFrontOptimiser):
    """Selects the point on a GPy model's Pareto front that maximises EI.

    See standard_acq_funcs_minimize.EI for details of the EI method.
    """

    def __call__(self, model):
        X, mu, sigma = self.get_front(model)
        ei = standard_acq_funcs_minimize.EI(mu, sigma, y_best=np.min(model.Y))
        return X[np.argmax(ei), :]


class UCB(ParetoFrontOptimiser):
    """Selects the point on a GPy model's Pareto front that maximises UCB.

    See standard_acq_funcs_minimize.UCB for details of the UCB method and its
    optional arguments.
    """

    def __call__(self, model):
        X, mu, sigma = self.get_front(model)
        ucb = standard_acq_funcs_minimize.UCB(
            mu,
            sigma,
            lb=self.lb,
            ub=self.ub,
            t=model.X.shape[0] + 1,
            d=model.X.shape[1],
            **self.acquisition_args
        )
        return X[np.argmax(ucb), :]


class MOEE(BaseOptimiser):
    def __call__(self, model):
        X_front, acf1_acf2_front = two_ac_pf(model, self.acq_budget, self.lb, self.ub, self.cf, self.acquisition_args)
        dim = self.lb.size

        if self.acquisition_args["type2"] == "exploit":
            return X_front[0]
        elif self.acquisition_args["type2"] == "topsis":
            # if trap into local optimal, execute explore
            Ymax = np.max(model.Y)
            Ymin = np.min(model.Y)
            Ynew = model.Y[-1]
            N_Ynew = np.abs(Ynew - np.min(model.Y[:-1])) / (Ymax - Ymin)
            print("dif:", N_Ynew)
            if N_Ynew < 0.1 ** 4:
                self.acquisition_args['explore'] += 1

            print("explore", self.acquisition_args['explore'])

            # only one point
            if acf1_acf2_front.shape[0] != 1:
                if "denom" in self.acquisition_args:
                    if self.acquisition_args["denom"] == 0:
                        condition = 0
                    else:
                        condition = 1 / self.acquisition_args["denom"]
                else:
                    condition = 1 / 2

                # if self.acquisition_args['explore'] >= np.sqrt(dim):
                if self.acquisition_args['explore'] >= dim ** (condition):
                    # get the TOPSIS point
                    tops = get_topsis(X_front, acf1_acf2_front, self.acquisition_args)
                    return tops
                else:
                    print("exploit")
                    return X_front[0]
            else:
                print("one point from PF")
                return X_front[0]

        elif self.acquisition_args["type2"] == "random":
            Ymax = np.max(model.Y)
            Ymin = np.min(model.Y)
            Ynew = model.Y[-1]
            N_Ynew = np.abs(Ynew - np.min(model.Y[:-1])) / (Ymax - Ymin)
            if N_Ynew < 0.1 ** 4:
                self.acquisition_args['explore'] += 1

            if self.acquisition_args['explore'] >= np.sqrt(dim):
                self.acquisition_args['explore'] = 0
                return X_front[np.random.choice(X_front.shape[0]), :]
            else:
                return X_front[0]

        elif self.acquisition_args["type2"] == "ePF":
            Ymax = np.max(model.Y)
            Ymin = np.min(model.Y)
            Ynew = model.Y[-1]
            N_Ynew = np.abs(Ynew - np.min(model.Y[:-1])) / (Ymax - Ymin)
            if N_Ynew >= 0.1 ** 4:
                return X_front[0]
            else:
                return X_front[np.random.choice(X_front.shape[0]), :]

        elif self.acquisition_args["type2"] == "PF":
            print("random select from PF")
            return X_front[np.random.choice(X_front.shape[0]), :]

        elif self.acquisition_args["type2"] == "switch":
            if np.random.random() < 0.5:
                print("exploit")
                return X_front[0]
            else:
                print("explore")
                # dim = self.lb.size
                # find elbow point using distance
                tops = get_topsis(X_front, acf1_acf2_front, self.acquisition_args)
                return tops


class eFront(ParetoFrontOptimiser):
    """Selects the point on a GPy model's Pareto front via the eFront method.

    eFront greedily selects a point an (estimated) Pareto front that has the
    best (lowest) mean predicted value with probability (1 - epsilon) and
    randomly selects a point on the front with probability epsilon.
    """

    def __call__(self, model):
        X, mu, sigma = self.get_front(model)
        Xnew = egreedy_acq_funcs_minimize.eFront(X, mu, sigma, **self.acquisition_args)
        return Xnew


class eRandom(ParetoFrontOptimiser):
    """Selects the point on a GPy model's Pareto front via the eRandom method.

    eRandom greedily selects a point an (estimated) Pareto front that has the
    best (lowest) mean predicted value with probability (1 - epsilon) and
    randomly selects a point in decision space with probability epsilon.
    """

    def __call__(self, model):
        X, mu, sigma = self.get_front(model)
        Xnew = egreedy_acq_funcs_minimize.eRandom(
            X, mu, sigma, lb=self.lb, ub=self.ub, cf=self.cf, **self.acquisition_args
        )
        return Xnew


class PFRandom(ParetoFrontOptimiser):
    """Selects the point on a GPy model's Pareto front via the PFRandom method.

    PFRandom randomly selects a point on the Pareto front.
    """

    def __call__(self, model):
        X, _, _ = self.get_front(model)
        Xnew = egreedy_acq_funcs_minimize.PFRandom(X)
        return Xnew


class Explore(ParetoFrontOptimiser):
    """Selects the point on a GPy model's Pareto front via the Explore method.

    Explore selects the most exploratory point on the front, i.e. the location
    with the largest standard deviation.
    """

    def __call__(self, model):
        X, _, sigma = self.get_front(model)
        Xnew = egreedy_acq_funcs_minimize.Explore(X, sigma)
        return Xnew


class Exploit(ParetoFrontOptimiser):
    """Selects the point on a GPy model's Pareto front via the Exploit method.

    Exploit selects the most exploitative point on the front, i.e. the location
    with the best (lowest) mean predicted value.
    """

    def __call__(self, model):
        X, mu, _ = self.get_front(model)
        Xnew = egreedy_acq_funcs_minimize.Exploit(X, mu)
        return Xnew


class PI(BaseOptimiser):
    """Maximises the PI acquisition function for a given GPy model.

    See standard_acq_funcs_minimize.PI for details of the PI method.

    Notes
    -----
    PI is maximised using the typical multi-restart approach of drawing a
    large number of samples from across the decision space (X), evaluating the
    locations with the acquisition function, and locally optimising the best 10
    of these with L-BFGS-B. Here we make the assumption that each local
    optimisation run will take ~100 evaluations -- emperically we found this to
    be the case.

    PI is not maximised using NSGA-II because the location that maximises PI is
    not guaranteed to be on the Pareto front; see the paper for full details.
    """

    def __call__(self, model):
        D = model.X.shape[1]
        incumbent = model.Y.min()

        # objective function wrapper for L-BFGS-B
        def min_obj(x):
            # if we have a constraint function and it is violated,
            # return a bad PI value
            if (self.cf is not None) and (not self.cf(x)):
                return np.inf

            mu, sigmaSQR = model.predict(np.atleast_2d(x), full_cov=False)

            # negate PI because we're using a minimiser
            pi = -standard_acq_funcs_minimize.PI(
                mu, np.sqrt(sigmaSQR), incumbent
            ).ravel()
            return pi

        # number of optimisation runs and *estimated* number of L-BFGS-B
        # function evaluations per run; note this was calculate empirically and
        # may not be true for all functions.
        N_opt_runs = 10
        fevals_assumed_per_run = 100

        N_samples = self.acq_budget - (N_opt_runs * fevals_assumed_per_run)
        if N_samples <= N_opt_runs:
            N_samples = N_opt_runs

        # initially perform a grid search for N_samples
        x0_points = np.random.uniform(self.lb, self.ub, size=(N_samples, D))
        fx0 = min_obj(x0_points).ravel()

        # select the top N_opt_runs to evaluate with L-BFGS-B
        x0_points = x0_points[np.argsort(fx0)[:N_opt_runs], :]

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


class EI_LBFGS(BaseOptimiser):
    def __call__(self, model):
        D = model.X.shape[1]
        incumbent = model.Y.min()

        # objective function wrapper for L-BFGS-B
        def min_obj(x):
            # if we have a constraint function and it is violated,
            # return a bad EI value
            if (self.cf is not None) and (not self.cf(x)):
                return np.inf

            mu, sigmaSQR = model.predict(np.atleast_2d(x), full_cov=False)

            # negate EI because we're using a minimiser
            ei = -standard_acq_funcs_minimize.EI(
                mu, np.sqrt(sigmaSQR), incumbent
            ).ravel()
            return ei

        # number of optimisation runs and *estimated* number of L-BFGS-B
        # function evaluations per run; note this was calculate empirically and
        # may not be true for all functions.
        N_opt_runs = 10
        fevals_assumed_per_run = 100

        N_samples = self.acq_budget - (N_opt_runs * fevals_assumed_per_run)
        if N_samples <= N_opt_runs:
            N_samples = N_opt_runs

        # initially perform a grid search for N_samples
        x0_points = np.random.uniform(self.lb, self.ub, size=(N_samples, D))
        fx0 = min_obj(x0_points).ravel()

        # select the top N_opt_runs to evaluate with L-BFGS-B
        x0_points = x0_points[np.argsort(fx0)[:N_opt_runs], :]

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


class UCB_LBFGS(BaseOptimiser):
    def __call__(self, model):
        D = model.X.shape[1]
        incumbent = model.Y.min()

        # objective function wrapper for L-BFGS-B
        def min_obj(x):
            # if we have a constraint function and it is violated,
            # return a bad UCB value
            if (self.cf is not None) and (not self.cf(x)):
                return np.inf

            mu, sigmaSQR = model.predict(np.atleast_2d(x), full_cov=False)

            # negate EI because we're using a minimiser
            ucb = -standard_acq_funcs_minimize.UCB(
                mu,
                sigmaSQR,
                lb=self.lb,
                ub=self.ub,
                t=model.X.shape[0] + 1,
                d=model.X.shape[1],
                **self.acquisition_args
            ).ravel()
            return ucb

        # number of optimisation runs and *estimated* number of L-BFGS-B
        # function evaluations per run; note this was calculate empirically and
        # may not be true for all functions.
        N_opt_runs = 10
        fevals_assumed_per_run = 100

        N_samples = self.acq_budget - (N_opt_runs * fevals_assumed_per_run)
        if N_samples <= N_opt_runs:
            N_samples = N_opt_runs

        # initially perform a grid search for N_samples
        x0_points = np.random.uniform(self.lb, self.ub, size=(N_samples, D))
        fx0 = min_obj(x0_points).ravel()

        # select the top N_opt_runs to evaluate with L-BFGS-B
        x0_points = x0_points[np.argsort(fx0)[:N_opt_runs], :]

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
