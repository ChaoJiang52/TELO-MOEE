"""Main experimental framework.

See perform_experiment for details.
"""
import os
import GPy as gp
from MOEE import acquisition_functions, test_problems
import matplotlib.pyplot as plt
from MOEE.test_problems.synthetic_problems import *

import torch
from torch.optim import LBFGS
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels.matern_kernel import MaternKernel
from botorch.acquisition import qKnowledgeGradient
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.optim import optimize_acqf
from botorch.acquisition.utils import get_optimal_samples
import gpytorch.settings as settings
from gpytorch.utils.errors import NotPSDError
from copy import deepcopy


def build_and_fit_GP_torch(train_X, train_Y):
    restarts = 10
    best_model = None
    best_loss = float('inf')

    for restart in range(restarts):
        print(f"Restart {restart + 1}/{restarts}")

        # Define the GP model
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=MaternKernel(nu=2.5),
            outcome_transform=Standardize(m=1),
        )

        # Manually initialize hyperparameters to avoid extreme values
        model.covar_module.lengthscale = torch.tensor([1e-1]).double()
        model.likelihood.noise_covar.noise = torch.tensor([1e-1]).double()

        # Define the Marginal Log Likelihood
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        model.train()

        # Define the LBFGS optimizer
        optimizer = LBFGS([{"params": model.parameters()}], lr=0.1, max_iter=200)

        # Define the closure function
        def closure():
            optimizer.zero_grad()  # Clear previous gradients
            output = model(train_X)  # Forward pass
            loss = -mll(output, model.train_targets)  # Compute the negative marginal log likelihood
            loss.backward()  # backprop gradients
            return loss

        try:
            # Increase jitter setting during model training
            with settings.cholesky_jitter(1e-2):
                optimizer.step(closure)
            # optimizer.step(closure)
        except NotPSDError as e:
            print("Non-positive definite matrix encountered. Skipping this restart.")
            continue

        # Check the final loss and update the best model if this one is better
        model.eval()
        with torch.no_grad():
            final_output = model(train_X)
            final_loss = -mll(final_output, model.train_targets).item()
            if final_loss < best_loss:
                best_loss = final_loss
                best_model = model

    if best_loss == np.inf:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # fit_gpytorch_mll by default uses L-BFGS
        fit_gpytorch_mll(mll)
        return model
    else:
        return best_model  # Return the best model after all restarts


def plot_gp(model):
    model.plot()
    plt.show()


def perform_experiment(
        problem_name,
        run_no,
        acquisition_name,
        acquisition_args={},
        budget=250,
        continue_runs=False,
        verbose=False,
        save=True,
):
    r"""Performs a Bayesian optimisation experiment.

    Performs Bayesian optimisation on a specified problem with given
    initial training data (typically Latin hypercube samples) and a
    specified acquisition function. The initial training data should be in
    the 'data' directory, be named '{problem_name}_{run_no}.npz', and contain
    two ``numpy.ndarray``s 'arr_0' of shape (N, D) and 'arr_1' of shape (N, 1)
    containing the N D-dimensional training points and their function values
    respectively. The training data may also have an optional dictionary
    call 'arr_2' which contains key-value pairs to be passed in as arguments
    to the ``problem_name`` function at instantiation.

    The optimisation results will be stored at each iteration in the form
    of a .npz file, in the 'results' directory with the name:
    '{problem_name:}_{run_no:}_{budget:}_{acquisition_name:}.npz'
    which contains arrays 'Xtr' of shape (budget, D) and 'Ytr' of shape
    (budget, 1). These contain the expensively evaluated D-dimensional
    locations and their corresponding function values. If the acquisition_args
    dictionary contains the key 'epsilon', e.g. for the eFront and eRandom
    methods, then its value will be appended to the end of the file name, e.g:
    '{problem_name:}_{run_no:}_{budget:}_{acquisition_name:}_eps0.1.npz'. Note
    that the first rows of 'Xtr' and 'Ytr' will contain the initial training
    data.

    If the file to save the experimental data to already exists and
    ``continue_runs`` is set to True then training data will be loaded from
    the saved run and the optimisation process will continue where it left off.
    If the save file exists and ``continue_runs`` is set to False, an
    exception will be raised.

    Parameters
    ----------
    problem_name : string
        Test problem name to perform the optimisation run on. This will
        attempt to import problem_name from the test_problems module.
    run_no : int
        Optimisation run number. The function will attempt to load training
        data from a file called '{problem_name}_{run_no}.npz' in the
        data directory in order to initialise the Gaussian process model.
    acquisition_name : string
        Name of the acquisition function to be used to select the next
        location to expensively evaluate. This will attempt to import
        acquisition_name from the acquisition_functions module.
    acquisition_args : dict, optional
        Keyword arguments to be passed to the acquisition function.
    budget : int
        Total number of expensive evaluations to carry out. Note that
        the budget includes points in the initial training data.
    continue_runs : bool
        If set to true then this allows for a saved run to be resumed.
    verbose : bool
        Prints information relating to the optimisation procedure as the
        algorithm progresses.
    save : bool
        Whether the optimisation run should be saved.

    Raises
    ------
    ValueError
        Raised if the function wants to save to a file that already exists and
        cannot resume the optimisation run as ``continue_runs`` is set to
        False.

    Examples
    --------
    Optimising the WangFreitas test problem, with training data
    corresponding to run 1, using the Expected Improvement (EI)
    acquisition function:

    >>> perform_experiment('WangFreitas', 1, 'EI', verbose=True)
    Loaded training data from: data/WangFreitas_1.npz
    Loaded test problem: WangFreitas
    Using acquisition function: EI
    Training a GP model with 2 data points.
    Optimising the acquisition function.
    Expensively evaluating the new selected location.
    New location  : [0.03726201]
    Function value: -1.64271
    ...

    Optimising the WangFreitas test problem, with training data
    corresponding to run 1, using the eFront algorithm (greedily maximises
    the GP mean response (1 - epsilon) of the time and otherwise selects a
    location on the [estimated] Pareto front) with epsilon = 0.1.

    >>> perform_experiment('WangFreitas', 1, 'eFront', verbose=True, acquisition_args={'epsilon': 0.1})
    Loaded training data from: data/WangFreitas_1.npz
    Loaded test problem: WangFreitas
    Using acquisition function: eFront
            with optional arguments: {'epsilon': 0.1}
    Training a GP model with 2 data points.
    Optimising the acquisition function.
    Expensively evaluating the new selected location.
    New location  : [0.08415271]
    Function value: -1.97504
    ...
    """
    # paths to data
    data_file = f"../training_data/{problem_name:}_{run_no:}.npz"
    save_file = f"../results/{problem_name:}_{run_no:}_{budget:}_{acquisition_name:}"

    if "epsilon" in acquisition_args:
        save_file += "_eps{:g}".format(acquisition_args["epsilon"])
    if "type" in acquisition_args:
        save_file += "_{}".format(acquisition_args['type'])
    # if "knee" in acquisition_args:
    #     if acquisition_args["knee"]:
    #         save_file += "_knee"
    if "type2" in acquisition_args:
        save_file += "_{}".format(acquisition_args['type2'])
    if "weights" in acquisition_args:
        save_file += "_{}".format(acquisition_args['weights'])
    if "denom" in acquisition_args:
        save_file += f"_denom{acquisition_args['denom']}"
    save_file += ".npz"

    # if the file we're going to be saving to already exists
    if os.path.exists(save_file):
        # if we've explicitly allowed the continuation of runs then load the
        # saved data. and optional function arguments from the original
        if continue_runs:
            # load the previously saved data
            with np.load(save_file, allow_pickle=True) as data:
                Xtr = data["Xtr"]
                Ytr = data["Ytr"]

            _print("Continuing saved run from: {:s}".format(save_file), verbose)

        # if we weren't expecting saved data to be there, raise an exception
        # instead of overwriting it
        else:
            raise ValueError(
                "Saved data found but continue_runs not set: {:s}".format(save_file)
            )

    # else just load the training data
    else:
        with np.load(data_file, allow_pickle=True) as data:
            Xtr = data["arr_0"]
            Ytr = data["arr_1"]

        _print("Loaded training data from: {:s}".format(data_file), verbose)

    # load the function's optional arguments, if there are any
    with np.load(data_file, allow_pickle=True) as data:
        if "arr_2" in data:
            f_optional_arguments = data["arr_2"].item()
        else:
            f_optional_arguments = {}

    # load the problem instance
    f_class = getattr(test_problems, problem_name)
    f = f_class(**f_optional_arguments)
    f_lb = f.lb
    f_ub = f.ub
    f_cf = f.cf
    f_dim = f.dim

    _print("Loaded test problem: {:s}".format(problem_name), verbose)
    if f_optional_arguments:
        _print("\twith optional arguments: {:}".format(f_optional_arguments), verbose)

    # load the acquisition function
    acq_budget = 5000 * f_dim
    if acquisition_name not in ["KG", "JES"]:
        acq_class = getattr(acquisition_functions, acquisition_name)
        acq_func = acq_class(
            lb=f_lb,
            ub=f_ub,
            cf=f_cf,
            acq_budget=acq_budget,  # nsga2 run number
            acquisition_args=acquisition_args,
        )

        _print("Using acquisition function: {:s}".format(acquisition_name), verbose)
    if acquisition_args:
        _print("\twith optional arguments: {:}".format(acquisition_args), verbose)

    # perform the Bayesian optimisation loop
    while Xtr.shape[0] < budget:
        if acquisition_name in ["KG", "JES"]:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _print(
                "Training a GP model with " "{:d} data points.".format(Xtr.shape[0]),
                verbose,
            )
            train_Y = torch.from_numpy(-Ytr).to(device)

            # gp = SingleTaskGP(
            #     train_X=torch.from_numpy(Xtr).to(device),
            #     train_Y=train_Y,
            #     covar_module=MaternKernel(nu=2.5),
            #     outcome_transform=Standardize(m=1),
            # )
            # mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            # # fit_gpytorch_mll by default uses L-BFGS
            # fit_gpytorch_mll(mll)

            build_and_fit_GP_torch(torch.from_numpy(Xtr).to(device), train_Y)

            bounds = torch.stack([torch.from_numpy(f.lb), torch.from_numpy(f.ub)]).to(torch.double).to(device)

            if acquisition_name == "KG":
                acf = qKnowledgeGradient(gp, num_fantasies=128)
            elif acquisition_name == "JES":
                num_samples = 100
                optimal_inputs, optimal_outputs = get_optimal_samples(
                    gp,
                    bounds=bounds,
                    num_optima=num_samples
                )
                acf = qJointEntropySearch(
                    model=gp,
                    optimal_inputs=optimal_inputs,
                    optimal_outputs=optimal_outputs,
                    estimation_type="LB",
                )
            # number of optimisation runs and *estimated* number of L-BFGS-B
            # function evaluations per run; note this was calculate empirically and
            # may not be true for all functions.
            N_opt_runs = 10
            fevals_assumed_per_run = 100

            # (5000D - 1000) samples
            samples = acq_budget - (N_opt_runs * fevals_assumed_per_run)

            _print("Optimising the acquisition function.", verbose)
            Xnew, acq_value = optimize_acqf(acf, bounds=bounds, q=1, num_restarts=N_opt_runs, raw_samples=samples)
            Xnew = Xnew.cpu().detach().numpy()
            # expensively evaluate it
            _print("Expensively evaluating the new selected location.", verbose)
            Ynew = f(Xnew)

            _print("New location  : {:}".format(Xnew.ravel()), verbose)
            _print("Function value: {:g}".format(Ynew.flat[0]), verbose)

        else:
            Xnew, Ynew, _ = perform_BO_iteration(Xtr, Ytr, f, acq_func, verbose, acquisition_args)

        # augment the training data and repeat
        Xtr = np.concatenate((Xtr, np.atleast_2d(Xnew)))
        Ytr = np.concatenate((Ytr, np.atleast_2d(Ynew)))

        # save the current training data
        if save:
            np.savez(save_file, Xtr=Xtr, Ytr=Ytr)

        min_idx = np.argmin(Ytr)
        _print("Best variables value so far: {}".format(Xtr[min_idx]), verbose)
        _print("Best function value so far: {}".format(Ytr[min_idx]), verbose)
        _print("", verbose)

    _print("Finished optimisation run", verbose)


def perform_BO_iteration(Xtr, Ytr, f, acq_func, verbose=False, acquisition_args=None):
    """Performs one iteration of Bayesian optimisation.

    The function trains a GPy GPRegression model with a Matern 5/2 kernel that
    has its length-scale and variance bounded in [1e-6, 1e6], and GP's
    Gaussian noise fixed at 1e-6. These parameters are optimised using L-BFGS-B
    with 10 restarts (the default GPy settings). Once the model has been
    trained, an acquisition function is optimised over the model, selecting a
    new location ``Xnew`` to expensively evaluate the function ``f`` at.
    Finally, the location is expensively evaluated.

    If the ``f(Xnew)`` raises an exception then the iteration will start again,
    up to a maximum of 10 times. If the maximum number of attempts has been
    reached then a RuntimeError will be raised.

    Parameters
    ----------
    Xtr : (N, D) numpy.ndarray
        Array of N previously evaluated locations in D-dimensional space.
    Ytr : (N, 1) numpy.ndarray
        The previously evaluated locations' function evaluations.
    f : callable
        A function that performs an expensive evaluation of its argument.
    acq_func : callable
        A function that maximises an acquisition function over the GP model and
        returns the next location to expensively evaluate.
    verbose : bool
        Prints information as to what is happening in the BO iteration.

    Returns
    -------
    Xnew : (D, ) numpy.ndarray
        Location that has been expensively evaluated
    Ynew : (1, 1) numpy.ndarray
        Function value at ``Xnew``
    model : GPy.models.GPRegression
        The trained gp model that was used by the acqusition function to
        select the next location to expensively evaluate

    Raises
    ------
    RuntimeError
        If either ``Xnew`` has violated the constraint function 10 times or
        f(Xnew) has raised an exception 10 times then the run is deemed a
        failure and the exception is raised.
    """
    N_FAILED = 0
    MAX_FAILED = 10

    while N_FAILED < MAX_FAILED:
        try:
            _print(
                "Training a GP model with " "{:d} data points.".format(Xtr.shape[0]),
                verbose,
            )

            # create a gp model with the training data and fit it
            kernel = gp.kern.Matern52(input_dim=Xtr.shape[1], ARD=False)
            model = gp.models.GPRegression(Xtr, Ytr, kernel, normalizer=False)
            model.constrain_positive("")
            (kern_variance, kern_lengthscale, gaussian_noise) = model.parameter_names()
            model[kern_variance].constrain_bounded(1e-6, 1e6, warning=False)
            model[kern_lengthscale].constrain_bounded(1e-6, 1e6, warning=False)
            model[gaussian_noise].constrain_fixed(1e-6, warning=False)

            model.optimize_restarts(
                optimizer="lbfgs", num_restarts=10, num_processes=1, verbose=False
            )
            # plot_gp(model)
            # optimise the acquisition function to get a new point to evaluate
            _print("Optimising the acquisition function.", verbose)
            Xnew = acq_func(model)

            # expensively evaluate it
            _print("Expensively evaluating the new selected location.", verbose)
            Ynew = f(Xnew)

            _print("New location  : {:}".format(Xnew.ravel()), verbose)
            _print("Function value: {:g}".format(Ynew.flat[0]), verbose)

            return Xnew, Ynew, model

        except KeyboardInterrupt:
            _print("Interrupted with CTRL+C - stopping run", verbose)
            raise

        # except:
        #     _print(
        #         "Something failed on Attempt "
        #         + "{:d}/{:d}".format(N_FAILED + 1, MAX_FAILED),
        #         verbose,
        #     )
        #     N_FAILED += 1

    raise RuntimeError(
        "The optimiser was unable to find a valid " + "location to evaluate"
    )


def _print(s, verbose=False):
    """Prints the argument if an a flag is set to True.

    Parameters
    ----------
    s : printable object (e.g. string)
        Text to print
    verbose : bool, optional
        Indicates whether ``s`` should be printed.
    """
    if verbose:
        print(s)


if __name__ == "__main__":
    problem_names = ['WangFreitas', 'Branin', 'GoldsteinPrice', 'Cosines', 'ACKLEY_2', 'GRIEWANK_2',
                     'SixHumpCamel', 'Hartmann6', 'GSobol', 'Attractive_Sector', 'Ellipsoidal', 'Schwefel',
                     'ACKLEY_10', 'GRIEWANK_10', 'ACKLEY_20', 'GRIEWANK_20', 'push4', 'push8', 'svm_wdbc']

    # MOEE
    types = ['UCB', 'EI']
    all_type2 = ['topsis', 'PF', 'exploit', 'ePF', 'switch']
    for name in problem_names:
        for i in range(1, 31):
            for _acf in types:
                perform_experiment(
                    name,  # problem name
                    i,  # problem instance (LHS samples and optional args)
                    "MOEE",  # method name
                    verbose=True,  # print status
                    continue_runs=True,  # resume runs
                    save=True,  # whether to save the run
                    acquisition_args={"type": _acf, "type2": "topsis", "explore": 0, "denom": 0},  # denom: 0, 3, 2, 1, 0.5
                    # acquisition_args={"type": 'UCB', "type2": "topsis", "explore": 0, "weights": [0.5, 0.5]}  #
                    # acquisition_args={"type": 'UCB', "type2": "topsis", "explore": 0, "denom": 0.5},
                )

    # epsilon-PF and epsilon-RS
    methods = ['eFront', 'eRandom']
    for method in methods:
        for name in problem_names:
            for i in range(1, 31):
                perform_experiment(
                    name,  # problem name
                    i,  # problem instance (LHS samples and optional args)
                    method,  # method name
                    verbose=True,  # print status
                    continue_runs=True,  # resume runs
                    save=True,  # whether to save the run
                    acquisition_args={"epsilon": 0.1},  # acq func args
                )

    # PI, EI, UCB, KG, JES, MACE
    methods = ['PI', 'EI_LBFGS', 'UCB_LBFGS', 'KG', 'JES', 'MACE']
    for method in methods:
        for name in problem_names:
            for i in range(1, 31):
                perform_experiment(
                    name,  # problem name
                    i,  # problem instance (LHS samples and optional args)
                    method,  # method name
                    verbose=True,  # print status
                    continue_runs=True,  # resume runs
                    save=True  # whether to save the run
                )

    # MGFI
    for name in problem_names:
        for i in range(1, 31):
            perform_experiment(
                name,  # problem name
                i,  # problem instance (LHS samples and optional args)
                "MGFI",  # method name
                verbose=True,  # print status
                continue_runs=True,  # resume runs
                save=True,  # whether to save the run
                acquisition_args={"t": 2},  # MGFI
            )

    # SAWEI
    for name in problem_names:
        for i in range(1, 31):
            perform_experiment(
                name,  # problem name
                i,  # problem instance (LHS samples and optional args)
                "SAWEI",  # method name
                verbose=True,  # print status
                continue_runs=True,  # resume runs
                save=True,  # whether to save the run
                acquisition_args={"ubr": [], "alpha": 0.5, "wei_pi_pure_term": 0, "wei_ei_term": 0},  # AWEI
            )

    # for name in problem_names:
    #     for i in range(1, 31):
    #         perform_experiment(
    #             name,  # problem name
    #             i,  # problem instance (LHS samples and optional args)
    #             "MOEE",  # method name
    #             verbose=True,  # print status
    #             continue_runs=True,  # resume runs
    #             save=True,  # whether to save the run
    #             acquisition_args={"type": 'UCB', "type2": "topsis", "explore": 0, "denom": 0},  # acq func args
    #             # acquisition_args={"type": 'UCB', "type2": "topsis", "explore": 0, "weights": [0.2,0.8], "denom": 1/3},  # acq func args
    #             # acquisition_args={"type": 'UCB', "type2": "topsis", "explore": 0, "weights": [0.8, 0.2]},  # acq func args
    #             # acquisition_args={"type": 'UCB', "type2": "topsis", "explore": 0, "denom": 0.5},  # acq func args
    #         )
    #
    # # WangFreitas BraninForrester
    # perform_experiment(
    #     "svm_wdbc",  # problem name
    #     1,  # problem instance (LHS samples and optional args)
    #     "UCB",  # method name
    #     verbose=True,  # print status
    #     continue_runs=True,  # resume runs
    #     save=True,  # whether to save the run
    #     # acquisition_args={"type": 'UCB', "knee": True, "type2": "switch", "explore": 0},  # acq func args
    #     # acquisition_args={"type": 'UCB', "type2": "topsis", "explore": 0},  # acq func args
    #     # acquisition_args={"ubr": [], "alpha": 0.5, "wei_pi_pure_term": 0, "wei_ei_term": 0},  # AWEI
    #     # acquisition_args={'t': 2},
    # )
