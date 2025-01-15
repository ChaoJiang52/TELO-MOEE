import numpy as np
import matplotlib.pyplot as plt
from MOEE.util.plotting import *

# settings define all the results we wish to process
results_dir = r'..\results'

problem_names = ['WangFreitas', 'Branin', 'GoldsteinPrice', 'Cosines', 'ACKLEY_2', 'GRIEWANK_2',
                 'SixHumpCamel', 'Hartmann6', 'GSobol', 'Attractive_Sector', 'Ellipsoidal', 'Schwefel',
                 'ACKLEY_10', 'GRIEWANK_10', 'ACKLEY_20', 'GRIEWANK_20', 'push4', 'push8']

problem_names_for_paper = ['WangFreitas', 'Branin', 'GoldsteinPrice', 'Cosines', 'Ackley', 'Griewank',
                           'SixHumpCamel', 'Hartmann6', 'GSobol', 'Attractive_Sector', 'Ellipsoidal', 'Schwefel',
                           'Ackley', 'Griewank', 'Ackley', 'Griewank', 'Push4', 'Push8']


# boolean indicating whether the problem should be plotted with a log axis
# number of problems equals numbers of booleans
problem_logplot = [True, True, True, True, True, True,
                   True, True, True, True, True, True,
                   True, True, True, True, True, True]


# exploration condition
# method_names = ['MOEE_UCB_topsis_denom3', 'MOEE_UCB_topsis', 'MOEE_UCB_topsis_denom1', 'MOEE_UCB_topsis_denom0.5', 'MOEE_UCB_topsis_denom0']
# method_names_for_paper = ['MOEE$_{d^{1/3}}$', 'MOEE$_{d^{1/2}}$', 'MOEE$_{d}$', 'MOEE$_{d^2}$', 'MOEE']

# Comparison of our proposal and the state-of-the-art acquisition functions (RQ1)
method_names = ['PI', 'EI_LBFGS', 'UCB_LBFGS', 'eFront_eps0.1', 'eRandom_eps0.1', 'AWEI', 'MGFI', 'KG', 'JES', 'MACE', 'MOEE_UCB_topsis_denom0']
method_names_for_paper = ['PI', 'EI', 'LCB', '$\epsilon$-PF', '$\epsilon$-RS', 'SAWEI', 'MGFI', 'KG', 'JES', 'MACE', 'MOEE']

# Ablation study (RQ2)
# method_names = ['MOEE_UCB_PF', 'MOEE_UCB_exploit', 'MOEE_UCB_random', 'MOEE_UCB_switch', 'MOEE_UCB_topsis_denom0']
# method_names_for_paper = ['MOEE$_{PF}$', 'MOEE$_{exploit}$', 'MOEE$_{e\_PF}$', 'MOEE$_{switch}$', 'MOEE']

# weights (RQ3)
# method_names = ['MOEE_UCB_topsis_[0.2, 0.8]', 'MOEE_UCB_topsis_[0.5, 0.5]', 'MOEE_UCB_topsis_[0.6, 0.4]', 'MOEE_UCB_topsis_[0.8, 0.2]', 'MOEE_UCB_topsis_denom0']
# method_names_for_paper = ['MOEE$_{28}$', 'MOEE$_{55}$', 'MOEE$_{64}$', 'MOEE$_{82}$', 'MOEE']

# MOEE_EI (RQ4)
# method_names = ['PI', 'EI_LBFGS', 'UCB_LBFGS', 'eFront_eps0.1', 'eRandom_eps0.1', 'AWEI', 'MGFI', 'KG', 'JES', 'MACE', 'MOEE_UCB_topsis_denom0', 'MOEE_EI_topsis_denom0']
# method_names_for_paper = ['PI', 'EI', 'LCB', '$\epsilon$-PF', '$\epsilon$-RS', 'SAWEI', 'MGFI', 'KG', 'JES', 'MACE', 'MOEE', 'MOEE$_{EI}$']

save_images = True
# load in all the optimisation results
results = process_results(results_dir, problem_names, method_names, budget=250, exp_no_start=1, exp_no_end=30)

with np.load(r'../training_data/push8_best_solutions.npz') as data:
    push8_estimated_optima = data['results']

for method_name in method_names:
    dist = results['push8'][method_name] - push8_estimated_optima[:, None][:30]

    # simple sanity checking - check the distance between them is >= 0, meaning that
    # the estimated optima are better or equal to the evaluated function values

    assert np.all(dist >= 0)

    results['push8'][method_name] = dist

plot_boxplots(results,
              [250],
              problem_names,
              problem_names_for_paper,
              problem_logplot,
              method_names,
              method_names_for_paper,
              LABEL_FONTSIZE=22,
              TITLE_FONTSIZE=25,
              TICK_FONTSIZE=20,
              save=save_images)

# plot_boxplots_combined(results,
#               [50, 150, 250],
#               ['Cosines', 'logSixHumpCamel', 'logGSobol'],
#               ['Cosines', 'logSixHumpCamel', 'logGSobol'],
#               [True, True, False],
#               method_names,
#               method_names_for_paper,
#               LABEL_FONTSIZE=22,
#               TITLE_FONTSIZE=25,
#               TICK_FONTSIZE=20,
#               save=save_images)
