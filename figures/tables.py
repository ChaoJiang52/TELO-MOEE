# Main results table
from MOEE.util.plotting import *

results_dir = r'..\results'

# 16 problems
problem_rows = [['WangFreitas', 'Branin', 'GoldsteinPrice', 'Cosines', 'ACKLEY_2', 'GRIEWANK_2'],
                ['SixHumpCamel', 'Hartmann6', 'GSobol', "Attractive_Sector", "Ellipsoidal", "Schwefel"],
                ['ACKLEY_10', 'GRIEWANK_10', 'ACKLEY_20', 'GRIEWANK_20', 'push4', 'push8']]
#
problem_names = ['WangFreitas', 'Branin', 'GoldsteinPrice', 'Cosines', 'ACKLEY_2', 'GRIEWANK_2', 'SixHumpCamel',
                 'Hartmann6', 'GSobol',
                 "Attractive_Sector", "Ellipsoidal", "Schwefel", 'ACKLEY_10', 'GRIEWANK_10', 'ACKLEY_20', 'GRIEWANK_20',
                 'push4', 'push8']

problem_paper_rows = [['WangFreitas', 'Branin', 'GoldsteinPrice', 'Cosines', 'Ackley', 'Griewank'],
                      ['SixHumpCamel', 'Hartmann6', 'GSobol', "Attractive Sector", "Ellipsoidal", "Schwefel"],
                      ['Ackley', 'Griewank', 'Ackley', 'Griewank', 'Push4', 'Push8']]

problem_dim_rows = [[1, 2, 2, 2, 2, 2],
                    [2, 6, 10, 10, 10, 10],
                    [10, 10, 20, 20, 4, 8]]

# exploration trigger condition
# method_names = ['MOEE_UCB_topsis', 'MOEE_UCB_topsis_denom0', 'MOEE_UCB_topsis_denom3', 'MOEE_UCB_topsis_denom1', 'MOEE_UCB_topsis_denom0.5']
# method_names_for_table = ['MOEE', 'MOEE$_{d^{0}}$', 'MOEE$_{d^{1/3}}$', 'MOEE$_{d}$', 'MOEE$_{d^2}$']

# MOEE (RQ1)
method_names = ['MOEE_UCB_topsis_denom0', 'PI', 'EI_LBFGS', 'UCB_LBFGS', 'eFront_eps0.1', 'eRandom_eps0.1', 'AWEI', 'MGFI', 'KG', 'JES', 'MACE']
method_names_for_table = ['MOEE', 'PI', 'EI', 'LCB', '$\epsilon$-PF', '$\epsilon$-RS', 'SAWEI', 'MGFI', 'KG', 'JES', 'MACE']

# Ablation study (RQ2)
# method_names = ['MOEE_UCB_topsis_denom0', 'MOEE_UCB_PF', 'MOEE_UCB_exploit', 'MOEE_UCB_ePF', 'MOEE_UCB_switch']
# method_names_for_table = ['MOEE', 'MOEE$_{PF}$', 'MOEE$_{exploit}$', 'MOEE$_{e\_PF}$', 'MOEE$_{switch}$']

# weights (RQ3)
# method_names = ['MOEE_UCB_topsis_denom0', 'MOEE_UCB_topsis_[0.2, 0.8]', 'MOEE_UCB_topsis_[0.5, 0.5]', 'MOEE_UCB_topsis_[0.6, 0.4]', 'MOEE_UCB_topsis_[0.8, 0.2]']
# method_names_for_table = ['MOEE', 'MOEE$_{28}$', 'MOEE$_{55}$', 'MOEE$_{64}$', 'MOEE$_{82}$']

# MOEE_EI (RQ4)
# method_names = ['MOEE_EI_topsis_denom0', 'MOEE_UCB_topsis_denom0', 'PI', 'EI_LBFGS', 'UCB_LBFGS', 'eFront_eps0.1', 'eRandom_eps0.1', 'AWEI', 'MGFI', 'KG', 'JES', 'MACE']
# method_names_for_table = ['MOEE$_{EI}$', 'MOEE', 'PI', 'EI', 'LCB', '$\epsilon$-PF', '$\epsilon$-RS', 'SAWEI', 'MGFI', 'KG', 'JES', 'MACE']


results = process_results(results_dir, problem_names, method_names, budget=250, exp_no_start=1, exp_no_end=30)

with np.load(r'../training_data/push8_best_solutions.npz') as data:
    push8_estimated_optima = data['results']

for method_name in method_names:
    dist = results['push8'][method_name] - push8_estimated_optima[:, None][:30]

    # simple sanity checking - check the distance between them is >= 0, meaning that
    # the estimated optima are better or equal to the evaluated function values

    assert np.all(dist >= 0)

    results['push8'][method_name] = dist

table_data = create_table_data_MOEE(results, problem_names, method_names, 30, process="median")

create_table_MOEE(table_data, problem_rows, problem_paper_rows,
                  problem_dim_rows, method_names, method_names_for_table, process="median")
