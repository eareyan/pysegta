import json
import math
import pprint
import names
import os
import time
import pandas as pd

from non_uniform import psp
from poker_game import draw_game, compute_game_stats, poker_dealer, poker_player_utility
from prettytable import PrettyTable


def compute_n(c, delta, epsilon, size_of_game):
    return (c / epsilon) * (c / epsilon) * 0.5 * math.log((2 * size_of_game) / delta)


def compute_eps(c, delta, size_of_game, number_of_samples):
    return c * math.sqrt((math.log((2.0 * size_of_game) / delta)) / (2.0 * number_of_samples))


def create_ptable(dictionary):
    ptable = PrettyTable()
    ptable.title = 'Running experiments with parameters'
    ptable.field_names = ['Parameter', 'Value']
    for k, v in dictionary.items():
        ptable.add_row([k, v])
    return ptable


def sample_hands():
    number_samples_mean = 10000
    sample = []
    for _ in range(0, 5000):
        p1_hand, p2_hand, _ = get_initial_game(size_hand=4)
        strat_prof = (p1_hand, p2_hand)

        expectation_p1 = expected_utility(1, strat_prof)
        expectation_p2 = expected_utility(2, strat_prof)

        m, U, V = poker_player_utility(1, strat_prof, poker_dealer(number_samples_mean))
        empirical_mean_p1 = U / m

        m, U, V = poker_player_utility(2, strat_prof, poker_dealer(number_samples_mean))
        empirical_mean_p2 = U / m

        variance_p1 = variance_utility(1, strat_prof)
        variance_p2 = variance_utility(2, strat_prof)

        print(strat_prof)
        print(f"{expectation_p1 : .4f} \t {empirical_mean_p1: .4f} \t {variance_p1 : .4f}")
        print(f"{expectation_p2 : .4f} \t {empirical_mean_p2 : .4f} \t {variance_p2 : .4f}")
        sample += [[p1_hand, p2_hand, expectation_p1, expectation_p2, empirical_mean_p1, empirical_mean_p2, variance_p1, variance_p2]]

    pd.DataFrame(sample, columns=['hand_p1', 'hand_p2', 'E[p1]', 'E[p2]', 'emp_p1', 'emp_p2', 'VAR[p1]', 'VAR[p2]']).to_csv('hands_stats.csv', index=False)


if __name__ == '__main__':

    # Exogenous Parameters.
    exogenous_parameters = {'experiment_name': names.get_full_name().replace(' ', '_'),
                            'size_of_game': 50,
                            'target_epsilon': 0.15,
                            'c': 2,
                            'delta': 0.1,
                            'number_of_games': 100,
                            'number_of_psp_runs_per_game': 30}

    # Check if an experiment with this name already exists so as not to overwrite any previous results.
    if os.path.exists(f"poker_experiments_data/experiment_{exogenous_parameters['experiment_name']}"):
        print(f"Experiment {exogenous_parameters['experiment_name']} already exists, try again. ")
        exit()

    # Endogenous Parameters.
    endogenous_parameters = {}
    endogenous_parameters['m'] = compute_n(c=exogenous_parameters['c'],
                                           delta=exogenous_parameters['delta'],
                                           epsilon=exogenous_parameters['target_epsilon'],
                                           size_of_game=exogenous_parameters['size_of_game'])
    endogenous_parameters['failure_probability_schedule'] = [exogenous_parameters['delta'] / 4 for _ in range(0, 4)]
    endogenous_parameters['sampling_schedule'] = [math.ceil(endogenous_parameters['m'] / 4),
                                                  math.ceil(endogenous_parameters['m'] / 2),
                                                  math.ceil(endogenous_parameters['m']),
                                                  math.ceil(2 * endogenous_parameters['m'])]

    # Print parameters
    parameter_ptable = create_ptable({**exogenous_parameters, **endogenous_parameters})
    print(parameter_ptable)
    with open(f"poker_experiments_data/experiment_{exogenous_parameters['experiment_name']}.txt", 'w') as the_file:
        the_file.write(str(parameter_ptable))

    # Run experiments
    list_of_games = []
    list_of_psp_results = []
    for game in range(0, exogenous_parameters['number_of_games']):
        # Time how long it takes to produce results for a single game
        t0_game = time.time()

        # Draw a game
        p1_poker_hand, p2_poker_hand, the_active_set = draw_game()

        # Compute the game's statistics
        max_variance, sum_variance, _ = compute_game_stats(the_active_set)

        # Print info about the game for debugging purposes.
        print(f"Game # {game}")
        print(f"\tmax_variance = {max_variance} \t sum_variance = {sum_variance}")
        print(f"\tp1_poker_hand = {p1_poker_hand} \t p2_poker_hand = {p2_poker_hand}")

        list_of_games += [[game, p1_poker_hand, p2_poker_hand, max_variance, sum_variance]]

        for run in range(0, exogenous_parameters['number_of_psp_runs_per_game']):
            # Run PSP
            _, _, results = psp(active_set=the_active_set,
                                target_epsilon=exogenous_parameters['target_epsilon'],
                                sampling_schedule=endogenous_parameters['sampling_schedule'],
                                failure_probability_schedule=endogenous_parameters['failure_probability_schedule'],
                                c=exogenous_parameters['c'],
                                sample_condition=poker_dealer,
                                compute_utility=poker_player_utility)

            # Print results for debugging purposes
            # pprint.pprint(statistics)
            # print(f"len_active_set = {results['len_active_set']}")
            list_of_psp_results += [[game] + results['len_active_set']]
        print(f"done!, took {time.time() - t0_game: .4f} sec")

    # Save games and results.
    pd.DataFrame(list_of_games, columns=['game_id', 'p1_poker_hand', 'p2_poker_hand', 'max_variance', 'sum_variance']). \
        to_csv(f"poker_experiments_data/experiment_{exogenous_parameters['experiment_name']}_games.csv", index=False)
    pd.DataFrame(list_of_psp_results, columns=['game_id'] + [f"psp_iter_{i}" for i, _ in enumerate(endogenous_parameters['sampling_schedule'])]). \
        to_csv(f"poker_experiments_data/experiment_{exogenous_parameters['experiment_name']}_psp.csv", index=False)
