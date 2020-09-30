import math
import pprint
import pandas as pd
import numpy as np
import poker_game
import scipy.stats


def compute_conf_interval(sample, confidence_level=0.95):
    degrees_freedom = len(sample) - 1
    sample_mean = np.mean(sample)
    sample_standard_error = scipy.stats.sem(sample)
    lb, ub = scipy.stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)

    return sample_mean, lb, ub, (ub - lb) / 2


def psp(active_set, target_epsilon, sampling_schedule, failure_probability_schedule, c, sample_condition, compute_utility):
    """
    Progressive sampling with pruning.
    :param active_set:
    :param target_epsilon:
    :param sampling_schedule:
    :param failure_probability_schedule:
    :param c:
    :param sample_condition:
    :param compute_utility:
    :return:
    """
    # Assume the size of the game is the size of the first active set, which is supposed to be the entire game.
    size_of_game = len(active_set)

    # Initialize estimates.
    statistics = {(player, strategy_profile): (0, 0, 0) for player, strategy_profile in active_set}
    confidence = {(player, strategy_profile): math.inf for player, strategy_profile in active_set}

    # Collect results.
    result = {'len_active_set': [len(active_set)]}

    # Iterate through the sampling schedule.
    for current_number_of_samples, current_delta in zip(sampling_schedule, failure_probability_schedule):
        print(f"len(active_set) = {len(active_set)}")

        # Sample the condition.
        current_condition = sample_condition(current_number_of_samples)

        # For each active (player, strategy profile) pair, refine estimates.
        for player, strategy_profile in active_set:
            current_estimates = compute_utility(player=player, strategy_profile=strategy_profile, condition=current_condition)

            # Add estimate to current counts.
            statistics[player, strategy_profile] = [sum(x) for x in zip(statistics[player, strategy_profile], current_estimates)]

            # m is the total number of samples for this (player, strategy profile), U its total sum, and V its squared sum.
            m, U, V = statistics[player, strategy_profile]

            # Math stuff.
            v_hat = (V - ((U * U) / m)) / (m - 1)
            epsilon_v = (c * c * math.log(3 * size_of_game / current_delta)) / (m - 1) + \
                        math.sqrt(math.pow((c * c * math.log(3 * size_of_game / current_delta)) / (m - 1), 2) + ((2 * c * c * v_hat * math.log(3 * size_of_game / current_delta)) / (m - 1)))
            confidence[player, strategy_profile] = min(c * math.sqrt((math.log(3 * size_of_game / current_delta)) / (2 * m)),
                                                       ((c * math.log(3 * size_of_game / current_delta)) / (3 * m))
                                                       + math.sqrt(((2 * (v_hat + epsilon_v)) * math.log(3 * size_of_game / current_delta)) / m))
            # Debug print, output what the estimates look like so far.
            # print(f"{player, strategy_profile} \t {m} \t {statistics[player, strategy_profile]} \t\t\t {confidence[player, strategy_profile]:.6f}")

        active_set = set(filter(lambda pair: confidence[pair] > target_epsilon, active_set))
        result['len_active_set'].append(len(active_set))

        if len(active_set) == 0:
            print(f"No more active pairs")
            return statistics, confidence, result

    return statistics, confidence, result


if __name__ == '__main__':

    data = pd.read_csv('all_results.csv')
    print([compute_conf_interval(data[str(i)]) for i in [0, 1, 2, 3]])
    exit()

    # Parameters.
    the_target_epsilon = 0.15
    the_c = 2
    the_sampling_schedule = [100, 400, 1600, 3200]
    the_failure_probability_schedule = [0.1 / 4 for _ in range(0, 4)]

    # the_target_epsilon = 0.05
    # the_c = 2
    # the_sampling_schedule = [100, 400, 1600, 3200, 6400]
    # the_failure_probability_schedule = [0.1 / 5 for _ in range(0, 4)]

    all_results = []
    for _ in range(0, 1000):
        # Draw the initial hands.
        the_active_set = poker_game.get_initial_game()
        # pprint.pprint(the_active_set)
        # print(len(the_active_set))
        # print(the_sample_condition(100))

        # Run PSP
        statistics, confidence, result = psp(active_set=the_active_set,
                                             target_epsilon=the_target_epsilon,
                                             sampling_schedule=the_sampling_schedule,
                                             failure_probability_schedule=the_failure_probability_schedule,
                                             c=the_c,
                                             sample_condition=poker_game.poker_dealer,
                                             compute_utility=poker_game.poker_player_utility)
        print(f"result = {result}")
        all_results.append(result['len_active_set'])

    pd.DataFrame(all_results).to_csv('all_results.csv')
