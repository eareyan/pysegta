import math
import pprint
import pandas as pd
import numpy as np
import poker_game
import scipy.stats


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
    results = {}
    len_active_set = [0 for _ in range(0, len(sampling_schedule))]

    # Iterate through the sampling schedule.
    for t, (current_number_of_samples, current_delta) in enumerate(zip(sampling_schedule, failure_probability_schedule)):

        # Print progress. Debug only.
        # print(f"len(active_set) = {len(active_set)}")

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
        len_active_set[t] = len(active_set)

        if len(active_set) == 0:
            # No more active pairs, break and finish the algorithm. 
            break

    results['len_active_set'] = len_active_set
    return statistics, confidence, results
