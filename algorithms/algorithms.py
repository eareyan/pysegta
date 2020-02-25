import numpy as np
from typing import List, Tuple
from structures.gamestructure import EstimatedGame
from structures.bounds import Bound


def global_sampling(estimated_game: EstimatedGame, bound: Bound, m: int, delta: float, c: float) -> Tuple[float, int]:
    """
    Implementation of global sampling.
    :param estimated_game: object over which the sampling is going to take place.
    :param bound: object of type Bound which is going to be used to compute a global bound.
    :param m: number of samples per active strategy profile.
    :param delta: error tolerance.
    :param c: bound on the range of payoffs.
    :return: epsilon guarantee, i.e., the radius of the bound that holds over all payoffs simultaneously, and the total number of samples
    """
    assert m > 0 and 0 < delta < 1
    samples, means = {}, {}
    for active_strategy_profile in estimated_game.active_profiles:
        samples[active_strategy_profile] = estimated_game.sample(active_strategy_profile, m)
        means[active_strategy_profile] = [np.mean(sample_player_payoff) for sample_player_payoff in samples[active_strategy_profile]]
    epsilon = bound.compute_bound({'estimated_game': estimated_game,
                                   'm': m,
                                   'delta': delta,
                                   'c': c,
                                   'means': means,
                                   'samples': samples})
    for strategy_profile in estimated_game.active_profiles:
        estimated_game.update_estimated_payoff(strategy_profile, means[strategy_profile], epsilon)
    return epsilon, m * estimated_game.size_active_game()


def psp(estimated_game: EstimatedGame, bound: Bound, m_schedule: List[int], delta_schedule: List[float], target_epsilon: float, c: float, verbose: bool = False) \
        -> Tuple[bool, int, int, float]:
    """
    Implements progressive sampling with pruning.
    :param estimated_game: object over which the sampling is going to take place.
    :param bound: object of type Bound which is going to be used to compute a global bound.
    :param m_schedule: a list with number of samples to take at each iteration.
    :param delta_schedule: a list with delta budget for each iteration.
    :param target_epsilon: epsilon guaranteed desired
    :param c: bound on the range of payoffs.
    :param verbose: boolean to indicate whether human-readable output should be printed, mainly for debugging purposes
    :return: bool indicating if the epsilon guaranteed was meet and the total number of samples
    """
    assert len(m_schedule) == len(delta_schedule)
    total_num_profiles_pruned = 0
    total_num_samples = 0
    current_epsilon = np.inf
    if verbose:
        print('////// Starting PSP //////')
    for t, (m_t, delta_t) in enumerate(zip(m_schedule, delta_schedule)):
        current_epsilon, gs_num_samples = global_sampling(estimated_game, bound, m_t, delta_t, c)
        total_num_samples += gs_num_samples
        if verbose:
            print(f'\t***** iteration {t} ***** \t eps_t = {current_epsilon}, m_t = {m_t}, delta_t = {delta_t}')
        if current_epsilon <= target_epsilon:
            # Algorithm terminates and outputs the current estimated game as the final game.
            if verbose:
                print(f'////// eps guarantee meet, finishing //////\n')
            return True, total_num_samples, total_num_profiles_pruned, current_epsilon
        if verbose:
            estimated_game.pp_estimated_payoffs()
        # Iterated removal of dominated strategies, i.e., strategy pruning.
        # This means deactivate strategy profiles that are dominated and repeat until no more deactivation happens.
        pruning = True
        pruning_round = 0
        while pruning:
            pruning = False
            pruning_round += 1
            if verbose:
                print(f'\t\t pruning round = {pruning_round}')
            for candidate_strategy_profile in list(estimated_game.active_profiles):
                # print(f'\t\t\t candidate: {candidate_strategy_profile}')
                if estimated_game.is_profile_eps_dominated(candidate_strategy_profile, 2 * current_epsilon):
                    pruning = True
                    total_num_profiles_pruned += 1
                    estimated_game.deactivate_profile(candidate_strategy_profile)
                    if verbose:
                        print(f'\t\t pruning {candidate_strategy_profile}')
    if verbose:
        print(f'////// WARNING eps guarantee NOT meet, finishing //////\n')
    return False, total_num_samples, total_num_profiles_pruned, current_epsilon
