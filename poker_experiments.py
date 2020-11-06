import itertools as it
import math
import os
import time

import names
import pandas as pd
from prettytable import PrettyTable

from non_uniform import psp
from poker_game import (
    draw_game,
    compute_game_stats,
    poker_dealer,
    poker_player_utility,
    compute_neighbors,
)


def compute_n(c, delta, epsilon, size_of_game):
    return (c / epsilon) * (c / epsilon) * 0.5 * math.log((2 * size_of_game) / delta)


def compute_eps(c, delta, size_of_game, number_of_samples):
    return c * math.sqrt(
        (math.log((2.0 * size_of_game) / delta)) / (2.0 * number_of_samples)
    )


def create_ptable(dictionary):
    ptable = PrettyTable()
    ptable.title = "Running experiments with parameters"
    ptable.field_names = ["Parameter", "Value"]
    for k, v in dictionary.items():
        ptable.add_row([k, v])
    return ptable


# Exogenous Parameters.
exogenous_parameters = {
    "experiment_name": names.get_full_name().replace(" ", "_"),
    "size_of_game": 50,
    "target_epsilon": 0.15,
    "c": 2,
    "delta": 0.1,
    # 'number_of_games': 100,
    "number_of_games": 1,
    "number_of_psp_runs_per_game": 30,
    # "number_of_psp_runs_per_game": 10,
}

# Endogenous Parameters.
endogenous_parameters = {
    "m": compute_n(
        c=exogenous_parameters["c"],
        delta=exogenous_parameters["delta"],
        epsilon=exogenous_parameters["target_epsilon"],
        size_of_game=exogenous_parameters["size_of_game"],
    ),
    "fail_prob_schedule": [exogenous_parameters["delta"] / 4 for _ in range(0, 4)],
}
endogenous_parameters["sampling_schedule"] = [
    math.ceil(endogenous_parameters["m"] / 4),
    math.ceil(endogenous_parameters["m"] / 2),
    math.ceil(endogenous_parameters["m"]),
    math.ceil(2 * endogenous_parameters["m"]),
]

if __name__ == "__main__":

    # Check if an experiment with this name already exists so as not to overwrite any previous results.
    if os.path.exists(
        f"poker_experiments_data/experiment_{exogenous_parameters['experiment_name']}"
    ):
        print(
            f"Experiment {exogenous_parameters['experiment_name']} already exists, try again. "
        )
        exit()

    # Print parameters
    parameter_ptable = create_ptable({**exogenous_parameters, **endogenous_parameters})
    print(parameter_ptable)
    with open(
        f"poker_experiments_data/experiment_{exogenous_parameters['experiment_name']}.txt",
        "w",
    ) as the_file:
        the_file.write(str(parameter_ptable))

    # Run experiments
    list_of_games = []
    list_of_psp_results = []
    for game in range(0, exogenous_parameters["number_of_games"]):
        # Time how long it takes to produce results for a single game
        t0_game = time.time()

        # Draw a game
        p1_poker_hand, p2_poker_hand, the_active_set = draw_game()
        neighborhood = compute_neighbors(p1_poker_hand, p2_poker_hand, the_active_set)

        # Compute the game's statistics
        max_variance, sum_variance, _ = compute_game_stats(the_active_set)

        # Print info about the game for debugging purposes.
        print(f"Game # {game}")
        print(f"\tmax_variance = {max_variance} \t sum_variance = {sum_variance}")
        print(f"\tp1_poker_hand = {p1_poker_hand} \t p2_poker_hand = {p2_poker_hand}")

        list_of_games += [
            [game, p1_poker_hand, p2_poker_hand, max_variance, sum_variance]
        ]
        for allow_regret_prune, allow_well_est_prune in it.product(
            [False, True], [False, True]
        ):
            for run in range(0, exogenous_parameters["number_of_psp_runs_per_game"]):
                # Run PSP
                _, _, results = psp(
                    neighborhood=neighborhood,
                    active_set=the_active_set,
                    target_epsilon=exogenous_parameters["target_epsilon"],
                    sampling_schedule=endogenous_parameters["sampling_schedule"],
                    fail_prob_schedule=endogenous_parameters["fail_prob_schedule"],
                    c=exogenous_parameters["c"],
                    sample_condition=poker_dealer,
                    compute_utility=poker_player_utility,
                    allow_regret_prune=allow_regret_prune,
                    allow_well_est_prune=allow_well_est_prune,
                )

                # Print results for debugging purposes
                # pprint.pprint(statistics)
                # print(f"size_active_set = {results['size_active_set']}")
                list_of_psp_results += [
                    [
                        game,
                        1 if allow_regret_prune else 0,
                        1 if allow_well_est_prune else 0,
                    ]
                    + results["size_active_set"]
                    + results["size_well_estimated_set"]
                    + results["size_dominated_set"]
                ]
        print(f"done!, took {time.time() - t0_game: .4f} sec")

    # Save games and results.
    pd.DataFrame(
        list_of_games,
        columns=[
            "game_id",
            "p1_poker_hand",
            "p2_poker_hand",
            "max_variance",
            "sum_variance",
        ],
    ).to_csv(
        f"poker_experiments_data/experiment_{exogenous_parameters['experiment_name']}_games.csv",
        index=False,
    )
    pd.DataFrame(
        list_of_psp_results,
        columns=["game_id", "allow_regret_prune", "allow_well_est_prune"]
        + [
            f"size_active_set{i}"
            for i, _ in enumerate(endogenous_parameters["sampling_schedule"])
        ]
        + [
            f"size_well_estimated_set{i}"
            for i, _ in enumerate(endogenous_parameters["sampling_schedule"])
        ]
        + [
            f"size_dominated_set{i}"
            for i, _ in enumerate(endogenous_parameters["sampling_schedule"])
        ],
    ).to_csv(
        f"poker_experiments_data/experiment_{exogenous_parameters['experiment_name']}_psp.csv",
        index=False,
    )
