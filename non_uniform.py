import math


def compute_confidence(m, U, V, size_of_game, c, current_delta):
    """
    :param m:
    :param U:
    :param V:
    :param size_of_game:
    :param c:
    :param current_delta:
    """
    v_hat = (V - ((U * U) / m)) / (m - 1)
    epsilon_v = (c * c * math.log(3 * size_of_game / current_delta)) / (
        m - 1
    ) + math.sqrt(
        math.pow((c * c * math.log(3 * size_of_game / current_delta)) / (m - 1), 2)
        + ((2 * c * c * v_hat * math.log(3 * size_of_game / current_delta)) / (m - 1))
    )
    return min(
        c * math.sqrt((math.log(3 * size_of_game / current_delta)) / (2 * m)),
        ((c * math.log(3 * size_of_game / current_delta)) / (3 * m))
        + math.sqrt(
            ((2 * (v_hat + epsilon_v)) * math.log(3 * size_of_game / current_delta)) / m
        ),
    )


def prune_by_regret(un_dominated_set, statistics, confidence, neighborhood):

    # To start with, the set of pruning pairs is empty.
    prune_by_regret_set = set()

    # Only un_dominated strategies are candidates for regret pruning.
    for p, (s1, s2) in un_dominated_set:

        # Compute the maximum deviation, up to estimation errors, in the corresponding neighborhood.
        best_alternative = max(
            [
                (
                    statistics[p, (x, s2) if p == 1 else (s1, x)][1]
                    / statistics[p, (x, s2) if p == 1 else (s1, x)][0]
                )
                - confidence[p, (x, s2) if p == 1 else (s1, x)]
                for x in filter(
                    lambda x: (p, (x, s2))
                    if p == 1
                    else (p, (s1, x)) in un_dominated_set,
                    neighborhood[p, (s1, s2)],
                )
            ],
            default=-math.inf,
        )

        # Prune the (player, strategy profile) pair in case the regret, up to errors, is too big.
        avg = statistics[p, (s1, s2)][1] / statistics[p, (s1, s2)][0]
        if avg + confidence[p, (s1, s2)] < best_alternative:
            prune_by_regret_set.add((p, (s1, s2)))

    return prune_by_regret_set


def psp(
    neighborhood,
    active_set,
    target_epsilon,
    sampling_schedule,
    fail_prob_schedule,
    c,
    sample_condition,
    compute_utility,
    allow_regret_prune=True,
    allow_well_est_prune=True,
):
    """
    Progressive sampling with pruning.
    :param neighborhood:
    :param active_set:
    :param target_epsilon:
    :param sampling_schedule:
    :param fail_prob_schedule:
    :param c:
    :param sample_condition:
    :param compute_utility:
    :param allow_regret_prune:
    :param allow_well_est_prune:
    :return:
    """
    # Assume the size of the game is the size of the first active set, which is supposed to be the entire game.
    size_of_game = len(active_set)

    # Initialize estimates.
    statistics = {
        (player, strategy_profile): (0, 0, 0) for player, strategy_profile in active_set
    }
    confidence = {
        (player, strategy_profile): math.inf for player, strategy_profile in active_set
    }

    # Collect results.
    size_active_set = [-1 for _ in range(0, len(sampling_schedule))]
    size_inactive_set = [-1 for _ in range(0, len(sampling_schedule))]
    size_well_estimated_set = [-1 for _ in range(0, len(sampling_schedule))]
    size_dominated_set = [-1 for _ in range(0, len(sampling_schedule))]

    # Keep track of categories.
    well_estimated_set = set()
    un_dominated_set = active_set.copy()
    dominated_set = set()

    # Iterate through the sampling schedule.
    for t, (current_number_of_samples, current_delta) in enumerate(
        zip(sampling_schedule, fail_prob_schedule)
    ):

        # Print progress. Debug only.
        # print(f"len(active_set) = {len(active_set)}")

        # Sample the condition.
        current_condition = sample_condition(current_number_of_samples)

        # For each active (player, strategy profile) pair, refine estimates.
        for player, strategy_profile in active_set:

            # Compute utilities of pair (player, strategy_profile) only in case the pair is not well estimated and not dominated.
            current_estimates = compute_utility(
                player=player,
                strategy_profile=strategy_profile,
                condition=current_condition,
            )

            # Add estimate to current counts.
            statistics[player, strategy_profile] = [
                sum(x)
                for x in zip(statistics[player, strategy_profile], current_estimates)
            ]

            # m is the total number of samples for this (player, strategy profile), U its total sum, and V its squared sum.
            m, U, V = statistics[player, strategy_profile]

            # Compute the confidence of the (player, strategy_profile) pair.
            confidence[player, strategy_profile] = compute_confidence(
                m, U, V, size_of_game, c, current_delta
            )

            # Debug print, output what the estimates look like so far.
            # print(f"{player, strategy_profile} \t {m} \t {statistics[player, strategy_profile]} \t\t\t {confidence[player, strategy_profile]:.6f}")

        # Prune by regret
        if allow_regret_prune:
            prune_by_regret_set = prune_by_regret(
                un_dominated_set=un_dominated_set,
                statistics=statistics,
                confidence=confidence,
                neighborhood=neighborhood,
            )

            # Update the dominated/un_dominated set.
            dominated_set = dominated_set.union(prune_by_regret_set)
            un_dominated_set = un_dominated_set - dominated_set
            assert len(un_dominated_set) + len(dominated_set) == size_of_game
            size_dominated_set[t] = len(dominated_set)

        # Update the different sets.
        if allow_well_est_prune:
            well_estimated_set = well_estimated_set.union(
                set(filter(lambda pair: confidence[pair] <= target_epsilon, active_set))
            )
            size_well_estimated_set[t] = len(well_estimated_set)
            inactive_set = well_estimated_set.union(dominated_set)
        else:
            inactive_set = dominated_set

        active_set = (active_set - well_estimated_set) - dominated_set

        assert len(active_set) + len(inactive_set) == size_of_game
        size_active_set[t] = len(active_set)
        size_inactive_set[t] = len(inactive_set)

        # Debug prints, to manually inspect the progress of the algorithm.
        # print(f"active_set \t\t\t= {len(active_set)}")
        # print(f"inactive_set \t\t\t= {len(inactive_set)}")
        # print(f"well_estimated_set \t= {len(well_estimated_set)}")
        # print(f"dominated_set \t\t= {len(dominated_set)}")
        # print(f"un_dominated_set \t= {len(un_dominated_set)}")
        # print(f"well or dominated \t= {len(well_estimated_set.union(dominated_set))}")

        if len(active_set) == 0:
            # No more active pairs, break and finish the algorithm.
            break

    # Collect and return and results.
    results = {
        "size_active_set": size_active_set,
        "size_inactive_set": size_inactive_set,
        "size_well_estimated_set": size_well_estimated_set,
        "size_dominated_set": size_dominated_set,
    }
    return statistics, confidence, results
