import itertools as it
import random

card_deck = [
    (suit, number) for suit, number in it.product(["C", "D", "H", "S"], range(2, 15))
]


def sort_hand(hand):
    """
    :param hand:
    :return: sorted hand, first by suit then by number.
    """
    return tuple(sorted(sorted(hand, key=lambda x: x[0]), key=lambda x: x[1]))


def poker_dealer(number_of_samples):
    """
    Returns number_of_samples many conditions samples, which in the poker game means number_of_samples many card samples.
    :param number_of_samples: how many sample cards from the dealer deck.
    :return: a list with number_of_samples many card sampled from the dealer's deck.
    """
    return [random.sample(card_deck, 1)[0] for _ in range(0, number_of_samples)]


def poker_player_utility(player, strategy_profile, condition):
    """
    Given a player and a pure strategy profile, returns the utility of the player in the strategy profile.
    :param player:
    :param strategy_profile:
    :param condition:
    :return:
    """
    u = []
    u_squared = []
    winner_point = 1

    player_hand = strategy_profile[0] if player == 1 else strategy_profile[1]
    opponent_hand = strategy_profile[1] if player == 1 else strategy_profile[0]

    for dealer_card in condition:
        player_hand_rank = hand_rank(sort_hand(player_hand + (dealer_card,)))
        oppone_hand_rank = hand_rank(sort_hand(opponent_hand + (dealer_card,)))
        if player_hand_rank == oppone_hand_rank:
            u.append(0)
            u_squared.append(0)
        elif player_hand_rank < oppone_hand_rank:
            u.append(winner_point)
            u_squared.append(winner_point * winner_point)
        else:
            u.append(-winner_point)
            u_squared.append(-winner_point * -winner_point)

    return len(condition), sum(u), sum(u_squared)


def hand_rank(hand):
    """
    Computes the rank of a hand, a number between 1 and 9, where lower numbers indicate better hands.
    Assumes the hand is given in ascending order of number.
    :param hand: a tuple with 5 tuples, each inner tuple representing a card from a standard deck.
    :return: the rank of the hand, a number between 1 and 9, where lower numbers indicate better hands.
    """
    # A hand must consist of 5 cards.
    assert len(hand) == 5

    # Straight flush.
    if all(hand[0][0] == card[0] for card in hand) and all(
        hand[i + 1][1] - hand[i][1] == 1 for i in [0, 1, 2, 3]
    ):
        return 1

    # Four of a kind.
    elif all(hand[0][1] == hand[i][1] for i in [1, 2, 3]) or all(
        hand[1][1] == hand[i][1] for i in [2, 3, 4]
    ):
        return 2

    # Full house.
    elif (
        all(hand[0][1] == hand[i][1] for i in [1, 2]) and hand[3][1] == hand[4][1]
    ) or (all(hand[2][1] == hand[i][1] for i in [3, 4]) and hand[0][1] == hand[1][1]):
        return 3

    # Flush.
    elif all(hand[0][0] == card[0] for card in hand):
        return 4

    # Straight.
    elif all(hand[i + 1][1] - hand[i][1] == 1 for i in [0, 1, 2, 3]):
        return 5

    # Three of a king
    elif (
        all(hand[0][1] == hand[i][1] for i in [1, 2])
        or all(hand[1][1] == hand[i][1] for i in [2, 3])
        or all(hand[2][1] == hand[i][1] for i in [3, 4])
    ):
        return 6

    # Tow pair
    elif (
        (hand[0][1] == hand[1][1] and hand[2][1] == hand[3][1])
        or (hand[1][1] == hand[2][1] and hand[3][1] == hand[4][1])
        or (hand[0][1] == hand[1][1] and hand[3][1] == hand[4][1])
    ):
        return 7

    # One pair
    elif any(hand[i][1] == hand[i + 1][1] for i in range(0, 4)):
        return 8

    # High card
    return 9


def expected_utility(player, strategy_profile):
    """
    Given a player and a strategy profile, returns the player's expected utility for that strategy profile.
    :param player: an integer in {1, 2}
    :param strategy_profile: a tuple of two 4-hands.
    :return: the player's expected utility for that strategy profile.
    """
    # Make sure the input is right.
    assert player == 1 or player == 2
    assert (
        len(strategy_profile) == 2
        and len(strategy_profile[0]) == 4
        and len(strategy_profile[1]) == 4
    )

    players_hand = strategy_profile[0] if player == 1 else strategy_profile[1]
    opponent_hand = strategy_profile[1] if player == 1 else strategy_profile[0]
    prob = 0
    # For each possible card of the dealer.
    for dealer_card in card_deck:
        player_complete_hand = sort_hand(players_hand + (dealer_card,))
        opponent_complete_hand = sort_hand(opponent_hand + (dealer_card,))
        player_hand_rank = hand_rank(player_complete_hand)
        opponent_hand_rank = hand_rank(opponent_complete_hand)
        if player_hand_rank == opponent_hand_rank:
            prob += 0
        elif player_hand_rank < opponent_hand_rank:
            prob += 1 / 52
        else:
            prob += -1 / 52
    return prob


def variance_utility(player, strategy_profile):
    """

    :param player:
    :param strategy_profile:
    :return:
    """
    players_hand = strategy_profile[0] if player == 1 else strategy_profile[1]
    opponent_hand = strategy_profile[1] if player == 1 else strategy_profile[0]
    expectation = expected_utility(player, strategy_profile)
    acum = 0
    # For each possible card of the dealer.
    for dealer_card in card_deck:
        player_complete_hand = sort_hand(players_hand + (dealer_card,))
        opponent_complete_hand = sort_hand(opponent_hand + (dealer_card,))
        player_hand_rank = hand_rank(player_complete_hand)
        opponent_hand_rank = hand_rank(opponent_complete_hand)
        if player_hand_rank == opponent_hand_rank:
            acum += (0 - expectation) * (0 - expectation) * (1 / 52)
        elif player_hand_rank < opponent_hand_rank:
            acum += (1 - expectation) * (1 - expectation) * (1 / 52)
        else:
            acum += (-1 - expectation) * (-1 - expectation) * (1 / 52)
    return acum


def draw_game(size_hand=5, p1_initial_hand=None, p2_initial_hand=None):
    """
    :param size_hand:
    :param p1_initial_hand:
    :param p2_initial_hand:
    Draw a random hand for player 1, a random hand for player 2, and construct the active set.
    :return: a tuple (p1_hand, p2_hand, game) where game is a set of tuples (player, p1_chosen_4_hand, p2_chosen_4_hand)
    """
    p1_poker_hand = (
        tuple(sort_hand(random.sample(card_deck, size_hand)))
        if p1_initial_hand is None
        else p1_initial_hand
    )
    p2_poker_hand = (
        tuple(
            sort_hand(
                random.sample(
                    [card for card in card_deck if card not in p1_poker_hand], size_hand
                )
            )
        )
        if p2_initial_hand is None
        else p2_initial_hand
    )

    return (
        p1_poker_hand,
        p2_poker_hand,
        {
            (p, (s1, s2))
            for p, s1, s2 in it.product(
                [1, 2],
                it.combinations(p1_poker_hand, 4),
                it.combinations(p2_poker_hand, 4),
            )
        },
    )


def compute_neighbors(p1_hand, p2_hand, game):
    """
    Computes the neighborhood structure of the poker game.
    :param p1_hand:
    :param p2_hand:
    :param game:
    """
    neighborhood = {(p, (s1, s2)): [] for p, (s1, s2) in game}
    for p, (s1, s2) in game:
        s = s1 if p == 1 else s2
        hand = p1_hand if p == 1 else p2_hand
        missing_card = tuple(card for card in hand if card not in s)
        for remove_card in s:
            neighborhood[p, (s1, s2)].append(
                tuple(
                    sort_hand(
                        tuple(card for card in s if card != remove_card) + missing_card
                    )
                )
            )

    return neighborhood


def compute_game_stats(game):
    """
    Compute statistics of a game. A game is a set of tuples (player, (strategy_p1, strategy_p2)).
    :param game:
    :return: a pair
    """
    statistics = {}
    for profile in game:
        if profile[1] not in statistics:
            statistics[profile[1]] = {
                "expected_utility_p1": expected_utility(1, profile[1]),
                "expected_utility_p2": expected_utility(2, profile[1]),
                "variance_utility_p1": variance_utility(1, profile[1]),
                "variance_utility_p2": variance_utility(2, profile[1]),
            }

    max_variance = max(
        [
            max(
                statistics[profile[1]]["variance_utility_p1"],
                statistics[profile[1]]["variance_utility_p2"],
            )
            for profile in game
        ]
    )
    sum_variance = sum(
        [
            sum(
                [
                    statistics[profile[1]]["variance_utility_p1"],
                    statistics[profile[1]]["variance_utility_p2"],
                ]
            )
            for profile in game
        ]
    )
    return max_variance, sum_variance, statistics
