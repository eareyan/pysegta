import random
import itertools as it


def poker_dealer(number_of_samples):
    """
    Returns number_of_samples many conditions samples, which in the poker game means number_of_samples many card samples.
    :param number_of_samples:
    :return: a list of number_of_samples card samples.
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
    for dealer_card in condition:
        player_1_hand = sorted(strategy_profile[0] + (dealer_card,), key=lambda x: x[1])
        player_2_hand = sorted(strategy_profile[1] + (dealer_card,), key=lambda x: x[1])
        player_1_hand_rank = hand_rank(player_1_hand)
        player_2_hand_rank = hand_rank(player_2_hand)
        if player_1_hand_rank == player_2_hand_rank:
            u.append(0)
            u_squared.append(0)
        elif player == 1:
            if player_1_hand_rank > player_2_hand_rank:
                u.append(winner_point)
                u_squared.append(winner_point * winner_point)
            else:
                u.append(-winner_point)
                u_squared.append(-winner_point * -winner_point)
        else:
            if player_2_hand_rank > player_1_hand_rank:
                u.append(winner_point)
                u_squared.append(winner_point)
            else:
                u.append(-winner_point)
                u_squared.append(-winner_point * -winner_point)

    return len(condition), sum(u), sum(u_squared)


def hand_rank(hand):
    """
    Assumes the hand is given in ascending order of number.
    :param hand:
    :return:
    """
    # Straight flush.
    if all(hand[0][0] == card[0] for card in hand) and all(hand[i + 1][1] - hand[i][1] == 1 for i in [0, 1, 2, 3]):
        return 1

    # Four of a kind.
    elif all(hand[0][1] == hand[i][1] for i in [1, 2, 3]) or all(hand[1][1] == hand[i][1] for i in [2, 3, 4]):
        return 2

    # Full house.
    elif (all(hand[0][1] == hand[i][1] for i in [1, 2]) and hand[3][1] == hand[4][1]) or (all(hand[2][1] == hand[i][1] for i in [3, 4]) and hand[0][1] == hand[1][1]):
        return 3

    # Flush.
    elif all(hand[0][0] == card[0] for card in hand):
        return 4

    # Straight.
    elif all(hand[i + 1][1] - hand[i][1] == 1 for i in [0, 1, 2, 3]):
        return 5

    # Three of a king
    elif all(hand[0][1] == hand[i][1] for i in [1, 2]) or all(hand[2][1] == hand[i][1] for i in [3, 4]):
        return 6

    # Tow pair
    elif (hand[0][1] == hand[1][1] and hand[2][1] == hand[3][1]) \
            or (hand[1][1] == hand[2][1] and hand[3][1] == hand[4][1]) \
            or (hand[0][1] == hand[1][1] and hand[3][1] == hand[4][1]):
        return 7

    # One pair
    elif any(hand[i][1] == hand[i + 1][1] for i in range(0, 4)):
        return 8

    # High card
    return 9


def get_initial_game():
    """
    Draw a random hand for player 1, a random hand for player 2, and construct the active set. 
    :return: 
    """
    p1_hand = sorted(random.sample(card_deck, 5), key=lambda x: x[1])
    p2_hand = sorted(random.sample([card for card in card_deck if card not in p1_hand], 5), key=lambda x: x[1])
    # p1_hand = (('D', 1), ('D', 2), ('D', 3), ('D', 4), ('D', 5))
    # p2_hand = (('C', 1), ('C', 1), ('H', 3), ('H', 4), ('C', 5))
    print(f"p1_hand = {p1_hand}, rank = {hand_rank(p1_hand)}")
    print(f"p2_hand = {p2_hand}, rank = {hand_rank(p2_hand)}")

    return {(p, (s1, s2)) for p, s1, s2 in it.product([1, 2], it.combinations(p1_hand, 4), it.combinations(p2_hand, 4))}


# Initially, all (player, strategy_profile) pairs are active.
card_deck = [(suit, number) for suit, number in it.product(['C', 'D', 'H', 'S'], range(2, 15))]
# pprint.pprint(card_deck)
