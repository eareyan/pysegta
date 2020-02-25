import itertools as it
import numpy as np
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
from config.config import gambit_location, path_to_nfg_files
import pprint
import pulp
import subprocess


class GameStructure:
    """
    A structure of a normal-form game. Implements methods that apply to any NFG plus helper methods to communicate with gambit.
    """

    def __init__(self,
                 title: str,
                 players: List[str],
                 num_strategies: List[int]):
        assert len(players) == len(num_strategies)
        self.title = title
        self.players = players
        self.num_strategies = num_strategies
        self.num_players = len(players)
        assert self.num_players > 1
        self.total_num_strategy_profiles = np.prod(self.num_strategies)
        self.total_size_of_game = self.total_num_strategy_profiles * self.num_players
        self.neighbours = {}
        self.payoffs = {}

    def __str__(self):
        """
        :return: A string representation of the game.
        """
        return f'Game: {self.title}. \n' \
               f'\tPlayers: {self.players}, \n' \
               f'\tnum_strategies: {self.num_strategies}, \n' \
               f'\ttotal size of game = {self.total_size_of_game}'

    def set_payoffs(self, payoffs: Dict[Tuple, List]):
        """
        Sets the game's payoffs.
        :param payoffs: a dictionary with payoffs for the game
        :return: None
        """
        self.payoffs = payoffs

    def is_profile_valid(self, strategy_profile: Tuple[int, ...]) -> bool:
        """
        A profile is valid if it is a tuple where each entry is a valid strategy for the corresponding player.
        :param strategy_profile: a tuple of integers
        :return: true if and only if the strategy profile is valid.
        """
        return len(strategy_profile) == self.num_players and all(list(map(lambda x: 0 <= x[1] <= self.num_strategies[x[0]] - 1, enumerate(strategy_profile))))

    def get_neighbours(self, strategy_profile: Tuple[int, ...], player: int) -> List[Tuple[int, ...]]:
        """
        Return the neighbouring strategy profiles of the given strategy profile where the given player changes strategies.
        :param strategy_profile: a strategy profile, i.e., a tuple of integers
        :param player: an integer
        :return: the neighbouring strategy profiles of the given strategy profile where the given player changes strategies.
        """
        assert self.is_profile_valid(strategy_profile) and 0 <= player <= self.num_players - 1
        if (strategy_profile, player) not in self.neighbours:
            neigh = [[strategy] for strategy in strategy_profile]
            neigh[player] = list(filter(lambda s: s != strategy_profile[player], [s for s in range(0, self.num_strategies[player])]))
            self.neighbours[(strategy_profile, player)] = list(it.product(*neigh))
        return self.neighbours[(strategy_profile, player)]

    def regret_at_pure(self, player: int, pure: Tuple[int, ...]):
        """
        Returns the regret of the player at the strategy profile considering deviations only to pure strategies.
        :param player: the player
        :param pure: a pure strategy profile defined as a tuple of integers
        :return: a float, the max regret
        """
        return max([self.payoffs[neigh][player] - self.payoffs[pure][player] for neigh in self.get_neighbours(pure, player) + [pure]])

    def max_regret_at_pure(self, pure: Tuple[int, ...]):
        """
        Returns the maximum regret of the strategy profile M(s) over pure strategies. M(s) is defined as:
        M(s) = max_{players: p} max_{pure neigh: n} (u_p(n) - u_p(s))
        :param pure: a pure strategy profile
        :return: a float, the maximum regret at s over pure strategies
        """
        return max([self.regret_at_pure(p, pure) for p in range(self.num_players)])

    def regret(self, player: int, mix: List[Tuple[float, ...]]) -> float:
        """
        Computes the regret of the player at the given strategy profile mix.
        :param player: an integer.
        :param mix: a list of tuple of floats denoting the mix strategies of players.
        :return: the regret of the player at the given strategy profile mix.
        """
        # Solve a linear program to maximize payoffs for the given player fixing the mix of other players.
        assert len(self.payoffs) > 0
        prob = pulp.LpProblem("Regret maximization problem", pulp.LpMaximize)
        mixing_vars = pulp.LpVariable.dict('mass', range(self.num_strategies[player]), 0.0, 1.0)
        prob += pulp.lpSum([mixing_vars[profile[player]] * self.payoffs[profile][player] * np.prod([mix[p][s] if p != player else 1.0 for p, s in enumerate(profile)])
                            for profile, _ in self.payoffs.items()])
        prob += sum(v for i, v in mixing_vars.items()) == 1.0
        prob.solve()
        assert pulp.LpStatus[prob.status] == 'Optimal'

        # Get the optimal mix for the player fixing the mix of all other players
        player_new_mix = tuple([var.varValue for _, var in mixing_vars.items()])

        # Expected payoff with the new mix:
        new_mix_payoff = sum([self.payoffs[profile][player] * np.prod([mix[p][s] if p != player else player_new_mix[s] for p, s in enumerate(profile)])
                              for profile, _ in self.payoffs.items()])

        # Current expected payoff:
        current_payoff = sum([self.payoffs[profile][player] * np.prod([mix[p][s] for p, s in enumerate(profile)]) for profile, _ in self.payoffs.items()])
        assert new_mix_payoff >= current_payoff

        return new_mix_payoff - current_payoff

    def pure_profile_to_mix(self, pure: Tuple[int, ...]):
        """
        Converts a pure profile to a mix by casting ints to floats
        :param pure:
        :return:
        """
        the_mix_profile = []
        for player, strategy in enumerate(pure):
            the_mix_profile += [tuple([0.0 if i != strategy else 1.0 for i in range(self.num_strategies[player])])]
        return the_mix_profile

    @staticmethod
    def is_eq_pure(mix: List[Tuple[float, ...]]):
        """
        Checks if a given profile is pure.
        :param mix: a mix profile
        :return: true if mix is actually pure
        """
        return all([item for sublist in [[i.is_integer() for i in mix] for mix in mix] for item in sublist])

    @staticmethod
    def add_one_to_profile(current_strategy, bases):
        """
        Given a strategy profile, it returns "the next" profile, which is defined as the profile
        where the strategy of the first player is incremented by one accounting for any "carry" over
        :param current_strategy: a strategy profiles
        :param bases: a list with the number of strategies of each player.
        :return:
        """
        if len(current_strategy) == 0:
            raise Exception(f'overflow, no more strategies after {current_strategy}')
        if (current_strategy[0] + 1) % bases[0] == 0:
            return [0] + GameStructure.add_one_to_profile(current_strategy[1:], bases[1:])
        else:
            current_strategy[0] = current_strategy[0] + 1
            return current_strategy

    def nfg_representation(self) -> str:
        """
        Produces a NFG representation of the game that Gambit can read.
        :return: a NFG representation of the game.
        """
        nfg = f'NFG 1 R \"{self.title}\" ' + "{ " + ' '.join(['"' + player_name + '"' for player_name in self.players]) + " }\n\n"
        nfg += '{ ' + '\n'.join([' '.join(["{"] + [f'"{i}"' for i in range(num_strats)] + ["}"]) for num_strats in self.num_strategies]) + '\n}'
        nfg += '\n""\n\n'
        nfg += '{\n'
        current_profile = [0] * self.num_players
        for _ in range(self.total_num_strategy_profiles - 1):
            nfg += '{ "' + str(current_profile) + '" ' + ', '.join([str(self.payoffs[tuple(current_profile)][p]) for p in range(0, self.num_players)]) + " }\n"
            current_profile = GameStructure.add_one_to_profile(current_profile, self.num_strategies)
        nfg += '{ "' + str(current_profile) + '" ' + ', '.join([str(self.payoffs[tuple(current_profile)][p]) for p in range(0, self.num_players)]) + " }\n"
        nfg += '}\n'
        nfg += ' '.join([str(i) for i in range(1, self.total_num_strategy_profiles + 1)])
        return nfg

    def save_nfg_representation(self, location: str):
        """
        Saves the NFG representation of the game to a file.
        :param location: the location of the file where the NFG of the game is to be saved.
        :return: None
        """
        with open(location, 'w') as the_file:
            the_file.write(self.nfg_representation())

    def solve_nash(self):
        """
        Solves for the Nash equilibria of this game by calling gambit. Takes the output of gambit and stores in a file in a format we can use.
        :return: None
        """
        self.save_nfg_representation(f'{path_to_nfg_files}{self.title.replace(" ","_")}.nfg')
        equilibria = subprocess.check_output([gambit_location, '-q', f'{path_to_nfg_files}{self.title.replace(" ","_")}.nfg']).decode("utf-8").split('\n')
        # Take the output from gambit and turn it into a list of list of tuples.
        list_of_nashs = []
        for equilibrium in equilibria:
            if equilibrium != '':
                eq_data = equilibrium.split(',')[1:]
                i = 0
                x = []
                for p in range(self.num_players):
                    player_nash = []
                    for s in range(self.num_strategies[p]):
                        player_nash += [float(eq_data[i])]
                        i += 1
                    x += [tuple(player_nash)]
                list_of_nashs += [x]
        # Removing duplicate Nashs.
        list_of_nashs.sort()
        list_of_nashs = list(k for k, _ in it.groupby(list_of_nashs))
        # Write back a file with a list of lists with all Nashs of the games, as given by the solver, with no duplicates.
        with open(path_to_nfg_files + self.title.replace(" ", "_") + '_sol', 'w') as the_file:
            the_file.write(str(list_of_nashs))
        return list_of_nashs


class EstimatedGame(GameStructure, ABC):
    """
    An estimated game enhances the GameStructure to maintain various bookkeeping structure
    necessary when estimating a game from data. It also defines an abstract method sample
    that must be defined by the extending class to articulate how sampling is performed for the game.
    """

    def __init__(self,
                 title: str,
                 players: List[str],
                 num_strategies: List[int]):
        GameStructure.__init__(self, title, players, num_strategies)
        self.active_profiles = set(it.product(*[[a for a in range(0, self.num_strategies[p])] for p in range(0, self.num_players)]))
        self.inactive_profiles = set()
        assert self.total_size_of_game == len(self.active_profiles) * self.num_players
        self.estimated_payoffs = {}

    @abstractmethod
    def sample(self, strategy_profile: Tuple[int], m: int) -> List:
        """
        Compute m samples of payoffs for each player in the given strategy profile
        :param strategy_profile: a tuple of integers
        :param m: the number of samples for each player at the given strategy profile
        :return: a list of lists, the ith inner list with m samples of payoffs for the ith player.
        """
        pass

    def is_profile_active(self, strategy_profile: Tuple[int, ...]) -> bool:
        """
        :param strategy_profile: a list of integers
        :return: True if and only if the given strategy profile is active
        """
        assert self.is_profile_valid(strategy_profile)
        return strategy_profile in self.active_profiles

    def deactivate_profile(self, strategy_profile: Tuple[int, ...]) -> None:
        """
        To be able to keep track of profiles whose eps guarantee are already meet, we deactivate them.
        :param strategy_profile: a list of integers.
        :return: None
        """
        assert self.is_profile_valid(strategy_profile)
        assert self.is_profile_active(strategy_profile)
        self.active_profiles.remove(strategy_profile)
        self.inactive_profiles.add(strategy_profile)

    def size_active_game(self) -> int:
        """
        :return: the size of the game defines as the number of individual payoffs necessary to completely describe the game.
        """
        return len(self.active_profiles) * self.num_players

    def update_estimated_payoff(self, strategy_profile: Tuple[int, ...], means: List, epsilon: float) -> None:
        """
        Updates the estimated payoffs at a strategy profile.
        :param strategy_profile: a list of integers.
        :param means: a list of floats
        :param epsilon: a float.
        :return: None
        """
        assert self.is_profile_valid(strategy_profile)
        assert self.is_profile_active(strategy_profile)
        assert epsilon >= 0
        self.estimated_payoffs[strategy_profile] = (means, epsilon)

    def is_profile_eps_dominated(self, strategy_profile: Tuple[int, ...], eps: float):
        """
        Given a strategy profile and an epsilon value, return true iff for every player, the
        corresponding strategy of the player is eps dominated.
        :param strategy_profile: a tuple of integers denoting the strategy profile
        :param eps: a floating value
        :return: boolean
        """
        assert eps >= 0
        return all([self.is_strategy_eps_dominated(strategy_profile, player, eps) for player in range(0, self.num_players)])

    def is_strategy_eps_dominated(self, strategy_profile: Tuple[int, ...], player: int, eps: float) -> bool:
        """
        Determines whether the strategy of a given player at a given strategy profile is epsilon dominated or not.
        :param strategy_profile: a tuple of integers.
        :param player: the index of a player (integer)
        :param eps: a floating value.
        :return: boolean
        """
        assert 0 <= player <= self.num_players and eps >= 0
        active_neighs = list(filter(lambda profile: self.is_profile_active(profile), self.get_neighbours(strategy_profile, player)))
        return any([self.estimated_payoffs[neighbour][0][player] - eps >= self.estimated_payoffs[strategy_profile][0][player] for neighbour in active_neighs])

    def set_payoffs(self) -> None:
        """
        sets the game payoffs to be the mean payoffs as estimated so far.
        """
        self.payoffs = {profile: mean_payoff for profile, (mean_payoff, _) in self.estimated_payoffs.items()}

    def pp_estimated_payoffs(self):
        """
        pretty print the game payoffs for human readability.
        """
        pp = pprint.PrettyPrinter(indent=4)
        print(f'\t estimated payoffs')
        pp.pprint(self.estimated_payoffs)
