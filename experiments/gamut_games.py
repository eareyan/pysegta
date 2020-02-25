from abc import ABC, abstractmethod
from structures.gamestructure import EstimatedGame, GameStructure
from typing import Dict, Tuple, List, Optional
from experiments.noise import Noise
from config.config import path_to_nfg_files, gamut_location, java_location
import os


class GamutGame(EstimatedGame, ABC):
    """
    A GamutGame class provides an abstraction to games drawn by Gamut.
    """

    def __init__(self, title: str, players: List[str], num_strategies: List[int], max_payoff: float, min_payoff: float, noise: Optional[Noise],
                 do_setup: bool = True,
                 base_payoffs: Dict = None):
        # Setup basic parameters of the game
        EstimatedGame.__init__(self, title, players, num_strategies)
        self.max_payoff = max_payoff
        self.min_payoff = min_payoff
        self.noise = noise
        if do_setup:
            self.__setup__()
        if base_payoffs is not None:
            self.base_payoffs = base_payoffs

    def __setup__(self):
        # Set the location for the .nfg file that is going to be produced by gamut.
        self.nfg_output_file_location = f'{path_to_nfg_files}{self.title.replace(" ", "_")}.nfg'

        # The following call to gamut is common to all gamut-based games
        self.initial_gamut_command = f"{java_location} -jar {gamut_location}/gamut.jar -f {self.nfg_output_file_location} -output GambitOutput " \
                                     f"-min_payoff {str(self.min_payoff)} -max_payoff {str(self.max_payoff)} -normalize 1 "
        self.command = self.initial_gamut_command + self.complete_gamut_command()
        # Call gamut. This call should produce a .nfg file in self.nfg_output_file_location
        os.system(self.command + ' >/dev/null 2>&1')  # Run this if you want to suppress output from gamut
        # os.system(self.command)  # Run this if you do not want to suppress output from gamut

        # Read the .nfg file back again and store the payoffs ad base payoffs. On top of these payoffs, we will add noise when sampling.
        self.base_payoffs = self.parse_gamut_payoffs(nfg_output_file_location=self.nfg_output_file_location)
        self.payoffs = self.base_payoffs

    def parse_gamut_payoffs(self, nfg_output_file_location: str) -> Dict[Tuple, List]:
        """
        Reads the nfg file produced by gamut with '-output GambitOutput' option and returns a dictionary strategy_profile -> [player 1 payoff, ..., player n payoff].
        This call should be made after calling self.gamut_cmd_call(), which is the function that produces the required .nfg file.
        :param nfg_output_file_location: the location of the .nfg file
        :return: a dictionary
        """
        f = open(f'{nfg_output_file_location}', "r")
        payoffs_from_gamut = f.readlines()[-1].split(' ')
        cur = [0] * self.num_players
        gamut_payoffs = {}
        i = 0
        for s in range(self.total_num_strategy_profiles - 1):
            gamut_payoffs[tuple(cur)] = [float(payoffs_from_gamut[i + j]) for j in range(0, self.num_players)]
            cur = GameStructure.add_one_to_profile(cur, self.num_strategies)
            i += self.num_players
        # @TODO Unfortunately, we have to account for the last profile outside the loop.
        gamut_payoffs[tuple(cur)] = [float(payoffs_from_gamut[i + j]) for j in range(0, self.num_players)]
        # pprint.pprint(gamut_payoffs)
        return gamut_payoffs

    @abstractmethod
    def complete_gamut_command(self) -> str:
        """
        @TODO Unfortunately, for all implementing methods I couldn't call gamut the appropriate way, i.e., by using subprocess.call(),
        @TODO b/c the number of actions is a list, e.g., -actions 5 3, would create a game where the first player has 5 actions and the second 3,
        @TODO but this list 5 3 gets quoted by subprocess which then makes gamut complain. I had to resort back to os.system()
        :return:
        """
        pass

    @abstractmethod
    def clone(self):
        pass

    def sample(self, strategy_profile: Tuple[int, ...], m: int):
        """
        Given a strategy profile (tuple of integers) and a number of samples (integer)
        return a list of sample m sample payoffs for each player at the profile.
        :param strategy_profile: tuple of integers
        :param m: number of sample, an integer
        :return: a list of lists, where the pth inner list is of length m containing m samples of payoff for the pth player at the profile.
        """
        assert self.is_profile_valid(strategy_profile)
        assert m > 0
        # The noise can be a dictionary, in which case the noise is at the strategy profile level. Otherwise, noise is global
        noise = self.noise[strategy_profile] if isinstance(self.noise, dict) else self.noise
        return [self.base_payoffs[strategy_profile][p] + noise.get_samples(m) for p in range(0, self.num_players)]

    @staticmethod
    def list_player_from_num(num_players: int) -> List[str]:
        """
        Given an integer, n, return a list of strings ['P1', 'P2', ..., 'Pn']
        :param num_players: an integer
        :return: a list of strings
        """
        return ['P' + str(i) for i in range(int(num_players))]

    @staticmethod
    def list_strategies_from_num(num_players: int, num_strategies: int) -> List[int]:
        return [int(num_strategies)] * int(num_players)


class RandomGames(GamutGame):
    """ Interfaces with Gamut via command line to obtain a RandomGame. """

    def __init__(self, title: str, num_players: int, num_strategies: int, max_payoff: float, min_payoff: float, noise: Noise, do_setup: bool = True, base_payoffs: Dict = None):
        GamutGame.__init__(self, title, GamutGame.list_player_from_num(num_players), GamutGame.list_strategies_from_num(num_players, num_strategies),
                           max_payoff, min_payoff, noise, do_setup, base_payoffs)

    def complete_gamut_command(self) -> str:
        list_actions_for_gamut = ' '.join([str(num_strategies_player) for num_strategies_player in self.num_strategies])
        return f"-g RandomGame -players {str(len(self.players))} -actions {list_actions_for_gamut}"

    def clone(self):
        return RandomGames(self.title, self.num_players, self.num_strategies[0], self.max_payoff, self.min_payoff, self.noise, do_setup=False, base_payoffs=self.base_payoffs)


class PrisonersDilemma(GamutGame):
    """ Interfaces with Gamut via command line to obtain a Prisoner's Dilemma game. Note that this game has 2 players with 2 strategies always. """

    def __init__(self, title: str, max_payoff: float, min_payoff: float, noise: Noise, do_setup: bool = True, base_payoffs: Dict = None):
        GamutGame.__init__(self, title, GamutGame.list_player_from_num(2), GamutGame.list_strategies_from_num(2, 2), max_payoff, min_payoff, noise, do_setup, base_payoffs)

    def complete_gamut_command(self) -> str:
        return f"-g PrisonersDilemma"

    def clone(self):
        return PrisonersDilemma(self.title, self.max_payoff, self.min_payoff, self.noise, do_setup=False, base_payoffs=self.base_payoffs)


class BattleOfTheSexes(GamutGame):
    """ Interfaces with Gamut via command line to obtain a Battle of the sexes game. Note that this game has 2 players with 2 strategies always. """

    def __init__(self, title: str, max_payoff: float, min_payoff: float, noise: Noise, do_setup: bool = True, base_payoffs: Dict = None):
        GamutGame.__init__(self, title, GamutGame.list_player_from_num(2), GamutGame.list_strategies_from_num(2, 2), max_payoff, min_payoff, noise, do_setup, base_payoffs)

    def complete_gamut_command(self) -> str:
        return f"-g BattleOfTheSexes"

    def clone(self):
        return BattleOfTheSexes(self.title, self.max_payoff, self.min_payoff, self.noise, do_setup=False, base_payoffs=self.base_payoffs)


class CongestionGame(GamutGame):
    """ Interfaces with Gamut via command line to obtain a Congestion game."""

    def __init__(self, title: str, num_players: int, num_facilities: int, max_payoff: float, min_payoff: float, noise: Noise, do_setup: bool = True, base_payoffs: Dict = None):
        num_players = int(num_players)
        num_facilities = int(num_facilities)
        assert 2 <= num_players <= 100 and 1 <= num_facilities <= 5
        self.num_facilities = num_facilities
        GamutGame.__init__(self, title, GamutGame.list_player_from_num(num_players), [(2 ** num_facilities) - 1] * num_players, max_payoff, min_payoff, noise,
                           do_setup, base_payoffs)

    def complete_gamut_command(self) -> str:
        return f"-g CongestionGame -players {self.num_players} -facilities {self.num_facilities} -func DecreasingWrapper " \
               f"-func_params [-base_func LogFunction -base_params [-alpha 1.0 -beta 1.0]] -sym_funcs 0 -random_params 1"

    def clone(self):
        return CongestionGame(self.title, self.num_players, self.num_facilities, self.max_payoff, self.min_payoff, self.noise, do_setup=False, base_payoffs=self.base_payoffs)


class TravelersDilemma(GamutGame):
    """ Interfaces with Gamut via command line to obtain a Travelers Dilemma game."""

    def __init__(self, title: str, num_players: int, num_strategies: int, max_payoff: float, min_payoff: float, noise: Noise, do_setup: bool = True, base_payoffs: Dict = None):
        assert 2 <= int(num_players) <= 100 and 1 <= int(num_strategies) <= 10
        GamutGame.__init__(self, title, GamutGame.list_player_from_num(num_players), GamutGame.list_strategies_from_num(num_players, num_strategies),
                           max_payoff, min_payoff, noise, do_setup, base_payoffs)

    def complete_gamut_command(self) -> str:
        return f"-g TravelersDilemma -players {self.num_players} -actions {self.num_strategies[0]} -random_params 1"

    def clone(self):
        return TravelersDilemma(self.title, self.num_players, self.num_strategies[0], self.max_payoff, self.min_payoff, self.noise,
                                do_setup=False, base_payoffs=self.base_payoffs)


class Chicken(GamutGame):
    """ Interfaces with Gamut via command line to obtain a Chicken game."""

    def __init__(self, title: str, max_payoff: float, min_payoff: float, noise: Noise, do_setup: bool = True, base_payoffs: Dict = None):
        GamutGame.__init__(self, title, GamutGame.list_player_from_num(2), GamutGame.list_strategies_from_num(2, 2), max_payoff, min_payoff, noise, do_setup, base_payoffs)

    def complete_gamut_command(self) -> str:
        return f"-g Chicken -random_params 1"

    def clone(self):
        return Chicken(self.title, self.max_payoff, self.min_payoff, self.noise, do_setup=False, base_payoffs=self.base_payoffs)


class MinimumEffort(GamutGame):
    """ Interfaces with Gamut via command line to obtain a Minimum Effort game."""

    def __init__(self, title: str, num_players: int, num_strategies: int, max_payoff: float, min_payoff: float, noise: Noise, do_setup: bool = True, base_payoffs: Dict = None):
        GamutGame.__init__(self, title, GamutGame.list_player_from_num(num_players), GamutGame.list_strategies_from_num(num_players, num_strategies),
                           max_payoff, min_payoff, noise, do_setup, base_payoffs)

    def complete_gamut_command(self) -> str:
        return f"-g MinimumEffortGame -players {self.num_players} -actions {self.num_strategies[0]} -random_params 1"

    def clone(self):
        return MinimumEffort(self.title, self.num_players, self.num_strategies[0], self.max_payoff, self.min_payoff, self.noise, do_setup=False, base_payoffs=self.base_payoffs)


class GrabTheDollar(GamutGame):
    """ Interfaces with Gamut via command line to obtain a Grab The Dollar game."""

    def __init__(self, title: str, num_strategies: int, max_payoff: float, min_payoff: float, noise: Noise, do_setup: bool = True, base_payoffs: Dict = None):
        GamutGame.__init__(self, title, GamutGame.list_player_from_num(2), GamutGame.list_strategies_from_num(2, num_strategies),
                           max_payoff, min_payoff, noise, do_setup, base_payoffs)

    def complete_gamut_command(self) -> str:
        return f"-g GrabTheDollar -actions {self.num_strategies[0]}"

    def clone(self):
        return GrabTheDollar(self.title, self.num_strategies[0], self.max_payoff, self.min_payoff, self.noise, do_setup=False, base_payoffs=self.base_payoffs)


class ZeroSum(GamutGame):
    """ Interfaces with Gamut via command line to obtain a random Zero sum game."""

    def __init__(self, title: str, num_strategies: int, max_payoff: float, min_payoff: float, noise: Noise, do_setup: bool = True, base_payoffs: Dict = None):
        GamutGame.__init__(self, title, GamutGame.list_player_from_num(2), GamutGame.list_strategies_from_num(2, num_strategies),
                           max_payoff, min_payoff, noise, do_setup, base_payoffs)

    def complete_gamut_command(self) -> str:
        return f"-g RandomZeroSum -actions {self.num_strategies[0]}"

    def clone(self):
        return ZeroSum(self.title, self.num_strategies[0], self.max_payoff, self.min_payoff, self.noise, do_setup=False, base_payoffs=self.base_payoffs)


class CompoundGame(GamutGame):
    """ Interfaces with Gamut via command line to obtain a random compound game."""

    def __init__(self, title: str, num_players: int, max_payoff: float, min_payoff: float, noise: Noise, do_setup: bool = True, base_payoffs: Dict = None):
        GamutGame.__init__(self, title, GamutGame.list_player_from_num(num_players), GamutGame.list_strategies_from_num(num_players, 2),
                           max_payoff, min_payoff, noise, do_setup, base_payoffs)

    def complete_gamut_command(self) -> str:
        return f"-g RandomCompoundGame -players {self.num_players}"

    def clone(self):
        return CompoundGame(self.title, self.num_players, self.max_payoff, self.min_payoff, self.noise, do_setup=False, base_payoffs=self.base_payoffs)


class BertrandOligopoly(GamutGame):
    """ Interfaces with Gamut via command line to obtain a RandomGame. """

    def __init__(self, title: str, num_players: int, num_strategies: int, max_payoff: float, min_payoff: float, noise: Noise, do_setup: bool = True, base_payoffs: Dict = None):
        GamutGame.__init__(self, title, GamutGame.list_player_from_num(num_players), GamutGame.list_strategies_from_num(num_players, num_strategies),
                           max_payoff, min_payoff, noise, do_setup, base_payoffs)

    def complete_gamut_command(self) -> str:
        list_actions_for_gamut = ' '.join([str(num_strategies_player) for num_strategies_player in self.num_strategies])
        return f"-g BertrandOligopoly -players {str(len(self.players))} -actions {list_actions_for_gamut} -random_params 1"

    def clone(self):
        return BertrandOligopoly(self.title, self.num_players, self.num_strategies[0], self.max_payoff, self.min_payoff, self.noise, do_setup=False, base_payoffs=self.base_payoffs)
