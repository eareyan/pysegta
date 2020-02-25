from experiments.gamut_games import *
from prettytable import PrettyTable
from algorithms.algorithms import global_sampling, psp
from structures.gamestructure import path_to_nfg_files
from experiments.noise import UniformNoise
from structures.bounds import HoeffdingBound
import pandas as pd
import ast
import time


def get_rg_ground_truth_game(params: Dict, game_params: Dict):
    return RandomGames(title=game_params['title'],
                       num_players=params['num_players'],
                       num_strategies=params['num_strategies'],
                       max_payoff=params['max_payoff'],
                       min_payoff=params['min_payoff'],
                       noise=game_params['noise'])


def get_pd_ground_truth_game(params: Dict, game_params: Dict):
    return PrisonersDilemma(title=game_params['title'],
                            max_payoff=params['max_payoff'],
                            min_payoff=params['min_payoff'],
                            noise=game_params['noise'])


def get_bs_ground_truth_game(params: Dict, game_params: Dict):
    return BattleOfTheSexes(title=game_params['title'],
                            max_payoff=params['max_payoff'],
                            min_payoff=params['min_payoff'],
                            noise=game_params['noise'])


def get_cn_ground_truth_game(params: Dict, game_params: Dict):
    return CongestionGame(title=game_params['title'],
                          num_players=params['num_players'],
                          num_facilities=params['num_facilities'],
                          max_payoff=params['max_payoff'],
                          min_payoff=params['min_payoff'],
                          noise=game_params['noise'])


def get_td_ground_truth_game(params: Dict, game_params: Dict):
    return TravelersDilemma(title=game_params['title'],
                            num_players=params['num_players'],
                            num_strategies=params['num_strategies'],
                            max_payoff=params['max_payoff'],
                            min_payoff=params['min_payoff'],
                            noise=game_params['noise'])


def get_ch_ground_truth_game(params: Dict, game_params: Dict):
    return Chicken(title=game_params['title'],
                   max_payoff=params['max_payoff'],
                   min_payoff=params['min_payoff'],
                   noise=game_params['noise'])


def get_me_ground_truth_game(params: Dict, game_params: Dict):
    return MinimumEffort(title=game_params['title'],
                         num_players=params['num_players'],
                         num_strategies=params['num_strategies'],
                         max_payoff=params['max_payoff'],
                         min_payoff=params['min_payoff'],
                         noise=game_params['noise'])


def get_gd_ground_truth_game(params: Dict, game_params: Dict):
    return GrabTheDollar(title=game_params['title'],
                         num_strategies=params['num_strategies'],
                         max_payoff=params['max_payoff'],
                         min_payoff=params['min_payoff'],
                         noise=game_params['noise'])


def get_zs_ground_truth_game(params: Dict, game_params: Dict):
    return ZeroSum(title=game_params['title'],
                   num_strategies=params['num_strategies'],
                   max_payoff=params['max_payoff'],
                   min_payoff=params['min_payoff'],
                   noise=game_params['noise'])


def get_cg_ground_truth_game(params: Dict, game_params: Dict):
    return CompoundGame(title=game_params['title'],
                        num_players=params['num_players'],
                        max_payoff=params['max_payoff'],
                        min_payoff=params['min_payoff'],
                        noise=game_params['noise'])


class Experiment(ABC):
    # Declare the type of games we are allowed to experiment with.
    game_generators_dict = {'rg': get_rg_ground_truth_game,
                            'pd': get_pd_ground_truth_game,
                            'bs': get_bs_ground_truth_game,
                            'cn': get_cn_ground_truth_game,
                            'td': get_td_ground_truth_game,
                            'ch': get_ch_ground_truth_game,
                            'me': get_me_ground_truth_game,
                            'gd': get_gd_ground_truth_game,
                            'zs': get_zs_ground_truth_game,
                            'cg': get_cg_ground_truth_game}

    def __init__(self, params: Dict):
        self.params = params
        self.gt_generator = Experiment.game_generators_dict[self.params['ground_truth_game_generator']]
        Experiment.generate_params_prettytable(params=self.params, meta_file_location=self.params['result_file_location'] + '.meta')

    @abstractmethod
    def run_experiment(self):
        pass

    @staticmethod
    def generate_params_prettytable(params: Dict, meta_file_location: str) -> None:
        """
        Generate a pretty table with the parameters of an experiment, print it and save it to a file.
        :param params: a list of tuples (param, value)
        :param meta_file_location: the location of the file where the pretty table of parameters will be stored
        :return:
        """
        #
        t = PrettyTable()
        t.field_names = ["Param", "Value"]
        for param, value in params.items():
            t.add_row([param, str(value)])
        print(t)

        # Save meta info file so we know what parameters were used to run the experiment.
        with open(meta_file_location, 'w+') as meta_file:
            meta_file.write(str(t))


class GSExperiments(Experiment):

    def run_experiment(self):
        # List for results
        results = []
        # Draw some number of ground-truth games.
        for i in range(0, self.params['num_games']):
            print(f'Game #{i}')
            # Test different noise models.
            for j, noise in enumerate(self.params['noise_models']):
                print(f'Noise #{j}', end='\t ')
                game = self.gt_generator(self.params,
                                         {'title': 'expt_gs_game_' + self.params['ground_truth_game_generator'] + '_' + self.params['experiment_name'], 'noise': noise})
                c = noise.get_c(self.params['max_payoff'], self.params['min_payoff'])
                # For fix noise model and ground-truth game, perform multiple trials defined as runs of GS.
                for t in range(0, self.params['num_trials']):
                    if t % 10 == 0:
                        print(t, end='\t')
                        df = pd.DataFrame(results, columns=['game', 'variance', 'bound', 'm', 'eps'])
                        df.to_csv(self.params['result_file_location'], index=False)
                    for m in self.params['m_test']:
                        # Run GS for each type of bound.
                        for bound in self.params['bounds']:
                            g = game.clone()
                            epsilon_gs, total_num_samples_gs = global_sampling(estimated_game=g, bound=bound, m=m, delta=self.params['delta'], c=c)
                            # Collect results in the form (game index, variance of the noise model, name of bound, number of samples, epsilon).
                            results += [[i, noise.get_variance(), str(bound)[0], m, epsilon_gs]]
                print('')

        # Convert results to DataFrame and save to a csv file
        df = pd.DataFrame(results, columns=['game', 'variance', 'bound', 'm', 'eps'])
        df.to_csv(self.params['result_file_location'], index=False)


class RegretExperiments(Experiment):
    def run_experiment(self):
        # List for results
        results = []

        # Draw some number of ground-truth games.
        for i in range(self.params['num_games']):
            print(f'Game #{i}')
            ground_truth_game = None
            list_of_nashs = []
            while len(list_of_nashs) == 0:
                # Create the ground truth game
                ground_truth_game = self.gt_generator(self.params,
                                                      {'title': 'expt_regret_game_' + self.params['ground_truth_game_generator'] + '_' + self.params['experiment_name'],
                                                       'noise': None})
                # Compute the Nash  of the ground-truth game by calling gambit and read it back.
                ground_truth_game.solve_nash()
                with open(path_to_nfg_files + 'expt_regret_game_' + self.params['ground_truth_game_generator'] + '_' + self.params['experiment_name'] + '_sol', 'r') as sol_file:
                    list_of_nashs = ast.literal_eval(sol_file.read())
                print(f'The game has {len(list_of_nashs)} many nashs')
                for nash in list_of_nashs:
                    print(f'\t {nash}')

            # Test different noise models.
            for j, noise in enumerate(self.params['noise_models']):
                print(f'Noise #{j}', end='\t ')

                # Construct the game which we are going to estimate.
                ground_truth_game.noise = noise
                estimated_game = ground_truth_game.clone()
                c = noise.get_c(self.params['max_payoff'], self.params['min_payoff'])

                # Start Experiment
                for t in range(0, self.params['num_trials']):
                    if t % 10 == 0:
                        print(t, end='\t')
                        # Convert results to DataFrame and save to a csv file
                        df = pd.DataFrame(results, columns=['game', 'variance', 'bound', 'm', 'eps', 'num_nash', 'max_regret'])
                        df.to_csv(self.params['result_file_location'], index=False)
                    for m in self.params['m_test']:
                        for bound in self.params['bounds']:
                            g = estimated_game.clone()
                            epsilon_gs, total_num_samples_gs = global_sampling(estimated_game=g, bound=bound, m=m, delta=self.params['delta'], c=c)
                            g.set_payoffs()
                            regrets = [g.regret(p, nash) for p in range(g.num_players) for nash in list_of_nashs]
                            max_regret = max(regrets)
                            # print(f'\t 2*eps = {2.0 * epsilon_gs:.4f} ', "\t".join(f'{regret_p:.4f}' for regret_p in regrets), f'max_regret = {max_regret:.4f}')
                            results += [[i, noise.get_variance(), str(bound)[0], m, epsilon_gs, len(list_of_nashs), max_regret]]
                print('')
        # Convert results to DataFrame and save to a csv file
        df = pd.DataFrame(results, columns=['game', 'variance', 'bound', 'm', 'eps', 'num_nash', 'max_regret'])
        df.to_csv(self.params['result_file_location'], index=False)


class PSPExperiments(Experiment):
    def run_experiment(self):
        # List for results
        results = []
        # Draw some number of ground-truth games.
        for i in range(0, self.params['num_games']):
            print(f'Game #{i}')
            # Test different noise models.
            for j, noise in enumerate(self.params['noise_models']):
                print(f'Noise #{j}', end='\t ')
                game = self.gt_generator(self.params, {'title': 'exp_psp_game_' + self.params['ground_truth_game_generator'] + '_' + self.params['experiment_name'],
                                                       'noise': noise})
                c = noise.get_c(self.params['max_payoff'], self.params['min_payoff'])
                # For fix noise model and ground-truth game, perform multiple trials defined as runs of GS.
                for t in range(0, self.params['num_trials']):
                    if t % 10 == 0:
                        print(t, end='\t')
                        # Convert results to DataFrame and save to a csv file
                        df = pd.DataFrame(results, columns=['game', 'algo', 'variance', 'bound', 'm', 'eps', 'num_pruned', 'success'])
                        df.to_csv(self.params['result_file_location'], index=False)
                    for m in self.params['m_test']:
                        # Run GS for each type of bound.
                        for bound in self.params['bounds']:
                            # First, run GS
                            g = game.clone()
                            epsilon_gs, total_num_samples_gs = global_sampling(estimated_game=g, bound=bound, m=m, delta=self.params['delta'], c=c)
                            # Collect gs results
                            results += [[i, 'gs', noise.get_variance(), str(bound)[0], total_num_samples_gs, epsilon_gs, -1, True]]

                            # Second, run PSP with epsilon given by GS and a schedule that ends in the number of samples used by GS.
                            g = game.clone()
                            psp_success, total_num_samples, total_num_profiles_pruned, psp_epsilon = psp(estimated_game=g,
                                                                                                         bound=bound,
                                                                                                         # m_schedule=[int(m / 2 ** (3 - i)) for i in range(1, 6)], # Old Schedule.
                                                                                                         m_schedule=[int((m / 4) * 2 ** i) for i in range(4)],
                                                                                                         delta_schedule=[self.params['delta'] / 4.0] * 4,
                                                                                                         # target_epsilon=epsilon_gs, # Old target epsilon.
                                                                                                         target_epsilon=0.0,
                                                                                                         c=c)
                            # Collect pss results
                            results += [[i, 'psp', noise.get_variance(), str(bound)[0], total_num_samples, psp_epsilon, total_num_profiles_pruned, psp_success]]
                print('')

        # Convert results to DataFrame and save to a csv file
        df = pd.DataFrame(results, columns=['game', 'algo', 'variance', 'bound', 'm', 'eps', 'num_pruned', 'success'])
        df.to_csv(self.params['result_file_location'], index=False)


class PSPExperimentsPart2(Experiment):
    def run_experiment(self):
        # List for results
        results = []
        noise = UniformNoise(low=-.5, high=.5)
        game_index = 0
        t0 = time.time()
        for num_actions in self.params['num_actions']:
            print(f'num_strategies #{num_actions}')
            # self.params['num_strategies'] = num_actions
            self.params['num_facilities'] = num_actions
            # Draw some number of ground-truth games.
            for _ in range(0, self.params['num_games']):
                print(f'Game #{game_index}')
                game = self.gt_generator(self.params, {'title': 'exp_psp_part2_game_' + self.params['ground_truth_game_generator'] + '_' + self.params['experiment_name'],
                                                       'noise': noise})
                c = noise.get_c(self.params['max_payoff'], self.params['min_payoff'])
                if game_index % 10 == 0:
                    print(f'Saving..., time so far = {time.time() - t0:.4f}')
                    # Convert results to DataFrame and save to a csv file
                    df = pd.DataFrame(results, columns=['game', 'num_strategies', 'algo', 'variance', 'bound', 'm', 'eps_index', 'eps', 'num_pruned'])
                    df.to_csv(self.params['result_file_location'], index=False)
                for j, eps in enumerate(self.params['eps']):
                    m = int(HoeffdingBound.number_of_samples({'c': c, 'delta': self.params['delta'], 'estimated_game': game, 'eps': eps}))
                    for bound in self.params['bounds']:
                        # First, run GS
                        g = game.clone()
                        epsilon_gs, total_num_samples_gs = global_sampling(estimated_game=g, bound=bound, m=m, delta=self.params['delta'], c=c)

                        # Collect gs results
                        results += [[game_index, num_actions, 'gs', noise.get_variance(), str(bound)[0], total_num_samples_gs, j, epsilon_gs, -1]]

                        # Second, run PSP with epsilon given by GS and a schedule that ends in the number of samples used by GS.
                        g = game.clone()
                        psp_success, total_num_samples, total_num_profiles_pruned, psp_epsilon = psp(estimated_game=g,
                                                                                                     bound=bound,
                                                                                                     m_schedule=[int((m / 4) * 2 ** i) for i in range(4)],
                                                                                                     delta_schedule=[self.params['delta'] / 4.0] * 4,
                                                                                                     target_epsilon=0.0,
                                                                                                     c=c)
                        # Collect pss results
                        results += [[game_index, num_actions, 'psp', noise.get_variance(), str(bound)[0], total_num_samples, j, psp_epsilon, total_num_profiles_pruned]]
                game_index += 1
            print('')

        # Convert results to DataFrame and save to a csv file
        print(f'Saving..., time so far = {time.time() - t0:.4f}')
        df = pd.DataFrame(results, columns=['game', 'num_strategies', 'algo', 'variance', 'bound', 'm', 'eps_index', 'eps', 'num_pruned'])
        df.to_csv(self.params['result_file_location'], index=False)
