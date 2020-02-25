from algorithms.algorithms import global_sampling, psp
from experiments.noise import UniformNoise
from experiments.gamut_games import RandomGames
from structures.bounds import HoeffdingBound
import math

# A test game: Friday the 13th.
# Taken from https://felixmunozgarcia.files.wordpress.com/2017/08/slides_71.pdf
g = RandomGames(title='Friday the 13th with noise',
                num_players=3,
                num_strategies=2,
                max_payoff=3.0,
                min_payoff=-4.0,
                noise=UniformNoise(low=-1.0, high=1.0),
                do_setup=False)

g.base_payoffs = {(0, 0, 0): [0, 0, 0],
                  (0, 0, 1): [3, 3, -2],
                  (0, 1, 0): [-4, 1, 2],
                  (1, 0, 0): [1, -4, 2],
                  (1, 1, 0): [2, 2, -2],
                  (1, 0, 1): [-4, 1, 2],
                  (0, 1, 1): [1, -4, 2],
                  (1, 1, 1): [0, 0, 0]}

# First test global sampling, then test progressive sampling with pruning.
print(f'{g}\n')

# Set parameters of Global Sampling.
m_test = 250
delta_test = 0.1
c_test = 100
bound = HoeffdingBound()

# Run Global Sampling
eps, total_samples = global_sampling(g, bound, m_test, delta_test, c_test)
assert total_samples == m_test * g.size_active_game()
print(f'Global Sampling. ')
print(f'total_samples = {total_samples}, m_test = {m_test}, size = {g.size_active_game()}')
print(f'eps = {eps}\n')

# Set parameters of Progressive Sampling with Pruning
initial_num_samples = 250
m_schedule_test = [int(math.pow(initial_num_samples, i)) for i in range(1, 10)]
delta_schedule_test = [0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005]
target_epsilon_test = 2.0
print('Progressive Sampling with Pruning. ')
print(f'm_schedule_test = {m_schedule_test}')
print(f'delta_schedule_test = {delta_schedule_test}')
print(f'target_epsilon_test = {target_epsilon_test}')

# Run Progressive Sampling with Pruning
success = psp(g, bound, m_schedule_test, delta_schedule_test, target_epsilon_test, c_test)
pruned_game = g
print(f'PSP converged? = {success}')
print(f'pruned game = {pruned_game}')
pruned_game.pp_estimated_payoffs()
print(f'active profiles = {pruned_game.active_profiles}')

# Set the games payoffs so that we can compute Nash equilibria.
pruned_game.set_payoffs()
print(f'pruned game payoffs = {pruned_game.payoffs}')

# Compute and print Nash equilibria
list_of_nashs = pruned_game.solve_nash()
print(f'\nlist of nash equilibria = {list_of_nashs}')
