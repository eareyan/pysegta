from experiments.noise import UniformNoise
from experiments.gamut_games import *
import pprint

max_payoff_test = 10.0
min_payoff_test = -10.0
noise_test = UniformNoise(-.5, .5)

g = RandomGames('gamut_rg_test', 3, 2, max_payoff_test, min_payoff_test, noise_test)
g.solve_nash()
pprint.pprint(g.payoffs)
pprint.pprint(g.sample((0, 0, 0), 10))

g = PrisonersDilemma('gamut_pd_test', max_payoff_test, min_payoff_test, noise_test)
g.solve_nash()
pprint.pprint(g.payoffs)
pprint.pprint(g.sample((0, 0), 10))

g = BattleOfTheSexes('gamut_bos_test', max_payoff_test, min_payoff_test, noise_test)
g.solve_nash()
pprint.pprint(g.payoffs)
pprint.pprint(g.sample((0, 0), 10))

g = CongestionGame('gamut_con_test', 2, 3, max_payoff_test, min_payoff_test, noise_test)
g.solve_nash()
pprint.pprint(g.payoffs)
pprint.pprint(g.sample((0, 0), 10))

g = TravelersDilemma('gamut_td_test', 5, 3, max_payoff_test, min_payoff_test, noise_test)
g.solve_nash()
pprint.pprint(g.payoffs)
pprint.pprint(g.sample((0, 0, 0, 0, 0), 10))

g = Chicken('gamut_ch_test', max_payoff_test, min_payoff_test, noise_test)
g.solve_nash()
pprint.pprint(g.payoffs)
pprint.pprint(g.sample((0, 0), 10))

g = MinimumEffort('gamut_me_test', 3, 5, max_payoff_test, min_payoff_test, noise_test)
g.solve_nash()
pprint.pprint(g.payoffs)
pprint.pprint(g.sample((0, 0, 0), 10))

g = GrabTheDollar('gamut_gd_test', 5, max_payoff_test, min_payoff_test, noise_test)
g.solve_nash()
pprint.pprint(g.payoffs)
pprint.pprint(g.sample((0, 0), 10))

g = ZeroSum('gamut_zs_test', 5, max_payoff_test, min_payoff_test, noise_test)
g.solve_nash()
pprint.pprint(g.payoffs)
pprint.pprint(g.sample((0, 0), 10))

g = CompoundGame('gamut_cg_test', 6, max_payoff_test, min_payoff_test, noise_test)
g.solve_nash()
pprint.pprint(g.payoffs)
pprint.pprint(g.sample((0, 0, 0, 0, 0, 0), 10))

g = BertrandOligopoly('gamut_bo_test', 6, 2, max_payoff_test, min_payoff_test, noise_test)
g.solve_nash()
pprint.pprint(g.payoffs)
pprint.pprint(g.sample((0, 0, 0, 0, 0, 0), 10))
