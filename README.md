# pySEGTA
Algorithms to approximately learn normal-form games from data. 

This library implements methods to sample arbitrary games and to produce approximations that approximately preserve equilibria.

The library interfaces with GAMUT (http://gamut.stanford.edu/) and Gambit(http://www.gambit-project.org/) to both sample games and solve for their equilibria.

For a quick start, first look at config/config.py and set your paths appropriately.
For an example use of the library, look at files tests/gamut_game_tests.py and tests/algorithms_tests.py 
