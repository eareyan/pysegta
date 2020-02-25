from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import math
import itertools as it


class Bound(ABC):
    """
    A bound must implement function compute bound which takes a dictionary of arguments and returns the radius of the diameter bound.
    """

    @abstractmethod
    def compute_bound(self, arguments: Dict) -> float:
        pass


class HoeffdingBound(Bound):
    """
    Implements Hoeffding plus union bound.
    """

    def compute_bound(self, arguments: Dict) -> float:
        return arguments['c'] * math.sqrt(math.log((2.0 * arguments['estimated_game'].size_active_game()) / arguments['delta']) / (2.0 * arguments['m']))

    @staticmethod
    def number_of_samples(arguments: Dict) -> float:
        return ((arguments['c'] ** 2) / (2 * (arguments['eps'] ** 2))) * math.log((2.0 * arguments['estimated_game'].size_active_game()) / arguments['delta'])

    def __repr__(self):
        return 'HoeffdingBound'


class RademacherBound(Bound):
    """
    Implements Rademacher bound with 1-ERA.
    """

    @staticmethod
    def n_era(samples: Dict, n: int, m: int) -> float:
        """
        Compute n-ERA. This is a static method.
        :param samples: a dictionary {strategy profile: [[sample payoff player 1, ..],..,[sample payoff player n, ...]]
        :param n: number of rows of Rademacher matrix
        :param m: number of cols of Rademacher matrix, equivalently, number of samples.
        :return: n-ERA
        """
        rademacher_vars = np.random.choice([1.0, -1.0], (n, m))
        return (1.0 / n) * sum([max(it.chain.from_iterable([[abs(np.dot(var, sample) / m) for sample in samples] for _, samples in samples.items()])) for var in rademacher_vars])

    def compute_bound(self, arguments: Dict) -> float:
        one_era = RademacherBound.n_era(samples=arguments['samples'], n=1, m=arguments['m'])
        return 2.0 * one_era + 3.0 * arguments['c'] * math.sqrt(math.log(1.0 / arguments['delta']) / (2.0 * arguments['m']))

    def __repr__(self):
        return 'RademacherBound'


class BennettsBound(Bound):

    def compute_bound(self, arguments: Dict) -> float:
        # Compute the empirical wimpy variance across all samples
        v_hat = max([max(np.var(sample, axis=1, ddof=1)) for _, sample in arguments['samples'].items()])

        # The 3.0 in the log factor is a one-tail variance and a two-tail expectation bound.
        log_factor = math.log((3.0 * arguments['estimated_game'].size_active_game()) / arguments['delta'])
        return math.sqrt((2.0 * v_hat * log_factor) / arguments['m']) + arguments['c'] * ((7.0 * log_factor) / (3.0 * (arguments['m'] - 1.0)))

    def __repr__(self):
        return 'BennetsBound'


class WimpyBernsteinBound(Bound):
    def compute_bound(self, arguments: Dict):
        c = arguments['c']
        delta = arguments['delta']
        m = arguments['m']
        ldm = math.log(3.0 / delta) / (m - 1.0)

        # Compute eps_v
        v_hat = max([max(np.var(sample, axis=1, ddof=1)) for _, sample in arguments['samples'].items()])
        eps_v = c * ldm + math.sqrt((c * ldm) ** 2 + (2.0 * ldm * v_hat / (m - 1.0)))
        log_factor = math.log((3.0 * arguments['estimated_game'].size_active_game()) / delta)

        # Compute Hoeffding component
        hoeffding = c * math.sqrt(log_factor / (2.0 * m))

        # Compute Bernstein component
        ber = (c * log_factor / (3.0 * m)) + math.sqrt((((c * log_factor) / (3.0 * m)) ** 2) + (((2 * log_factor) * (v_hat + eps_v)) / m))
        return min(hoeffding, ber)

    def __repr__(self):
        return 'WimpyBernsteinBound'
