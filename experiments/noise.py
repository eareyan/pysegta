from abc import ABC, abstractmethod
from scipy.stats import uniform
from typing import List
import numpy as np


class Noise(ABC):
    """An abstract base class to implement noise distribution used when sampling games."""

    @abstractmethod
    def get_samples(self, m: int) -> List[float]:
        """
        Return m samples of noise.
        :param m: an integer
        :return: a list of m values, each value corresponding to a sample noise value.
        """
        pass

    @abstractmethod
    def get_mean(self) -> float:
        """
        Return the mean of the noise distribution.
        :return: a float corresponding to the mean value of the distribution
        """
        pass

    @abstractmethod
    def get_variance(self):
        """
        Return the variance of the noise distribution.
        :return: a float corresponding to the variance of the distribution.
        """
        pass

    @abstractmethod
    def get_c(self, max_utility: float, min_utility: float):
        """
        Compute the range of utilities, including noise, of utilities.
        :param max_utility: the max utility of the ground-truth game (or an upper-bound)
        :param min_utility: the min utility of the ground-truth game (or a lower-bound)
        :return: a float.
        """
        pass


class UniformNoise(Noise):
    """Implements uniform noise. """

    def __init__(self, low: float, high: float):
        assert low <= high
        # We concentrate on noise that is centered at zero so that we don't have to shift the games' payoffs around.
        assert low + high == 0.0
        self.low = low
        self.high = high
        self.uniform_distribution = uniform(loc=self.low, scale=self.high - self.low)

    def get_samples(self, m: int):
        # return self.uniform_distribution.rvs(size=m) # This is much slower than using the following line!
        return np.random.uniform(self.low, self.high, m)

    def get_mean(self):
        """
        Compute mean of the uniform distribution.
        :return:
        """
        return self.uniform_distribution.mean()

    def get_variance(self):
        """
        Compute variance of the uniform distribution.
        :return:
        """
        return self.uniform_distribution.var()

    def get_c(self, max_utility: float, min_utility: float):
        """
        Range of the noise.
        :param max_utility:
        :param min_utility:
        :return:
        """
        return max_utility - min_utility + self.high - self.low

    def __repr__(self):
        return f'UniformNoise, Variance = {self.get_variance():.4f}'
