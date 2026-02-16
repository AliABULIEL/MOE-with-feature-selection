from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    def __init__(self, name: str):
        """
        Initialize the estimator with a name.
        """
        self.name = name

    @abstractmethod
    def fit(self, data):
        """
        Abstract method to fit the model to data.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def pdf(self, x):
        """
        Abstract method to calculate the Probability Density Function.
        Must be implemented by subclasses.
        """

    @abstractmethod
    def cdf(self, x):
        """
        Abstract method to calculate the Cumulative Distribution Function.
        Must be implemented by subclasses.
        """
        pass
