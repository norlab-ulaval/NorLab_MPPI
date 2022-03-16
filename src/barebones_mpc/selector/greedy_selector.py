from src.barebones_mpc.selector.abstract_selector import AbstractSelector
import numpy as np

class GreedySelector(AbstractSelector):
    def __init__(self, number_samples):
        self.number_samples = number_samples


    def select_next_input(self, sample_cost):
        """ select the optimal next input and state arrays

        :param sample_cost: sample cost array
        :return new nominal input and nominal state arrays
        """
        return np.argmax(sample_cost)