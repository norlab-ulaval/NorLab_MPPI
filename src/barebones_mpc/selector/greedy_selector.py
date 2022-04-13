from typing import List, Tuple, Union

from src.barebones_mpc.selector.abstract_selector import AbstractSelector
import numpy as np


class GreedySelector(AbstractSelector):
    def select_next_input(
        self, sample_states: np.ndarray, sample_inputs: np.ndarray, sample_costs: np.ndarray
    ) -> Tuple[Union[int, float, List[Union[int, float]]], np.ndarray]:
        """ select the optimal next input and state arrays

        :param sample_states:
        :param sample_inputs:
        :param sample_costs: sample cost array
        :return new nominal input and nominal state arrays
        """
        trajectories_cumulative_cost = sample_costs.sum(axis=0)
        optimal_trajetory_id = np.argmin(trajectories_cumulative_cost)
        optimal_inputs = sample_inputs[:, optimal_trajetory_id, :]
        nominal_states = sample_states[:, optimal_trajetory_id, :]

        return optimal_inputs, nominal_states
