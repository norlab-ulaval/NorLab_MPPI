from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
from gym import wrappers as gym_wrappers
from gym import make as gym_make
from random import randint

from src.barebones_mpc.abstract_model_predictive_control_component import AbstractModelPredictiveControlComponent


class AbstractSelector(ABC, AbstractModelPredictiveControlComponent):
    def __init__(self):
        super().__init__()

    @classmethod
    def _specialized_config_key(cls) -> str:
        return "selector_hparam"

    @classmethod
    def _specialized_config_required_fields(cls) -> List[str]:
        return []

    def _config_pre__init__callback(self, config: Dict, specialized_config: Dict,
                                    init__signature_values_from_config: Dict) -> Dict:
        return {}

    def _config_post__init__callback(self, config: Dict) -> None:
        pass

    @abstractmethod
    def select_next_input(
        self, sample_states, sample_inputs, sample_costs
    ) -> Tuple[Union[int, float, List[Union[int, float]]], np.ndarray]:
        """ select the optimal next input and state arrays

        :param sample_states: sample state array
        :param sample_inputs:
        :param sample_costs: sample cost array
        :return new nominal input array and nominal state arrays
        """
        pass


class MockSelector(AbstractSelector):
    """ For testing purpose only"""

    def __init__(self):
        super().__init__()

    def _config_post__init__callback(self, config: Dict) -> None:
        super()._config_post__init__callback(config)
        try:
            if self._config["environment"]["type"] == "gym":
                self.env: gym_wrappers.time_limit.TimeLimit = gym_make(self._config["environment"]["name"])
            else:
                raise NotImplementedError
        except AttributeError:
            pass

    def select_next_input(self, sample_states, sample_inputs, sample_costs) -> Tuple[Union[int, float], np.ndarray]:

        mock_optimal_trajectory = randint(a=0, b=sample_states.shape[1]-1)

        return sample_inputs[:, mock_optimal_trajectory, :], sample_states[:, mock_optimal_trajectory, :]
