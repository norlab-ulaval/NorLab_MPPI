# coding=utf-8

from typing import TypeVar, Union, Any, Type, Tuple, List, Dict
import numpy as np
from gym import wrappers as gym_wrappers
from gym import make as gym_make

from src.barebones_mpc.nominal_path.abstract_NP import AbstractNominalPath


class RandomNominalPath(AbstractNominalPath):
    """ For testing purpose only"""

    def __init__(self, sample_length, input_dimension):
        super().__init__(sample_length, input_dimension)

    def _config_post__init__callback(self, config: Dict) -> None:
        super()._config_post__init__callback(config)
        try:
            if self._config["environment"]["type"] == "gym":
                self.env: gym_wrappers.time_limit.TimeLimit = gym_make(
                    self._config["environment"]["name"]
                )
            else:
                raise NotImplementedError
        except AttributeError:
            pass

    def bootstrap_single_input(self, state_t=None) -> gym_wrappers.time_limit.TimeLimit.action_space:
        return self.env.action_space.sample()

    def bootstrap(self, state_t0=None) -> np.ndarray:
        initial_nominal_inputs = np.random.choice(
            a=self._config["environment"]["input_space"]["legal_actions"],
            size=(self.sample_length, self.input_dimension),
        )
        return initial_nominal_inputs
