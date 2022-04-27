# coding=utf-8

from abc import ABC, ABCMeta, abstractmethod
from typing import TypeVar, Union, Any, Type, Tuple, List, Dict
import numpy as np
from gym import wrappers as gym_wrappers
from gym import make as gym_make

from src.barebones_mpc.abstract_model_predictive_control_component import AbstractModelPredictiveControlComponent


class AbstractNominalPath(ABC, AbstractModelPredictiveControlComponent):
    def __init__(self, sample_length, input_dimension):
        super().__init__()

        self.sample_length = sample_length
        self.input_dimension = input_dimension

    @classmethod
    def _specialized_config_key(cls) -> str:
        return "nominal_path_bootstrap"

    @classmethod
    def _specialized_config_required_fields(cls) -> List[str]:
        return []

    def _config_pre__init__callback(
        self, config: Dict, specialized_config: Dict, init__signature_values_from_config: Dict
    ) -> Dict:
        try:
            horizon = config["hparam"]["sampler_hparam"]["horizon"]
            prediction_step = config["hparam"]["sampler_hparam"]["prediction_step"]
        except KeyError as e:
            raise KeyError(
                f"{self.NAMED_ERR()} There's required baseclass parameters missing in the config file. Make sure that "
                f"both following key exist: "
                f"`hparam:sampler_hparam:horizon` and `hparam:sampler_hparam:prediction_step`\n"
                f"{e}"
            ) from e

        values_from_callback = {
            "sample_length": int(horizon / prediction_step),
            "input_dimension": config["environment"]["input_space"]["dim"],
        }
        return values_from_callback

    def _config_post__init__callback(self, config: Dict) -> None:
        pass

    @abstractmethod
    def bootstrap(self, state_t0=None) -> Any:
        """ Bootstrap the initial nominal path

        :param state_t0:
        :return initial_nominal_path
        """
        pass

    @abstractmethod
    def bootstrap_single_input(self, state_t=None) -> Any:
        """ Bootstrap a single input

        :param state_t:
        :return a single input
        """
        pass


class MockNominalPath(AbstractNominalPath):
    """ For testing purpose only"""

    def __init__(self, sample_length, input_dimension):
        super().__init__(sample_length, input_dimension)

    def _config_post__init__callback(self, config: Dict) -> None:
        super()._config_post__init__callback(config)
        try:
            if self._config["environment"]["type"] == "gym":
                self.env: gym_wrappers.time_limit.TimeLimit = gym_make(self._config["environment"]["name"])
            else:
                raise NotImplementedError
        except AttributeError:
            pass

    def bootstrap_single_input(self, state_t=None) -> Any:
        return self.env.action_space.sample()

    # def bootstrap(self, state_t0) -> Tuple[Union[int, float, np.ndarray], np.ndarray]:
    def bootstrap(self, state_t0=None) -> np.ndarray:
        initial_nominal_input = self.bootstrap_single_input()
        initial_nominal_inputs = np.full(
            shape=(self.sample_length, self.input_dimension), fill_value=initial_nominal_input
        )
        return initial_nominal_inputs
