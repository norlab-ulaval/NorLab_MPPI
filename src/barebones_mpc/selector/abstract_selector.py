from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
from gym import wrappers as gym_wrappers
from gym import make as gym_make

from src.barebones_mpc.abstract_model_predictive_control_component import AbstractModelPredictiveControlComponent


class AbstractSelector(ABC, AbstractModelPredictiveControlComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _subclass_config_key(cls) -> str:
        return "selector_hparam"

    @classmethod
    def _config_file_required_field(cls) -> List[str]:
        return []

    def _config_pre_init_callback(
        self, config: Dict, subclass_config: Dict, signature_values_from_config: Dict
    ) -> Dict:
        return {}

    def _config_post_init_callback(self, config: Dict) -> None:
        pass

    @abstractmethod
    def select_next_input(self, sample_cost) -> Tuple[Union[int, float, List[Union[int, float]]], np.ndarray]:
        """ select the optimal next input and state arrays

        :param sample_cost: sample cost array
        :return new nominal input and nominal state arrays
        """
        # :param sample_state: sample state array  (legacy)
        # :param sample_cost: sample cost array  (legacy)
        pass


class MockSelector(AbstractSelector):
    """ For testing purpose only"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _config_post_init_callback(self, config: Dict) -> None:
        try:
            if self._config["environment"]["type"] == "gym":
                self.env: gym_wrappers.time_limit.TimeLimit = gym_make(self._config["environment"]["name"])
            else:
                raise NotImplementedError
        except AttributeError:
            pass

        try:
            self.input_dimension = config["environment"]["input_space"]["dim"]
            horizon: int = config["hparam"]["sampler_hparam"]["horizon"]
            time_step: int = config["hparam"]["sampler_hparam"]["steps_per_prediction"]
        except KeyError as e:
            raise KeyError(
                f"{self.ERR_S()} There's required baseclass parameters missing in the config file. Make sure that "
                f"both following key exist: "
                f"`environment:input_space:dim`, `hparam:sampler_hparam:horizon`,"
                f"`hparam:sampler_hparam:steps_per_prediction`\n"
                f"{e}"
            ) from e

        self.sample_length = int(horizon / time_step)

    def select_next_input(self, sample_cost) -> int:
        return self.env.action_space.sample(), np.array(self.sample_length)
