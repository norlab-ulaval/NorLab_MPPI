from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, List

import numpy as np

from src.barebones_mpc.abstract_model_predictive_control_component import AbstractModelPredictiveControlComponent


class AbstractModel(ABC, AbstractModelPredictiveControlComponent):
    def __init__(self, time_step: int, number_samples: int, sample_length: int, state_dimension: int):
        super().__init__()
        self.time_step = time_step
        self.number_samples = number_samples
        self.sample_length = sample_length

        # self.input_dimension = None
        self.state_dimension = state_dimension

    @classmethod
    def _specialized_config_key(cls) -> str:
        return "model_hparam"

    @classmethod
    def _specialized_config_required_fields(cls) -> List[str]:
        return []

    def _config_pre__init__callback(self, config: Dict, specialized_config: Dict,
                                    init__signature_values_from_config: Dict) -> Dict:

        try:
            observation_dim: int = config["environment"]["observation_space"]["dim"]
            number_samples: int = config["hparam"]["sampler_hparam"]["number_samples"]
            horizon: int = config["hparam"]["sampler_hparam"]["horizon"]
            time_step: int = config["hparam"]["sampler_hparam"]["prediction_step"]
        except KeyError as e:
            raise KeyError(
                f"{self.NAMED_ERR()} There's required baseclass parameters missing in the config file. Make sure that "
                f"both following key exist: "
                f"`environment:observation_space:dim`, `hparam:sampler_hparam:horizon` and "
                f"`hparam:sampler_hparam:prediction_step`\n"
                f"{e}"
            ) from e

        values_from_callback = {
            "sample_length": int(horizon / time_step),
            "time_step": time_step,
            "state_dimension": observation_dim,
            "number_samples": number_samples,
        }

        return values_from_callback

    def _config_post__init__callback(self, config: Dict) -> None:
        pass

    @abstractmethod
    def predict_states(self, init_state, sample_input):
        """ predicts a sample state array based on a nominal input array

        :param init_state: initial state array
        :param sample_input: sample input array
        :return: sample state array
        """
        pass

    @abstractmethod
    def _predict(self, init_state, initial_input):
        """ makes a single state prediction based on initial state and input

        :param init_state: initial state array
        :param initial_input: input array
        :return: predicted state array
        """
        pass


class MockModel(AbstractModel):
    """ For testing purpose only """

    def predict_states(self, init_state, sample_input):
        # return np.array(
        #     [
        #         [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        #         [[0.0, 0.05, 0.0, -0.05], [0.0, 0.2, 0.0, -0.2], [0.0, 0.35, 0.0, -0.35]],
        #         [[0.0025, 0.15, -0.0025, -0.15], [0.01, 0.45, -0.01, -0.45], [0.0175, 0.75, -0.0175, -0.75]],
        #         [
        #             [0.01, 0.3000625, -0.01, -0.300125],
        #             [0.0325, 0.75025, -0.0325, -0.7505],
        #             [0.055, 1.2004375, -0.055, -1.200875],
        #         ],
        #     ]
        # )
        return np.zeros((self.sample_length + 1, self.number_samples, self.state_dimension))

    def _predict(self, init_state, initial_input):
        return np.zeros((self.state_dimension, 1))
