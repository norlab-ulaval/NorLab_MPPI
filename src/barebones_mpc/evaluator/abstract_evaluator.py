# coding=utf-8

from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np

from src.barebones_mpc.abstract_model_predictive_control_component import AbstractModelPredictiveControlComponent


class AbstractEvaluator(ABC, AbstractModelPredictiveControlComponent):
    def __init__(
        self, number_samples: int, input_dimension: int, sample_length: int, state_dimension: int
    ):
        super().__init__()

        self.input_dimension = input_dimension
        self.sample_length = sample_length
        self.number_samples = number_samples
        self.state_dimension = state_dimension

    @classmethod
    def _specialized_config_key(cls) -> str:
        return "evaluator_hparam"

    @classmethod
    def _specialized_config_required_fields(cls) -> List[str]:
        return []

    def _config_pre__init__callback(self, config: Dict, specialized_config: Dict,
                                    init__signature_values_from_config: Dict) -> Dict:

        try:
            input_dimension: int = config["environment"]["input_space"]["dim"]
            observation_dim: int = config["environment"]["observation_space"]["dim"]
            number_samples: int = config["hparam"]["sampler_hparam"]["number_samples"]
            horizon: int = config["hparam"]["sampler_hparam"]["horizon"]
            time_step: int = config["hparam"]["sampler_hparam"]["prediction_step"]
        except KeyError as e:
            raise KeyError(
                f"{self.NAMED_ERR()} There's required baseclass parameters missing in the config file. Make sure that "
                f"both following key exist: "
                f"`environment:input_space:dim`,`environment:observation_space:dim`, "
                f"`hparam:sampler_hparam:number_samples`, `hparam:sampler_hparam:horizon` and "
                f"`hparam:sampler_hparam:prediction_step`\n"
                f"{e}"
            ) from e

        values_from_callback = {
            "number_samples": number_samples,
            "input_dimension": input_dimension,
            "sample_length": int(horizon / time_step),
            "state_dimension": observation_dim,
        }
        return values_from_callback

    def _config_post__init__callback(self, config: Dict) -> None:
        pass

    @abstractmethod
    def compute_sample_costs(self, sample_input: np.ndarray, sample_states: np.ndarray) -> None:
        """ computes the cost related to every sample

        :param sample_input: sample input_array array
        :param sample_states: sample state array
        :return None
        """
        pass

    @abstractmethod
    def _compute_input_cost(self, input_array: np.ndarray) -> Union[float, np.ndarray]:
        """ computes a single input cost

        :param input_array: single input array
        :return input_cost: input cost
        """
        pass

    @abstractmethod
    def _compute_state_cost(self, state: np.ndarray) -> float:
        """ compute a single state cost

        :param state: single state array
        :return state_cost: state cost
        """
        pass

    @abstractmethod
    def compute_final_state_cost(self, final_state: np.ndarray) -> float:
        """ compute a final state cost

        :param final_state: final state array
        :return final_state_cost: final state cost
        """
        pass

    @abstractmethod
    def get_trajectories_cost(self):
        pass

    @abstractmethod
    def get_trajectories_cumulative_cost(self):
        pass


class MockEvaluator(AbstractEvaluator):
    """ For testing purpose only"""

    def compute_sample_costs(self, sample_input: np.ndarray, sample_states: np.ndarray) -> None:
        return None

    def _compute_input_cost(self, input_array: np.ndarray) -> float:
        return np.random.random((1,))

    def _compute_state_cost(self, state: np.ndarray) -> float:
        return np.random.random((1,))

    def compute_final_state_cost(self, final_state: np.ndarray) -> float:
        raise NotImplementedError("(NICE TO HAVE) ToDo:implement >> mock return value")  # todo

    def get_trajectories_cost(self):
        return np.random.random((self.sample_length, self.number_samples))

    def get_trajectories_cumulative_cost(self):
        return np.random.random((1, self.number_samples))
