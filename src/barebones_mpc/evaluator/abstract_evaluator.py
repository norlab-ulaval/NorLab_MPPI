# coding=utf-8

from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, List

from src.barebones_mpc.abstract_model_predictive_control_component import AbstractModelPredictiveControlComponent


class AbstractEvaluator(ABC, AbstractModelPredictiveControlComponent):
    def __init__(
        self, number_samples: int, input_dimension: int, sample_length: int, state_dimension: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.input_dimension = input_dimension
        self.sample_length = sample_length
        self.number_samples = number_samples
        self.state_dimension = state_dimension

    @classmethod
    def _subclass_config_key(cls) -> str:
        return "evaluator_hparam"

    @classmethod
    def _config_file_required_field(cls) -> List[str]:
        return []

    def _config_pre_init_callback(
        self, config: Dict, subclass_config: Dict, signature_values_from_config: Dict
    ) -> Dict:

        try:
            input_dimension: int = config["environment"]["input_space"]["dim"]
            observation_dim: int = config["environment"]["observation_space"]["dim"]
            number_samples: int = config["hparam"]["sampler_hparam"]["number_samples"]
            horizon: int = config["hparam"]["sampler_hparam"]["horizon"]
            time_step: int = config["hparam"]["sampler_hparam"]["steps_per_prediction"]
        except KeyError as e:
            raise KeyError(
                f"{self.ERR_S()} There's required baseclass parameters missing in the config file. Make sure that "
                f"both following key exist: "
                f"`environment:input_space:dim`,`environment:observation_space:dim`, "
                f"`hparam:sampler_hparam:number_samples`, `hparam:sampler_hparam:horizon` and "
                f"`hparam:sampler_hparam:steps_per_prediction`"
            ) from e

        values_from_callback = {
            "number_samples": number_samples,
            "input_dimension": input_dimension,
            "sample_length": int(horizon / time_step),
            "state_dimension": observation_dim,
        }
        return values_from_callback

    def _config_post_init_callback(self, config: Dict) -> None:
        pass

    @abstractmethod
    def compute_sample_costs(self, sample_input, sample_states):
        """ computes the cost related to every sample

        :param sample_input: sample input array
        :param sample_state: sample state array
        :return sample cost array
        """
        pass

    @abstractmethod
    def compute_input_cost(self, input):
        """ computes a single input cost

        :param input: single input array
        :return input_cost: input cost
        """
        pass

    @abstractmethod
    def compute_state_cost(self, state):
        """ compute a single state cost

        :param state: single state array
        :return state_cost: state cost
        """
        pass

    @abstractmethod
    def compute_final_state_cost(self, final_state):
        """ compute a final state cost

        :param final_state: final state array
        :return final_state_cost: final state cost
        """
        pass


class MockEvaluator(AbstractEvaluator):
    """ For testing purpose only"""

    def compute_sample_costs(self, sample_input, sample_states):
        raise NotImplementedError("(NICE TO HAVE) ToDo:implement >> mock return value")  # todo

    def compute_input_cost(self, input):
        raise NotImplementedError("(NICE TO HAVE) ToDo:implement >> mock return value")  # todo

    def compute_state_cost(self, state):
        raise NotImplementedError("(NICE TO HAVE) ToDo:implement >> mock return value")  # todo

    def compute_final_state_cost(self, final_state):
        raise NotImplementedError("(NICE TO HAVE) ToDo:implement >> mock return value")  # todo
