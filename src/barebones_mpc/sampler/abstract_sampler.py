from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type

import numpy as np
from gym import wrappers as gym_wrappers
from gym import make as gym_make

from src.barebones_mpc.abstract_model_predictive_control_component import AbstractModelPredictiveControlComponent
from src.barebones_mpc.model.abstract_model import AbstractModel
from src.barebones_mpc.config_files.config_utils import import_controler_component_class


class AbstractSampler(ABC, AbstractModelPredictiveControlComponent):
    def __init__(
        self,
        model: Type[AbstractModel],
        number_samples: int,
        input_dimension: int,
        sample_length: int,
        init_state: np.ndarray,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.number_samples = number_samples
        self.input_dimension = input_dimension
        self.sample_length = sample_length
        self.init_state = init_state

        assert isinstance(model, AbstractModel), f"{self.ERR_S()} {model} is not and instance of AbstractModel"
        self.model = model

    @classmethod
    def _subclass_config_key(cls) -> str:
        return "sampler_hparam"

    @classmethod
    def _config_file_required_field(cls) -> List[str]:
        return ["number_samples", "horizon", "steps_per_prediction"]

    def _config_pre_init_callback(
        self, config: Dict, subclass_config: Dict, signature_values_from_config: Dict
    ) -> Dict:
        horizon: int = subclass_config["horizon"]
        time_step: int = subclass_config["steps_per_prediction"]
        values_from_callback = {
            "input_dimension": config["environment"]["input_space"]["dim"],
            "sample_length": int(horizon / time_step),
            "init_state": np.zeros(config["environment"]["observation_space"]["dim"]),
        }

        return values_from_callback

    def _config_post_init_callback(self, config: Dict) -> None:
        pass

    @classmethod
    def config_init(cls, config: Dict, model: Type[AbstractModel], *args, **kwargs):
        # kwargs.update({'model': import_controler_component_class(config, 'model')()})
        model_class = import_controler_component_class(config, "model")
        kwargs.update({"model": model_class.config_init(config=config)})
        return super().config_init(config, *args, **kwargs)

    @abstractmethod
    def sample_inputs(self, nominal_input):
        """ Sample inputs based on the nominal input array

        :param nominal_input: the nominal input array
        :return: sample input array
        """
        pass

    @abstractmethod
    def sample_states(self, sample_input, init_state):
        """ Sample states based on the sample input array through the model

        use self.model to sample states

        :param sample_input: the sampling input
        :param init_state: the initial state array
        :return: sample state array
        """
        pass


class MockSampler(AbstractSampler):
    """ For testing purpose only"""

    def __init__(
        self,
        model,
        number_samples,
        input_dimension,
        sample_length,
        init_state,
        test_arbitrary_param: Tuple[int] = (1, 2, 3,),
    ):

        super().__init__(model, number_samples, input_dimension, sample_length, init_state)

        # for unit testing  of base class AbstractModelPredictiveControlComponent in
        #   test_abstract_model_predictive_control_component.py
        self.computed_test_arbitrary_param = (np.array(list(test_arbitrary_param))).sum()

    def _config_post_init_callback(self, config: Dict) -> None:
        try:
            if self._config["environment"]["type"] == "gym":
                self.env: gym_wrappers.time_limit.TimeLimit = gym_make(self._config["environment"]["name"])
            else:
                raise NotImplementedError
        except AttributeError:
            pass

    def sample_inputs(self, nominal_input):
        sample = np.full(
            (self.sample_length, self.number_samples + 1, self.input_dimension), self.env.action_space.sample()
        )
        return sample

    def sample_states(self, sample_input, init_state):
        sample = np.zeros((self.sample_length, self.number_samples + 1, self.env.observation_space.shape[0]))
        return sample
