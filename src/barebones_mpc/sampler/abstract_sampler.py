from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type, Union

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
        input_type: str,
        input_space: Union[Tuple[int], np.ndarray],
    ):
        super().__init__()

        self.number_samples = number_samples
        self.input_dimension = input_dimension
        self.sample_length = sample_length
        self.init_state = init_state
        self.input_type = input_type
        self.input_space = input_space

        assert isinstance(model, AbstractModel), f"{self.NAMED_ERR()} {model} is not and instance of AbstractModel"
        self.model = model

    @classmethod
    def _specialized_config_key(cls) -> str:
        return "sampler_hparam"

    @classmethod
    def _specialized_config_required_fields(cls) -> List[str]:
        return ["number_samples", "horizon", "steps_per_prediction"]

    def _config_pre__init__callback(self, config: Dict, specialized_config: Dict,
                                    init__signature_values_from_config: Dict) -> Dict:
        try:
            input_dimension: int = config["environment"]["input_space"]["dim"]
            observation_dim: int = config["environment"]["observation_space"]["dim"]
            input_type = config["environment"]["input_space"]["type"]
            input_space = config["environment"]["input_space"]["legal_actions"]
        except KeyError as e:
            raise KeyError(
                f"{self.NAMED_ERR()} There's required baseclass parameters missing in the config file. Make sure that "
                f"both following key exist: "
                f"`environment:input_space:dim`,`environment:observation_space:dim`\n"
                f"{e}"
            ) from e

        horizon: int = specialized_config["horizon"]
        time_step: int = specialized_config["steps_per_prediction"]
        values_from_callback = {
            "input_dimension": input_dimension,
            "init_state": np.zeros(observation_dim),
            "sample_length": int(horizon / time_step),
            "input_type": input_type,
            "input_space": input_space,
        }

        return values_from_callback

    def _config_post__init__callback(self, config: Dict) -> None:
        pass

    @classmethod
    def config_init(cls, config: Dict, model: Type[AbstractModel], *args, **kwargs):
        """
        Alternative initialization method via configuration dictionary
        Return an instance of AbstractNominalPath
        Note: This is an overloaded version of the `config_init` with the added param `model`

        :param config: a dictionary of configuration
        :param model:
        :param args: pass arbitrary argument to the baseclass init method
        :param kwargs: pass arbitrary keyword argument to the baseclass init method
        """
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
        input_type,
        input_space,
        test_arbitrary_param: Tuple[int] = (1, 2, 3,),
    ):

        super().__init__(model, number_samples, input_dimension, sample_length, init_state, input_type, input_space)

        # for unit testing  of base class AbstractModelPredictiveControlComponent in
        #   test_abstract_model_predictive_control_component.py
        self.computed_test_arbitrary_param = (np.array(list(test_arbitrary_param))).sum()

    @classmethod
    def _specialized_config_required_fields(cls) -> List[str]:
        required_field: List[str] = super()._specialized_config_required_fields()
        required_field.extend(["test_arbitrary_param"])
        return required_field

    def _config_post__init__callback(self, config: Dict) -> None:
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
