from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, List

from src.barebones_mpc.abstract_model_predictive_control_component import AbstractModelPredictiveControlComponent


class AbstractModel(ABC, AbstractModelPredictiveControlComponent):

    # def __init__(self, time_step, number_samples, sample_length):
    def __init__(self):
        self.state_dimension = None
        self.input_dimension = None
        self.time_step = None

    @classmethod
    def _subclass_config_key(cls) -> str:
        return 'model_hparam'

    @classmethod
    def _init_method_registred_param(cls) -> List[str]:
        return ['self', 'state_dimension', 'input_dimension', 'time_step']

    def _config_pre_init_callback(self, config: Dict, subclass_config: Dict,
                                  signature_values_from_config: Dict) -> Dict:
        # raise NotImplementedError   # todo: implement
        pass

    def _config_post_init_callback(self, config: Dict) -> None:
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
    def _predict(self, init_state, input):
        """ makes a single state prediction based on initial state and input

        :param init_state: initial state array
        :param sample_input: input array
        :return: predicted state array
        """
        pass


class MockModel(AbstractModel):
    """ For testing purpose only"""

    def predict_states(self, init_state, sample_input):
        pass

    def _predict(self, init_state, input):
        pass
