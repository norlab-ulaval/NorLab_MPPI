# coding=utf-8

from abc import ABC, ABCMeta, abstractmethod
from typing import TypeVar, Union, Any, Type, Tuple, List, Dict
import numpy as np
from gym import wrappers as gym_wrappers
from gym import make as gym_make

from src.barebones_mpc.abstract_model_predictive_control_component import AbstractModelPredictiveControlComponent


class AbstractNominalPathBootstrap(ABC, AbstractModelPredictiveControlComponent):
    config: dict

    def __init__(self, sample_length, input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sample_length = sample_length
        self.input_shape = input_shape

    @classmethod
    def _subclass_config_key(cls) -> str:
        return "nominal_path_bootstrap"

    @classmethod
    def _init_method_registred_param(cls) -> List[str]:
        return ['self', 'sample_length', 'input_shape']

    def _config_init_callback(self, config: Dict, subclass_config: Dict, signature_values_from_config: Dict) -> Dict:
        horizon = config['hparam']['sampler_hparam']['horizon']
        time_step = config['hparam']['sampler_hparam']['steps_per_prediction']
        values_from_callback = {
            'sample_length': int(horizon/time_step),
            'input_shape':   config['environment']['input_space']['shape'],
            }
        return values_from_callback

    @abstractmethod
    def execute(self) -> Tuple[Any, Any]:
        """
        Bootstrap the initial nominal path

        Return (initial_nominal_input, initial_nominal_path)
        """
        pass


class MockNominalPathBootstrap(AbstractNominalPathBootstrap):
    """ For testing purpose only"""
    env: None

    def __init__(self, sample_length, input_shape):
        super().__init__(sample_length, input_shape)

        try:
            if self._config['environment']['type'] == 'gym':
                self.env: gym_wrappers.time_limit.TimeLimit = gym_make(self._config['environment']['name'])
            else:
                raise NotImplementedError
        except AttributeError:
            pass

    def execute(self) -> Tuple[int, np.ndarray]:
        initial_nominal_input = self.env.action_space.sample()
        initial_nominal_path = np.full(shape=(self.sample_length,), fill_value=initial_nominal_input)
        return initial_nominal_input, initial_nominal_path
