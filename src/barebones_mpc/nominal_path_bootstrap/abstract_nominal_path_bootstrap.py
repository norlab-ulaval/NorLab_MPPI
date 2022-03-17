# coding=utf-8

from abc import ABCMeta, abstractmethod
from typing import TypeVar, Union, Any, Type, Tuple, List, Dict
import numpy as np


class AbstractNominalPathBootstrap(metaclass=ABCMeta):

    def __init__(self, sample_length, input_space, config: dict = None):
        self._sample_length = sample_length
        self._input_space = input_space
        self._config = config

    @classmethod
    @abstractmethod
    def config_init(cls, config: dict):
        """
        Alternative initialization method via configuration dictionary
        Return an instance of AbstractNominalPathBootstrap

        Exemple
            >>> def config_init(cls, config: dict):
            >>>     horizon = config['hparam']['sampler_hparam']['horizon']
            >>>     time_step = config['hparam']['sampler_hparam']['steps_per_prediction']
            >>>     instance =  cls(sample_length=int(horizon/time_step),
            >>>                     input_space=config['input_space']['legal_actions'],
            >>>                     config=config,
            >>>                     arbitrary_path=config['nominal_path_bootstrap'],
            >>>                     )
            >>>     return instance

        :param config: a dictionary of configuration
        """
        pass

    @abstractmethod
    def execute(self) -> Tuple[Any, Any]:
        """
        Bootstrap the initial nominal path

        Return (initial_nominal_input, initial_nominal_path)
        """
        pass


class MockNominalPathBootstrap(AbstractNominalPathBootstrap):
    """ For testing purpose only"""

    def __init__(self, sample_length, input_space, config: dict = None, arbitrary_path=None):
        super().__init__(sample_length, input_space,  config)
        self._arbitrary_path = arbitrary_path

        if config['environment']['type'] == 'gym':
            import gym
            self.env: gym.wrappers.time_limit.TimeLimit = gym.make(config['environment']['name'])
        else:
            raise NotImplementedError

    @classmethod
    def config_init(cls, config: dict):
        horizon = config['hparam']['sampler_hparam']['horizon']
        time_step = config['hparam']['sampler_hparam']['steps_per_prediction']
        instance = cls(sample_length=int(horizon/time_step),
                       input_space=config['environment']['input_space']['legal_actions'],
                       arbitrary_path=config['hparam']['nominal_path_bootstrap'],
                       config=config
                       )
        return instance

    def execute(self) -> Tuple[int, np.ndarray]:
        initial_nominal_input = self.env.action_space.sample()
        initial_nominal_path = np.full(shape=(self._sample_length,), fill_value=initial_nominal_input)
        return initial_nominal_input, initial_nominal_path
