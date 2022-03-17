# coding=utf-8

from abc import ABCMeta, abstractmethod
from typing import TypeVar, Union, Any, Type, Tuple, List, Dict
import numpy as np


class AbstractNominalPathBootstrap(metaclass=ABCMeta):
    config: dict

    def __init__(self, sample_length, input_shape):
        self.sample_length = sample_length
        self.input_shape = input_shape

    @classmethod
    @abstractmethod
    def config_init(cls, config: dict):
        """
        Alternative initialization method via configuration dictionary
        Return an instance of AbstractNominalPathBootstrap

        Exemple
        >>>     @classmethod
        >>> def config_init(cls, config: dict):
        >>>     horizon = config['hparam']['sampler_hparam']['horizon']
        >>>     time_step = config['hparam']['sampler_hparam']['steps_per_prediction']
        >>>     cls.config = config
        >>>     instance =  cls(sample_length=int(horizon/time_step),
        >>>                     input_shape=config['environment']['input_space']['shape'],
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
    env: None

    def __init__(self, sample_length, input_shape, arbitrary_path=None, config: dict = None):
        super().__init__(sample_length, input_shape)
        self._arbitrary_path = arbitrary_path

        self.config = config

        if self.config['environment']['type'] == 'gym':
            import gym
            self.env: gym.wrappers.time_limit.TimeLimit = gym.make(self.config['environment']['name'])
        else:
            raise NotImplementedError

    @classmethod
    def config_init(cls, config: dict):
        horizon = config['hparam']['sampler_hparam']['horizon']
        time_step = config['hparam']['sampler_hparam']['steps_per_prediction']
        instance = cls(sample_length=int(horizon/time_step),
                       input_shape=config['environment']['input_space']['shape'],
                       arbitrary_path=config['hparam']['nominal_path_bootstrap'],
                       config=config
                       )
        return instance

    def execute(self) -> Tuple[int, np.ndarray]:
        initial_nominal_input = self.env.action_space.sample()
        initial_nominal_path = np.full(shape=(self.sample_length,), fill_value=initial_nominal_input)
        return initial_nominal_input, initial_nominal_path
