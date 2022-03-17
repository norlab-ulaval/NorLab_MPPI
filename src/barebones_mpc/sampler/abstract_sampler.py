from abc import ABCMeta, abstractmethod
from typing import Type

import numpy as np

from src.barebones_mpc.model.abstract_model import AbstractModel


class AbstractSampler(metaclass=ABCMeta):
    config: dict

    def __init__(self, model: AbstractModel, number_samples: int, input_dimension: int, sample_length: int,
                 init_state: np.ndarray):
        self.number_samples = number_samples
        self.input_dimension = input_dimension
        self.sample_length = sample_length
        self.init_state = init_state

        ERR_S = f"({self.__class__.__name__} ERROR): "
        assert isinstance(model, AbstractModel), f"{ERR_S} {model} is not and instance of AbstractModel"
        self.model = model

    @classmethod
    @abstractmethod
    def config_init(cls, model, config: dict):
        """
        Alternative initialization method via configuration dictionary
        Return an instance of AbstractNominalPathBootstrap

        Exemple
        >>>     @classmethod
        >>>     def config_init(cls, config: dict):
        >>>         from src.barebones_mpc.config_files.config_utils import import_controler_component_class
        >>>         horizon = config['hparam']['sampler_hparam']['horizon']
        >>>         time_step = config['hparam']['sampler_hparam']['steps_per_prediction']
        >>>         input_shape: tuple = config['environment']['input_space']['shape']
        >>>         cls.config = config
        >>>         instance = cls(model=import_controler_component_class(config, 'model')(),
        >>>                        number_samples=config['hparam']['sampler_hparam']['number_samples'],
        >>>                        input_dimension=len(input_shape),
        >>>                        sample_length=(int(horizon/time_step)),
        >>>                        init_state=np.zeros(config['environment']['observation_space']['shape'][0])
        >>>                        )
        >>>         return instance

        :param model:
        :param config: a dictionary of configuration
        """
        pass

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
    env: None
    config: dict

    def __init__(self, model, number_samples, input_dimension, sample_length, init_state):
        super().__init__(model, number_samples, input_dimension, sample_length, init_state)

        try:
            if self.config['environment']['type'] == 'gym':
                import gym
                self.env: gym.wrappers.time_limit.TimeLimit = gym.make(self.config['environment']['name'])
            else:
                raise NotImplementedError
        except AttributeError:
            pass

    @classmethod
    def config_init(cls, model, config: dict):
        from src.barebones_mpc.config_files.config_utils import import_controler_component_class

        horizon = config['hparam']['sampler_hparam']['horizon']
        time_step = config['hparam']['sampler_hparam']['steps_per_prediction']
        input_shape: tuple = config['environment']['input_space']['shape']
        cls.config = config

        instance = cls(model=model,
                       number_samples=config['hparam']['sampler_hparam']['number_samples'],
                       input_dimension=len(input_shape),
                       sample_length=(int(horizon/time_step)),
                       init_state=np.zeros(config['environment']['observation_space']['shape'][0])
                       )
        return instance

    def sample_inputs(self, nominal_input):
        sample = np.full((self.sample_length, self.number_samples + 1, self.input_dimension),
                         self.env.action_space.sample())
        return sample

    def sample_states(self, sample_input, init_state):
        sample = np.zeros((self.sample_length, self.number_samples + 1, self.env.observation_space.shape[0]))
        return sample
