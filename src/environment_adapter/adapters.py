# coding=utf-8
from abc import ABCMeta, abstractmethod
from typing import Union, Any, Type, Tuple, List, Dict
import gym
import numpy as np


def make_environment_adapter(config_dict) -> Any:
    environment_type = config_dict['environment']['type']

    if environment_type == 'gym':
        adapted_env = GymEnvironmentAdapter(config_dict)
    else:
        raise NotImplementedError  # todo: implement

    return adapted_env


class AbstractEnvironmentAdapter(metaclass=ABCMeta):

    def __init__(self, config_dict):
        self._config_dict = config_dict

        self._env = self._make()
        self.observation_space = self._init_observation_space()
        self.action_space = self._init_action_space()

    @abstractmethod
    def _make(self) -> Any:
        """
        Implement every step that must be executed to produce an instance of the environment
        """
        pass

    @abstractmethod
    def _init_observation_space(self) -> Type[gym.spaces.Space]:
        """
        Implement the environnement observation space.
        Must comply with `gym.spaces.Space`

        Exemple inspired from gym Pendulum-v0 observation space definition
            >>> def _init_observation_space(self):
            >>>     from gym import spaces
            >>>     max_speed = self._config_dict['observation_space']['max_speed']
            >>>     high = np.array([1., 1., max_speed], dtype=np.float32)
            >>>     observation_space = spaces.Box( low=-high,
            >>>                                     high=high,
            >>>                                     dtype=np.float32
            >>>     )
            >>>     return observation_space
        """
        pass

    @abstractmethod
    def _init_action_space(self) -> Type[gym.spaces.Space]:
        """
        Implement the environnement action space.
        Must comply with `gym.spaces.Space`

        Exemple inspired from gym Pendulum-v0 action space definition
            >>> def _init_action_space(self):
            >>>     from gym import spaces
            >>>     max_torque = self._config_dict['action_space']['max_torque']
            >>>     observation_space = spaces.Box( low=-max_torque,
            >>>                                     high=max_torque,
            >>>                                     shape=(1,),
            >>>                                     dtype=np.float32
            >>>     )
            >>>     return observation_space
        """
        pass

    @abstractmethod
    def step(self, action) -> Tuple[Union[np.ndarray, Any], Union[int, float], bool, Dict]:
        """
        Execute an action in the environment and observe the next state and the resulting reward.

        Note:   Implementing `step` in subclass is required in order to comply with the OpenAI Gym interface and make
                the usage uniform across environment type.

        ================================================================================================================

        (From OpenAI Gym)
        Run one timestep of the environment's dynamics. When end of episode is reached,
        you are responsible for calling `reset()` to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Returns:
         - observation (object): agent's observation of the current environment
         - reward (int or float) : amount of reward returned after previous action
         - done (bool): whether the episode has ended, in which case further step() calls will return undefined results
         - info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        ================================================================================================================

        :param action: the action to perform in that state at the curent time step
        :return: the resulting observation and reward consisting of a tupple (observation, reward, done, info)
        """
        pass

    @abstractmethod
    def reset(self) -> Tuple[Union[np.ndarray, List[int]], Union[int, float], bool, Dict]:
        """
        Reset the state of the environment to an initial state at timestep_0
        :return: the observation at timestep_0 (observation, reward, done, info)
        """
        pass

    @abstractmethod
    def render(self, mode: str = 'human') -> None:
        pass


class GymEnvironmentAdapter(AbstractEnvironmentAdapter):

    def __init__(self, config_dict):
        super().__init__(config_dict=config_dict)

    def _make(self) -> Type[gym.wrappers.time_limit.TimeLimit]:
        env = gym.make(self._config_dict['environment']['name'])
        return env

    def _init_observation_space(self) -> Type[gym.spaces.Space]:
        return self._env.observation_space

    def _init_action_space(self) -> Type[gym.spaces.Space]:
        return self._env.action_space

    def step(self, action) -> Tuple[Union[np.ndarray, List[int]], Union[int, float], bool, Dict]:
        return self._env.step(action)

    def reset(self) -> Tuple[Union[np.ndarray, List[int]], Union[int, float], bool, Dict]:
        return self._env.reset()

    def render(self, mode: str = 'human') -> None:
        return self._env.render(mode=mode)
