# coding=utf-8
from abc import ABCMeta, abstractmethod
from typing import Union, Any, Type, Tuple, List, Dict
import os
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
from pyvirtualdisplay import Display


def make_environment_adapter(config_dict) -> Any:
    environment_type = config_dict['environment']['type']

    if environment_type == 'gym':
        adapted_env = GymEnvironmentAdapter(config_dict)
    else:
        raise NotImplementedError  # todo: implement

    return adapted_env


class AbstractEnvironmentAdapter(metaclass=ABCMeta):
    _rollout_idx: int
    _record: bool
    _headless: bool
    _virtual_display: Display

    def __init__(self, config_dict: dict):
        self._config_dict = config_dict
        self._record = self._config_dict['record']
        self._headless = self._config_dict['force_headless_mode']
        if self._headless and self._record and (self._config_dict['environment']['rendering_interval'] > 0):
            self._virtual_display = Display(visible=False, size=(1400, 900))
            self._virtual_display.start()

        self._rollout_idx = 0
        self._env = self._make()
        self.observation_space = self._init_observation_space()
        self.action_space = self._init_action_space()

    @abstractmethod
    def _make(self) -> Any:
        """
        Implement every step that must be executed to produce an instance of the environment
        Executed once during init
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
        Implement the environnement input space.
        Must comply with `gym.spaces.Space`

        Exemple inspired from gym Pendulum-v0 input space definition
            >>> def _init_action_space(self):
            >>>     from gym import spaces
            >>>     max_torque = self._config_dict['input_space']['max_torque']
            >>>     observation_space = spaces.Box( low=-max_torque,
            >>>                                     high=max_torque,
            >>>                                     shape=(1,),
            >>>                                     dtype=np.float32
            >>>     )
            >>>     return observation_space
        """
        pass

    @abstractmethod
    def step(self, input) -> Tuple[Union[np.ndarray, Any], Union[int, float], bool, Dict]:
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

        :param input: the input to perform in that state at the curent time step
        :return: the resulting observation and reward consisting of a tupple (observation, reward, done, info)
        """
        pass

    @abstractmethod
    def reset(self) -> Union[np.ndarray, List[int]]:
        """
        Reset the state of the environment to an initial state at timestep_0
        :return: the observation at timestep_0
        """

        pass

    @abstractmethod
    def render(self, mode: str = 'human') -> None:
        pass

    @abstractmethod
    def _close(self) -> None:
        """
        Subclass own teardown process executed at environment clossing
        """
        pass

    def close(self) -> None:
        """
        Teardown process executed at environment clossing
        """
        self._close()
        if self._headless and self._record and (self._config_dict['environment']['rendering_interval'] > 0):
            self._virtual_display.stop()
        return None

    def __del__(self) -> None:
        self.close()
        return None


class GymEnvironmentAdapter(AbstractEnvironmentAdapter):
    _recorder: VideoRecorder

    def __init__(self, config_dict):
        """
        Adapter for gym environment. Specification are fetch from the configuration file

        ----

        # Ex. ref CartPole-v1 gym environment
            >>> environment:
            >>>   type: 'gym'
            >>>   name: 'CartPole-v1'
            >>>   rendering_interval: 1
            >>>   observation_space:
            >>>     cart_position: 4.8
            >>>     cart_velocity: Inf
            >>>     pole_angle: 0.418
            >>>     pole_angular_velocity: Inf
            >>>   input_space:
            >>>     shape: 1
            >>>     legal_actions: {0, 1}

        Space definition from gym CartPole-v1
        at https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

        ### Action Space

        The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction of the fixed force the cart is pushed with.

        | Num | Action                 |
        |-----|------------------------|
        | 0   | Push cart to the left  |
        | 1   | Push cart to the right |

        **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

        ### Observation Space

        The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

        | Num | Observation           | Min                  | Max                |
        |-----|-----------------------|----------------------|--------------------|
        | 0   | Cart Position         | -4.8                 | 4.8                |
        | 1   | Cart Velocity         | -Inf                 | Inf                |
        | 2   | Pole Angle            | ~ -0.418 rad (-24°)  | ~ 0.418 rad (24°)  |
        | 3   | Pole Angular Velocity | -Inf                 | Inf                |

        **Note:** While the ranges above denote the possible values for observation space of each element, it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
        -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates if the cart leaves the `(-2.4, 2.4)` range.
        -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

        ----

        # Ex. ref Pendulum-v1 gym environment
            >>> environment:
            >>>   type: 'gym'
            >>>   name: 'Pendulum-v1'
            >>>   rendering_interval: 1
            >>>   observation_space:
            >>>     cos_theta: 1.0
            >>>     sin_angle: 1.0
            >>>     angular_velocity: 8.0
            >>>   input_space:
            >>>     shape: 1
            >>>     max_torque: 2.0

        Space definition from gym Pendulum-v1
        at https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

        ### Action Space
        The action is the torque applied to the pendulum.

        | Num | Action | Min  | Max |
        |-----|--------|------|-----|
        | 0   | Torque | -2.0 | 2.0 |

        ### Observation Space
        The observations correspond to the x-y coordinate of the pendulum's end, and its angular velocity.

        | Num | Observation      | Min  | Max |
        |-----|------------------|------|-----|
        | 0   | x = cos(theta)   | -1.0 | 1.0 |
        | 1   | y = sin(angle)   | -1.0 | 1.0 |
        | 2   | Angular Velocity | -8.0 | 8.0 |

        ----

        :param config_dict:
        """
        super().__init__(config_dict=config_dict)

    def _make(self) -> Type[gym.wrappers.time_limit.TimeLimit]:
        config_name = self._config_dict['config_name'].replace(" ", "_")
        env = gym.make(self._config_dict['environment']['name'])
        if self._record:
            print('os.getcwd() >>>', os.getcwd())

            video_recording_path = os.path.join('experiment', config_name, 'video')
            if not os.path.exists(video_recording_path):
                os.makedirs(video_recording_path)

            recording_name = '{}_{}.mp4'.format(config_name, self._rollout_idx)
            recording_path = os.path.join(video_recording_path, recording_name)

            self._recorder = VideoRecorder(env, recording_path)

        return env

    def _init_observation_space(self) -> Type[gym.spaces.Space]:
        return self._env.observation_space

    def _init_action_space(self) -> Type[gym.spaces.Space]:
        return self._env.action_space

    def step(self, input) -> Tuple[Union[np.ndarray, List[int]], Union[int, float], bool, Dict]:
        return self._env.step(input)

    def reset(self) -> Tuple[Union[np.ndarray, List[int]], Union[int, float], bool, Dict]:
        self._rollout_idx += 1
        return self._env.reset()

    def render(self, mode: str = 'human') -> None:
        if self._record and (self._config_dict['environment']['rendering_interval'] > 0):
            self._recorder.capture_frame()
        elif not self._headless:
            self._env.render(mode=mode)
        return self._env

    def _close(self) -> None:
        if self._record:
            self._recorder.close()
        self._env.close()
        return None
