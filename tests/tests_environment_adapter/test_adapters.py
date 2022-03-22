# coding=utf-8
from typing import Any, Dict, List, Tuple, Type, Union, TypeVar

import os
import pytest
import gym
import numpy as np

from src.environment_adapter.adapters import AbstractEnvironmentAdapter, GymEnvironmentAdapter, make_environment_adapter

EA = TypeVar("EA", AbstractEnvironmentAdapter, GymEnvironmentAdapter)


# ::: AbstractEnvironmentAdapter :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# @pytest.mark.skip(reason="tmp mute")
class TestAbstractEnvironmentAdapter:
    @pytest.fixture(scope="function")
    def setup_MockEnvironmentAdapter(self, setup_mock_config_dict_CartPole) -> EA:
        test_config_dict = setup_mock_config_dict_CartPole

        class MockEnvironmentAdapter(AbstractEnvironmentAdapter):
            def _init_observation_space(self) -> Type[gym.spaces.Space]:
                return True

            def _init_action_space(self) -> Type[gym.spaces.Space]:
                return True

            def _make(self) -> Any:
                return True

            def step(self, action) -> Tuple[Union[np.ndarray, Any], Union[int, float], bool, Dict]:
                pass

            def reset(self) -> Union[np.ndarray, List[int]]:
                pass

            def render(self, mode: str = "human") -> None:
                pass

            def _close(self) -> None:
                pass

        return MockEnvironmentAdapter(test_config_dict)

    def test__make(self, setup_MockEnvironmentAdapter):
        assert setup_MockEnvironmentAdapter._env is True

    def test__init_observation_space(self, setup_MockEnvironmentAdapter):
        assert setup_MockEnvironmentAdapter.observation_space is True

    def test__init_action_space(self, setup_MockEnvironmentAdapter):
        assert setup_MockEnvironmentAdapter.action_space is True


# ::: Gym rendering virtual display ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


def test_gym_headless_rendering_classic_control(setup_virtual_display):
    # from pyvirtualdisplay import Display
    # virtual_display = Display(visible=False, size=(1400, 900))
    # virtual_display.start()
    env = gym.make("Pendulum-v1")
    env.reset()
    env.render("human")
    env.close()


def test_gym_headless_rendering_classic_control_rgb(setup_virtual_display):
    env = gym.make("Pendulum-v1")
    env.reset()
    output = env.render("rgb_array")
    env.close()


def test_gym_headless_rendering_box2d(setup_virtual_display):
    env = gym.make("LunarLanderContinuous-v2")
    env.reset()
    env.render()
    env.close()


# ::: GymEnvironmentAdapter ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class TestGymEnvironmentAdapter:
    @pytest.fixture(scope="function")
    def gym_env_adapter_fixture(self, setup_mock_config_dict_CartPole) -> EA:
        test_config_dict = setup_mock_config_dict_CartPole
        env = GymEnvironmentAdapter(test_config_dict)
        yield env
        env.close()

    @pytest.fixture(scope="function")
    def gym_env_adapter_fixture_record_true(self, setup_mock_config_dict_CartPole):
        test_config_dict = setup_mock_config_dict_CartPole
        test_config_dict["record"] = True
        test_config_dict["hparam"]["experimental_hparam"]["experimental_window"] = 150
        env = GymEnvironmentAdapter(test_config_dict)
        yield env
        env.close()

    def test__make(self, gym_env_adapter_fixture):
        assert isinstance(gym_env_adapter_fixture._env, gym.wrappers.time_limit.TimeLimit)

    def test__init_observation_space(self, gym_env_adapter_fixture):
        assert isinstance(gym_env_adapter_fixture.observation_space, gym.spaces.Box)

    def test__init_action_space(self, gym_env_adapter_fixture):
        assert isinstance(gym_env_adapter_fixture.action_space, gym.spaces.Discrete)

    def test_reset(self, gym_env_adapter_fixture):
        obs = gym_env_adapter_fixture.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == gym.make("CartPole-v1").observation_space.shape

    def test_step(self, gym_env_adapter_fixture):
        _ = gym_env_adapter_fixture.reset()
        action = gym_env_adapter_fixture.action_space.sample()
        next_obs = gym_env_adapter_fixture.step(action)
        assert isinstance(next_obs[0], np.ndarray)
        assert next_obs[0].shape == gym.make("CartPole-v1").observation_space.shape
        assert isinstance(next_obs[1], float)
        assert isinstance(next_obs[2], bool)
        assert isinstance(next_obs[3], dict)

    def test_render_NO_record(self, gym_env_adapter_fixture):
        gym_env_adapter_fixture.reset()
        gym_env_adapter_fixture.render()
        gym_env_adapter_fixture.close()

    def test_render_and_record(self, gym_env_adapter_fixture_record_true):
        assert gym_env_adapter_fixture_record_true._record is True
        video_dir = os.path.join("experiment", "default", "video")
        assert os.path.isdir(video_dir), "_make() did not create the video recording directory"

        gym_env_adapter_fixture_record_true.reset()
        for global_step in range(1):
            gym_env_adapter_fixture_record_true.render()
            action = gym_env_adapter_fixture_record_true.action_space.sample()
            gym_env_adapter_fixture_record_true.step(action)

        # print("os.getcwd() >>>", os.getcwd())
        video_recording_path = gym_env_adapter_fixture_record_true._recorder.path
        gym_env_adapter_fixture_record_true.close()

        assert os.path.exists(video_recording_path)
        assert os.path.isfile(video_recording_path)
        print(os.path.abspath(video_recording_path))

        # ... Remove recorded video ....................................................................................
        os.remove(video_recording_path)
        json_file_path = f"{os.path.splitext(video_recording_path)[0]}.meta.json"
        assert os.path.exists(json_file_path)
        os.remove(json_file_path)


def test_make_environment_adapter_gym(setup_mock_config_dict_CartPole):
    assert isinstance(make_environment_adapter(setup_mock_config_dict_CartPole)._env, gym.wrappers.time_limit.TimeLimit)