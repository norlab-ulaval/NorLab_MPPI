# coding=utf-8
from typing import Any, Dict, List, Tuple, Type, Union

import pytest
import gym
import yaml
import numpy as np

from src.environment_adapter.adapters import AbstractEnvironmentAdapter, GymEnvironmentAdapter, make_environment_adapter


@pytest.fixture(scope="module")
def setup_mock_config_dict():
    """ Test config file specify environment as gym Pendulum-v1 """
    config_path = "tests/test_barebones_mpc/config_files/default_test_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = dict(yaml.safe_load(f))
    return config_dict


# @pytest.mark.skip(reason="tmp mute")
class TestAbstractEnvironmentAdapter:

    @pytest.fixture(scope="module")
    def setup_MockEnvironmentAdapter(self, setup_mock_config_dict):
        test_config_dict = setup_mock_config_dict


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

            def render(self, mode: str = 'human') -> None:
                pass


        return MockEnvironmentAdapter(test_config_dict)

    def test__make(self, setup_MockEnvironmentAdapter):
        assert setup_MockEnvironmentAdapter._env is True

    def test__init_observation_space(self, setup_MockEnvironmentAdapter):
        assert setup_MockEnvironmentAdapter.observation_space is True

    def test__init_action_space(self, setup_MockEnvironmentAdapter):
        assert setup_MockEnvironmentAdapter.action_space is True


class TestGymEnvironmentAdapter:

    @pytest.fixture(scope="function")
    def setup_GymEnvironmentAdapter(self, setup_mock_config_dict):
        test_config_dict = setup_mock_config_dict
        env = GymEnvironmentAdapter(test_config_dict)
        yield env
        env.close()

    def test__make(self, setup_GymEnvironmentAdapter):
        assert isinstance(setup_GymEnvironmentAdapter._env, gym.wrappers.time_limit.TimeLimit)

    def test__init_observation_space(self, setup_GymEnvironmentAdapter):
        assert isinstance(setup_GymEnvironmentAdapter.observation_space, gym.spaces.Box)

    def test__init_action_space(self, setup_GymEnvironmentAdapter):
        assert isinstance(setup_GymEnvironmentAdapter.action_space, gym.spaces.Box)

    def test_reset(self, setup_GymEnvironmentAdapter):
        obs = setup_GymEnvironmentAdapter.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (3,)

    def test_step(self, setup_GymEnvironmentAdapter):
        _ = setup_GymEnvironmentAdapter.reset()
        action = setup_GymEnvironmentAdapter.action_space.sample()
        next_obs = setup_GymEnvironmentAdapter.step(action)
        assert isinstance(next_obs[0], np.ndarray)
        assert next_obs[0].shape == (3,)
        assert isinstance(next_obs[1], float)
        assert isinstance(next_obs[2], bool)
        assert isinstance(next_obs[3], dict)

    @pytest.mark.skip(reason="mute until resolved NMO-125 ﹅→ OpenAi gym env.render()")
    def test_render(self, setup_GymEnvironmentAdapter):
        # setup_GymEnvironmentAdapter
        assert False


def test_make_environment_adapter_gym(setup_mock_config_dict):
    assert isinstance(make_environment_adapter(setup_mock_config_dict)._env, gym.wrappers.time_limit.TimeLimit)
