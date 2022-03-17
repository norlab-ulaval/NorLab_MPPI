# coding=utf-8

from typing import Tuple, Union

import pytest
from src.barebones_mpc.nominal_path_bootstrap.abstract_nominal_path_bootstrap import (
    AbstractNominalPathBootstrap,
    MockNominalPathBootstrap,
    )

from dataclasses import dataclass


@dataclass
class ConfigNominalPath:
    config: Union[dict, None]
    sample_length: int = 5
    discrete_input_space: Tuple[int] = (0, 1)


@pytest.fixture(scope="function")
def config_nominal_path(setup_mock_config_dict_CartPole):
    return ConfigNominalPath(config=setup_mock_config_dict_CartPole)


class TestMockNominalPathBoostrap:

    def test_init(self, config_nominal_path):
        MockNominalPathBootstrap(sample_length=config_nominal_path.sample_length,
                                 input_shape=config_nominal_path.discrete_input_space,
                                 config=config_nominal_path.config,
                                 arbitrary_path=None)

    def test_init_environment_FAIL(self, config_nominal_path):
        with pytest.raises(NotImplementedError):
            config_nominal_path.config['environment']['type'] = 'NOT_gym'
            MockNominalPathBootstrap(sample_length=config_nominal_path.sample_length,
                                     input_shape=config_nominal_path.discrete_input_space,
                                     config=config_nominal_path.config,
                                     arbitrary_path=None)

    def test_config_init(self, setup_mock_config_dict_CartPole):
        instance = MockNominalPathBootstrap.config_init(setup_mock_config_dict_CartPole)
        assert isinstance(instance, AbstractNominalPathBootstrap)

    def test_execute(self, setup_mock_config_dict_CartPole, config_nominal_path):
        instance = MockNominalPathBootstrap.config_init(setup_mock_config_dict_CartPole)
        nominal_input, nominal_path = instance.execute()
        assert type(nominal_input) is int
        assert nominal_path.size == config_nominal_path.sample_length
