# coding=utf-8

from typing import Tuple, Union

import numpy as np
import pytest
from src.barebones_mpc.nominal_path.abstract_NP import (
    AbstractNominalPath,
    MockNominalPath,
    )

from dataclasses import dataclass


@dataclass
class ConfigNominalPath:
    config: Union[dict, None]
    sample_length: int = 15
    discrete_input_space: Tuple[int] = (0, 1)


@pytest.fixture(scope="function")
def config_nominal_path(setup_mock_config_dict_CartPole):
    return ConfigNominalPath(config=setup_mock_config_dict_CartPole)


class TestMockNominalPathBoostrap:

    def test_init(self, config_nominal_path):
        MockNominalPath(sample_length=config_nominal_path.sample_length,
                        input_dimension=config_nominal_path.discrete_input_space)

    def test_config_init(self, setup_mock_config_dict_CartPole):
        instance = MockNominalPath.config_init(config=setup_mock_config_dict_CartPole)
        assert isinstance(instance, AbstractNominalPath)

    def test_config_init_environment_FAIL(self, config_nominal_path):
        with pytest.raises(NotImplementedError):
            config_nominal_path.config['environment']['type'] = 'NOT_gym'
            MockNominalPath.config_init(config=config_nominal_path.config, )

    def test_bootstrap(self, setup_mock_config_dict_CartPole, config_nominal_path):
        instance = MockNominalPath.config_init(config=setup_mock_config_dict_CartPole)
        nominal_input = instance.bootstrap()
        assert type(nominal_input) is np.ndarray
        assert nominal_input.shape[0] == config_nominal_path.sample_length
        # assert nominal_path.size == config_nominal_path.sample_length
