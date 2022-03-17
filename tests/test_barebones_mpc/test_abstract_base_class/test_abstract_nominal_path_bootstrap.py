# coding=utf-8

from collections import namedtuple
from typing import List, Type, Tuple, Dict, Union

import numpy as np
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
def cnp(setup_mock_config_dict_CartPole):
    return ConfigNominalPath(config=setup_mock_config_dict_CartPole)


class TestMockNominalPathBoostrap:

    def test_init(self, cnp):
        MockNominalPathBootstrap(sample_length=cnp.sample_length,
                                 input_space=cnp.discrete_input_space,
                                 config=cnp.config,
                                 arbitrary_path=None)

    def test_init_environment_FAIL(self, cnp):
        with pytest.raises(NotImplementedError):
            cnp.config['environment']['type'] = 'NOT_gym'
            MockNominalPathBootstrap(sample_length=cnp.sample_length,
                                     input_space=cnp.discrete_input_space,
                                     config=cnp.config,
                                     arbitrary_path=None)

    def test_config_init(self, setup_mock_config_dict_CartPole):
        instance = MockNominalPathBootstrap.config_init(setup_mock_config_dict_CartPole)
        assert isinstance(instance, AbstractNominalPathBootstrap)

    def test_execute(self, setup_mock_config_dict_CartPole, cnp):
        instance = MockNominalPathBootstrap.config_init(setup_mock_config_dict_CartPole)
        nominal_input, nominal_path = instance.execute()
        assert type(nominal_input) is int
        assert nominal_path.size == cnp.sample_length
