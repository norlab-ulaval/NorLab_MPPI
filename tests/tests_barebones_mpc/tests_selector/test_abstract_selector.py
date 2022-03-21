# coding=utf-8

from typing import Any, List, Type, Tuple, Dict, TypeVar, Union

import numpy as np
import pytest
from src.barebones_mpc.selector.abstract_selector import AbstractSelector, MockSelector

from dataclasses import dataclass


@dataclass
class ConfigSelector:
    config: Union[dict, None]
    number_samples: int = 1000
    sample_total_cost: np.ndarray = np.ones((1, number_samples))
    random_min: np.ndarray = np.random.randint(0, number_samples - 1)

    def __post_init__(self):
        self.sample_total_cost[0, self.random_min] = 0


@pytest.fixture(scope="function")
def config_selector(setup_mock_config_dict_CartPole):
    return ConfigSelector(config=setup_mock_config_dict_CartPole)


class TestMockSelector:
    def test_init(self, config_selector):
        instance = MockSelector()

    def test_config_init(self, setup_mock_config_dict_CartPole, config_selector):
        instance = MockSelector.config_init(config=setup_mock_config_dict_CartPole)
        assert isinstance(instance, AbstractSelector)

    def test_config_init(self, setup_mock_config_dict_CartPole, config_selector):
        instance = MockSelector.config_init(config=setup_mock_config_dict_CartPole)
        assert isinstance(instance, AbstractSelector)

    def test_select_next_input(self, setup_mock_config_dict_CartPole, config_selector):
        instance = MockSelector.config_init(config=setup_mock_config_dict_CartPole)
        nominal_input, nominal_path = instance.select_next_input(None)
        assert type(nominal_input) is int
        assert type(nominal_path) is np.ndarray
