from typing import Union

import pytest
import numpy as np
from dataclasses import dataclass

from src.barebones_mpc.selector.greedy_selector import GreedySelector

from src.barebones_mpc.selector.abstract_selector import AbstractSelector


# ... refactoring ......................................................................................................
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


def test_config_init(setup_mock_config_dict_CartPole, config_selector):
    instance = GreedySelector.config_init(config=setup_mock_config_dict_CartPole)
    assert isinstance(instance, AbstractSelector)


# ... implementation ...................................................................................................


@pytest.fixture
def greedy_selector_init_params():
    number_samples = 1000
    sample_total_cost = np.arange(0, number_samples).reshape(1, number_samples)

    return number_samples, sample_total_cost


def test_greedy_selector_init(greedy_selector_init_params):
    number_samples, sample_total_cost = greedy_selector_init_params

    greedy_selector = GreedySelector()
    return None


def test_greedy_selector_select_arange(greedy_selector_init_params):
    number_samples, sample_total_cost = greedy_selector_init_params

    greedy_selector = GreedySelector(number_samples)
    selected_input_id = greedy_selector.select_next_input(sample_total_cost)
    assert selected_input_id == 0
    return None


@pytest.fixture
def greedy_selector_init_params_random_max():
    number_samples = 1000
    sample_total_cost = np.ones((1, number_samples))
    random_min = np.random.randint(0, number_samples - 1)
    sample_total_cost[0, random_min] = 0

    return number_samples, sample_total_cost, random_min


def test_greedy_selector_select_arange(greedy_selector_init_params_random_max):
    number_samples, sample_total_cost, random_min = greedy_selector_init_params_random_max

    greedy_selector = GreedySelector()
    selected_input_id = greedy_selector.select_next_input(sample_total_cost)
    assert selected_input_id == random_min
    return None
