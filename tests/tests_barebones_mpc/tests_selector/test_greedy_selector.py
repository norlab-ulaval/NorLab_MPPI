from typing import Union

import pytest
import numpy as np
from dataclasses import dataclass

from src.barebones_mpc.selector.greedy_selector import GreedySelector

from src.barebones_mpc.selector.abstract_selector import AbstractSelector


# ... refactoring ......................................................................................................


def test_config_init(setup_mock_config_dict_CartPole):
    instance = GreedySelector.config_init(config=setup_mock_config_dict_CartPole)
    assert isinstance(instance, AbstractSelector)


def test_manual_init(arbitrary_size_manual_selector_10_5_1_2):
    instance = GreedySelector()
    assert isinstance(instance, AbstractSelector)


# ... implementation ...................................................................................................


@pytest.fixture
def greedy_selector_init_params(arbitrary_size_manual_selector_5_1000_1_2):
    number_samples = arbitrary_size_manual_selector_5_1000_1_2.number_samples
    sample_total_cost = np.arange(0, number_samples).reshape(1, number_samples)
    arbitrary_size_manual_selector_5_1000_1_2.sample_cost = sample_total_cost

    return arbitrary_size_manual_selector_5_1000_1_2


def test_greedy_selector_init():
    greedy_selector = GreedySelector()


def test_greedy_selector_select_arange(arbitrary_size_manual_selector_10_5_1_2):
    greedy_selector = GreedySelector()
    nominal_input, _ = greedy_selector.select_next_input(
        sample_states=arbitrary_size_manual_selector_10_5_1_2.sample_state,
        sample_inputs=arbitrary_size_manual_selector_10_5_1_2.sample_input,
        sample_costs=arbitrary_size_manual_selector_10_5_1_2.sample_cost,
    )
    selected_input = nominal_input[0]
    assert selected_input == 1

