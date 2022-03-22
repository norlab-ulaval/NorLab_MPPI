# coding=utf-8

from typing import Union

import numpy as np
import pytest
from src.barebones_mpc.selector.abstract_selector import AbstractSelector, MockSelector


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
        config_selector.mock_trajectories()

        nominal_input, nominal_path = instance.select_next_input(
            sample_states=config_selector.sample_state,
            sample_inputs=config_selector.sample_input,
            sample_costs=config_selector.sample_cost,
        )
        assert type(nominal_input) is np.ndarray
        assert type(nominal_path) is np.ndarray
        assert nominal_path.shape == config_selector.sample_state[:, 0, :].shape
        assert nominal_input.shape == config_selector.sample_input[:, 0, :].shape
