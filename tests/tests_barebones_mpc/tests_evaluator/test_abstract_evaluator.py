# coding=utf-8

from typing import Any, List, Type, Tuple, Dict, TypeVar, Union

import numpy as np
import pytest
from src.barebones_mpc.evaluator.abstract_evaluator import AbstractEvaluator, MockEvaluator

from dataclasses import dataclass


@dataclass
class ConfigEvaluator:
    config: Union[dict, None]
    number_samples: int = 3
    input_dimension: int = 1  # input array dimension
    time_step: float = 1 / 20  # 20 hz or 0.2 seconds
    horizon: int = time_step * 3  # s
    sample_length = int(horizon / time_step)
    state_dimension: int = 4  # state array dimension
    state_weights: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0])
    std_dev: np.ndarray = np.array([0.1])
    beta: int = 0
    inverse_temperature: float = 0.01
    reference_state: np.ndarray = np.zeros(4)


@pytest.fixture(scope="function")
def config_evaluator(setup_mock_config_dict_CartPole):
    return ConfigEvaluator(config=setup_mock_config_dict_CartPole)


class TestMockEvaluator:
    def test_init(self, config_evaluator):
        instance = MockEvaluator(
            number_samples=config_evaluator.number_samples,
            input_dimension=config_evaluator.input_dimension,
            sample_length=config_evaluator.sample_length,
            state_dimension=config_evaluator.state_dimension,
        )

    def test_config_init(self, setup_mock_config_dict_CartPole, config_evaluator):
        instance = MockEvaluator.config_init(config=setup_mock_config_dict_CartPole)
        assert isinstance(instance, AbstractEvaluator)

    def test_compute_sample_costs(self, setup_mock_config_dict_CartPole, config_evaluator):
        instance = MockEvaluator.config_init(config=setup_mock_config_dict_CartPole)
        instance.compute_sample_costs(None, None)

    def test_compute_input_cost(self, setup_mock_config_dict_CartPole, config_evaluator):
        instance = MockEvaluator.config_init(config=setup_mock_config_dict_CartPole)
        instance.compute_input_cost(None)

    def test_compute_state_cost(self, setup_mock_config_dict_CartPole, config_evaluator):
        instance = MockEvaluator.config_init(config=setup_mock_config_dict_CartPole)
        instance.compute_state_cost(None)

    def test_compute_final_state_cost(self, setup_mock_config_dict_CartPole, config_evaluator):
        instance = MockEvaluator.config_init(config=setup_mock_config_dict_CartPole)
        instance.compute_final_state_cost(None)
