# coding=utf-8
from typing import Union

import numpy as np
import pytest
from dataclasses import dataclass


@dataclass
class ConfigEvaluator:
    config: Union[dict, None]
    number_samples: int = 3
    input_dimension: int = 1  # input array dimension
    prediction_step: float = 1 / 20  # 20 hz or 0.2 seconds
    horizon: int = prediction_step * 3  # s
    sample_length = int(horizon / prediction_step)
    state_dimension: int = 4  # state array dimension
    state_weights: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0])
    std_dev: np.ndarray = np.array([0.1])
    beta: int = 0
    inverse_temperature: float = 0.01
    reference_state: np.ndarray = np.zeros(4)


@pytest.fixture(scope="function")
def config_evaluator(setup_mock_config_dict_CartPole):
    return ConfigEvaluator(config=setup_mock_config_dict_CartPole)
