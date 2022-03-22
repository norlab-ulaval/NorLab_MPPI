# coding=utf-8
from typing import Any, Union

import numpy as np
import pytest
from dataclasses import dataclass


@dataclass
class ConfigModel:
    config: Union[dict, None]
    time_step: float = 1 / 20  # 20 hz or 0.2 seconds
    horizon: int = time_step * 3  # s
    sample_length = int(horizon / time_step)
    commanded_lon_vel: float = 1.5  # m/s
    number_samples: int = 3
    input_dimension: int = 1  # input array dimension
    state_dimension: int = 4  # state array dimension
    init_state: Any = np.zeros(4)  # initial state vector
    cart_mass: int = 1  # kg
    pendulum_mass: int = 1  # kg
    sample_input: np.ndarray = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]]).reshape((3, 3, 1))


@pytest.fixture(scope="function")
def config_model(setup_mock_config_dict_CartPole):
    return ConfigModel(config=setup_mock_config_dict_CartPole)