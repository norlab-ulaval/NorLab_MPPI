# coding=utf-8
from typing import Any, Tuple, Type, Union

import numpy as np
import pytest
from dataclasses import dataclass

from src.barebones_mpc.model.abstract_model import MockModel


@dataclass
class ConfigSampler:
    config: Union[dict, None]
    init_state: Any = np.zeros(4)
    number_samples: int = 1000
    input_dimension: int = 1
    sample_length: int = int(0.75 / (1 / 20))
    input_type: str = "discrete"
    input_space: Tuple[int] = (0, 1)
    model: Type[MockModel] = MockModel(1, 1, 1, 1)


@pytest.fixture(scope="function")
def config_sampler(setup_mock_config_dict_CartPole):
    return ConfigSampler(config=setup_mock_config_dict_CartPole)
