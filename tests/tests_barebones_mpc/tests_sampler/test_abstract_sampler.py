# coding=utf-8

from typing import Any, List, Type, Tuple, Dict, TypeVar, Union

import numpy as np
import pytest
from src.barebones_mpc.sampler.abstract_sampler import AbstractSampler, MockSampler
from src.barebones_mpc.model.abstract_model import MockModel

from dataclasses import dataclass


@dataclass
class ConfigSampler:
    config: Union[dict, None]
    init_state: Any = np.zeros(4)
    number_samples: int = 1000
    input_dimension: int = 1
    sample_length: int = int(0.75 / (1 / 20))
    model: Type[MockModel] = MockModel(1, 1, 1, 1)


@pytest.fixture(scope="function")
def config_sampler(setup_mock_config_dict_CartPole):
    return ConfigSampler(config=setup_mock_config_dict_CartPole)


class TestMockSampler:
    def test_init(self, config_sampler):
        instance = MockSampler(
            model=config_sampler.model,
            number_samples=config_sampler.number_samples,
            input_dimension=config_sampler.input_dimension,
            sample_length=config_sampler.sample_length,
            init_state=config_sampler.init_state,
        )

    def test_config_init(self, setup_mock_config_dict_CartPole, config_sampler):
        instance = MockSampler.config_init(config=setup_mock_config_dict_CartPole, model=config_sampler.model)
        assert isinstance(instance, AbstractSampler)

    def test_sample_inputs(self, setup_mock_config_dict_CartPole, config_sampler):
        instance = MockSampler.config_init(config=setup_mock_config_dict_CartPole, model=config_sampler.model)
        instance.sample_inputs(None)

    def test_sample_sample_states(self, setup_mock_config_dict_CartPole, config_sampler):
        instance = MockSampler.config_init(config=setup_mock_config_dict_CartPole, model=config_sampler.model)
        instance.sample_states(None, None)
