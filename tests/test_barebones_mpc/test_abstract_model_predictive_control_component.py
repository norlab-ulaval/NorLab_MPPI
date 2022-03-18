# coding=utf-8

from typing import Any, List, Type, Tuple, Dict, TypeVar, Union

import numpy as np
import pytest
from src.barebones_mpc.sampler.abstract_sampler import AbstractSampler, MockSampler
from src.barebones_mpc.model.abstract_model import MockModel

from dataclasses import dataclass


@dataclass
class ConfigAMPCC:
    config: Union[dict, None]
    init_state: Any = np.zeros(4)
    number_samples: int = 1000
    input_dimension: int = 1
    sample_length: int = int(0.75/(1/20))
    model: MockModel = MockModel()


@pytest.fixture(scope="function")
def config_ampcc(setup_mock_config_dict_CartPole):
    return ConfigAMPCC(config=setup_mock_config_dict_CartPole)


class BadMockSampler(MockSampler):

    def _config_init_callback(self, config: Dict, subclass_config: Dict, signature_values_from_config: Dict) -> Dict:
        callback = super()._config_init_callback(self, config, subclass_config, signature_values_from_config)

        del callback['sample_length']
        return callback


class TestAbstractModelPredictiveControlComponent:
    """ Use MockSampler as a proxy for testing the AbstractModelPredictiveControlComponent class"""

    def test_init(self, config_ampcc):
        instance = MockSampler(model=config_ampcc.model,
                               number_samples=config_ampcc.number_samples,
                               input_dimension=config_ampcc.input_dimension,
                               sample_length=config_ampcc.sample_length,
                               init_state=config_ampcc.init_state,
                               )

        assert instance.computed_test_arbitrary_param == 6

    def test_config_init(self, setup_mock_config_dict_CartPole, config_ampcc):
        instance = MockSampler.config_init(config=setup_mock_config_dict_CartPole,
                                           model=config_ampcc.model)
        assert isinstance(instance, AbstractSampler)
        assert instance.computed_test_arbitrary_param == 9

    def test_config_init_unaccounted_parameter_check(self, setup_mock_config_dict_CartPole, config_ampcc):
        with pytest.raises(AssertionError):
            _ = BadMockSampler.config_init(config=setup_mock_config_dict_CartPole,
                                           model=config_ampcc.model)
