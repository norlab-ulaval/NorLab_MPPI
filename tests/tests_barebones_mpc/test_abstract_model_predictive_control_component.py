# coding=utf-8
from copy import deepcopy
from typing import Any, List, Type, Tuple, Dict, TypeVar, Union

import numpy as np
import pytest
from src.barebones_mpc.sampler.abstract_sampler import AbstractSampler, MockSampler
from src.barebones_mpc.model.abstract_model import MockModel

from dataclasses import dataclass


@dataclass
class ConfigAMPCC:
    config: Union[dict, None]
    different_config_instance: Union[dict, None]
    init_state: Any = np.zeros(4)
    number_samples: int = 1000
    input_dimension: int = 1
    sample_length: int = int(0.75 / (1 / 20))
    # model: Type[MockModel] = MockModel()
    model: MockModel = MockModel(1, 1, 1, 1)


@pytest.fixture(scope="function")
def config_ampcc(setup_mock_config_dict_CartPole):
    return ConfigAMPCC(config=setup_mock_config_dict_CartPole, different_config_instance=None)


@pytest.fixture(scope="function")
def config_ampcc_multi(setup_mock_config_dict_CartPole):
    not_setup_mock_config_dict_CartPole = deepcopy(setup_mock_config_dict_CartPole)
    not_setup_mock_config_dict_CartPole["config_name"] = "NOT THE SAME"

    return ConfigAMPCC(
        config=setup_mock_config_dict_CartPole, different_config_instance=not_setup_mock_config_dict_CartPole
    )


class BadMockSampler(MockSampler):
    def _config_pre_init_callback(
        self, config: Dict, subclass_config: Dict, signature_values_from_config: Dict
    ) -> Dict:
        original_callback_dict = super()._config_pre_init_callback(
            self, config, subclass_config, signature_values_from_config
        )

        del original_callback_dict["sample_length"]
        return original_callback_dict


class TestAbstractModelPredictiveControlComponent:
    """ Use MockSampler as a proxy for testing the AbstractModelPredictiveControlComponent class"""

    def test_init(self, config_ampcc):
        instance = MockSampler(
            model=config_ampcc.model,
            number_samples=config_ampcc.number_samples,
            input_dimension=config_ampcc.input_dimension,
            sample_length=config_ampcc.sample_length,
            init_state=config_ampcc.init_state,
        )

        assert instance.computed_test_arbitrary_param == 6

    def test_ERR_S_msg(self, setup_mock_config_dict_CartPole, config_ampcc):
        print("\n>>> On a class: ", MockSampler.ERR_S())
        print("\n>>> On a class: ", AbstractSampler.ERR_S())

        instance = MockSampler.config_init(config=setup_mock_config_dict_CartPole, model=config_ampcc.model)

        print("\n>>> On an instance: ", instance.ERR_S())

    def test_config_init(self, setup_mock_config_dict_CartPole, config_ampcc):
        instance = MockSampler.config_init(config=setup_mock_config_dict_CartPole, model=config_ampcc.model)
        assert isinstance(instance, AbstractSampler)
        assert instance.computed_test_arbitrary_param == 9

    def test_config_init_unaccounted_parameter_check(self, setup_mock_config_dict_CartPole, config_ampcc):
        with pytest.raises(AssertionError):
            _ = BadMockSampler.config_init(config=setup_mock_config_dict_CartPole, model=config_ampcc.model)

    @pytest.mark.skip(reason="There is no class variable anymore")
    def test_class_variable(self, config_ampcc_multi):
        ms1 = MockSampler.config_init(model=config_ampcc_multi.model, config=config_ampcc_multi.config)
        ms2 = MockSampler.config_init(
            model=config_ampcc_multi.model, config=config_ampcc_multi.different_config_instance
        )
        assert ms1 is not ms2

    def test_config_file_instance_variable(self, config_ampcc_multi):
        ms1 = MockSampler.config_init(model=config_ampcc_multi.model, config=config_ampcc_multi.config)

        assert ms1._config["config_name"] == "default CartPole-v1"

        ms2 = MockSampler.config_init(
            model=config_ampcc_multi.model, config=config_ampcc_multi.different_config_instance
        )

        assert ms1 is not ms2
        assert ms2._config["config_name"] == "NOT THE SAME"
        assert ms1._config["config_name"] == "default CartPole-v1"

        # assert ms1._config is ms2._config
