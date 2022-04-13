# coding=utf-8

from typing import Any, List, Type, Tuple, Dict, TypeVar, Union

import numpy as np
import pytest
from src.barebones_mpc.evaluator.abstract_evaluator import AbstractEvaluator, MockEvaluator


class TestMockEvaluator:
    def test_init(self, config_evaluator):
        instance = MockEvaluator(
            number_samples=config_evaluator.number_samples,
            input_dimension=config_evaluator.input_dimension,
            sample_length=config_evaluator.sample_length,
            state_dimension=config_evaluator.state_dimension,
        )

    def test_config_init(self, setup_mock_config_dict_CartPole):
        instance = MockEvaluator.config_init(config=setup_mock_config_dict_CartPole)
        assert isinstance(instance, AbstractEvaluator)

    def test_compute_sample_costs(self, setup_mock_config_dict_CartPole):
        instance = MockEvaluator.config_init(config=setup_mock_config_dict_CartPole)
        instance.compute_sample_costs(None, None)

    def test_compute_input_cost(self, setup_mock_config_dict_CartPole):
        instance = MockEvaluator.config_init(config=setup_mock_config_dict_CartPole)
        instance._compute_input_cost(None)

    def test_compute_state_cost(self, setup_mock_config_dict_CartPole):
        instance = MockEvaluator.config_init(config=setup_mock_config_dict_CartPole)
        instance._compute_state_cost(None)

    @pytest.mark.skip(reason="ToDo:implement >> mock return value")
    def test_compute_final_state_cost(self, setup_mock_config_dict_CartPole):
        instance = MockEvaluator.config_init(config=setup_mock_config_dict_CartPole)
        instance.compute_final_state_cost(None)
