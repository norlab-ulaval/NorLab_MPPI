# coding=utf-8

import pytest
import gym
import yaml

from src.barebones_mpc.controler import ModelPredictiveControler


@pytest.fixture(scope="function")
def setup_mock_barebones_mpc():
    config_path = "tests/test_barebones_mpc/config_files/default_test_config.yaml"
    return config_path

# model_cls=MockModel, sampler_cls=MockSampler, evaluator_cls=MockEvaluator, selector_cls=MockSelector,

def test_ModelPredictiveControler_init_PASS(setup_mock_barebones_mpc):
    config_path = setup_mock_barebones_mpc
    ModelPredictiveControler(config_path=config_path)


def test_ModelPredictiveControler_init_arg_config_path_exist_FAIL(setup_mock_barebones_mpc):
    config_path = "tests/test_barebones_mpc/BROKEN_PATH/default_test_config.yaml"
    with pytest.raises(AssertionError):
        ModelPredictiveControler(config_path=config_path)

def test_ModelPredictiveControler_init_arg_component_is_subclass_FAIL(setup_mock_barebones_mpc):
    config_path = "tests/test_barebones_mpc/config_files/broken_test_config.yaml"
    with pytest.raises(AssertionError):
        # model_cls = dict, sampler_cls = dict, evaluator_cls = dict,; selector_cls = dict,
        ModelPredictiveControler(config_path=config_path)


@pytest.mark.skip(reason="Todo: implement arbitrary state_t0")
def test_state_t0_is_None_PASS(setup_mock_barebones_mpc):
    config_path = setup_mock_barebones_mpc
    ModelPredictiveControler(config_path=config_path)


def test_execute_headless_PASS(setup_mock_barebones_mpc):
    config_path = setup_mock_barebones_mpc
    mpc = ModelPredictiveControler(config_path=config_path)
    mpc.execute()


@pytest.mark.skip(reason="mute")
def test_execute_record_PASS(setup_mock_barebones_mpc):
    # config_path = setup_mock_barebones_mpc
    # mpc = ModelPredictiveControler(config_path=config_path)
    # mpc.execute()
    assert False

