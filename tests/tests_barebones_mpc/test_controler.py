# coding=utf-8

import pytest

from src.barebones_mpc.controller.base_controler import ModelPredictiveControler


@pytest.fixture(scope="function")
def setup_mock_barebones_mpc_mock():
    config_path = "tests/tests_barebones_mpc/config_files/default_test_config_mock_CartPole-v1.yaml"
    return config_path


class TestModelPredictiveControlerMock:
    def test_MPC_controler_init(self, setup_mock_barebones_mpc_mock):
        config_path = setup_mock_barebones_mpc_mock
        ModelPredictiveControler(config_path=config_path)

    def test_MPC_controler_init_arg_config_path_exist(self, setup_mock_barebones_mpc_mock):
        config_path = "tests/tests_barebones_mpc/BROKEN_PATH/default_test_config_mock_Pendulum-v1.yaml"
        with pytest.raises(AssertionError):
            ModelPredictiveControler(config_path=config_path)

    def test_MPC_controler_init_arg_component_is_subclass(self, setup_mock_barebones_mpc_mock):
        config_path = "tests/tests_barebones_mpc/config_files/broken_test_config_mock.yaml"
        with pytest.raises(AssertionError):
            # model_cls = dict, sampler_cls = dict, evaluator_cls = dict,; selector_cls = dict,
            ModelPredictiveControler(config_path=config_path)

    def test_state_t0_IS_None(self, setup_mock_barebones_mpc_mock):
        config_path = setup_mock_barebones_mpc_mock
        mpc = ModelPredictiveControler(config_path=config_path)
        mpc.execute()

    def test_state_t0_NOT_None(self, setup_mock_barebones_mpc_mock):
        config_path = setup_mock_barebones_mpc_mock
        mpc = ModelPredictiveControler(config_path=config_path)
        mpc.config["hparam"]["controler"]["state_t0"] = [1.0, 1.0, 1.0, 1.0]
        mpc.execute()

    def test_state_t0_NOT_None_and_wrong_shape(self, setup_mock_barebones_mpc_mock):
        config_path = setup_mock_barebones_mpc_mock
        mpc = ModelPredictiveControler(config_path=config_path)
        mpc.config["hparam"]["controler"]["state_t0"] = [1.0, 1.0, 1.0]
        with pytest.raises(AssertionError):
            mpc.execute()

    def test_execute_headless(self, setup_mock_barebones_mpc_mock):
        config_path = setup_mock_barebones_mpc_mock
        mpc = ModelPredictiveControler(config_path=config_path)
        mpc.execute()

    @pytest.mark.skip(reason="ToDo: iemplement mp4 recording")
    def test_execute_record(self, setup_mock_barebones_mpc_mock):
        # config_path = setup_mock_barebones_mpc_mock
        # mpc = ModelPredictiveControler(config_path=config_path)
        # mpc.execute()

        raise NotImplementedError  # todo: implement


@pytest.fixture(scope="function")
def setup_mock_barebones_mpc_real():
    config_path = "tests/tests_barebones_mpc/config_files/default_test_config_real_CartPole-v1.yaml"
    return config_path


class TestModelPredictiveControlerReal:
    def test_MPC_controler_init(self, setup_mock_barebones_mpc_real):
        config_path = setup_mock_barebones_mpc_real
        ModelPredictiveControler(config_path=config_path)

    def test_execute_headless(self, setup_mock_barebones_mpc_real):
        config_path = setup_mock_barebones_mpc_real
        mpc = ModelPredictiveControler(config_path=config_path)
        mpc.execute()

    @pytest.mark.skip(reason="ToDo: iemplement mp4 recording")
    def test_execute_record(self, setup_mock_barebones_mpc_real):
        # config_path = setup_mock_barebones_mpc_real
        # mpc = ModelPredictiveControler(config_path=config_path)
        # mpc.execute()

        raise NotImplementedError  # todo: implement
