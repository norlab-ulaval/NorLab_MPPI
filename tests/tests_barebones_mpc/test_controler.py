# coding=utf-8

import pytest

from src.barebones_mpc.controller.base_controler import ModelPredictiveControler, TrajectoryCollector


@pytest.fixture(scope="function")
def path_to_config_with_MOCK_subclass():
    config_path = "tests/tests_barebones_mpc/config_files/default_test_config_mock_CartPole-v1.yaml"
    return config_path


class TestModelPredictiveControlerMock:
    def test_MPC_controler_init(self, path_to_config_with_MOCK_subclass):
        ModelPredictiveControler(config_path=path_to_config_with_MOCK_subclass)

    def test_MPC_controler_init_arg_config_path_exist(self):
        config_path = "tests/tests_barebones_mpc/BROKEN_PATH/default_test_config_mock_Pendulum-v1.yaml"
        with pytest.raises(AssertionError):
            ModelPredictiveControler(config_path=config_path)

    def test_MPC_controler_init_arg_component_is_subclass(self):
        config_path = "tests/tests_barebones_mpc/config_files/broken_test_config_mock.yaml"
        with pytest.raises(AssertionError):
            # model_cls = dict, sampler_cls = dict, evaluator_cls = dict,; selector_cls = dict,
            ModelPredictiveControler(config_path=config_path)

    def test_state_t0_IS_None(self, path_to_config_with_MOCK_subclass):
        mpc = ModelPredictiveControler(config_path=path_to_config_with_MOCK_subclass)
        mpc.execute()

    def test_state_t0_NOT_None(self, path_to_config_with_MOCK_subclass):
        for each in range(100):
            mpc = ModelPredictiveControler(config_path=path_to_config_with_MOCK_subclass)
            mpc.config["hparam"]["controler"]["state_t0"] = [1.0, 1.0, 1.0, 1.0]
            trj_collector: TrajectoryCollector = mpc.execute()
            print(f"{each} ›› Size", trj_collector.get_size(), sep=' ')

    def test_state_t0_NOT_None_and_wrong_shape(self, path_to_config_with_MOCK_subclass):
        mpc = ModelPredictiveControler(config_path=path_to_config_with_MOCK_subclass)
        mpc.config["hparam"]["controler"]["state_t0"] = [1.0, 1.0, 1.0]
        with pytest.raises(AssertionError):
            mpc.execute()

    def test_execute_headless(self, path_to_config_with_MOCK_subclass):
        mpc = ModelPredictiveControler(config_path=path_to_config_with_MOCK_subclass)
        mpc.execute()

    @pytest.mark.skip(reason="ToDo: implement mp4 recording")
    def test_execute_record(self, path_to_config_with_MOCK_subclass):
        # config_path = path_to_config_with_MOCK_subclass
        # mpc = ModelPredictiveControler(config_path=config_path)
        # mpc.execute()

        raise NotImplementedError  # todo: implement


@pytest.fixture(scope="function")
def path_to_config_with_REAL_subclass():
    config_path = "tests/tests_barebones_mpc/config_files/default_test_config_real_CartPole-v1.yaml"
    return config_path

@pytest.fixture(scope="function")
def path_to_config_with_REAL_subclass_ALT():
    config_path = "tests/tests_barebones_mpc/config_files/default_test_config_real_ALT_CartPole-v1.yaml"
    return config_path


class TestModelPredictiveControlerReal:
    def test_MPC_controler_init(self, path_to_config_with_REAL_subclass):
        ModelPredictiveControler(config_path=path_to_config_with_REAL_subclass)

    def test_execute_headless(self, path_to_config_with_REAL_subclass):
        mpc = ModelPredictiveControler(config_path=path_to_config_with_REAL_subclass)
        mpc.execute()

    def test_execute_headless_w_prediction_step_type_int(self, path_to_config_with_REAL_subclass_ALT):
        mpc = ModelPredictiveControler(config_path=path_to_config_with_REAL_subclass_ALT)
        trajectory_collector = mpc.execute()
        print("trj collector\n",  "\tsize", trajectory_collector.get_size())
        # print("\ttrjs_rewards", trajectory_collector.trjs_rewards)
        print("\ttrjs_actions", trajectory_collector.trjs_actions)



    @pytest.mark.skip(reason="ToDo: iemplement mp4 recording")
    def test_execute_record(self, path_to_config_with_REAL_subclass):
        # config_path = path_to_config_with_REAL_subclass
        # mpc = ModelPredictiveControler(config_path=config_path)
        # mpc.execute()

        raise NotImplementedError  # todo: implement
