# coding=utf-8

from src.barebones_mpc.model.abstract_model import AbstractModel, MockModel


class TestMockModel:
    def test_init(self, config_model):
        instance = MockModel(
            prediction_step=config_model.prediction_step,
            number_samples=config_model.number_samples,
            sample_length=config_model.sample_length,
            state_dimension=config_model.state_dimension,
        )

    def test_config_init(self, setup_mock_config_dict_CartPole):
        instance = MockModel.config_init(config=setup_mock_config_dict_CartPole)
        assert isinstance(instance, AbstractModel)

    def test_predict_states(self, setup_mock_config_dict_CartPole):
        instance = MockModel.config_init(config=setup_mock_config_dict_CartPole)
        instance.predict_states(None, None)

    def test__predict(self, setup_mock_config_dict_CartPole):
        instance = MockModel.config_init(config=setup_mock_config_dict_CartPole)
        instance._predict(None, None)
