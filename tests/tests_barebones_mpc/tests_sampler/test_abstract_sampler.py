# coding=utf-8

from src.barebones_mpc.sampler.abstract_sampler import AbstractSampler, MockSampler


class TestMockSampler:
    def test_init(self, config_sampler):
        instance = MockSampler(
            model=config_sampler.model,
            number_samples=config_sampler.number_samples,
            input_dimension=config_sampler.input_dimension,
            sample_length=config_sampler.sample_length,
            init_state=config_sampler.init_state,
            input_type=config_sampler.input_type,
            input_space=config_sampler.input_space,
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
