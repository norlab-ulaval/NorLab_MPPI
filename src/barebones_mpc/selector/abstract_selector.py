from abc import ABC, ABCMeta, abstractmethod

from src.barebones_mpc.abstract_model_predictive_control_component import AbstractModelPredictiveControlComponent


class AbstractSelector(ABC, AbstractModelPredictiveControlComponent):
    def __init__(self, number_samples):
        self.number_samples = number_samples

    @classmethod
    def _subclass_config_key(cls) -> str:
        return "selector_hparam"

    @abstractmethod
    def select_next_input(self, sample_cost):
        """ select the optimal next input and state arrays

        :param sample_cost: sample cost array
        :return new nominal input and nominal state arrays
        """
        # :param sample_state: sample state array  (legacy)
        # :param sample_cost: sample cost array  (legacy)
        pass


class MockSelector(AbstractSelector):
    """ For testing purpose only"""

    def __init__(self, number_samples):
        super().__init__(number_samples)

        import gym

        self.env = gym.make("Pendulum-v1")

    def select_next_input(self, sample_cost):
        return self.env.action_space.sample(), True
