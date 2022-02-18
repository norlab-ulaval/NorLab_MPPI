# coding=utf-8

from src.barebones_mpc.evaluator.abstract_evaluator import AbstractEvaluator
from src.barebones_mpc.model.abstract_model import AbstractModel
from src.barebones_mpc.sampler.abstract_sampler import AbstractSampler
from src.barebones_mpc.selector.abstract_selector import AbstractSelector


class MockModel(AbstractModel):

    def predict_states(self, init_state, sample_input):
        pass

    def _predict(self, init_state, input):
        pass


class MockSampler(AbstractSampler):

    def sample_inputs(self, nominal_input):
        pass

    def sample_states(self, sample_input, init_state):
        pass


class MockEvaluator(AbstractEvaluator):

    def sample_costs(self, sample_input, sample_states):
        pass

    def input_cost(self, input):
        pass

    def state_cost(self, state):
        pass

    def final_state_cost(self, final_state):
        pass


class MockSelector(AbstractSelector):

    def select_next_input(self, sample_input, sample_state, sample_cost):
        pass
