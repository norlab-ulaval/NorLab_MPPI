# coding=utf-8

from abc import ABCMeta, abstractmethod


class AbstractEvaluator(metaclass=ABCMeta):

    def __init__(self):
        self.state_weight = None
        self.input_weight = None

    @abstractmethod
    def compute_sample_costs(self, sample_input, sample_states):
        """ computes the cost related to every sample

        :param sample_input: sample input array
        :param sample_state: sample state array
        :return sample cost array
        """
        pass

    @abstractmethod
    def compute_input_cost(self, input):
        """ computes a single input cost

        :param input: single input array
        :return input_cost: input cost
        """
        pass

    @abstractmethod
    def compute_state_cost(self, state):
        """ compute a single state cost

        :param state: single state array
        :return state_cost: state cost
        """
        pass

    @abstractmethod
    def compute_final_state_cost(self, final_state):
        """ compute a final state cost

        :param final_state: final state array
        :return final_state_cost: final state cost
        """
        pass


class MockEvaluator(AbstractEvaluator):
    """ For testing purpose only"""

    def compute_sample_costs(self, sample_input, sample_states):
        pass

    def compute_input_cost(self, input):
        pass

    def compute_state_cost(self, state):
        pass

    def compute_final_state_cost(self, final_state):
        pass
