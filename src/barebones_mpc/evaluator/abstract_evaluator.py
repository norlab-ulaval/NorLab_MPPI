from abc import ABCMeta, abstractmethod


class AbstractEvaluator(metaclass=ABCMeta):

    def __init__(self):
        self.state_weight = None
        self.input_weight = None

    @abstractmethod
    def sample_costs(self, sample_input, sample_states):
        """ computes the cost related to every sample

        :param sample_input: sample input array
        :param sample_state: sample state array
        :return sample cost array
        """
        pass

    @abstractmethod
    def input_cost(self, input):
        """ computes a single input cost

        :param input: single input array
        :return input_cost: input cost
        """
        pass

    @abstractmethod
    def state_cost(self, state):
        """ compute a single state cost

        :param state: single state array
        :return state_cost: state cost
        """
        pass

    @abstractmethod
    def final_state_cost(self, final_state):
        """ compute a final state cost

        :param final_state: final state array
        :return final_state_cost: final state cost
        """
        pass


class MockEvaluator(AbstractEvaluator):
    """ For testing purpose only"""

    def sample_costs(self, sample_input, sample_states):
        pass

    def input_cost(self, input):
        pass

    def state_cost(self, state):
        pass

    def final_state_cost(self, final_state):
        pass
