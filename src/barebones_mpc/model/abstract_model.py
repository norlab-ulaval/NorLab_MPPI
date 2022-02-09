from abc import ABCMeta, abstractmethod

class AbstractModel(metaclass=ABCMeta):
    def __init__(self):
        self.state_dim = None
        self.input_dim = None
        self.time_step = None

    @abstractmethod
    def predict_states(self, init_state, sample_input):
        """ predicts a sample state array based on a nominal input array

        :param init_state: initial state array
        :param sample_input: sample input array
        :return: sample state array
        """
        pass

    @abstractmethod
    def _predict(self, init_state, input):
        """ makes a single state prediction based on initial state and input

        :param init_state: initial state array
        :param sample_input: input array
        :return: predicted state array
        """
        pass