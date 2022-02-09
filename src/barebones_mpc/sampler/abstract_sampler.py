from abc import ABCMeta, abstractmethod
from src.barebones_mpc.model.abstract_model import AbstractModel

class AbstractSampler(metaclass=ABCMeta):
    def __init__(self, model):
        self.number_samples = None
        self.sample_length = None
        self.init_state = None

        assert isinstance(model, AbstractModel)
        self.model = model

    @abstractmethod
    def sample_inputs(self, nominal_input):
        """ Sample inputs based on the nominal input array

        :param nominal_input: the nominal input array
        :return: sample input array
        """
        pass

    @abstractmethod
    def sample_states(self, sample_input, init_state):
        """ Sample states based on the sample input array through the model

        use self.model to sample states

        :param sample_input: the sampling input
        :param init_state: the initial state array
        :return: sample state array
        """
        pass