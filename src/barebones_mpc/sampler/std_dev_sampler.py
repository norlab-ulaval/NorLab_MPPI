from src.barebones_mpc.model.abstract_model import AbstractModel
from src.barebones_mpc.sampler.abstract_sampler import AbstractSampler
import numpy as np
from typing import Type


class StandardDevSampler(AbstractSampler):

    def __init__(self, model: Type[AbstractModel], number_samples: int, input_dimension: int,
                 sample_length: int, init_state: np.ndarray, std_dev: float):
        super().__init__(model, number_samples, input_dimension, sample_length, init_state)

        self.std_dev = std_dev  #TODO : Define std_Dev for multiple dimensions of input
        self.sample_input = np.empty((self.sample_length, self.number_samples + 1, self.input_dimension))

    def sample_inputs(self, nominal_input):
        """ Sample inputs based on the nominal input array

        :param nominal_input: the nominal input array
        :return: sample input array
        """
        self.sample_input[:, 0, :] = nominal_input
        sample_noise = np.random.normal(loc=0, scale=self.std_dev,
                                        size=(self.sample_length, self.number_samples, self.input_dimension))
        self.sample_input[:, 1:, :] = sample_noise + np.tile(
            nominal_input.reshape(self.sample_length, 1, self.input_dimension), (1, self.number_samples, 1))
        return self.sample_input

    def sample_states(self, sample_input, init_state):
        """ Sample states based on the sample input array through the model

        use self.model to sample states

        :param sample_input: the sampling input
        :param init_state: the initial state array
        :return: sample state array
        """
        sample_states = self.model.predict_states(init_state, sample_input)
        return sample_states
