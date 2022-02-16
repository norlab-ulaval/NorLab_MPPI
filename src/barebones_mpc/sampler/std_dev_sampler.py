from src.barebones_mpc.sampler.abstract_sampler import AbstractSampler
import numpy as np

class StandardDevSampler(AbstractSampler):
    def __init__(self, model, number_samples, input_dimension, sample_length, init_state, std_dev):
        self.number_samples = number_samples
        self.input_dimension = input_dimension
        self.sample_length = sample_length
        self.init_state = init_state
        self.std_dev = std_dev

    def sample_inputs(self, nominal_input):
        """ Sample inputs based on the nominal input array

        :param nominal_input: the nominal input array
        :return: sample input array
        """
        sample_noise = np.random.normal(loc=0, scale=self.std_dev, size=(self.input_dimension, self.number_samples))
        sample_input = np.sum(nominal_input, sample_noise)
        sample_input = np.concatenate(nominal_input, sample_input, axis=0)
        return sample_input

    def sample_states(self, sample_input, init_state):
        """ Sample states based on the sample input array through the model

        use self.model to sample states

        :param sample_input: the sampling input
        :param init_state: the initial state array
        :return: sample state array
        """
        sample_states = self.model.predict_states(init_state, sample_input)
        return sample_states