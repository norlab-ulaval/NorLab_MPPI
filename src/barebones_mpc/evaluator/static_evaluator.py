from src.barebones_mpc.evaluator.abstract_evaluator import AbstractEvaluator
import numpy as np

class StaticEvaluator(AbstractEvaluator):
    def __init__(self, number_samples, input_dimension, sample_length, state_dimension, std_dev, beta, inverse_temperature):
        self.number_samples = number_samples
        self.input_dimension = input_dimension
        self.sample_length = sample_length
        self.state_dimension = state_dimension
        self.std_dev = std_dev #TODO : Define std_Dev for multiple dimensions of input
        self.input_covariance = np.empty((self.input_dimension, self.input_dimension))
        np.fill_diagonal(self.input_covariance, self.std_dev)
        self.sample_costs = np.zeros((1, self.number_samples))
        self.input_cost = np.zeros((self.sample_length, self.number_samples))
        self.state_cost = np.zeros((self.sample_length, self.number_samples))
        self.final_state_cost = np.zeros((1, self.number_samples))
        self.beta = np.array([[beta]])
        self.inverse_temperature = inverse_temperature
        self.half_inverse_temperature = inverse_temperature / 2

    def compute_sample_costs(self, sample_input, sample_states):
        """ computes the cost related to every sample

        :param sample_input: sample input array
        :param sample_state: sample state array
        :return sample cost array
        """
        pass

    def compute_input_cost(self, input):
        """ computes a single input cost

        :param input: single input array
        :return input_cost: input cost
        """
        for j in range(0, input.shape[1]):
            for i in range(0, input.shape[0]):
                self.input_cost[i,j] = self.half_inverse_temperature * (input[i,j,:].transpose() @ self.input_covariance @ input[i,j,:] +
                                                                        self.beta.transpose() @ input[i,j,:])

    def compute_state_cost(self, state):
        """ compute a single state cost

        :param state: single state array
        :return state_cost: state cost
        """
        pass

    def compute_final_state_cost(self, final_state):
        """ compute a final state cost

        :param final_state: final state array
        :return final_state_cost: final state cost
        """
        pass
