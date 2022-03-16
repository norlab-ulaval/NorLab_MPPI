from src.barebones_mpc.evaluator.abstract_evaluator import AbstractEvaluator
import numpy as np

class QuadraticEvaluator(AbstractEvaluator):
    def __init__(self, number_samples, input_dimension, sample_length, state_dimension, state_weights, std_dev, beta, inverse_temperature, reference_state):
        self.number_samples = number_samples
        self.input_dimension = input_dimension
        self.sample_length = sample_length
        self.state_dimension = state_dimension
        self.std_dev = std_dev #TODO : Define std_Dev for multiple dimensions of input
        self.input_covariance = np.empty((self.input_dimension, self.input_dimension))
        np.fill_diagonal(self.input_covariance, self.std_dev)
        self.input_covariance_inverse = np.linalg.inv(self.input_covariance)
        self.sample_costs = np.zeros((self.sample_length, self.number_samples))
        self.sample_total_costs = np.zeros((1, self.number_samples))
        self.state_weights = np.zeros((self.state_dimension, self.state_dimension))
        np.fill_diagonal(self.state_weights, state_weights)
        self.final_state_cost = np.zeros((1, self.number_samples))
        self.beta = np.array([[beta]])
        self.inverse_temperature = inverse_temperature
        self.half_inverse_temperature = inverse_temperature / 2
        self.reference_state = reference_state

    def compute_sample_costs(self, sample_input, sample_states):
        """ computes the cost related to every sample

        :param sample_input: sample input array
        :param sample_state: sample state array
        :return sample cost array
        """
        for j in range(0, self.number_samples):
            for i in range(0, self.sample_length):
                self.sample_costs[i, j] = self.compute_state_cost(sample_states[i, j, :], self.reference_state) + \
                                          self.compute_input_cost(sample_input[i, j, :])
            self.sample_total_costs[0,j] = np.sum(self.sample_costs[:,j])

    def compute_input_cost(self, input):
        """ computes a single input cost via a quadratic input cost

        :param input: single input array
        :return input_cost: input cost
        """
        return self.half_inverse_temperature * (input.transpose() @ self.input_covariance_inverse @ input +
                                                  self.beta.transpose() @ input)

    def compute_state_cost(self, state, reference):
        """ compute a single state cost via a quadartic state cost

        :param state: single state array
        :return state_cost: state cost
        """
        error = state - reference
        return error.transpose() @ self.state_weights @ error

    def compute_final_state_cost(self, final_state):
        """ compute a final state cost

        :param final_state: final state array
        :return final_state_cost: final state cost
        """
        pass
