import pytest
import numpy as np
from src.barebones_mpc.evaluator.static_evaluator import StaticEvaluator

@pytest.fixture
def static_evaluator_init_params():
    time_step = 1/20  # 20 hz or 0.2 seconds
    horizon = 0.75  # s
    sample_length = int(horizon/time_step)
    number_samples = 1000
    input_dimension = 1 # input array dimension
    state_dimension = 4 # state array dimension
    beta = 0
    std_dev = np.array([0.1])
    inverse_temperature = 0.01

    sample_input = np.ones((sample_length, number_samples, input_dimension))

    return sample_length, number_samples, input_dimension, state_dimension, \
           std_dev, beta, inverse_temperature, sample_input

def test_static_evaluator_init(static_evaluator_init_params):
    sample_length, number_samples, input_dimension, state_dimension, std_dev, beta, inverse_temperature, sample_input = static_evaluator_init_params

    static_evaluator = StaticEvaluator(number_samples, input_dimension, sample_length, state_dimension, std_dev, beta, inverse_temperature)
    return None

def test_static_evaluator_compute_cost(static_evaluator_init_params):
    sample_length, number_samples, input_dimension, state_dimension, std_dev, beta, inverse_temperature, sample_input = static_evaluator_init_params

    static_evaluator = StaticEvaluator(number_samples, input_dimension, sample_length, state_dimension, std_dev, beta, inverse_temperature)
    static_evaluator.compute_input_cost(sample_input)

    assert static_evaluator.input_cost.shape == (sample_length, number_samples)
    return None