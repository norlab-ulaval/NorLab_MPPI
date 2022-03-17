import pytest
import numpy as np
from src.barebones_mpc.evaluator.quadratic_evaluator import QuadraticEvaluator

@pytest.fixture
def quadratic_evaluator_init_params():
    time_step = 1/20  # 20 hz or 0.2 seconds
    horizon = 0.75  # s
    sample_length = int(horizon/time_step)
    number_samples = 1000
    input_dimension = 1 # input array dimension
    state_dimension = 4 # state array dimension
    beta = 0
    std_dev = np.array([0.1])
    inverse_temperature = 0.01

    sample_state = np.ones((sample_length, number_samples, state_dimension))
    sample_input = np.ones((sample_length, number_samples, input_dimension))

    state_weights = np.array([1.0, 1.0, 1.0, 1.0])
    reference_state = np.zeros(4)

    return sample_length, number_samples, input_dimension, state_dimension, \
           std_dev, beta, inverse_temperature, sample_input, sample_state, reference_state, state_weights

def test_quadratic_evaluator_init(quadratic_evaluator_init_params):
    sample_length, number_samples, input_dimension, state_dimension, std_dev, beta, \
    inverse_temperature, sample_input, sample_state, reference_state, state_weights = quadratic_evaluator_init_params

    quadratic_evaluator = QuadraticEvaluator(number_samples, input_dimension, sample_length, state_dimension, state_weights,
                                             std_dev, beta, inverse_temperature, reference_state)
    return None

def test_quadratic_evaluator_compute_cost(quadratic_evaluator_init_params):
    sample_length, number_samples, input_dimension, state_dimension, std_dev, beta, \
    inverse_temperature, sample_input, sample_state, reference_state, state_weights = quadratic_evaluator_init_params

    quadratic_evaluator = QuadraticEvaluator(number_samples, input_dimension, sample_length, state_dimension, state_weights,
                                             std_dev, beta, inverse_temperature, reference_state)
    quadratic_evaluator.compute_sample_costs(sample_input, sample_state)

    assert quadratic_evaluator.sample_costs.shape == (sample_length, number_samples)
    assert quadratic_evaluator.sample_total_costs.shape == (1, number_samples)
    return None

@pytest.fixture
def quadratic_evaluator_simple_params():
    sample_length = 3
    number_samples = 3
    input_dimension = 1 # input array dimension
    state_dimension = 4 # state array dimension
    beta = 0
    std_dev = np.array([0.1])
    inverse_temperature = 0.01

    sample_state = np.ones((sample_length, number_samples, state_dimension))
    sample_input = np.ones((sample_length, number_samples, input_dimension))

    state_weights = np.array([1.0, 1.0, 1.0, 1.0])
    reference_state = np.zeros(4)

    return sample_length, number_samples, input_dimension, state_dimension, \
           std_dev, beta, inverse_temperature, sample_input, sample_state, reference_state, state_weights

def test_quadratic_evaluator_compute_cost(quadratic_evaluator_simple_params):
    sample_length, number_samples, input_dimension, state_dimension, std_dev, beta, \
    inverse_temperature, sample_input, sample_state, reference_state, state_weights = quadratic_evaluator_simple_params

    quadratic_evaluator = QuadraticEvaluator(number_samples, input_dimension, sample_length, state_dimension, state_weights,
                                             std_dev, beta, inverse_temperature, reference_state)
    quadratic_evaluator.compute_sample_costs(sample_input, sample_state)

    assert np.all(quadratic_evaluator.sample_costs == 4.05)
    assert np.all(quadratic_evaluator.sample_total_costs == 4.05 * sample_length)
    return None