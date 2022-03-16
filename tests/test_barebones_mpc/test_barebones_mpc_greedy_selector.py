import pytest
import numpy as np
from src.barebones_mpc.selector.greedy_selector import GreedySelector

@pytest.fixture
def greedy_selector_init_params():

    number_samples = 1000
    sample_total_cost = np.arange(0, number_samples).reshape(1,number_samples)

    return number_samples, sample_total_cost

def test_greedy_selector_init(greedy_selector_init_params):
    number_samples, sample_total_cost = greedy_selector_init_params

    greedy_selector = GreedySelector(number_samples)
    return None

def test_greedy_selector_select_arange(greedy_selector_init_params):
    number_samples, sample_total_cost = greedy_selector_init_params

    greedy_selector = GreedySelector(number_samples)
    selected_input_id = greedy_selector.select_next_input(sample_total_cost)
    assert selected_input_id == number_samples-1
    return None

@pytest.fixture
def greedy_selector_init_params_random_max():

    number_samples = 1000
    sample_total_cost = np.zeros((1, number_samples))
    random_max = np.random.randint(0,number_samples-1)
    sample_total_cost[0,random_max] = 1

    return number_samples, sample_total_cost, random_max

def test_greedy_selector_select_arange(greedy_selector_init_params_random_max):
    number_samples, sample_total_cost, random_max = greedy_selector_init_params_random_max

    greedy_selector = GreedySelector(number_samples)
    selected_input_id = greedy_selector.select_next_input(sample_total_cost)
    assert selected_input_id == random_max
    return None