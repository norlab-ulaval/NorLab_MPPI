# coding=utf-8

import pytest
import numpy as np
from src.barebones_mpc.model.inv_pendulum_model import InvPendulumModel

import pandas as pd
from matplotlib import pyplot as plt

@pytest.fixture
def model_init_params():
    time_step = 1/20  # 20 hz or 0.2 seconds
    commanded_lon_vel = 1.5  # m/s
    horizon = 0.75  # s
    sample_length = int(horizon/time_step)
    number_samples = 1000
    input_dimension = 1 # input array dimension
    state_dimension = 4 # state array dimension
    init_state = np.zeros(4) # initial state vector
    cart_mass = 1 # kg
    pendulum_mass = 1 # kg
    sample_input = np.random.normal(loc=0, scale=1, size=(sample_length, number_samples, input_dimension))

    return time_step, commanded_lon_vel, horizon, sample_length, number_samples, input_dimension, init_state, cart_mass, pendulum_mass, sample_input

def test_mpc_inv_pend_model_init(model_init_params):
    time_step, commanded_lon_vel, horizon, sample_length, number_samples, input_dimension, init_state, cart_mass, pendulum_mass, sample_input = model_init_params

    inv_pendulum_model = InvPendulumModel(time_step, number_samples, sample_length, cart_mass, pendulum_mass)
    return None


@pytest.mark.skip(reason="TODO: dev in progress")
def test_mpc_inv_pend_model_predict(model_init_params):
    time_step, commanded_lon_vel, horizon, sample_length, number_samples, input_dimension, init_state, cart_mass, pendulum_mass, sample_input = model_init_params

    inv_pendulum_model = InvPendulumModel(time_step, number_samples, sample_length, cart_mass, pendulum_mass)
    sample_states = inv_pendulum_model.predict_states(init_state, sample_input)

    assert sample_states.shape == (sample_length, number_samples, init_state.shape[0])
    return None

