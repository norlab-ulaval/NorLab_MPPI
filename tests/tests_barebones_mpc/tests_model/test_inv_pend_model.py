# coding=utf-8
from typing import Any, Union

import pytest
import numpy as np

from src.barebones_mpc.model.abstract_model import AbstractModel
from src.barebones_mpc.model.inv_pendulum_model import InvPendulumModel

import pandas as pd
from matplotlib import pyplot as plt


# ... refactoring ......................................................................................................
def test_config_init(setup_mock_config_dict_CartPole):
    instance = InvPendulumModel.config_init(config=setup_mock_config_dict_CartPole)
    assert isinstance(instance, AbstractModel)


# ... implementation ...................................................................................................
@pytest.fixture
def model_init_params():
    prediction_step = 1 / 20  # 20 hz or 0.2 seconds
    horizon = 0.75  # s
    sample_length = int(horizon / prediction_step)
    commanded_lon_vel = 1.5  # m/s
    number_samples = 1000
    input_dimension = 1  # input array dimension
    state_dimension = 4  # state array dimension
    init_state = np.zeros(4)  # initial state vector
    cart_mass = 1  # kg
    pendulum_mass = 1  # kg
    sample_input = np.random.normal(loc=0, scale=1, size=(sample_length, number_samples, input_dimension))

    return (
        prediction_step,
        commanded_lon_vel,
        horizon,
        sample_length,
        number_samples,
        input_dimension,
        state_dimension,
        init_state,
        cart_mass,
        pendulum_mass,
        sample_input,
    )


def test_mpc_inv_pend_model_init(model_init_params):
    (
        prediction_step,
        commanded_lon_vel,
        horizon,
        sample_length,
        number_samples,
        input_dimension,
        state_dimension,
        init_state,
        cart_mass,
        pendulum_mass,
        sample_input,
    ) = model_init_params

    # inv_pendulum_model = InvPendulumModel(prediction_step, number_samples, sample_length, cart_mass, pendulum_mass)
    inv_pendulum_model = InvPendulumModel(
        prediction_step=prediction_step,
        number_samples=number_samples,
        sample_length=sample_length,
        state_dimension=state_dimension,
        cart_mass=cart_mass,
        pendulum_mass=pendulum_mass,
    )
    return None


# @pytest.mark.skip(reason="TODO: dev in progress")
def test_mpc_inv_pend_model_predict(model_init_params):
    (
        prediction_step,
        commanded_lon_vel,
        horizon,
        sample_length,
        number_samples,
        input_dimension,
        state_dimension,
        init_state,
        cart_mass,
        pendulum_mass,
        sample_input,
    ) = model_init_params

    # inv_pendulum_model = InvPendulumModel(prediction_step, number_samples, sample_length, cart_mass, pendulum_mass)
    inv_pendulum_model = InvPendulumModel(
        prediction_step=prediction_step,
        number_samples=number_samples,
        sample_length=sample_length,
        state_dimension=state_dimension,
        cart_mass=cart_mass,
        pendulum_mass=pendulum_mass,
    )
    sample_states = inv_pendulum_model.predict_states(init_state, sample_input)

    assert sample_states.shape == (sample_length + 1, number_samples, init_state.shape[0])
    return None


@pytest.fixture
def simple_model_init_params():
    prediction_step = 1 / 20  # 20 hz or 0.2 seconds
    commanded_lon_vel = 1.5  # m/s
    horizon = prediction_step * 3  # s
    sample_length = int(horizon / prediction_step)
    number_samples = 3
    input_dimension = 1  # input array dimension
    state_dimension = 4  # state array dimension
    init_state = np.zeros(4)  # initial state vector
    cart_mass = 1  # kg
    pendulum_mass = 1  # kg
    sample_input = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]]).reshape((3, 3, 1))
    return (
        prediction_step,
        commanded_lon_vel,
        horizon,
        sample_length,
        number_samples,
        input_dimension,
        state_dimension,
        init_state,
        cart_mass,
        pendulum_mass,
        sample_input,
    )


def test_mpc_inv_pend_model_simple_sampling(simple_model_init_params):
    (
        prediction_step,
        commanded_lon_vel,
        horizon,
        sample_length,
        number_samples,
        input_dimension,
        state_dimension,
        init_state,
        cart_mass,
        pendulum_mass,
        sample_input,
    ) = simple_model_init_params

    # inv_pendulum_model = InvPendulumModel(prediction_step, number_samples, sample_length, cart_mass, pendulum_mass)
    inv_pendulum_model = InvPendulumModel(
        prediction_step=prediction_step,
        number_samples=number_samples,
        sample_length=sample_length,
        state_dimension=state_dimension,
        cart_mass=cart_mass,
        pendulum_mass=pendulum_mass,
    )
    sample_states = inv_pendulum_model.predict_states(init_state, sample_input)

    print(sample_states[3, 1, :])
    tolerance = 0.000000001
    assert np.allclose(sample_states[0, 0, :], np.array([0, 0, 0, 0]), atol=tolerance)
    assert np.allclose(sample_states[1, 0, :], np.array([0, 0.05, 0, -0.05]), atol=tolerance)
    assert np.allclose(sample_states[2, 0, :], np.array([0.0025, 0.15, -0.0025, -0.15]), atol=tolerance)
    assert np.allclose(sample_states[3, 0, :], np.array([0.01, 0.3000625, -0.01, -0.300125]), atol=tolerance)
    assert np.allclose(sample_states[1, 1, :], np.array([0, 0.2, 0, -0.2]), atol=tolerance)
    assert np.allclose(sample_states[2, 1, :], np.array([0.01, 0.45, -0.01, -0.45]), atol=tolerance)
    assert np.allclose(sample_states[3, 1, :], np.array([0.0325, 0.75025, -0.0325, -0.7505]), atol=tolerance)
    return None
