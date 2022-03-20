# coding=utf-8
from dataclasses import dataclass
from typing import Any, Type, Union

import pytest
import numpy as np

from src.barebones_mpc.sampler.abstract_sampler import AbstractSampler
from src.barebones_mpc.sampler.std_dev_sampler import StandardDevSampler
from src.barebones_mpc.model.abstract_model import AbstractModel, MockModel
from src.barebones_mpc.model.inv_pendulum_model import InvPendulumModel

import pandas as pd
from matplotlib import pyplot as plt


# ... refactoring ......................................................................................................
@dataclass
class ConfigSampler:
    config: Union[dict, None]
    init_state: Any = np.zeros(4)
    number_samples: int = 1000
    input_dimension: int = 1
    sample_length: int = int(0.75 / (1 / 20))
    # model: Type[MockModel] = MockModel()
    model: MockModel = MockModel(1, 1, 1, 1)


@pytest.fixture(scope="function")
def config_sampler(setup_mock_config_dict_CartPole):
    return ConfigSampler(config=setup_mock_config_dict_CartPole)


def test_config_init(setup_mock_config_dict_CartPole, config_sampler):
    instance = StandardDevSampler.config_init(config=setup_mock_config_dict_CartPole, model=config_sampler.model)
    assert isinstance(instance, AbstractSampler)


# ... implementation ...................................................................................................


@pytest.fixture
def sampler_init_params():
    time_step = 1 / 20  # 20 hz or 0.2 seconds
    commanded_lon_vel = 1.5  # m/s
    horizon = 0.75  # s
    sample_length = int(horizon / time_step)
    number_samples = 1000
    input_dimension = 2  # input array dimension
    state_dimension = 2
    init_state = np.zeros(state_dimension)  # initial state vector
    std_dev = np.array([1.5])  # rad/s

    return (
        time_step,
        commanded_lon_vel,
        horizon,
        sample_length,
        number_samples,
        input_dimension,
        state_dimension,
        init_state,
        std_dev,
    )


@pytest.fixture
def std_dev_inv_pendulum_model_init_params():
    time_step = 1 / 20  # 20 hz or 0.2 seconds
    commanded_lon_vel = 1.5  # m/s
    horizon = 0.75  # s
    sample_length = int(horizon / time_step)
    number_samples = 1000
    input_dimension = 1  # input array dimension
    state_dimension = 4  # state array dimension
    init_state = np.zeros(state_dimension)  # initial state vector
    cart_mass = 1  # kg
    pendulum_mass = 1  # kg
    nominal_input = np.zeros((sample_length, input_dimension))
    std_dev = np.array([0.1])

    return (
        time_step,
        commanded_lon_vel,
        horizon,
        sample_length,
        number_samples,
        input_dimension,
        state_dimension,
        init_state,
        cart_mass,
        pendulum_mass,
        nominal_input,
        std_dev,
    )


def test_mpc_stddev_init(sampler_init_params, config_sampler):
    (
        time_step,
        commanded_lon_vel,
        horizon,
        sample_length,
        number_samples,
        input_dimension,
        state_dimension,
        init_state,
        std_dev,
    ) = sampler_init_params

    standard_dev_sampler = StandardDevSampler(
        model=config_sampler.model,
        number_samples=number_samples,
        input_dimension=input_dimension,
        sample_length=sample_length,
        init_state=init_state,
        std_dev=std_dev,
    )
    return None


def test_mpc_stddev_inv_pendulum_sample_input(std_dev_inv_pendulum_model_init_params):
    (
        time_step,
        commanded_lon_vel,
        horizon,
        sample_length,
        number_samples,
        input_dimension,
        state_dimension,
        init_state,
        cart_mass,
        pendulum_mass,
        nominal_input,
        std_dev,
    ) = std_dev_inv_pendulum_model_init_params

    inv_pendulum_model = InvPendulumModel(time_step, number_samples, sample_length, cart_mass, pendulum_mass)
    standard_dev_sampler = StandardDevSampler(
        inv_pendulum_model, number_samples, input_dimension, sample_length, init_state, std_dev
    )

    sample_input = standard_dev_sampler.sample_inputs(nominal_input)
    assert sample_input.shape == (sample_length + 1, number_samples, input_dimension)
    return None


def test_mpc_stddev_inv_pendulum_sample_input(std_dev_inv_pendulum_model_init_params):
    (
        time_step,
        commanded_lon_vel,
        horizon,
        sample_length,
        number_samples,
        input_dimension,
        state_dimension,
        init_state,
        cart_mass,
        pendulum_mass,
        nominal_input,
        std_dev,
    ) = std_dev_inv_pendulum_model_init_params

    inv_pendulum_model = InvPendulumModel(
        time_step=time_step,
        number_samples=number_samples,
        sample_length=sample_length,
        state_dimension=state_dimension,
        cart_mass=cart_mass,
        pendulum_mass=pendulum_mass,
    )
    standard_dev_sampler = StandardDevSampler(
        inv_pendulum_model, number_samples, input_dimension, sample_length, init_state, std_dev
    )

    sample_input = standard_dev_sampler.sample_inputs(nominal_input)
    sample_state = standard_dev_sampler.sample_states(sample_input, init_state)
    assert sample_state.shape == (sample_length + 1, number_samples, state_dimension)
    return None


@pytest.fixture
def std_dev_sample_straight_line_params():
    time_step = 1 / 20  # 20 hz or 0.2 seconds
    commanded_lon_vel = 1.5  # m/s
    horizon = 0.75  # s
    sample_length = int(horizon / time_step)
    number_samples = 1000
    input_dimension = 1  # input array dimension
    state_dimension = 4  # state array dimension
    init_state = np.zeros(state_dimension)  # initial state vector
    cart_mass = 1  # kg
    pendulum_mass = 1  # kg
    nominal_input = np.arange(0, sample_length).reshape((sample_length, input_dimension))
    std_dev = np.array([0.0])

    return (
        time_step,
        commanded_lon_vel,
        horizon,
        sample_length,
        number_samples,
        input_dimension,
        state_dimension,
        init_state,
        cart_mass,
        pendulum_mass,
        nominal_input,
        std_dev,
    )


def test_mpc_barebones_sample(std_dev_sample_straight_line_params):
    (
        time_step,
        commanded_lon_vel,
        horizon,
        sample_length,
        number_samples,
        input_dimension,
        state_dimension,
        init_state,
        cart_mass,
        pendulum_mass,
        nominal_input,
        std_dev,
    ) = std_dev_sample_straight_line_params

    inv_pendulum_model = InvPendulumModel(
        time_step=time_step,
        number_samples=number_samples,
        sample_length=sample_length,
        state_dimension=state_dimension,
        cart_mass=cart_mass,
        pendulum_mass=pendulum_mass,
        )
    standard_dev_sampler = StandardDevSampler(
        inv_pendulum_model, number_samples, input_dimension, sample_length, init_state, std_dev
    )

    sample_input = standard_dev_sampler.sample_inputs(nominal_input)
    assert (sample_input[:, 0, :] == nominal_input).all()
    return None
