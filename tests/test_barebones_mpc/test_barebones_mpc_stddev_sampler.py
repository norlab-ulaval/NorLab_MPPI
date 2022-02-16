# coding=utf-8

import pytest
import numpy as np
from src.barebones_mpc.sampler.std_dev_sampler import StandardDevSampler
from src.barebones_mpc.model.abstract_model import AbstractModel

import pandas as pd
from matplotlib import pyplot as plt

@pytest.fixture
def sampler_init_params():
    time_step = 1/20  # 20 hz or 0.2 seconds
    commanded_lon_vel = 1.5  # m/s
    horizon = 0.75  # s
    sample_length = int(horizon/time_step)
    number_samples = 1000
    input_dimension = 2 # input array dimension
    init_state = np.zeros(2) # initial state vector
    std_dev = 1.5  # rad/s

    return time_step, commanded_lon_vel, horizon, sample_length, number_samples, input_dimension, init_state, std_dev

def test_mpc_sampler_original_init(sampler_init_params):
    time_step, commanded_lon_vel, horizon, sample_length, number_samples, input_dimension, init_state, std_dev = sampler_init_params

    abstract_model = AbstractModel
    standard_dev_sampler = StandardDevSampler(abstract_model, number_samples, input_dimension, sample_length, init_state, std_dev)
    return None

def test_manualy_triggered_FAIL():
    assert False