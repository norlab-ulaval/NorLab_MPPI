# coding=utf-8

from typing import Union

import pytest
import numpy as np
from dataclasses import dataclass, field


# === SetUp from config file ===========================================================================================
@dataclass
class SetUpConfigSelector:
    config: Union[dict, None]
    number_samples: int = None
    sample_length: int = None
    input_dimension: int = None
    state_dimension: int = None
    sample_state: np.ndarray = None
    sample_input: np.ndarray = None
    sample_cost: np.ndarray = None
    random_min: np.ndarray = None
    optimal_trj_id: int = 1

    def mock_trajectories(self) -> None:
        assert self.config is not None
        self.number_samples = self.config["hparam"]["sampler_hparam"]["number_samples"]
        horizon = self.config["hparam"]["sampler_hparam"]["horizon"]
        time_step = self.config["hparam"]["sampler_hparam"]["steps_per_prediction"]
        self.sample_length = int(horizon / time_step)
        self.input_dimension = self.config["environment"]["observation_space"]["dim"]
        self.state_dimension = self.config["environment"]["input_space"]["dim"]

        self.sample_state = np.zeros((self.sample_length, self.number_samples, self.state_dimension))
        self.sample_input = np.zeros((self.sample_length, self.number_samples, self.input_dimension))

        self.sample_cost = np.ones((self.sample_length, self.number_samples, 1))
        self.sample_cost[:, self.optimal_trj_id, :] = 0
        return None


@pytest.fixture(scope="function")
def config_selector(setup_mock_config_dict_CartPole):
    return SetUpConfigSelector(config=setup_mock_config_dict_CartPole)


# === Manual setup =====================================================================================================
@dataclass
class ArbitrarySizeManualSelector:
    number_samples: int
    sample_length: int
    input_dimension: int  # input array dimension
    state_dimension: int  # state array dimension
    sample_input: np.ndarray = field(init=False)
    sample_state: np.ndarray = field(init=False)
    sample_cost: np.ndarray = field(init=False)
    random_min: np.ndarray = field(init=False)

    def __post_init__(self):
        self.sample_input = np.zeros((self.sample_length, self.number_samples, self.input_dimension))
        self.sample_state = np.zeros((self.sample_length, self.number_samples, self.state_dimension))
        self.sample_cost: np.ndarray = np.ones((self.sample_length, self.number_samples))
        self.random_min: np.ndarray = np.random.randint(0, self.number_samples - 1)

        self.sample_input[:, self.random_min, :] = 1
        self.sample_cost[0, self.random_min] = 0


@pytest.fixture(scope="function")
def arbitrary_size_manual_selector_10_5_1_2():
    return ArbitrarySizeManualSelector(sample_length=10, number_samples=5, input_dimension=1, state_dimension=2)

@pytest.fixture(scope="function")
def arbitrary_size_manual_selector_100_10_2_2():
    return ArbitrarySizeManualSelector(sample_length=100, number_samples=2, input_dimension=2, state_dimension=2)


@pytest.fixture
def greedy_selector_init_params(arbitrary_size_manual_selector_100_10_2_2):
    number_samples = arbitrary_size_manual_selector_100_10_2_2.number_samples
    sample_total_cost = np.arange(0, number_samples).reshape(1, number_samples)

    return arbitrary_size_manual_selector_100_10_2_2, sample_total_cost