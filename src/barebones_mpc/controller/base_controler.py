# coding=utf-8
import os
from typing import List, Type

import numpy as np
import yaml
from dataclasses import dataclass, field

from src.barebones_mpc.config_files.config_utils import import_controler_component_class

from src.barebones_mpc.model.abstract_model import AbstractModel
from src.barebones_mpc.sampler.abstract_sampler import AbstractSampler
from src.barebones_mpc.evaluator.abstract_evaluator import AbstractEvaluator
from src.barebones_mpc.selector.abstract_selector import AbstractSelector
from src.barebones_mpc.nominal_path.abstract_NP import AbstractNominalPath

from src.environment_adapter.adapters import make_environment_adapter, AbstractEnvironmentAdapter


@dataclass
class TrajectoryCollector:
    trjs_observations: List = field(init=False)
    trjs_actions: List = field(init=False)
    trjs_rewards: List = field(init=False)

    def __post_init__(self):
        self.trjs_observations = []
        self.trjs_actions = []
        self.trjs_rewards = []

    def append(self, trj_observations, trj_actions, trj_rewards):
        self.trjs_observations.append(trj_observations)
        self.trjs_actions.append(trj_actions)
        self.trjs_rewards.append(trj_rewards)

    def get_size(self):
        return len(self.trjs_rewards)


@dataclass
class TimestepCollector:
    observations: List = field(init=False)
    actions: List = field(init=False)
    rewards: List = field(init=False)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def append(self, observation, action, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)

    def get_trajectory_length(self):
        return len(self.rewards)


class ModelPredictiveControler(object):
    @classmethod
    def MPC_ERROR(cls):
        return f"({cls.__qualname__} ERROR): "

    @classmethod
    def MPC_MSG(cls):
        return f"({cls.__qualname__} MSG): "

    @classmethod
    def MPC_feadbackloop_MSG(cls):
        return f":: MPC MSG: "

    def __init__(self, config_path: str):
        """ Execute a barebones mpc run using the provided component.
        The function manage the instanciation of class component and execute the run with respect to the config file.

        :param config_path: path to the configuration file in YAML
        """

        # ... Setup controller .........................................................................................
        assert type(config_path) is str
        assert os.path.exists(os.path.relpath(config_path))
        with open(config_path, "r") as f:
            self.config = dict(yaml.safe_load(f))

        nominal_path_cls: Type[AbstractNominalPath] = import_controler_component_class(self.config, "nominalPath")
        model_cls: Type[AbstractModel] = import_controler_component_class(self.config, "model")
        sampler_cls: Type[AbstractSampler] = import_controler_component_class(self.config, "sampler")
        evaluator_cls: Type[AbstractEvaluator] = import_controler_component_class(self.config, "evaluator")
        selector_cls: Type[AbstractSelector] = import_controler_component_class(self.config, "selector")

        # ... Config validation ........................................................................................
        assert issubclass(nominal_path_cls, AbstractNominalPath), (
            f"{self.MPC_ERROR()} argument passed to param 'nominal_path_cls' must be a subclass of "
            f"'AbstractNominalPath'"
        )
        assert issubclass(
            model_cls, AbstractModel
        ), f"{self.MPC_ERROR()} argument passed to param 'model_cls' must be a subclass of 'AbstractModel'"
        assert issubclass(
            sampler_cls, AbstractSampler
        ), f"{self.MPC_ERROR()} argument passed to param 'sampler_cls' must be a subclass of 'AbstractSampler'"
        assert issubclass(
            evaluator_cls, AbstractEvaluator
        ), f"{self.MPC_ERROR()} argument passed to param 'evaluator_cls' must be a subclass of 'AbstractEvaluator'"
        assert issubclass(
            selector_cls, AbstractSelector
        ), f"{self.MPC_ERROR()} argument passed to param 'selector_cls' must be a subclass of 'AbstractSelector'"

        self.nominal_path = nominal_path_cls.config_init(self.config)
        self.model = model_cls.config_init(self.config)
        self.sampler = sampler_cls.config_init(model=self.model, config=self.config)
        self.evaluator = evaluator_cls.config_init(self.config)
        self.selector = selector_cls.config_init(self.config)
        self.environment = self._setup_environment()

    def execute(self) -> TrajectoryCollector:
        """ Execute feedback loop """

        try:
            headless_mode_ = self.config["force_headless_mode"]
            experimental_window_ = self.config["hparam"]["experimental_hparam"]["experimental_window"]
        except KeyError as e:
            raise KeyError(
                f"{self.MPC_ERROR()} There's required key-value missing in the config file. " f"Missing key >> {e}"
            ) from e

        # ::: Initialize observation and nominal input at timestep 0 :::::::::::::::::::::::::::::::::::::::::::::::::::
        state_t0 = self.config["hparam"]["controler"]["state_t0"]
        if state_t0 is None:
            observation = self.environment.reset()
        else:
            observation = np.array(state_t0)

            expected_obs_shape = self.environment.reset().shape
            assert (
                expected_obs_shape == observation.shape
            ), f"{self.MPC_ERROR()} The `state_t0` list provided in config_file as the wrong shape"

        trajectory_collector = TrajectoryCollector()
        timestep_collector = TimestepCollector()

        nominal_inputs = self.nominal_path.bootstrap(state_t0)

        # ::: Start feedback loop ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        f"{self.MPC_feadbackloop_MSG()} Starting feedback loop"
        for global_step in range(experimental_window_):

            if headless_mode_:
                self.environment.render()

            # ... Virtualy sample many alternate trajectories...........................................................
            sample_inputs = self.sampler.sample_inputs(nominal_input=nominal_inputs)
            sample_states = self.sampler.sample_states(sample_input=sample_inputs, init_state=observation)

            # ... Evaluate samples .....................................................................................
            self.evaluator.compute_sample_costs(sample_input=sample_inputs, sample_states=sample_states)
            sample_costs = self.evaluator.get_trajectories_cumulative_cost()

            # ... Choose the optimal trajectory and commit to the first input command ..................................
            nominal_inputs, nominal_states = self.selector.select_next_input(
                sample_states=sample_states, sample_inputs=sample_inputs, sample_costs=sample_costs
            )
            first_input = nominal_inputs.ravel()[0]

            assert self.environment.action_space.contains(
                first_input
            ), f"{self.MPC_ERROR()} The input {first_input} is not a legal input"

            next_observation, cost, done, info = self.environment.step(action=first_input)

            # ... Trajectory data book keeping .........................................................................
            timestep_collector.append(observation=observation, action=first_input, reward=cost)

            if done:
                next_observation = self.environment.reset()
                trajectory_collector.append(
                    trj_observations=timestep_collector.observations.copy(),
                    trj_actions=timestep_collector.actions.copy(),
                    trj_rewards=timestep_collector.rewards.copy(),
                )
                print(
                    f"{self.MPC_feadbackloop_MSG()} Trj {trajectory_collector.get_size()}: "
                    f"Terminal state reached with return={sum(timestep_collector.rewards)}"
                )
                timestep_collector.reset()

            observation = next_observation

        self.environment.close()

        if trajectory_collector.get_size() == 0:
            print(f"{self.MPC_feadbackloop_MSG()} Did not reach any terminal state during the experimental window.")

        return trajectory_collector

    def _setup_environment(self) -> Type[AbstractEnvironmentAdapter]:
        return make_environment_adapter(self.config)
