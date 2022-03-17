# coding=utf-8
import os
from typing import Any, Type
import yaml

from src.barebones_mpc.config_files.config_utils import import_controler_component_class
from src.barebones_mpc.model.abstract_model import AbstractModel
from src.barebones_mpc.sampler.abstract_sampler import AbstractSampler
from src.barebones_mpc.evaluator.abstract_evaluator import AbstractEvaluator
from src.barebones_mpc.selector.abstract_selector import AbstractSelector
from src.environment_adapter.adapters import make_environment_adapter, AbstractEnvironmentAdapter


class ModelPredictiveControler(object):

    def __init__(self, config_path: str):
        """
        Execute a barebones mpc run using the provided component.

        The function manage the instanciation of class component and execute the run with respect to the config file.

        :param config_path: path to the configuration file in YAML
        """

        # ... Setup controller .........................................................................................
        assert type(config_path) is str
        assert os.path.exists(os.path.relpath(config_path))
        with open(config_path, 'r') as f:
            self.config = dict(yaml.safe_load(f))

        model_cls: Type[AbstractModel] = import_controler_component_class(self.config, 'model')
        sampler_cls: Type[AbstractSampler] = import_controler_component_class(self.config, 'sampler')
        evaluator_cls: Type[AbstractEvaluator] = import_controler_component_class(self.config, 'evaluator')
        selector_cls: Type[AbstractSelector] = import_controler_component_class(self.config, 'selector')

        # ... Config validation ........................................................................................
        ERR_S = "(barebones_mpc ERROR): "
        assert issubclass(model_cls, AbstractModel), (
            f'{ERR_S} argument passed to param \'model_cls\' must be a subclass of \'AbstractModel\'')
        assert issubclass(sampler_cls, AbstractSampler), (
            f'{ERR_S} argument passed to param \'sampler_cls\' must be a subclass of \'AbstractSampler\'')
        assert issubclass(evaluator_cls, AbstractEvaluator), (
            f'{ERR_S} argument passed to param \'evaluator_cls\' must be a subclass of \'AbstractEvaluator\'')
        assert issubclass(selector_cls, AbstractSelector), (
            f'{ERR_S} argument passed to param \'selector_cls\' must be a subclass of \'AbstractSelector\'')

        self.model = model_cls.config_init(self.config)
        self.sampler = sampler_cls.config_init(self.model, self.config)
        self.evaluator = evaluator_cls.config_init(self.config)
        self.selector = selector_cls.config_init(self.config)
        self.environment = self._setup_environment()

    def execute(self, state_t0: Any = None) -> None:
        """ Execute feedback loop

        :param state_t0: (optional) will be generated from the environment if not provided
        """

        observations = []
        actions = []
        rewards = []

        if state_t0 is not None:
            # (CRITICAL) todo:implement
            raise NotImplementedError("(CRITICAL) todo ← Feeback loop initial state is given by the environment")
        else:
            observation = self.environment.reset()

        observations.append(observation)

        nominal_input = self._init_nominal_input()
        experimental_window = self.config['hparam']['experimental_hparam']['experimental_window']
        for global_step in range(experimental_window):

            if self.config['force_headless_mode']:
                self.environment.render()

            sample_input = self.sampler.sample_inputs(nominal_input=nominal_input)
            sample_states = self.sampler.sample_states(sample_input=sample_input, init_state=observation)
            sample_cost = self.evaluator.compute_sample_costs(sample_input=sample_input, sample_states=sample_states)
            nominal_input, nominal_states = self.selector.select_next_input(sample_cost=sample_cost)

            next_observation, reward, done, info = self.environment.step(input=nominal_input)

            actions.append(nominal_input)
            rewards.append(reward)
            observations.append(next_observation)

            if done:
                next_observation = self.environment.reset()

            observation = next_observation

        self.environment.close()

        return observations, actions, rewards

    def _setup_environment(self) -> Type[AbstractEnvironmentAdapter]:
        return make_environment_adapter(self.config)

    def _init_nominal_input(self):
        pass
        # raise NotImplementedError  # (CRITICAL) todo:implement <-- we are here

