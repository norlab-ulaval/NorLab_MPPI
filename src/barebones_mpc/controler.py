# coding=utf-8
import os
from typing import Union, Any, Type
import yaml
import gym

from src.barebones_mpc.model.abstract_model import AbstractModel
from src.barebones_mpc.sampler.abstract_sampler import AbstractSampler
from src.barebones_mpc.evaluator.abstract_evaluator import AbstractEvaluator
from src.barebones_mpc.selector.abstract_selector import AbstractSelector


class ModelPredictiveControler(object):

    def __init__(self, config_path: str,
                 environment: Union[Type[gym.Env]],
                 state_t0: Any = None):
        """
        Execute a barebones mpc run using the provided component.

        The function manage the instanciation of class component and execute the run with respect to the config file.

        :param config_path: path to the configuration file in YAML
        :param environment: must respect the OpenAI gym API signature
        :param state_t0: (optional) will be generated from the environment if not provided
        """

        # ... Setup controller .........................................................................................
        assert type(config_path) is str
        assert os.path.exists(os.path.relpath(config_path))
        with open(config_path, 'r') as f:
            self.config = dict(yaml.safe_load(f))

        model_cls: Type[AbstractModel] = self._import_controler_component_class('model')
        sampler_cls = self._import_controler_component_class('sampler')
        evaluator_cls = self._import_controler_component_class('evaluator')
        selector_cls = self._import_controler_component_class('selector')

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

        self.model = model_cls()
        self.sampler = sampler_cls(self.model)
        self.evaluator = evaluator_cls()
        self.selector = selector_cls()

        # ... Feedback loop ............................................................................................

        self.execute(state_t0)

    def execute(self, state_t0):
        if state_t0 is not None:
            raise NotImplementedError("(CRITICAL) todo ← Feeback loop initial state is given by the environment")

        hparam_horizon = self.config['hparam']['experimental-hparam']['horizon']
        for global_step in range(hparam_horizon):
            pass

    def _import_controler_component_class(self, component_key: str) -> Union[
        Type[AbstractModel], Type[AbstractSampler], Type[AbstractEvaluator], Type[AbstractSelector]]:
        """ Dynamicaly import a controler-component class from config key-value

        :param component_key: the controler-component class corresponding key in self.config
        :return: the controler-component class
        """
        model_cls_k: str = self.config['controler-component'][component_key]
        model_module_k, model_cls_k = model_cls_k.rsplit('.', maxsplit=1)
        model_module = __import__(model_module_k, fromlist=[model_cls_k])
        model_cls = getattr(model_module, model_cls_k)
        return model_cls
