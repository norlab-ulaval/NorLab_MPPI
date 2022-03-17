# coding=utf-8
from typing import Type, Union

from src.barebones_mpc.evaluator.abstract_evaluator import AbstractEvaluator
from src.barebones_mpc.model.abstract_model import AbstractModel
from src.barebones_mpc.sampler.abstract_sampler import AbstractSampler
from src.barebones_mpc.selector.abstract_selector import AbstractSelector


def import_controler_component_class(config, component_key: str) -> Union[Type[AbstractModel], Type[AbstractSampler], Type[AbstractEvaluator], Type[AbstractSelector]]:
    """ Dynamicaly import a controler_component class from config key-value

    :param config: a configuration dictionary
    :param component_key: the controler_component class corresponding key in self.config
    :return: the controler_component class
    """
    model_cls_k: str = config['controler_component'][component_key]
    model_module_k, model_cls_k = model_cls_k.rsplit('.', maxsplit=1)
    model_module = __import__(model_module_k, fromlist=[model_cls_k])
    model_cls = getattr(model_module, model_cls_k)
    return model_cls
