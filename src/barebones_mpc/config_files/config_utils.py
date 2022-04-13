# coding=utf-8
from typing import Dict

from src.barebones_mpc.abstract_model_predictive_control_component import AbstractModelPredictiveControlComponent

def import_controler_component_class(config: Dict, component_key: str) -> AbstractModelPredictiveControlComponent:
    """ Dynamicaly import a controler_component class from config dictionary key-value

    :param config: a configuration dictionary
    :param component_key: the controler_component class corresponding key in self._config
    :return: the controler_component class
    """
    model_cls_k: str = config["controler_component"][component_key]
    model_module_k, model_cls_k = model_cls_k.rsplit(".", maxsplit=1)
    model_module = __import__(model_module_k, fromlist=[model_cls_k])
    model_cls = getattr(model_module, model_cls_k)
    return model_cls
