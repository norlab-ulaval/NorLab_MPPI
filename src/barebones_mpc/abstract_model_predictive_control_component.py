# coding=utf-8

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List


class AbstractModelPredictiveControlComponent(metaclass=ABCMeta):
    _config: Dict

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @classmethod
    @abstractmethod
    def _subclass_config_key(cls) -> str:
        """ The configuration file dictionary key pointing to the subclass component

        Exemple

        >>>     @classmethod
        >>>     def _subclass_config_key(cls) -> str:
        >>>         return 'sampler_hparam'

        """
        pass

    @classmethod
    @abstractmethod
    def _init_method_registred_param(cls) -> List[str]:
        """ The list of paramenters from the subclass __init__ signature

        Exemple

        >>>    @classmethod
        >>>    def _init_method_registred_param(cls) -> List[str]:
        >>>        return ['self', 'model', 'number_samples', 'input_dimension', 'sample_length', 'init_state']

        """
        pass

    @abstractmethod
    def _config_init_callback(self, config: Dict, subclass_config: Dict, signature_values_from_config: Dict) -> Dict:
        """ Add any custom code that you want executed with the config value just before object instanciation.

        Exemple

        >>>     def _config_init_callback(self, config: Dict, subclass_config: Dict, value_from_config: Dict) -> Dict:
        >>>     horizon: int = subclass_config['horizon']
        >>>     time_step: int = subclass_config['steps_per_prediction']
        >>>     input_shape: tuple = config['environment']['input_space']['shape']
        >>>
        >>>     values_from_callback = {
        >>>         'sample_length':   int(horizon/time_step),
        >>>         'init_state':      np.zeros(config['environment']['observation_space']['shape'][0]),
        >>>         'input_dimension': len(input_shape),
        >>>         }
        >>>
        >>>     return values_from_callback

        :param config: the complete configuration dictionary
        :param subclass_config: the configuration dictionary with only the key:value of the subclass
        :param signature_values_from_config: the key:value computed from the callback
        """
        pass

    @classmethod
    def config_init(cls, config: Dict, *args, **kwargs):
        """
        Alternative initialization method via configuration dictionary
        Return an instance of AbstractNominalPathBootstrap

        :param config: a dictionary of configuration
        :param args: pass arbitrary argument to the baseclass init method
        :param kwargs: pass arbitrary keyword argument to the baseclass init method
        """
        cls._config = config
        subclass_config = config['hparam'][cls._subclass_config_key()]

        # ... Fetch subclass __init__ signature and corresponding value ................................................
        subclasse_init_param_list = list(cls.__init__.__code__.co_varnames)
        subclasse_init_param_list.remove('self')
        signature_values_from_config = {param: value for param, value in subclass_config.items() if
                                        param in subclasse_init_param_list}

        signature_values_from_callback = cls._config_init_callback(cls, config=config,
                                                                   subclass_config=subclass_config,
                                                                   signature_values_from_config=signature_values_from_config)

        kwargs.update(signature_values_from_config)
        kwargs.update(signature_values_from_callback)

        # ... unaccounted parameter check ..............................................................................
        unaccounted_param = [each for each in subclasse_init_param_list if each not in kwargs]
        assert len(unaccounted_param) == 0, (
            f"{cls.ERR_S()} There's __init__ signature parameters unacounted by the config_init method >> "
            f"{unaccounted_param}")

        instance = cls(*args, **kwargs)
        return instance

    @classmethod
    def ERR_S(cls) -> str:
        return f"({cls.__qualname__} ERROR): "
