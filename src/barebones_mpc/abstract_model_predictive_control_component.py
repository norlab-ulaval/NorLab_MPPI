# coding=utf-8

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List


class AbstractModelPredictiveControlComponent(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._config = None

    @classmethod
    def ERR_S(cls) -> str:
        return f"({cls.__qualname__} ERROR): "

    @classmethod
    @abstractmethod
    def _subclass_config_key(cls) -> str:
        """ The configuration file dictionary key pointing to the subclass component

        Exemple

        >>> class MyCoolMPCComponent:
        >>>     @classmethod
        >>>     def _subclass_config_key(cls) -> str:
        >>>         return 'sampler_hparam'

        """
        pass

    @classmethod
    @abstractmethod
    def _config_file_required_field(cls) -> List[str]:
        """ The list of paramenters from the subclass __init__ signature

        Exemple

        >>> class MyCoolMPCComponent:
        >>>    @classmethod
        >>>    def _config_file_required_field(cls) -> List[str]:
        >>>        return ['number_samples', 'sample_length', 'input_dimension']

        """
        pass

    @abstractmethod
    def _config_pre_init_callback(
        self, config: Dict, subclass_config: Dict, signature_values_from_config: Dict
    ) -> Dict:
        """ Add any custom code that you want executed with the config value just before object instanciation.

        Exemple

        >>> class MyCoolMPCComponent:
        >>>     def _config_pre_init_callback(self, config: Dict, subclass_config: Dict, signature_values_from_config:
        Dict) -> Dict:
        >>>     horizon: int = subclass_config['horizon']
        >>>     time_step: int = subclass_config['steps_per_prediction']
        >>>
        >>>     values_from_callback = {
        >>>         'sample_length':   int(horizon/time_step),
        >>>         'init_state':      np.zeros(config['environment']['observation_space']['shape'][0]),
        >>>         'input_dimension': config['environment']['input_space']['dim'],
        >>>         }
        >>>
        >>>     return values_from_callback

        :param config: the complete configuration dictionary
        :param subclass_config: the configuration dictionary with only the key:value of the subclass
        :param signature_values_from_config: the key:value computed from the callback
        """
        pass

    @abstractmethod
    def _config_post_init_callback(self, config: Dict) -> None:
        """ Add any custom code that you want executed with the config value just after object instanciation.

        Exemple

        >>> class MyCoolMPCComponent:
        >>>     def _config_post_init_callback(self, config: Dict, subclass_config: Dict, signature_values_from_config:
        Dict) -> None:
        >>>     try:
        >>>         if self._config['environment']['type'] == 'gym':
        >>>             self.env: gym_wrappers.time_limit.TimeLimit = gym_make(self._config['environment']['name'])
        >>>         else:
        >>>             raise NotImplementedError
        >>>     except AttributeError:
        >>>         pass
        >>>
        >>>     return None

        :param config: the complete configuration dictionary
        """
        pass

    @classmethod
    def config_init(cls, config: Dict, *args, **kwargs):
        """
        Alternative initialization method via configuration dictionary
        Return an instance of AbstractNominalPath

        :param config: a dictionary of configuration
        :param args: pass arbitrary argument to the baseclass init method
        :param kwargs: pass arbitrary keyword argument to the baseclass init method
        """
        subclass_config: Dict = config["hparam"][cls._subclass_config_key()]
        if (subclass_config is None) or (subclass_config == "None"):
            subclass_config = {}

        # ... Fetch subclass __init__ signature and corresponding value ................................................
        subclasse_init_param_list = list(cls.__init__.__code__.co_varnames)
        subclasse_init_param_list.remove("self")

        try:
            subclasse_init_param_list.remove("args")
        except ValueError as e:
            # No param `args` in the __init__ signature
            pass
        try:
            subclasse_init_param_list.remove("kwargs")
        except ValueError as e:
            # No param `kwargs` in the __init__ signature
            pass

        # ... Check for required base class parameter missing from the config ..........................................
        if len(cls._config_file_required_field()) > 0:
            assert set(cls._config_file_required_field()).issubset(set(subclass_config.keys())), (
                f"{cls.ERR_S()} There's required baseclass parameters missing in the config file >> Hint: execute `"
                f"{cls.__qualname__}._config_file_required_field()`"
            )

        try:
            signature_values_from_config = {
                param: value for param, value in subclass_config.items() if param in subclasse_init_param_list
            }
        except AttributeError as e:
            raise e
            signature_values_from_config = {}

        signature_values_from_callback = cls._config_pre_init_callback(
            cls,
            config=config,
            subclass_config=subclass_config,
            signature_values_from_config=signature_values_from_config,
        )

        kwargs.update(signature_values_from_config)
        try:
            kwargs.update(signature_values_from_callback)
        except TypeError as e:
            raise TypeError(
                f"{cls.ERR_S()} Something is wrong with the `{cls.__qualname__}._config_pre_init_callback()` return "
                f"value. Be sure to return a dict\n"
                f"{e}"
            ) from e

        # ... unaccounted parameter check ..............................................................................
        unaccounted_param = [each for each in subclasse_init_param_list if each not in kwargs]
        assert len(unaccounted_param) == 0, (
            f"{cls.ERR_S()} There's __init__ signature parameters unacounted by the config_init method >> "
            f"{unaccounted_param}"
        )

        instance = cls(*args, **kwargs)
        instance._config = config
        instance._config_post_init_callback(config=config)

        return instance
