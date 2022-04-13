# coding=utf-8

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List


class AbstractModelPredictiveControlComponent(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = None

    @classmethod
    def NAMED_ERR(cls) -> str:
        return f"({cls.__qualname__} ERROR): "

    @classmethod
    @abstractmethod
    def _specialized_config_key(cls) -> str:
        """ The configuration file dictionary key pointing to the subclass component

        Exemple

        >>> class MyCoolMPCComponent:
        >>>     @classmethod
        >>>     def _specialized_config_key(cls) -> str:
        >>>         return 'sampler_hparam'

        """
        pass

    @classmethod
    @abstractmethod
    def _specialized_config_required_fields(cls) -> List[str]:
        """ The list of paramenters from the subclass __init__ signature

        Exemple

        >>> class MyCoolMPCComponent:
        >>>    @classmethod
        >>>    def _specialized_config_required_fields(cls) -> List[str]:
        >>>        return ['number_samples', 'sample_length', 'input_dimension']

        """
        pass

    @abstractmethod
    def _config_pre__init__callback(
        self, config: Dict, specialized_config: Dict, init__signature_values_from_config: Dict
    ) -> Dict:
        """ Configuration file computation executed PRIOR to instanciation.
        Add any custom code that you want executed with the config value just before object instanciation.

        Exemple

        >>> class MyCoolMPCComponent:
        >>>     def _config_pre__init__callback(self, config: Dict, subclass_config: Dict, signature_values_from_config:
        Dict) -> Dict:
        >>>     horizon: int = specialized_config['horizon']
        >>>     prediction_step: int = specialized_config['prediction_step']
        >>>
        >>>     values_from_callback = {
        >>>         'sample_length':   int(horizon/prediction_step),
        >>>         'init_state':      np.zeros(config['environment']['observation_space']['shape'][0]),
        >>>         'input_dimension': config['environment']['input_space']['dim'],
        >>>         }
        >>>
        >>>     return values_from_callback

        :param config: the complete configuration dictionary
        :param specialized_config: the configuration dictionary with only the key:value of the subclass
        :param init__signature_values_from_config: the key:value computed from the callback
        """
        pass

    @abstractmethod
    def _config_post__init__callback(self, config: Dict) -> None:
        """ Configuration file computation executed AFTER instanciation.
        Add any custom code that you want executed with the config value just after object instanciation.

        Exemple

        >>> class MyCoolMPCComponent:
        >>>     def _config_post__init__callback(self, config: Dict, subclass_config: Dict,
        signature_values_from_config:
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
        Alternative initialization method via configuration dictionary.
        Return a subclass instance of AbstractModelPredictiveControlComponent.

        :param config: a dictionary of configuration
        :param args: pass arbitrary argument to the baseclass init method
        :param kwargs: pass arbitrary keyword argument to the baseclass init method
        """

        # ... Fetch subclass __init__ signature and corresponding value ................................................
        init__signature = list(cls.__init__.__code__.co_varnames)
        init__signature.remove("self")

        try:
            init__signature.remove("args")
        except ValueError as e:
            # No param `args` in the __init__ signature
            pass
        try:
            init__signature.remove("kwargs")
        except ValueError as e:
            # No param `kwargs` in the __init__ signature
            pass

        # ... Check for required specialized parameter missing from the config .........................................
        specialized_config: Dict = config["hparam"][cls._specialized_config_key()]
        if (specialized_config is None) or (specialized_config == "None"):
            specialized_config = {}

        if len(cls._specialized_config_required_fields()) > 0:
            assert set(cls._specialized_config_required_fields()).issubset(set(specialized_config.keys())), (
                f"{cls.NAMED_ERR()} There's required init parameters missing in the config file >> Hint: execute `"
                f"{cls.__qualname__}._specialized_config_required_fields()`"
            )

        # ... Fetch specialized config value ...........................................................................
        try:
            init__signature_values_from_config = {
                param: value for param, value in specialized_config.items() if param in init__signature
            }
        except AttributeError as e:
            raise e
            init__signature_values_from_config = {}

        # ... Execute pre __init__ callback ............................................................................
        init__signature_values_from_pre_callback = cls._config_pre__init__callback(
            cls,
            config=config,
            specialized_config=specialized_config,
            init__signature_values_from_config=init__signature_values_from_config,
        )

        kwargs.update(init__signature_values_from_config)
        try:
            kwargs.update(init__signature_values_from_pre_callback)
        except TypeError as e:
            raise TypeError(
                f"{cls.NAMED_ERR()} Something is wrong with the `{cls.__qualname__}._config_pre__init__callback()` "
                f"return "
                f"value. Be sure to return a dict\n"
                f"{e}"
            ) from e

        # ... unaccounted parameter check ..............................................................................
        unaccounted_param = [each for each in init__signature if each not in kwargs]
        assert len(unaccounted_param) == 0, (
            f"{cls.NAMED_ERR()} There's __init__ signature parameters unacounted by the config_init method >> "
            f"{unaccounted_param}"
        )

        # ... Instanciate MPC controler component ......................................................................
        instance = cls(*args, **kwargs)
        instance._config = config
        instance._config_post__init__callback(config=config)

        return instance
