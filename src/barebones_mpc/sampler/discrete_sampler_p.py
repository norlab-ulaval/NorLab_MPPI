from src.barebones_mpc.model.abstract_model import AbstractModel
from src.barebones_mpc.sampler.abstract_sampler import AbstractSampler
import numpy as np
from typing import List, Type


class DiscreteSamplerP(AbstractSampler):
    def __init__(
        self,
        model: Type[AbstractModel],
        number_samples: int,
        input_dimension: int,
        sample_length: int,
        init_state: np.ndarray,
        input_type,
        input_space,
        nominal_input_change_probability: float,
    ):
        super().__init__(
            model=model,
            number_samples=number_samples,
            input_dimension=input_dimension,
            sample_length=sample_length,
            init_state=init_state,
            input_type=input_type,
            input_space=input_space,
        )

        assert (
            self.input_type == "discrete"
        ), f"{self.NAMED_ERR()}  Wrong input_space type! Check the config file"

        self.nominal_input_change_probability = nominal_input_change_probability
        self.sample_input = np.empty(
            (self.sample_length, self.number_samples + 1, self.input_dimension),
            dtype=np.int,
        )

    @classmethod
    def _specialized_config_required_fields(cls) -> List[str]:
        required_field: List[str] = super()._specialized_config_required_fields()
        required_field.extend(["nominal_input_change_probability"])
        return required_field

    def sample_inputs(self, nominal_input):
        """ Sample inputs based on the nominal input array

        :param nominal_input: the nominal input array
        :return: sample input array
        """
        # self.sample_input[:, 0, :] = np.expand_dims(nominal_input, axis=1)
        self.sample_input[:, 0, :] = nominal_input
        self.sample_input[:, 1:, :] = np.tile(
            nominal_input.reshape(self.sample_length, 1, self.input_dimension), (1, self.number_samples, 1)
            )

        # (CRITICAL) ToDo:implement >> input space constraint
        sample_input_change_map = np.random.binomial(
            n=1, p=self.nominal_input_change_probability, size=self.sample_input.shape
        )
        sample_input_change_map[:, 0, :] = 0

        sample_input_change_map = sample_input_change_map.ravel()
        sample_input_ = self.sample_input.ravel()

        sample_input_[sample_input_change_map == 1] += 1
        sample_input_[sample_input_ == 2] = 0
        self.sample_input = sample_input_.reshape(self.sample_input.shape)

        return self.sample_input

    def sample_states(self, sample_input, init_state):
        """ Sample states based on the sample input array through the model

        use self.model to sample states

        :param sample_input: the sampling input
        :param init_state: the initial state array
        :return: sample state array
        """
        sample_states = self.model.predict_states(init_state, sample_input)
        return sample_states
