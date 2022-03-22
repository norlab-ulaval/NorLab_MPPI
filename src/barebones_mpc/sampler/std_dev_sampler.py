from src.barebones_mpc.model.abstract_model import AbstractModel
from src.barebones_mpc.sampler.abstract_sampler import AbstractSampler
import numpy as np
from typing import List, Type


class StandardDevSampler(AbstractSampler):
    def __init__(
        self,
        model: Type[AbstractModel],
        number_samples: int,
        input_dimension: int,
        sample_length: int,
        init_state: np.ndarray,
        input_type,
        input_space,
        std_dev: float,
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

        self.std_dev = std_dev  # TODO : Define std_Dev for multiple dimensions of input
        self.sample_input = np.empty((self.sample_length, self.number_samples + 1, self.input_dimension))

    @classmethod
    def _specialized_config_required_fields(cls) -> List[str]:
        required_field: List[str] = super()._specialized_config_required_fields()
        required_field.extend(["std_dev"])
        return required_field

    def sample_inputs(self, nominal_input):
        """ Sample inputs based on the nominal input array

        :param nominal_input: the nominal input array
        :return: sample input array
        """
        # self.sample_input[:, 0, :] = np.expand_dims(nominal_input, axis=1)
        self.sample_input[:, 0, :] = nominal_input

        # (CRITICAL) ToDo:implement >> input space type: continuous case and discrete case
        # (CRITICAL) ToDo:implement >> input space constraint
        if self.input_type == "discrete":
            self.sample_input = np.random.randint(
                low=self.input_space[0], high=self.input_space[1], size=self.sample_input.shape
            )

            # self.sample_input = np.rint(self.sample_input).astype(int)
        elif self.input_type == "continuous":
            sample_noise = np.random.normal(
                loc=0, scale=self.std_dev, size=(self.sample_length, self.number_samples, self.input_dimension)
            )

            self.sample_input[:, 1:, :] = sample_noise + np.tile(
                nominal_input.reshape(self.sample_length, 1, self.input_dimension), (1, self.number_samples, 1)
            )

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


