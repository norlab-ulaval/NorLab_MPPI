from typing import Dict, List, Union

from src.barebones_mpc.evaluator.abstract_evaluator import AbstractEvaluator
import numpy as np


class QuadraticEvaluator(AbstractEvaluator):
    def __init__(
        self,
        number_samples: int,
        input_dimension: int,
        sample_length: int,
        state_dimension: int,
        std_dev: Union[float, np.ndarray],
        beta: int,
        inverse_temperature: float,
        state_weights: Union[List, np.ndarray, float],
        reference_state: np.ndarray = None,
    ):
        super().__init__(
            number_samples=number_samples,
            input_dimension=input_dimension,
            sample_length=sample_length,
            state_dimension=state_dimension,
        )

        self.std_dev = std_dev  # TODO : Define std_Dev for multiple dimensions of input

        self.input_covariance = np.empty((self.input_dimension, self.input_dimension))
        np.fill_diagonal(self.input_covariance, self.std_dev)
        self.input_covariance_inverse = np.linalg.inv(self.input_covariance)
        self.sample_costs = np.full(shape=(self.sample_length, self.number_samples), fill_value=np.infty)
        self.sample_total_costs = np.full(shape=(1, self.number_samples), fill_value=np.infty)

        if type(state_weights) is float:
            state_weights = np.full(shape=(state_dimension,), fill_value=state_weights)
        elif type(state_weights) is list:
            state_weights = np.array(state_weights)

        self.state_weights = np.zeros((self.state_dimension, self.state_dimension))
        np.fill_diagonal(self.state_weights, state_weights)
        self.final_state_cost = np.zeros((1, self.number_samples))
        self.beta = np.array([[beta]])
        self.inverse_temperature = inverse_temperature
        self.half_inverse_temperature = inverse_temperature / 2

        if reference_state is None:
            reference_state = np.zeros(shape=(state_dimension,))
        self.reference_state = reference_state

    @classmethod
    def _specialized_config_required_fields(cls) -> List[str]:
        required_field: List[str] = super()._specialized_config_required_fields()
        required_field.extend(["std_dev",
                               "beta",
                               "inverse_temperature",
                               "state_weights"
                               ])
        return required_field

    def _config_pre__init__callback(
        self, config: Dict, specialized_config: Dict, init__signature_values_from_config: Dict
    ) -> Dict:
        values_from_callback: dict = super()._config_pre__init__callback(
            self,
            config=config,
            specialized_config=specialized_config,
            init__signature_values_from_config=init__signature_values_from_config,
        )

        values_from_callback.update(
            {
                # "state_weights": subclass_config["state_weights"], # ToDo:investigate??
                "reference_state": np.zeros(shape=(values_from_callback["state_dimension"]))
            }
        )
        return values_from_callback

    def compute_sample_costs(self, sample_input: np.ndarray, sample_states: np.ndarray) -> None:
        """ computes the cost related to every sample

        :param sample_input: sample input array
        :param sample_states: sample state array
        :return None
        """
        for j in range(0, self.number_samples):
            for i in range(0, self.sample_length):
                self.sample_costs[i, j] = self._compute_state_cost(
                    sample_states[i, j, :], self.reference_state
                ) + self._compute_input_cost(sample_input[i, j, :])
            self.sample_total_costs[0, j] = np.sum(self.sample_costs[:, j])

        return None

    def _compute_input_cost(self, input_array: np.ndarray) -> float:
        """ computes a single input cost via a quadratic input cost

        :param input_array: single input array
        :return input_cost: input cost
        """
        cost_array = self.half_inverse_temperature * (
                input_array.transpose()@self.input_covariance_inverse@input_array + self.beta.transpose()@input_array
        )
        return cost_array[0]

    def _compute_state_cost(self, state: np.ndarray, reference: np.ndarray) -> float:
        """ compute a single state cost via a quadartic state cost

        :param state: single state array
        :return state_cost: state cost
        """
        error = state - reference
        state_cost = error.transpose()@self.state_weights@error
        return state_cost

    def compute_final_state_cost(self, final_state: np.ndarray) -> float:
        """ compute a final state cost

        :param final_state: final state array
        :return final_state_cost: final state cost
        """
        pass

    def get_trajectories_cost(self):
        return self.sample_costs

    def get_trajectories_cumulative_cost(self):
        return self.sample_total_costs
