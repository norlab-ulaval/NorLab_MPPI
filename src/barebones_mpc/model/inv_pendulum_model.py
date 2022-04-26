from typing import List

from src.barebones_mpc.model.abstract_model import AbstractModel
import numpy as np


class InvPendulumModel(AbstractModel):
    def __init__(
        self,
        prediction_step: int,
        number_samples: int,
        sample_length: int,
        state_dimension: int,
        cart_mass: int,
        pendulum_mass: int,
    ):
        """ Inverted pendulum model. The states are [y, v, theta, q], input is [u]
        y: cart position
        v: cart velocity
        theta: pendulum angle
        q: pendulum angle rate of change
        u: force applied to the cart

        """
        super().__init__(
            prediction_step=prediction_step,
            number_samples=number_samples,
            sample_length=sample_length,
            state_dimension=state_dimension,
        )

        self.prediction_step = prediction_step
        self.number_samples = number_samples
        self.sample_length = sample_length

        self.input_dimension = 1
        self.state_dimension = state_dimension

        self.cart_mass = cart_mass
        self.pendulum_mass = pendulum_mass
        self.epsilon = pendulum_mass / (cart_mass + pendulum_mass)
        print("::: InvPendulumModel.epsilon", self.epsilon)

        self.state_transition_matrix = np.array(
            [  # fmt: skip
                [0, 1, 0, 0],
                [0, 0, -self.epsilon, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ]
        )

        self.input_transition = np.array([0, 1, 0, -1]).reshape((4, 1))

        self.sample_states = np.empty(
            (self.sample_length + 1, self.number_samples, self.state_dimension)
        )

    @classmethod
    def _specialized_config_required_fields(cls) -> List[str]:
        required_field: List[str] = super()._specialized_config_required_fields()
        required_field.extend(["cart_mass", "pendulum_mass"])
        return required_field

    def predict_states(self, init_state, sample_input):
        """ predicts a sample state array based on a nominal input array

        :param init_state: initial state array
        :param sample_input: sample input array
        :return: sample state array
        """
        self.sample_states[0, :, :] = init_state
        for j in range(0, self.number_samples):
            for i in range(1, self.sample_length + 1):
                self.sample_states[i, j, :] = self._predict(
                    self.sample_states[i - 1, j, :], sample_input[i - 1, j, :]
                )[:, 0]

        return self.sample_states

    def _predict(self, init_state, initial_input):
        """ makes a single state prediction based on initial state and input

        :param init_state: initial state array
        :param initial_input: input array
        :return: predicted state array
        """
        init_state = init_state.reshape((self.state_dimension, 1))
        state_diff = (
            self.state_transition_matrix @ init_state
            + self.input_transition * initial_input
        )

        return init_state + state_diff * self.prediction_step
