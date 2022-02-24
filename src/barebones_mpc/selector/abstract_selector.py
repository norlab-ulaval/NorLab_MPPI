from abc import ABCMeta, abstractmethod


class AbstractSelector(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def select_next_input(self, sample_input, sample_state, sample_cost):
        """ select the optimal next input and state arrays

        :param sample_input: sample input array
        :param sample_state: sample state array
        :param sample_cost: sample cost array
        :return new nominal input and nominal state arrays
        """
        pass


class MockSelector(AbstractSelector):
    """ For testing purpose only"""

    def __init__(self):
        super().__init__()
        import gym
        self.env = gym.make('Pendulum-v1')

    def select_next_input(self, sample_input, sample_state, sample_cost):
        return self.env.action_space.sample(), True
