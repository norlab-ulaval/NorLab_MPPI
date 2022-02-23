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


class MockSelector(AbstractSelector):
    """ For testing purpose only"""

    def select_next_input(self, sample_input, sample_state, sample_cost):
        pass
