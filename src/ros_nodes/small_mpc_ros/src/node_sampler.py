import numpy as np

SAMPLING_STANDARD_DEVIATION = 1.5

class Sampler:
    def __init__(self, nb_samples, nb_steps):
        self.nb_samples = nb_samples
        self.nb_steps = nb_steps

    def generate_commands(self, nominal_commands):
        commands = np.empty((self.nb_samples, self.nb_steps, 2))
        for i in range(self.nb_samples):
            trajectory_standard_deviation = np.random.normal(loc=0.0, scale=SAMPLING_STANDARD_DEVIATION, size=(1, nominal_commands.shape[1]))
            commands[i, :, :] = nominal_commands.transpose()
            commands[i, :, 1] += trajectory_standard_deviation.reshape((commands.shape[1],))
        return commands
