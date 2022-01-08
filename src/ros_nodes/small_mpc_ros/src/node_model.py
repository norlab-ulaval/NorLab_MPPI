import numpy as np


class Model:
    def __init__(self, rate):
        self.rotation_matrix = np.eye(3)
        self.time_step = 1.0 / rate

    def _update_rotation_matrix(self, yaw):
        self.rotation_matrix[0, 0] = self.rotation_matrix[1, 1] = np.cos(yaw)
        sin_theta = np.sin(yaw)
        self.rotation_matrix[0, 1] = -sin_theta
        self.rotation_matrix[1, 0] = sin_theta

    def simulate(self, current_state, sampled_commands):
        commands = np.empty((sampled_commands.shape[0], sampled_commands.shape[1], 3))
        commands[:, :, 0] = sampled_commands[:, :, 0]
        commands[:, :, 1] = 0
        commands[:, :, 2] = sampled_commands[:, :, 1]

        trajectories = np.zeros((commands.shape[0], commands.shape[1] + 1, commands.shape[2]))
        trajectories[:, 0, 0].fill(current_state[0])
        trajectories[:, 0, 1].fill(current_state[1])
        trajectories[:, 0, 2].fill(current_state[2])

        for i in range(trajectories.shape[0]):
            for j in range(1, trajectories.shape[1]):
                self._update_rotation_matrix(commands[i, j - 1, 2])
                trajectories[i, j, :] = trajectories[i, j - 1, :] + self.rotation_matrix.transpose() @ \
                                        commands[i, j - 1, :] * self.time_step
        return trajectories
