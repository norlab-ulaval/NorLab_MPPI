# coding=utf-8
import copy

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def mpc_sampler(dt, n_samples, n_steps, std_dev_cmd, v_x_c):

    # /// Utility function /////////////////////////////////////////////////////////////////////////////////////////////
    def update_R(R, theta):
        R[0, 0] = R[1, 1] = np.cos(theta)
        sin_theta = np.sin(theta)
        R[0, 1] = -sin_theta
        R[1, 0] = sin_theta
        return R

    def sample_trajs(u_nom, x_init, n_samples, n_steps, std_dev_cmd, min_cost):
        R_init = np.eye(3)
        pool = []
        cost_list = []
        min_cost_traj = -1

        for j in range(0, n_samples):
            u_std_dev = np.random.normal(loc=0.0, scale=std_dev_cmd, size=(1, n_steps))
            u_s = copy.deepcopy(u_nom)
            u_s[2, :] = u_s[2, :] + u_std_dev

            traj = []

            curr_x = x_init

            for i in range(0, n_steps):
                R = update_R(R_init, curr_x[2])
                dx = u_s[:, i]
                new_x = R.T@dx*dt + curr_x
                #             cost = cost + dd
                traj.append(new_x)
                curr_x = new_x

            traj = np.transpose(np.asarray(traj))
            pool.append(traj)

            # dd, _ = ref_tree.query(traj[:2, :].T, k=1,
                                   # workers=-1,
                                   # )
            # cost = np.sum(dd)

            if j > min_cost_traj and cost < min_cost:
                min_cost_traj = j
                min_cost = cost
        return pool, min_cost_traj, cost

    # /// initialize state /////////////////////////////////////////////////////////////////////////////////////////////
    x_init = np.array([0.0, 0.0, 0.0])
    curr_x = x_init

    # ❯❯❯ Nominal trajectory ❯❯❯ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # (!) As discuss with Luc and Simon, the MPC box should receive in input a nominal trajectory express in therme of
    #  state only otherwise, the problem of fiding the optimal command u for t+1 would be solved already.
    u_nom = np.zeros((3, n_steps))
    u_nom[0, :] = v_x_c
    u_nom[2, :] = -np.pi/2

    #  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .❮❮❮ Nominal trajectory ❮❮❮

    traj_nom = []
    R_init = np.eye(3)
    for i in range(0, n_steps):
        nom_cost = 0
        R = update_R(R_init, curr_x[2])
        dx = u_nom[:, i]
        new_x = R.T@dx*dt + curr_x
        traj_nom.append(new_x)
        curr_x = new_x
    traj_nom = np.transpose(np.asarray(traj_nom))
    # dd, _ = ref_tree.query(traj_nom[:2, :].T, k=1,
                           # workers=-1,
    #                        )
    # nom_cost = np.sum(dd)
    # %timeit sample_trajs(u_nom, x_init, n_samples, n_steps, std_dev_cmd, nom_cost)
    pool, min_cost_traj, cost = sample_trajs(u_nom, x_init, n_samples, n_steps, std_dev_cmd, nom_cost)

    return update_R, x_init, traj_nom, pool, min_cost_traj
