# coding=utf-8
import copy

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.spatial import KDTree
from typing import Any, Callable, List, Tuple


def mpc_sampler_optimized(dt: float, n_samples: int, n_steps: int, std_dev_cmd: float, v_x_c: float) -> Tuple[
    ndarray, ndarray, ndarray, List[ndarray], int]:
    # /// create ref traj //////////////////////////////////////////////////////////////////////////////////////////////
    df = pd.read_csv('src/snow_small_mpc/data/boreal_smooth.csv', header=None)
    # xs = df['Points:1']
    xs = df[0]
    # ys = df['Points:0']
    ys = df[1]
    ref_traj = np.array([[x, y] for x, y in zip(xs, ys)])
    # print(ref_traj)
    # create kdtree from traj
    ref_tree = KDTree(ref_traj)

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

            dd, _ = ref_tree.query(traj[:2, :].T, k=1,
                                   # workers=-1,
                                   )
            cost = np.sum(dd)

            if j > min_cost_traj and cost < min_cost:
                min_cost_traj = j
                min_cost = cost
        return pool, min_cost_traj, cost

    # /// initialize state /////////////////////////////////////////////////////////////////////////////////////////////
    x_init = np.array([0.0, 0.0, 0.0])
    curr_x = x_init
    u_nom = np.zeros((3, n_steps))
    u_nom[0, :] = v_x_c
    u_nom[2, :] = -np.pi/2
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
    dd, _ = ref_tree.query(traj_nom[:2, :].T, k=1,
                           # workers=-1,
                           )
    nom_cost = np.sum(dd)
    # %timeit sample_trajs(u_nom, x_init, n_samples, n_steps, std_dev_cmd, nom_cost)
    pool, min_cost_traj, cost = sample_trajs(u_nom, x_init, n_samples, n_steps, std_dev_cmd, nom_cost)

    return ref_traj, x_init, traj_nom, pool, min_cost_traj
