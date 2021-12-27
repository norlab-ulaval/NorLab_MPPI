# coding=utf-8
import copy

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.spatial import KDTree
from typing import Any, Callable, List, Tuple

# (CRITICAL) todo: might be better to ditch this module and start from scratch (!)

# todo:refactor compute_nominal_path >> parameters should be expressed as `trj_so_far`, `n_steps_foward`
# todo:refactor compute_nominal_path >> with turn radius constraint: v_x_c/omega >= r_min
def compute_nominal_path(x_init, n_samples, n_steps, v_x_c, dt):
    """

    :param x_init:
    :param n_samples:
    :param n_steps:
    :param v_x_c:
    :param dt:
    :return:
    """
    curr_x = x_init
    u_nom = np.zeros((n_samples, n_steps, 2))

    # ... Nominal command sampling .....................................................................................
    u_nom[:, :, 0] = v_x_c  # linear speed
    u_nom[:, :, 1] = -np.pi/2  # angular speed

    R_init = np.eye(3)

    def update_R(R, theta):
        R[0, 0] = R[1, 1] = np.cos(theta)
        sin_theta = np.sin(theta)
        R[0, 1] = -sin_theta
        R[1, 0] = sin_theta
        return R

    traj_nom = []
    for timestep in range(0, n_steps):
        nom_cost = 0
        R = update_R(R_init, curr_x[2])

        dx = u_nom[:, timestep, :]
        new_x = R.T@dx*dt + curr_x
        traj_nom.append(new_x)
        curr_x = new_x
    traj_nom = np.transpose(np.asarray(traj_nom))
    return traj_nom, u_nom


def heading(trajs: np.ndarray, ts):
    """
    :param trajs: the full 3D array compose of (n_samples trj, n_steps, [x_s, y_s, theta])
    :return: The computed the heading in the world frame
    """
    thetas = trajs.view()[:, ts, 2]

    return np.vstack(np.cos(thetas), np.sin(thetas))


def sample_trajs(u_nom, x_init, n_samples, n_steps, std_dev_cmd, min_cost, update_R, dt, ref_tree):
    R_init = np.eye(3)
    pool = []
    cost_list = []
    min_cost_traj = -1

    for j in range(0, n_samples):

        # # ❯❯❯ Trj command sampling ❯❯❯ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        u_std_dev = np.random.normal(loc=0.0, scale=std_dev_cmd, size=(1, n_steps))
        u_s = copy.deepcopy(u_nom)
        u_s[2, :] = u_s[2, :] + u_std_dev
        #  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .❮❮❮ Trj command sampling ❮❮❮

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


def mpc_sampler_optimized(dt: float, n_samples: int, n_steps: int, std_dev_cmd: float, v_x_c: float,
                          x_init: np.ndarray = None) -> Tuple[
    ndarray, ndarray, ndarray, List[ndarray], int]:
    """
    Most barebone version of MPC sampler
    :param x_init:
    :param dt:
    :param n_samples:
    :param n_steps:
    :param std_dev_cmd:
    :param v_x_c:
    :return:
    """
    # /// create ref traj /////////////////////////////////////////////////////////////////////////////////////////////
    df = pd.read_csv('src/snow_small_mpc/data/boreal_smooth.csv', header=None)
    # xs = df['Points:1']
    xs = df[0]
    # ys = df['Points:0']
    ys = df[1]
    ref_traj = np.array([[x, y] for x, y in zip(xs, ys)])

    ref_tree = KDTree(ref_traj)  # create kdtree from traj

    if x_init is None:
        x_init = np.array([0.0, 0.0, 0.0])

    # /// initialize ///////////////////////////////////////////////////////////////////////////////////////////////////

    # ... Trajectory pool init .........................................................................................
    trajs = np.empty((n_samples, n_steps, 3), dtype=np.float)

    trajs[:, 0, 0].fill(x_init[0])
    trajs[:, 0, 1].fill(x_init[1])
    trajs[:, 0, 2].fill(x_init[2])

    # ... Init trajs commands ..........................................................................................

    trajs_u = np.empty((n_samples, n_steps, 2), dtype=np.float)
    trajs_u[:, :, 0].fill(v_x_c)
    u_std_dev = np.random.normal(loc=0.0, scale=std_dev_cmd, size=(1, n_steps))
    trajs_u[:, :, 1] = trajs_u[:, :, 1] + u_std_dev


    # /// Nominal path /////////////////////////////////////////////////////////////////////////////////////////////////
    traj_nom, u_nom = compute_nominal_path(x_init, n_samples, n_steps, v_x_c, dt)


    dd, _ = ref_tree.query(traj_nom[:2, :].T, k=1,
                           # workers=-1,
                           )
    nom_cost = np.sum(dd)


    # %timeit sample_trajs(u_nom, x_init, n_samples, n_steps, std_dev_cmd, nom_cost)
    pool, min_cost_traj, cost = sample_trajs(u_nom, x_init, n_samples, n_steps, std_dev_cmd, nom_cost)
    return ref_traj, x_init, traj_nom, pool, min_cost_traj
