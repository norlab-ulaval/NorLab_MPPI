# coding=utf-8

import pytest
import numpy as np
from src.snow_small_mpc.script.sampler_optimized import mpc_sampler_optimized

import pandas as pd
import matplotlib.pyplot as plt


# (Priority) todo:fixme!!

@pytest.fixture
def ref_traj_1():
    dt = 1/20  # 20 hz or 0.2 seconds
    v_x_c = 1.5  # m/s
    horizon = 0.75  # s
    n_steps = int(horizon/dt)
    n_samples = 1000
    std_dev_cmd = 1.5  # rad/s

    ref_traj_path = 'src/snow_small_mpc/data/boreal_smooth.csv'
    df = pd.read_csv(ref_traj_path, header=None)
    # xs = df['Points:1']
    xs = df[0]
    # ys = df['Points:0']
    ys = df[1]
    ref_traj = np.array([[x, y] for x, y in zip(xs, ys)])

    return dt, v_x_c, horizon, n_steps, n_samples, std_dev_cmd, ref_traj


def test_mpc_sampler_optimized_init(ref_traj_1):
    dt, v_x_c, horizon, n_steps, n_samples, std_dev_cmd, ref_traj = ref_traj_1

    _, x_init, traj_nom, pool, min_cost_traj = mpc_sampler_optimized(dt, n_samples, n_steps, std_dev_cmd, v_x_c)

    return None


def test_mpc_sampler_optimized_plot(ref_traj_1):
    dt, v_x_c, horizon, n_steps, n_samples, std_dev_cmd, ref_traj = ref_traj_1

    _, x_init, traj_nom, pool, min_cost_traj = mpc_sampler_optimized(dt, n_samples, n_steps, std_dev_cmd, v_x_c)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.plot(ref_traj[:15, 1], ref_traj[:15, 0], c='C1', linewidth=5, label='Ref. traj.')

    for i in range(0, n_samples):
        if i == min_cost_traj:
            ax.plot(pool[i][1, :], pool[i][0, :], linewidth=5, label='Selected traj.')
        else:
            ax.plot(pool[i][1, :], pool[i][0, :], linewidth=1, alpha=0.05)

    ax.plot(traj_nom[1, :], traj_nom[0, :], linewidth=5, c='C0', label='Nominal traj.')
    ax.scatter(x_init[1], x_init[0], c='C2', label='Init. pose')

    ax.set_aspect('equal')
    ax.set_xlabel('y_g')
    ax.set_ylabel('x_g')
    ax.legend()

    plt.show()

    return None


# def test_fail():
#    raise AssertionError


def test_pass():
    assert 1 == 1
