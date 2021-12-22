# coding=utf-8

import matplotlib.pyplot as plt

# /// set parameters ///////////////////////////////////////////////////////////////////////////////////////////////////
from src.snow_small_mpc.script.sampler import mpc_sampler

dt = 1/20  # 20 hz or 0.2 seconds
v_x_c = 1.5  # m/s
horizon = 0.75  # s
n_steps = int(horizon/dt)
n_samples = 1000
std_dev_cmd = 1.5  # rad/s

ref_traj, x_init, traj_nom, pool, min_cost_traj = mpc_sampler(dt, n_samples, n_steps, std_dev_cmd, v_x_c)

# /// plot trajectories ////////////////////////////////////////////////////////////////////////////////////////////////
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
