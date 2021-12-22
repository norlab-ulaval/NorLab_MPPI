# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# /// set parameters ///////////////////////////////////////////////////////////////////////////////////////////////////
from src.snow_small_mpc.script.sampler import mpc_sampler
from src.snow_small_mpc.script.cost_map import mpc_costmap

dt = 1/20  # 20 hz or 0.2 seconds
v_x_c = 1.5  # m/s
horizon = 0.75  # s
n_steps = int(horizon/dt)
n_samples = 1000
std_dev_cmd = 1.5  # rad/s
costmap_res = 0.1 # m
extra_dim = 20 # m
cost_gain = 100

ref_traj_path = 'src/snow_small_mpc/data/boreal_smooth.csv'

# /// create ref traj //////////////////////////////////////////////////////////////////////////////////////////////
df = pd.read_csv(ref_traj_path, header=None)
# xs = df['Points:1']
xs = df[0]
# ys = df['Points:0']
ys = df[1]
ref_traj = np.array([[x, y] for x, y in zip(xs, ys)])

update_R, x_init, traj_nom, pool, min_cost_traj = mpc_sampler(dt, n_samples, n_steps, std_dev_cmd, v_x_c)
# cost_map, map_extent, ref_traj = mpc_costmap(ref_traj_path, costmap_res, extra_dim, cost_gain)
## save costmap to increase compute time
# np.save('src/snow_small_mpc/data/boreal_smooth_costmap', cost_map)
cost_map = np.load('src/snow_small_mpc/data/boreal_smooth_costmap.npy')
map_extent = -51, 51, -51, 51

# /// plot trajectories ////////////////////////////////////////////////////////////////////////////////////////////////
fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(ref_traj[:, 1], ref_traj[:, 0], c='C1', linewidth=5, label='Ref. traj.')

ax.imshow(cost_map, extent=map_extent)

#
# for i in range(0, n_samples):
#     if i == min_cost_traj:
#         ax.plot(pool[i][1, :], pool[i][0, :], linewidth=5, label='Selected traj.')
#     else:
#         ax.plot(pool[i][1, :], pool[i][0, :], linewidth=1, alpha=0.05)
#
# ax.plot(traj_nom[1, :], traj_nom[0, :], linewidth=5, c='C0', label='Nominal traj.')
# ax.scatter(x_init[1], x_init[0], c='C2', label='Init. pose')
#
ax.set_aspect('equal')
ax.set_xlabel('y_g')
ax.set_ylabel('x_g')
ax.legend()

plt.show()