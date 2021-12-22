import copy

import numpy as np
import pandas as pd
import math

from scipy.spatial import KDTree

def mpc_costmap(traj_path, res, extra_dim, cost_gain):
    # /// create ref traj //////////////////////////////////////////////////////////////////////////////////////////////
    df = pd.read_csv(traj_path, header=None)
    # xs = df['Points:1']
    xs = df[0]
    # ys = df['Points:0']
    ys = df[1]
    ref_traj = np.array([[x, y] for x, y in zip(xs, ys)])
    ref_tree = KDTree(ref_traj)

    # /// Define costmap dimensions //////////////////////////////////////////////////////////////////////////////////////////////
    traj_max_x = int(math.ceil(np.max(ref_traj[:, 0])))
    traj_min_x = int(math.ceil(np.min(ref_traj[:, 0])))

    traj_max_y = int(math.ceil(np.max(ref_traj[:, 1])))
    traj_min_y = int(math.ceil(np.min(ref_traj[:, 1])))

    traj_dim_array = np.array([traj_min_x, traj_max_x, traj_min_y, traj_max_y])

    print('x_dim : ', traj_max_x, traj_min_x)
    print('y_dim : ', traj_max_y, traj_min_y)

    # /// Create costmap array //////////////////////////////////////////////////////////////////////////////////////////////

    max_dir = np.argmax(np.abs(traj_dim_array))
    max_dim = traj_dim_array[max_dir] + extra_dim
    n_cells = math.ceil(max_dim * 2 / res)

    costmap_extent= -max_dim, max_dim, -max_dim, max_dim

    costmap = np.zeros((n_cells, n_cells))

    print('costmap shape : ', costmap.shape)
    print('max_dir : ', max_dir)
    print(costmap_extent[max_dir])

    # /// Assign costs //////////////////////////////////////////////////////////////////////////////////////////////

    for i in range(0, costmap.shape[0]):
        x_cell = max_dim - (i * res + (res / 2))
        for j in range(0, costmap.shape[1]):
            y_cell = j * res + (res/2) - max_dim
            cell_cost, _ = ref_tree.query(np.array([x_cell, y_cell]), k=1)
            costmap[i, j] = cost_gain * cell_cost**0.5

    return costmap, costmap_extent, ref_traj


