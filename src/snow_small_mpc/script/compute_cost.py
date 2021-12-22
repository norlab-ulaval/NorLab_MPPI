import copy

import numpy as np
import math

def mpc_compute_cost(cost_map, pool):
    for traj in pool:
