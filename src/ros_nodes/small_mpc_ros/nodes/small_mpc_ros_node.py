#! /usr/bin/env python3.6

import rospy
import pandas as pd
import numpy as np
import scipy.spatial.kdtree
import math
from nav_msgs.msg import Odometry
from threading import Thread, Lock
import time
from src.ros_nodes.small_mpc_ros.src.node_sampler import Sampler
from src.ros_nodes.small_mpc_ros.src.node_model import Model
import matplotlib.pyplot as plt


RATE = 20.0
TIME_HORIZON = 0.75
NB_STEPS = int(TIME_HORIZON * RATE)
LINEAR_VELOCITY = 1.5
NB_SAMPLES = 10
COST_MAP_RESOLUTION = 1
COST_MAP_BUFFER = 20
COST_FUNCTION_GAIN = 100
# TRAJECTORY_FILE_NAME = "/media/dominic/norlab_dominic/rosbags/snow/fm_12_2021/slide.csv"
TRAJECTORY_FILE_NAME = "samples/rosbags/snow/fm_12_2021/slide.csv"

state = None
state_lock = Lock()


def quaternion_to_euler(w, x, y, z):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x ** 2 + y ** 2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1,
                     np.sign(sinp) * np.pi / 2,
                     np.arcsin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y ** 2 + z ** 2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def wrap2pi(angle):
    if angle <= np.pi and angle >= -np.pi:
        return (angle)
    elif angle < -np.pi:
        return (wrap2pi(angle + 2 * np.pi))
    else:
        return (wrap2pi(angle - 2 * np.pi))


def localization_callback(msg):
    state_lock.acquire()
    global state
    state = (msg.pose.pose.position.x, msg.pose.pose.position.y,
             quaternion_to_euler(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                 msg.pose.pose.orientation.z)[0])
    state_lock.release()


def read_trajectory_file(trajectory_file_name):
    df = pd.read_csv(trajectory_file_name, header=None)
    xs = df[0]
    ys = df[1]
    ref_traj = np.array([[x, y] for x, y in zip(xs, ys)])
    return ref_traj


def compute_cost_map(reference_trajectory):
    ref_tree = scipy.spatial.KDTree(reference_trajectory)

    # /// Define costmap dimensions //////////////////////////////////////////////////////////////////////////////////////////////
    traj_max_x = int(math.ceil(np.max(reference_trajectory[:, 0])))
    traj_min_x = int(math.ceil(np.min(reference_trajectory[:, 0])))

    traj_max_y = int(math.ceil(np.max(reference_trajectory[:, 1])))
    traj_min_y = int(math.ceil(np.min(reference_trajectory[:, 1])))

    traj_dim_array = np.array([traj_min_x, traj_max_x, traj_min_y, traj_max_y])

    # /// Create costmap array //////////////////////////////////////////////////////////////////////////////////////////////

    max_dir = np.argmax(np.abs(traj_dim_array))
    max_dim = traj_dim_array[max_dir] + COST_MAP_BUFFER
    n_cells = math.ceil(max_dim * 2 / COST_MAP_RESOLUTION)

    costmap_extent = -max_dim, max_dim, -max_dim, max_dim

    costmap = np.zeros((n_cells, n_cells))

    # /// Assign costs //////////////////////////////////////////////////////////////////////////////////////////////

    for i in range(0, costmap.shape[0]):
        x_cell = max_dim - (i * COST_MAP_RESOLUTION + (COST_MAP_RESOLUTION / 2))
        for j in range(0, costmap.shape[1]):
            y_cell = j * COST_MAP_RESOLUTION + (COST_MAP_RESOLUTION / 2) - max_dim
            cell_cost, _ = ref_tree.query(np.array([x_cell, y_cell]), k=1)
            costmap[i, j] = COST_FUNCTION_GAIN * cell_cost ** 0.5
    return costmap, costmap_extent


if __name__ == "__main__":
    rospy.init_node("small_mpc_ros_node")

    rospy.Subscriber("icp_odom", Odometry, localization_callback)

    reference_trajectory = read_trajectory_file(TRAJECTORY_FILE_NAME)[:30]
    # cost_map, cost_map_extent = compute_cost_map(reference_trajectory)
    sampler = Sampler(NB_SAMPLES, NB_STEPS)
    model = Model(RATE)

    t = Thread(target=rospy.spin)
    t.start()

    nominal_commands = np.empty((2, NB_STEPS), dtype=np.float)
    nominal_commands[0].fill(LINEAR_VELOCITY)
    nominal_commands[1].fill(0.0)
    while not rospy.is_shutdown():
        state_lock.acquire()
        current_state = state
        state_lock.release()
        if current_state is not None:
            sampled_commands = sampler.generate_commands(nominal_commands)
            sampled_trajectories = model.simulate(current_state, sampled_commands)
            fig, ax = plt.subplots(figsize=(10, 10))

            ax.plot(reference_trajectory[:, 1], reference_trajectory[:, 0], c='C1', linewidth=5, label='Ref. traj.')

            # ax.imshow(cost_map, extent=cost_map_extent)

            for i in range(0, NB_SAMPLES):
                ax.plot(sampled_trajectories[i, :, 1], sampled_trajectories[i, :, 0], linewidth=1, alpha=0.05)

            # ax.plot(traj_nom[1, :], traj_nom[0, :], linewidth=5, c='C0', label='Nominal traj.')
            ax.scatter(current_state[1], current_state[0], c='C2', label='Init. pose')
            ax.set_aspect('equal')
            ax.set_xlabel('y_g')
            ax.set_ylabel('x_g')
            ax.legend()

            plt.show()
            quit()
        else:
            time.sleep(0.1)
