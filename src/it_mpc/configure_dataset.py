# coding=utf-8
import pandas as pd
from dataclasses import dataclass, asdict

from multifeature_aggregator import (
    aggregate_multiple_features,
    StatePose2D,
    CmdSkidSteer,
    Velocity,
)
from multifeature_aggregator.data_containers import (
    AbstractMultifeature,
    AbstractFeatureDataclass,
)

# ::: Data wrangling :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
dataset_path = 'samples/data/marmotte/ga_hard_snow_25_01_a/slip_dataset_all.pkl'

dataset_marmotte = pd.read_pickle(dataset_path)

features_config_1 = {
    'body_vel_disturption': StatePose2D,
    'icp_interpolated': StatePose2D,
    'icp_vel': StatePose2D,
    'icp': ('StatePose3D', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'),
    'idd_vel': StatePose2D,
    'cmd': CmdSkidSteer,
    'left_wheel': ('WheelVelocity', 'vel'),
    'right_wheel': ('WheelVelocity', 'vel'),
}

# (NICE TO HAVE) ToDo: (in multifeature_aggregator) Implement mechanism for dataframe column label renaming with
# validation (ref task NLMPPI-61 )


# marm1_eda: AbstractMultifeature = aggregate_multiple_features(
marm1_eda = aggregate_multiple_features(
    dataset_marmotte,
    dataset_info="Robot› marmotte, Details› ga_hard_snow_25_01_a",
    features_config=features_config_1,
)

print(marm1_eda.summary)

# ::: IT-MPC state representation ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

itmpc_config_1 = {
    'icp': ('StatePoseMPPI', 'x', 'y', 'yaw', 'roll'),
    'icp_vel': StatePose2D,
    'cmd': CmdSkidSteer,
}

marm1_itmpc = aggregate_multiple_features(
    dataset_marmotte,
    dataset_info="IT-MPC state representation. Robot› marmotte, Details› ga_hard_snow_25_01_a. ",
    features_config=itmpc_config_1,
)

print(marm1_itmpc.summary)
