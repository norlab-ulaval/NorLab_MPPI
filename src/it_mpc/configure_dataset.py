# coding=utf-8
import pandas as pd
from multifeature_aggregator import aggregate_multiple_features, StatePose2D

# ::: Data wrangling :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
dataset_path = 'samples/data/marmotte/ga_hard_snow_25_01_a/slip_dataset_all.pkl'
# dataset_path = '/Users/redleader/PycharmProjects/NorLab_MPPI/samples/data/marmotte/ga_hard_snow_25_01_a
# /slip_dataset_all.pkl'

dataset_marmotte = pd.read_pickle(dataset_path)

features_config_1 = {
        'body_vel_disturption': StatePose2D,
        'icp_interpolated':     StatePose2D,
        'icp_vel':              StatePose2D,
        'idd_vel':              StatePose2D,
        'icp':                  ('StatePose3D', 'x', 'y', 'z', 'roll', 'pitch', 'yaw')
        }

mf1 = aggregate_multiple_features(dataset_marmotte,
                                  dataset_info="Robot: marmotte, Details: ga_hard_snow_25_01_a",
                                  features_config=features_config_1)
