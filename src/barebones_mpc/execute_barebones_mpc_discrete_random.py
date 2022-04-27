# coding=utf-8
import src.barebones_mpc.sampler.std_dev_sampler
from src.barebones_mpc.abstract_model_predictive_control_component import AbstractModelPredictiveControlComponent
from src.barebones_mpc.controller.base_controler import ModelPredictiveControler, TrajectoryCollector

config_path = "src/barebones_mpc/config_files/config_real_CartPole-v1_discrete_random_tuning.yaml"
mpc = ModelPredictiveControler(config_path=config_path)

print("\n:: std_sampler:")
std_sampler = mpc.sampler
print("    ↳ | sample_length", std_sampler.sample_length)

trajectory_collector = mpc.execute()

print("\n:: Trj collector:")
print("    ↳ | size", trajectory_collector.get_size())
# print("    ↳ | trjs_rewards", trajectory_collector.trjs_rewards)
print("    ↳ | trjs_actions", trajectory_collector.trjs_actions)
