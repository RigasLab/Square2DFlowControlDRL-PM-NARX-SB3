'''
Perform a single run of the flow without control
'''

import os
import socket
import numpy as np
import csv
import sys
import math
import argparse
import json

from dolfin import Expression
from gym.wrappers.time_limit import TimeLimit

from Env2DCylinderModified import Env2DCylinderModified
from probe_positions import probe_positions
from simulation_base.env import resume_env, nb_actuations, simulation_duration

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TQC
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# If previous evaluation results exist, delete them
if(os.path.exists("saved_models/test_strategy.csv")):
    os.remove("saved_models/test_strategy.csv")

if(os.path.exists("saved_models/test_strategy_avg.csv")):
    os.remove("saved_models/test_strategy_avg.csv")


if __name__ == '__main__':


    action_step_size = simulation_duration / nb_actuations  # Get action step size from the environment, not used
    horizon = 400 # Number of actions for single run. Non-dimensional time is horizon*action_step_size (by default action_step_size=0.5)
    action_steps = int(horizon)
    
    env = SubprocVecEnv([resume_env(plot=False, single_run=True, horizon=horizon, dump_vtu=100, n_env=99)], start_method='spawn')

    observations = env.reset()
    
    for k in range(action_steps):
        observations, rw, done, _ = env.step()
