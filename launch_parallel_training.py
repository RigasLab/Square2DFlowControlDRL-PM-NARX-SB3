import argparse
import os
import sys
import csv
import socket
import numpy as np
import torch

from simulation_base.env import resume_env, nb_actuations

from gym.wrappers.time_limit import TimeLimit

from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Logger, HumanOutputFormat, DEBUG
from stable_baselines3.sac import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
    ap.add_argument("-s", "--savedir", required=False,
                    help="Directory into which to save the NN. Defaults to 'saver_data'.", type=str,
                    default='saver_data')

    args = vars(ap.parse_args())

    number_servers = args["number_servers"]
    savedir = args["savedir"]


    config = {}

    config["learning_rate"] = 1e-4
    config["learning_starts"] = 26000
    config["batch_size"] = 128

    config["tau"] = 5e-3
    config["gamma"] = 0.99
    config["train_freq"] = 1
    config["target_update_interval"] = 1
    config["gradient_steps"] = 48

    config["buffer_size"] = int(10e5)
    config["optimize_memory_usage"] = False

    config["ent_coef"] = "auto_0.01"
    config["target_entropy"] = "auto"
    device = "cuda" # "cpu" if run the training on cpu
    policy_kwargs = dict(net_arch=dict(pi=[512,512,512], qf=[512,512,512]))
    checkpoint_callback = CheckpointCallback(
                                            save_freq=max(200, 1),
                                            #save_env_stats=True,
                                            #save_replay_buffer=True, # This is not tested, may be useful for resume
                                            save_vecnormalize=True,
                                            save_path=savedir,
                                            name_prefix='PMTQC27FSavgPR')

    env = SubprocVecEnv([resume_env(plot=False, dump_CL=False, dump_debug=10, n_env=i) for i in range(number_servers)], start_method='spawn')
    
    # Deactivate this if not use history observations
    env = VecFrameStack(env, n_stack=27)
    
    env = VecNormalize(env, gamma=0.99)

    # Replace 'TQC' by 'SAC' if want to use SAC
    model = TQC('MlpPolicy', env, policy_kwargs=policy_kwargs, tensorboard_log=savedir, device=device, **config)

    model.learn(15000000, callback=[checkpoint_callback], log_interval=1)


    print("Agent and Runner closed -- Learning complete -- End of script")
    os._exit(0)

