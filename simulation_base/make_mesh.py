import env
from stable_baselines3.common.vec_env import DummyVecEnv
from env import resume_env
Env = DummyVecEnv([resume_env(plot=False, dump_CL=False,remesh=True, dump_debug=10, n_env=999)])
