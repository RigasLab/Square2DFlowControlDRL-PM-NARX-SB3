# Square2DFlowControlDRL-PM-NARX-SB3
This repository contains the code corresponding to our JFM paper "Active Flow Control for Bluff Body Drag Reduction Using Reinforcement Learning with Partial Measurements", accessible at [here](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/active-flow-control-for-bluff-body-drag-reduction-using-reinforcement-learning-with-partial-measurements/F98233D07BAD238143B8C2544DE0BD03).

This code implements reinforcement learning control with a NARX-modelled controller with Soft Actor-Critic, to reduce the drag due to vortex shedding in the wake of a 2D square body.

This code is a further development on the work published:
 "Artificial Neural Networks trained through Deep Reinforcement Learning discover control strategies for active flow control", Rabault et. al., Journal of Fluid Mechanics (2019) (code at https://github.com/jerabaul29/Cylinder2DFlowControlDRL), 
and in "Accelerating Deep Reinforcement Learning Strategies of Flow Control through a multi-environment approach", Rabault and Kuhnle, Physics of Fluids (2019) (code at https://github.com/jerabaul29/Cylinder2DFlowControlDRLParallel). 

The Reinforcement Learning framework used in this code is based on Stable Baseline3 https://github.com/DLR-RM/stable-baselines3 and Stable Baseline3 Contrib https://github.com/Stable-Baselines-Team/stable-baselines3-contrib

If you find this work useful and/or use it in your own research, please cite these works:

```
Rabault, J., Kuhnle, A (2019).
Accelerating Deep Reinforcement Learning strategies of Flow Control through a
multi-environment approach.
Physics of Fluids.

Rabault, J., Kuchta, M., Jensen, A., Réglade, U., & Cerardi, N. (2019).
Artificial neural networks trained through deep reinforcement learning discover
control strategies for active flow control.
Journal of Fluid Mechanics, 865, 281-302. doi:10.1017/jfm.2019.62

Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021).
Stable-Baselines3: Reliable Reinforcement Learning Implementations.
Journal of Machine Learning Research.

Logg, A., & Wells, G. N. (2010).
DOLFIN: Automated Finite Element Computing.
ACM Transactions on Mathematical Software.
```

## Getting started

This code is developed for both running on the cluster (tested at Imperial College London HPC service) and on PC.

- The main reinforcement learning environment is in **Env2DCylinderModified.py**.
- The reinforcement learning parameters are set in **launch_parallel_training.py**.
- The simulation template to be set is in the **simulation_base/env.py**.
- The simulation solver is in **flow_solver.py**.

## Package installation in Python

Create a Python environment (in Conda) and activate it:

```
conda create -n <ENV_NAME> -c conda-forge fenics (This will create an environment with a user-defined name <ENV_NAME> and install FEniCS with its dependencies.)
conda activate <ENV_NAME> 

```

Install other packages required (python 3.7+ and PyTorch >= 1.11 recommended):

```
pip3 install torch torchvision torchaudio (Check https://pytorch.org/get-started/locally/ if the command does not work)
pip install sb3-contrib (This will install stable-baselines3 as well automatically. Check https://sb3-contrib.readthedocs.io/en/master/guide/install.html for details.)
pip install tensorboard
pip install gmsh=3.0.6 (Only needed if you want to generate new mesh files)
```

Few other packages like pickle5, scipy and peakutils are required, and they can be easily installed by:

```
pip install <PACKAGE>
```

- For details of all the packages and versions, please refer to **requirements.txt**.


## Main scripts

The scripts for implementing reinforcement learning and flow simulation are:

- **Env2DCylinderModified.py**: contains the reinforcement learning environment and data-saving functions.

- **simulation_base/env.py**: sets up the reinforcement learning environments, and defines all the simulation parameters and reward function

- **launch_parallel_training.py**: defines RL parameters and launching training (Learning parameters, Neural Network configurations etc.).

- **single_runner.py**: tests a trained controller (the policy at any Checkpoint).

- **flow_solver.py**: implements a 2D flow solver by FEniCS.

- **probe_positions.py**: defines different sensor configurations.

- **probes.py**: contains different types of sensors.

- **generate_msh.py**: generate mesh files according to **geometry_2d.template_geo**.

- **simulation_base/make_mesh.py**: calls a dummy environment to generate mesh files.


The scripts for launching the code as batch jobs on the cluster are (**modification needed if not run on Imperial HPC**):

- **script_launch_parallel_cluster.sh**: train a controller. This script specifies the settings of the job (Wall Time, Number of Processers, Number of Parallel Environments, etc.)

- **Run.sh**: test a trained controller

- **script_make_mesh_cluster.sh**: generate mesh files for new simulation environments


## Run the code on PC

- NOTE: before running, check all the parameters in **launch_parallel_training.py** and **simulation_base/env.py**.

Launch training (activate Python environment first; <NUM_ENV> defines the number of simulation environments to run):

```
python3 launch_parallel_training.py -n <NUM_ENV>
```

Launch a short test of a trained controller:

```
python3 single_runner.py
```

Make new mesh files:

```
python3 make_mesh.py
```

## Run the code on the cluster as batch jobs

- NOTE: before running, check all the parameters in **launch_parallel_training.py** and **simulation_base/env.py**.

To run the code on the cluster (Imperial HPC as an example):

```
qsub script_launch_parallel_cluster.sh
(This automatically launches the training as a parallel batch job and calls **launch_parallel_training.py**.)
```

- NOTE: Please set the job size reasonably. For a mesh of around 10000 elements and a timestep of dt=0.004, these conservative guidelines are a good starting point:
```
wall_time = 40 minutes * #_episodes / #_number of parallel environments
n_cpus = #_number of parallel environments + 2
```

- NOTE: The job submission requires an environment variable **NUM_PORT** to be set prior to execution. This variable determines <NUM_ENV> as the number of parallel environments for flow simulation.

```
qsub Run.sh
(This launches a job to test a saved policy, calling **single_runner.py**.)
```


## Troubleshooting

If you encounter problems, please:

- look for help in the .md README files of this repo
- if this is not enough to get your problem solved, feel free to open an issue and ask for help.


## Flow simulation

The CFD in this code is implemented by FEniCS. Please refer to its official website https://fenicsproject.org/ for more details. 


