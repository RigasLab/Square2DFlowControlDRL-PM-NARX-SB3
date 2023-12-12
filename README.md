# Square2DFlowControlDRL-PM-NARX-SB3
An open-source code release for the paper "Active Flow Control for Bluff Body Drag Reduction Using Reinforcement Learning with Partial Measurements". 

Note: This README document is not the final version and will be updated soon.

This repository contains code for reinforcement learning control aimed at reducing the drag due to vortex shedding in the wake of a 2D square body.


The code is a further development on the work published in:
 "Artificial Neural Networks trained through Deep Reinforcement Learning discover control strategies for active flow control", Rabault et. al., Journal of Fluid Mechanics (2019) (code at https://github.com/jerabaul29/Cylinder2DFlowControlDRL), 
and in "Accelerating Deep Reinforcement Learning strategies of Flow Control through a multi-environment approach", Rabault and Kuhnle, Physics of Fluids (2019) (code at https://github.com/jerabaul29/Cylinder2DFlowControlDRLParallel). 

The Reinforcement Learning framework used in this code is based on Stable Baseline3 https://github.com/DLR-RM/stable-baselines3 and Stable Baseline3 Contrib https://github.com/Stable-Baselines-Team/stable-baselines3-contrib

If you find this work useful and/or use it in your own research, please also cite these works:

```
Rabault, J., Kuhnle, A (2019).
Accelerating Deep Reinforcement Leaning strategies of Flow Control through a
multi-environment approach.
Physics of Fluids.

Rabault, J., Kuchta, M., Jensen, A., Réglade, U., & Cerardi, N. (2019).
Artificial neural networks trained through deep reinforcement learning discover
control strategies for active flow control.
Journal of Fluid Mechanics, 865, 281-302. doi:10.1017/jfm.2019.62

Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021).
Stable-Baselines3: Reliable Reinforcement Learning Implementations.
Journal of Machine Learning Research.
```

## Getting started

This code is developed for both running on the cluster (tested at Imperial College London HPC service) and on PC.

- The main reinforcement learning environment is in **Env2DCylinderModified.py**.
- The reinforcement learning parameters are set in **launch_parallel_training.py**.
- The simulation template to be set is in the **simulation_base** folder.
- The simulation solver is in **flow_solver.py**.

## Package installation in Python

```
- Create a Python environment (in Conda) and activate it:
```
conda create -n <ENV_NAME> -c conda-forge fenics (This will create an environment with a user-defined name "RLSB3" and install FEniCS with its dependencies.)
conda activate <ENV_NAME> 

```
- Install other packages required (python 3.7+ and PyTorch >= 1.11 recommended)
```
pip3 install torch torchvision torchaudio (Check https://pytorch.org/get-started/locally/ if it doesn't work)
pip install sb3-contrib (This will install stable-baselines3 as well automatically. Check https://sb3-contrib.readthedocs.io/en/master/guide/install.html for details.)
pip install tensorboard
pip install gmsh=3.0.6 (Only needed if you want to generate new mesh files)
(Few other packages like pickle5, scipy and peakutils are required and they can be easily installed by pip)
```
- For details of all the packages and versions, please refer to **requirements.txt**.

## How to run the code

- **launch_parallel_training.py**: define training parameters (Algorithm, Neural Network, hyperparameters etc.)

- **single_runner.py**: evaluate the latest saved policy.

## Implementing on cluster as batch jobs 

The scripts for launching batch jobs on cluster are:

```
script_launch_parallel_cluster.sh (Train a controller)
Run.sh (Test a trained controller)
script_make_mesh_cluster.sh (Generate mesh files for new simulation environments)
```

To run the code on cluster:
- **script_launch_parallel_cluster.sh**: automatically launch the training as a parallel batch job. This script calls **launch_parallel_training.py**.
- **Run.sh**: launch a job to evaluate a particular policy. This script calls **single_runner.py**.

The main script for launching trainings as batch jobs is the **script_launch_parallel_cluster.sh** script. This script specifies the settings of the job (Time, Number of Procs etc.) and calls **launch_parallel_training.py**, which actually setup and run the training process.

Make the job is sized correctly. For a mesh of around 10000 elements and a timestep of dt=0.004, these conservative guidelines are a good starting point:
- wall_time = 30 minutes * #_episodes / #_parallel environments
- n_cpus = #_parallel environments + 2

The job submission requires an environment variable **NUM_PORT** to be set prior to execution. This variable determines the number of parallel environments during training and can be modified in **script_launch_parallel_cluster.sh**.


## Troubleshooting

If you encounter problems, please:

- look for help in the .md readme files of this repo
- look for help on the github repo of the JFM paper used for serial training
- if this is not enough to get your problem solved, feel free to open an issue and ask for help.


## CFD simulation fenics, and user-defined user cases

The CFD simulation is implemented by FEniCS. Please refer to its official website https://fenicsproject.org/ for more details. 
For more details about the CFD simulation and how to build your own user-defined cases, please consult the Readme of the JFM code, available at https://github.com/jerabaul29/Cylinder2DFlowControlDRL.

