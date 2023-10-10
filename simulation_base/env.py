"""Initialize and use the environment.
"""

import sys
import os
import shutil
import numpy as np
cwd = os.getcwd()
sys.path.append(cwd + "/../")

from Env2DCylinderModified import Env2DCylinderModified
from probe_positions import probe_positions


from dolfin import Expression
from gym.wrappers.time_limit import TimeLimit

from stable_baselines3.common.monitor import Monitor

nb_actuations = 400 # Number of actions (NN actuations) taken per episode (Number of action intervals)

def resume_env(plot=False,  # To plot results (Field, controls, lift, drag, rec area) during training
               dump_vtu=False,  # If not False, create vtu files of area, velocity, pressure, every 'dump_vtu' steps
               dump_debug=500,  # If not False, output step info of ep,step,rec_area,L,D,jets Q* to saved_models/debug.csv, every 'dump_debug' steps
               dump_CL=500,  # If not False, output step info of ep,step,rec_area,L,D,jets Q* to command line, every 'dump_CL' steps
               remesh=False,
               random_start=True,
               single_run=False,
               horizon=nb_actuations,
               n_env=1):

    def _init():

        # ---------------------------------------------------------------------------------

        simulation_duration = 200  # In non-dimensional time unit
        dt = 0.004
        single_input = False
        single_output = False
        include_actions = True

        root = 'mesh/turek_2d'  # Root of geometry file path
        if (not os.path.exists('mesh')):
            os.mkdir('mesh')

        geometry_params = {'output': '.'.join([root, 'geo']),
                           # mesh/turek_2d.geo // relative output path of geometry file according to geo params
                           'template': 'geometry_2d.template_geo',  # relative path of geometry file template
                           'clscale': 1,
                           # mesh size scaling ratio (all mesh characteristic lenghts of geometry file scaled by this factor)
                           'remesh': remesh,  # remesh toggle (from resume_env args)
                           'jets_toggle': 1,  # toggle Jets --> 0 : No jets, 1: Yes jets
                           'jet_width': 0.1,  # Jet Width
                           'height_cylinder': 1,  # Cylinder Height
                           'ar': 1.0,  # Cylinder Aspect Ratio
                           'cylinder_y_shift': 0,  # Cylinder Center Shift from Centreline, Positive UP
                           'x_upstream': 20,  # Domain Upstream Length (from left-most rect point)
                           'x_downstream': 26,  # Domain Downstream Length (from right-most rect point)
                           'height_domain': 25,  # Domain Height
                           'mesh_size_cylinder': 0.075,  # Mesh Size on Cylinder Walls
                           'mesh_size_jets': 0.015,  # Mesh size on jet boundaries
                           'mesh_size_medium': 0.45,  # Medium mesh size (at boundary where coarsening starts)
                           'mesh_size_coarse': 1,  # Coarse mesh Size Close to Domain boundaries outside wake
                           'coarse_y_distance_top_bot': 4,  # y-distance from center where mesh coarsening starts
                           'coarse_x_distance_left_from_LE': 2.5}  # x-distance from upstream face where mesh coarsening starts

        profile = Expression(('1', '0'), degree=2)  # Inflow profile (defined as FEniCS expression)

        flow_params = {'mu': 1E-2,  # Dynamic viscosity. This in turn defines the Reynolds number: Re = U * D / mu
                       'rho': 1,  # Density
                       'inflow_profile': profile}  # flow_params['inflow_profile'] stores a reference to the profile function

        solver_params = {'dt': dt}

        # Define probes positions
        probe_distribution = {'distribution_type': 'base', # Or 'inflow64' for probes in the wake
                              'probes_at_jets': False,
                              # Whether to use probes at jets or not, usually not in use
                              'n_base': 64}  # Number of probes at cylinder base if 'base' distribution is used

        list_position_probes,list_position_probes_ob = probe_positions(probe_distribution, geometry_params)

        output_params = {'locations': list_position_probes,  # List of (x,y) np arrays with probe positions
                         'locations_ob':list_position_probes_ob, # List of probe positions only for observations from the wake
                         'probe_type': 'pressure',  # Set quantity measured by probes (pressure/velocity)
                         'single_input': single_input,
                         # whether to feed as input probe values or difference between average top/bottom pressures
                         'single_output': single_output,  # whether policy network outputs one or two outputs
                         'symmetric': False,
                         'include_actions': include_actions
                         }

        optimization_params = {"num_steps_in_pressure_history": 1,
                               # Number of steps that constitute an environment state (state shape = this * len(locations))
                               "min_value_jet_MFR": -0.1,  # Set min and max Q* for weak actuation
                               "max_value_jet_MFR": 0.1,
                               "smooth_control": 0.1,  # parameter alpha to smooth out control
                               "zero_net_Qs": True,  # True for Q1 + Q2 = 0
                               "random_start": random_start}

        inspection_params = {"plot": plot,
                             "dump_vtu": dump_vtu,
                             "dump_debug": dump_debug,
                             "dump_CL": dump_CL,
                             "range_pressure_plot": [-2.0, 1],  # ylim for pressure dynamic plot
                             "range_drag_plot": [-0.175, -0.13],  # ylim for drag dynamic plot
                             "range_lift_plot": [-0.2, +0.2],  # ylim for lift dynamic plot
                             "line_drag": -0.7221,  # Mean drag without control
                             "line_lift": 0,  # Mean lift without control
                             "show_all_at_reset": False,
                             "single_run": single_run,
                             'index': n_env
                             }

        reward_function = 'Tavg_power_reward' # Check 'compute_reward()' in Env2DCylinderModified.py for available options

        # Ensure that SI is True only if probes on body base, and record pressure
        output_params['single_input'] = (single_input and probe_distribution['distribution_type'] == 'base' and output_params['probe_type'] == 'pressure')

        verbose = 0  # For detailed output (see Env2DCylinder)

        number_steps_execution = int((simulation_duration / dt) / nb_actuations)  # Duration in timesteps of action interval (Number of numerical timesteps over which NN action is kept constant, control being interpolated)

        # ---------------------------------------------------------------------------------
        # do the initialization

        # If remesh = True, we sim with no control until a well-developed unsteady wake is obtained. That state is saved and
        # used as a start for each subsequent learning episode.

        # If so, set the value of n-iter (no. iterations to calculate converged initial state)
        if (remesh):
            n_iter = int(225.0 / dt)  # default: 200
            if (os.path.exists('mesh')):
                shutil.rmtree('mesh')  # If previous mesh directory exists, we delete it
            os.mkdir('mesh')  # Create new empty mesh directory
            print("Make converge initial state for {} iterations".format(n_iter))
        else:
            n_iter = None

        # Processing the name of the simulation (to be used in outputs)
        simu_name = 'Simu'

        if geometry_params["ar"] != 1:
            next_param = 'AR' + str(geometry_params["ar"])
            simu_name = '_'.join([simu_name, next_param])  # e.g: if cyl_size (mesh) = 0.025 --> simu_name += '_M25'
        if optimization_params["max_value_jet_MFR"] != 0.01:
            next_param = 'maxF' + str(optimization_params["max_value_jet_MFR"])[2:]
            simu_name = '_'.join([simu_name, next_param])  # e.g: if max_MFR = 0.09 --> simu_name += '_maxF9'
        if nb_actuations != 80:
            next_param = 'NbAct' + str(nb_actuations)
            simu_name = '_'.join([simu_name, next_param])  # e.g: if max_MFR = 100 --> simu_name += '_NbAct100'

        next_param = 'drag'
        if reward_function == 'recirculation_area':
            next_param = 'area'
        if reward_function == 'max_recirculation_area':
            next_param = 'max_area'
        elif reward_function == 'drag':
            next_param = 'last_drag'
        elif reward_function == 'max_plain_drag':
            next_param = 'max_plain_drag'
        elif reward_function == 'drag_plain_lift':
            next_param = 'lift'
        elif reward_function == 'drag_avg_abs_lift':
            next_param = 'avgAbsLift'
        simu_name = '_'.join([simu_name, next_param])

        # Pass parameters to the Environment class
        env_2d_cylinder = Monitor(TimeLimit(Env2DCylinderModified(path_root=root,
                                                geometry_params=geometry_params,
                                                flow_params=flow_params,
                                                solver_params=solver_params,
                                                output_params=output_params,
                                                optimization_params=optimization_params,
                                                inspection_params=inspection_params,
                                                n_iter_make_ready=n_iter,
                                                verbose=verbose,
                                                reward_function=reward_function,
                                                number_steps_execution=number_steps_execution,
                                                simu_name=simu_name), max_episode_steps = horizon))

        return env_2d_cylinder


    return _init  # resume_env() returns instance of Environment object
