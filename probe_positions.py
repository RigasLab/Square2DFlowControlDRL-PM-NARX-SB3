import numpy as np


def probe_positions(probe_distribution, geometry_params):
    '''
    Function that returns the list of np arrays with probe position coordinate pairs for a given distribution.
    Distributions implemented:
    'rabault151' - Original distribution of 151 used by Rabault in his two first papers
    'rabault241' - Similar distribution to 'Rabault151' but with a increased number of probes in the wake
    'rabault9' - Partial information distribution tested by Rabault in the appendix of his first 2019 paper.
    'inflow8' - 8 probes in the wake
    'inflow64' - 64 probes in the wake more compact distributed close to the body, used in Chengwei et al. 2023
    'base' - Distribution where n_base evenly distributed probes are only located at the base of the cylinder

    In addition, probes close to the jets can also be enabled by the boolean probes_at_jets. Note that these
    probes are redundant in the case with 151 so this argument will be ignored in that case

    :param probe_distribution: String that identifies a given distribution
    :param probes_at_jets: boolean - whether to use probes close to jets or not
    :param geometry_params

    :return: list_position_probes: list of np arrays with probe position coordinate
    '''

    # Obtain relevant quantities

    distribution_type = probe_distribution['distribution_type']
    probes_at_jets = probe_distribution['probes_at_jets']
    n_base = probe_distribution['n_base']

    height_cylinder = geometry_params['height_cylinder']
    ar = geometry_params['ar']
    length_cylinder = ar * height_cylinder
    jet_width = geometry_params['jet_width']


    list_position_probes = []  # Initialise list of (x,y) np arrays with positions coordinate pairs
    
    # Initialize some observers in the wake
    list_position_probes_ob = [] # These probes are not inputs for NN, but just for collecting statistics in the wake

    x_dist_from_right_side = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    positions_probes_for_grid_x_ob = [length_cylinder / 2 + x for x in x_dist_from_right_side]
    positions_probes_for_grid_y_ob = [-1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5]
    
    for crrt_x in positions_probes_for_grid_x_ob:
        for crrt_y in positions_probes_for_grid_y_ob:
            list_position_probes_ob.append(np.array([crrt_x, crrt_y]))  # Append (x,y) pairs np array
            
    # Set probe positions based on 'distribution_type', which is defined in env.py
    if (distribution_type == 'rabault151' or distribution_type == 'rabault241'):

        if distribution_type == 'rabault151':
            # The 9 'columns' of 7 probes downstream of the cylinder
            positions_probes_x_dist_from_right_side = [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
            positions_probes_for_grid_x = [length_cylinder/2 + x for x in positions_probes_x_dist_from_right_side]
            positions_probes_for_grid_y = [-1.5, -1, -0.5, 0.0, 0.5, 1, 1.5]

        if distribution_type == 'rabault241':
            # The 17 'columns' of 9 probes downstream of the cylinder
            positions_probes_x_dist_from_right_side = [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10]
            positions_probes_for_grid_x = [length_cylinder / 2 + x for x in positions_probes_x_dist_from_right_side]
            positions_probes_for_grid_y = [-2, -1.5, -1, -0.5, 0.0, 0.5, 1, 1.5, 2]

        for crrt_x in positions_probes_for_grid_x:
            for crrt_y in positions_probes_for_grid_y:
                list_position_probes.append(np.array([crrt_x, crrt_y]))  # Append (x,y) pairs np array

        # The 4 'columns' of 4 probes on top and bottom of the cylinder
        positions_probes_for_grid_x = [-length_cylinder / 4, 0.0, length_cylinder / 4, length_cylinder / 2]
        positions_probes_for_grid_y = [-1.5, -1, 1, 1.5]

        for crrt_x in positions_probes_for_grid_x:
            for crrt_y in positions_probes_for_grid_y:
                list_position_probes.append(np.array([crrt_x, crrt_y]))  # Append (x,y) pairs np array

        # Two rectangles of probes around body of 36 probes each (for now, 10 probes on each side --> 36 total as corners are shared)
        # TODO: Make the distribution even as AR changes (scalable)

        for offset in [0.2, 0.4]:
            dist_probes_x = (length_cylinder + offset * 2) / 9  # Dist between probes on top and bottom sides of periferic
            dist_probes_y = (height_cylinder + offset * 2) / 9  # Dist between probes on left and right sides of periferic
            left_side_periferic_x = -length_cylinder / 2 - offset  # x coord of left side of periferic
            bot_side_periferic_y = -height_cylinder / 2 - offset  # y coord of bot side of periferic

            # Define top and bot sides probes
            positions_probes_for_grid_x = [left_side_periferic_x + dist_probes_x * i for i in range(10)]
            positions_probes_for_grid_y = [bot_side_periferic_y, height_cylinder / 2 + offset]

            for crrt_x in positions_probes_for_grid_x:
                for crrt_y in positions_probes_for_grid_y:
                    list_position_probes.append(np.array([crrt_x, crrt_y]))  # Append (x,y) pair array

            # Define left and right probes
            positions_probes_for_grid_x = [left_side_periferic_x, length_cylinder / 2 + offset]
            positions_probes_for_grid_y = [bot_side_periferic_y + dist_probes_y * i for i in range(1,9)]

            for crrt_x in positions_probes_for_grid_x:
                for crrt_y in positions_probes_for_grid_y:
                    list_position_probes.append(np.array([crrt_x, crrt_y]))  # Append (x,y) pair array

    elif (distribution_type == 'inflow8' or distribution_type == 'inflow64'):
        
        if distribution_type == 'inflow8':
            # The 4 'columns' of 2 probes downstream of the cylinder
            positions_probes_x_dist_from_right_side = [0.5, 1.0, 2.0, 4.0]
            positions_probes_for_grid_x = [length_cylinder/2 + x for x in positions_probes_x_dist_from_right_side]
            positions_probes_for_grid_y = [-0.5, 0.5]

        if distribution_type == 'inflow64':
            # The 8 'columns' of 8 probes downstream of the cylinder
            positions_probes_x_dist_from_right_side = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
            positions_probes_for_grid_x = [length_cylinder/2 + x for x in positions_probes_x_dist_from_right_side]
            positions_probes_for_grid_y = [-1.5, -1.0, -0.5, -0.25 , 0.25, 0.5, 1.0, 1.5]
        
        for crrt_x in positions_probes_for_grid_x:
            for crrt_y in positions_probes_for_grid_y:
                list_position_probes.append(np.array([crrt_x, crrt_y]))  # Append (x,y) pairs np array

    else:

        if probes_at_jets:

            x_probe_at_jet = length_cylinder/2 - jet_width/2
            list_position_probes.append(np.array([x_probe_at_jet, height_cylinder/2 + 0.2]))  # Top jet
            list_position_probes.append(np.array([x_probe_at_jet, -height_cylinder / 2 - 0.2]))  # Bottom jet

        if distribution_type == 'rabault9':

            positions_probes_x_dist_from_right_side = [0.25, 1, 3]
            positions_probes_for_grid_x = [length_cylinder / 2 + x for x in positions_probes_x_dist_from_right_side]
            positions_probes_for_grid_y = [-1, 0.0, 1]

            for crrt_x in positions_probes_for_grid_x:
                for crrt_y in positions_probes_for_grid_y:
                    list_position_probes.append(np.array([crrt_x, crrt_y]))  # Append (x,y) pairs np array

        elif distribution_type == 'base':

            positions_probes_for_grid_x = length_cylinder / 2
            positions_probes_for_grid_y = [-height_cylinder/2 + (height_cylinder/(n_base+1)) * i for i in range(1,n_base+1)]

            for crrt_y in positions_probes_for_grid_y:
                list_position_probes.append(np.array([positions_probes_for_grid_x, crrt_y]))  # Append (x,y) pairs np array

    return list_position_probes,list_position_probes_ob