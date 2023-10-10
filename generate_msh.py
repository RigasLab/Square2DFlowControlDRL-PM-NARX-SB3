import os, subprocess
# from printind.printind_function import printiv

# Args will be the geometry params (or cmd line args if called from there)
# dim is domain dimension (2 for 2D) for meshing
def generate_mesh(args, template='geometry_2d.template_geo', dim=2):
    '''Modify template according to args (geom_params) and make gmsh generate the mesh'''

    assert os.path.exists(template)  # Raise an error if no template
    args = args.copy()  # Create a copy of the dict as we will be popping pairs

    with open(template, 'r') as f: old = f.readlines()  # Read template and save each line as an item of the list 'old''

    # Lambda defines an anonymous function --> lambda arguments : expression
    # .startswith() method returns True if a string starts with the specified prefix
    # map(function, iterable) returns a list of the outputs of applying the function to each element of the iterable
    # .index() returns the index of an element in a list
    split = list(map(lambda s: s.startswith('DefineConstant'), old)).index(True)  # --> Return line number of DefineConstant

    body = ''.join(old[split:])  # generate body for .geo

    output = args.pop('output')  # -> output = 'mesh/turek_2d.geo'

    # os.path.splitext() splits the path name into a pair root , ext
    if not output:
        output = template
    assert os.path.splitext(output)[1] == '.geo'  # raise error if output doesnt have .geo extension

    with open(output, 'w') as f: f.write(body)  # write body to target ('output') .geo file

    scale = args.pop('clscale')  # Get mesh size scaling ratio

    cmd = 'gmsh -0 %s' % output  # Create cmd string to output unrolled geometry

    list_geometric_parameters = ['jets_toggle', 'jet_width', 'height_cylinder', 'ar', 'cylinder_y_shift',
                                 'x_upstream', 'x_downstream', 'height_domain',
                                 'mesh_size_cylinder', 'mesh_size_jets', 'mesh_size_medium', 'mesh_size_coarse',
                                 'coarse_y_distance_top_bot', 'coarse_x_distance_left_from_LE']

    constants = " "

    # Create cmd string to set params of DefineConstants
    for crrt_param in list_geometric_parameters:
        constants = constants + " -setnumber " + crrt_param + " " + str(args[crrt_param])

    # Create unrolled model with the geometry_params set
    subprocess.call(cmd + constants, shell=True)  # run the command to create unrolled

    # Assert that 'mesh/turek_2d.geo'_unrolled exists
    unrolled = '_'.join([output, 'unrolled'])
    assert os.path.exists(unrolled)

    return subprocess.call(['gmsh -%d -format msh2 -clscale %g %s' % (dim, scale, unrolled)], shell=True)
    # generate 2d mesh --> turek_2d.msh
    # -clscale <float> : Set global mesh element size scaling factor
    # -2: Perform 2D mesh generation, then exit

# -------------------------------------------------------------------

if __name__ == '__main__':  # This is only run when this file is executed as a script, not when imported
    import argparse, sys, petsc4py
    from math import pi

    # The argparse module makes it easy to write user-friendly command-line interfaces. The program defines what arguments it requires, and argparse will figure out how to parse those out of sys.argv.
    # generate argument help output for when "python generate_mesh.py -help" is called from the command line:

    parser = argparse.ArgumentParser(description='Generate msh file from GMSH',  # Text to display before the argument help
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter) # adds information about default values to each of the argument help messages
    # Optional output geo file
    parser.add_argument('-output', default='', type=str, help='A geofile for writing out geometry')

    # Geometry
    parser.add_argument('-jets_toggle', default=1, type=bool,
                        help='toggle Jets --> 0 : No jets, 1: Yes jets')
    parser.add_argument('-jet_width', default=0.1, type=float,
                        help='Jet Width')
    parser.add_argument('-height_cylinder', default=40, type=float,
                        help='Cylinder Height')
    parser.add_argument('-ar', default=1, type=float,
                        help='Cylinder Aspect Ratio')
    parser.add_argument('-cylinder_y_shift', default=80, type=float,
                        help='Cylinder Center Shift from Centerline, Positive UP')
    parser.add_argument('-x_upstream', default=0.25, type=float,
                        help='Domain Upstream Length (from left-most rect point)')
    parser.add_argument('-x_downstream', default=5, type=float,
                        help='Domain Downstream Length (from right-most rect point)')
    parser.add_argument('-height_domain', nargs='+', default=[0, 60, 120, 180, 240, 300],
                        help='Domain Height')
    parser.add_argument('-mesh_size_cylinder', default=0.005, type=float,
                        help='Mesh Size on Cylinder Walls')
    parser.add_argument('-mesh_size_coarse', default=1, type=float,
                        help='Mesh Size on boundaries outside wake')
    parser.add_argument('-mesh_size_medium', default=0.2, type=float,
                        help='Medium mesh size (at boundary where coarsening starts')
    parser.add_argument('-coarse_y_distance_top_bot', default=4, type=float,
                        help='y-distance from center where mesh coarsening starts')
    parser.add_argument('-coarse_x_distance_left_from_LE', default=2, type=float,
                        help='x-distance from upstream face where mesh coarsening starts')

    # Refine geometry
    parser.add_argument('-clscale', default=1, type=float,
                        help='Scale the mesh size relative to give')

    args = parser.parse_args()

    # Using geometry_2d.geo to produce geometry_2d.msh
    sys.exit(generate_mesh(args.__dict__))

    # FIXME: inflow profile
    # FIXME: test with turek's benchmark

    # IDEAS: More accureate non-linearity handling
    #        Consider splitting such that we solve for scalar components
