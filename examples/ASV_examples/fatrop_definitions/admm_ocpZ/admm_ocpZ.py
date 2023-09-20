from rockit import *
from casadi import *
import numpy as np

# Define problem parameters
boat_diam = 0.30

# Problem parameters
nx    = 2                   # the system is composed of 2 states per robot
Tf    = 2                   # control horizon [s]
Nhor  = 20                  # number of control intervals
dt    = Tf/Nhor             # sample time
number_of_robots = 7        # number of robots that are neighbors (without local)

mu = 1

# Create OCP object
ocpZ = Ocp(T=Tf)

# Variables of own robot
# z_x = ocpZ.variable(grid='control',include_last=True)
# z_y = ocpZ.variable(grid='control',include_last=True)
stupid_state = ocpZ.state()
ocpZ.set_der(stupid_state, 0)
ocpZ.add_objective(ocpZ.sum(stupid_state**2))
z_x = ocpZ.control()
z_y = ocpZ.control()

Z = vertcat(z_x,z_y)

# Variables of neighbors
# Z_ij = ocpZ.variable(nx*number_of_robots, grid='control',include_last=True)
Z_ij = ocpZ.control(nx*number_of_robots)

# Parameters for copies and multipliers
lambda_i = ocpZ.register_parameter(MX.sym('lambda_i', nx), grid='control', include_last=True)
trajectory_i = ocpZ.register_parameter(MX.sym('trajectory_i', nx), grid='control', include_last=True)
lambda_ij = ocpZ.register_parameter(MX.sym('lambda_ij', nx*number_of_robots), grid='control', include_last=True)
trajectory_ij = ocpZ.register_parameter(MX.sym('trajectory_ij', nx*number_of_robots), grid='control', include_last=True)

c_i = Z - trajectory_i
term_i = dot(lambda_i, c_i) + mu/2*sumsqr(c_i)
if ocpZ.is_signal(term_i):
    term_i = ocpZ.sum(term_i,include_last=False)
ocpZ.add_objective(term_i)

for j in range(number_of_robots):
    Z_j = vertcat(Z_ij[2*j], Z_ij[2*j+1])
    trajectory_j = vertcat(trajectory_ij[2*j], trajectory_ij[2*j+1])
    lambda_j = vertcat(lambda_ij[2*j], lambda_ij[2*j+1])
    c_j = Z_j - trajectory_j
    term_j = dot(lambda_j, c_j) + mu/2*sumsqr(c_j)
    if ocpZ.is_signal(term_j):
        term_j = ocpZ.sum(term_j,include_last=False)
    ocpZ.add_objective(term_j)
    distance_j = sqrt( (Z[0] - Z_j[0])**2 + (Z[1] - Z_j[1])**2 )
    ocpZ.subject_to( distance_j >= boat_diam )

method = external_method('fatrop',N=Nhor, intg='rk')
ocpZ.method(method)
ocpZ._method.set_name("Z")

#options = {"ipopt": {"print_level": 5}}
#options["expand"] = True
#options["print_time"] = False
#ocpZ.solver('ipopt',options)
#ocpZ.method(MultipleShooting(N=Nhor,M=1,intg='rk'))


lambda_value = np.zeros([nx, Nhor+1])
ocpZ.set_value(lambda_i, lambda_value)
trajectory_value = np.zeros([nx, Nhor+1])
ocpZ.set_value(trajectory_i, trajectory_value)

lambda_values = np.zeros([nx*number_of_robots, Nhor+1])
ocpZ.set_value(lambda_ij, lambda_values)
trajectory_values = np.zeros([nx*number_of_robots, Nhor+1])
for j in range(number_of_robots):
    trajectory_values[2*j] = j*10+10
    trajectory_values[2*j+1] = j*10+10

ocpZ.set_value(trajectory_ij, trajectory_values)
ocpZ.set_initial(Z_ij, trajectory_values)

ocpZ.solve()