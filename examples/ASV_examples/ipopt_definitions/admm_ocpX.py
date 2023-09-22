from rockit import *
from casadi import *
import numpy as np

# Define problem parameters
max_speed_limit = 0.04

# Problem parameters
nx    = 2                   # the system is composed of 2 states per robot
nu    = 2                   # the system has 2 inputs per robot
Tf    = 2                   # control horizon [s]
Nhor  = 20                  # number of control intervals
dt    = Tf/Nhor             # sample time
number_of_robots = 7        # number of robots that are neighbors (without local)

mu = 1

# Create OCP object
ocpX = Ocp(T=Tf)

# States
x = ocpX.state()
y = ocpX.state()
X = vertcat(x,y)

# Controls
u = ocpX.control()
v = ocpX.control()

# Derivatives
ocpX.set_der(x, u)
ocpX.set_der(y, v)

# Reference parameters
x_ref = ocpX.register_parameter(MX.sym('x_ref', 1))
y_ref = ocpX.register_parameter(MX.sym('y_ref', 1))

# Lagrange objective
ocpX.add_objective(ocpX.sum((x_ref-x)**2 + (y_ref-y)**2))
ocpX.add_objective(ocpX.at_tf((x_ref-x)**2 + (y_ref-y)**2))
ocpX.subject_to( (-max_speed_limit <= u) <= max_speed_limit )
ocpX.subject_to( (-max_speed_limit <= v) <= max_speed_limit )

# Initial condition
X_0 = ocpX.register_parameter(MX.sym('X_0', nx))
ocpX.subject_to(ocpX.at_t0(X)==X_0)

# Parameters for copies and multipliers
lambda_i = ocpX.register_parameter(MX.sym('lambda_i', nx), grid='control', include_last=True)
copy_i = ocpX.register_parameter(MX.sym('copy_i', nx), grid='control', include_last=True)
lambda_ji = ocpX.register_parameter(MX.sym('lambda_ji', nx*number_of_robots), grid='control', include_last=True)
copy_ji = ocpX.register_parameter(MX.sym('copy_ji', nx*number_of_robots), grid='control', include_last=True)

c_i = copy_i - X
term_i = dot(lambda_i, c_i) + mu/2*sumsqr(c_i)
if ocpX.is_signal(term_i):
    term_i = ocpX.sum(term_i,include_last=True)
ocpX.add_objective(term_i)

for j in range(number_of_robots):
    copy_j = vertcat(copy_ji[2*j], copy_ji[2*j+1])
    lambda_j = vertcat(lambda_ji[2*j], lambda_ji[2*j+1])
    c_j = copy_j - X
    term_j = dot(lambda_j, c_j) + mu/2*sumsqr(c_j)
    if ocpX.is_signal(term_j):
        term_j = ocpX.sum(term_j,include_last=True)
    ocpX.add_objective(term_j)

options = {"ipopt": {"print_level": 5}}
options["expand"] = True
options["print_time"] = False
ocpX.solver('ipopt',options)
ocpX.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

#ocpX._method.add_sampler("u", u)
#ocpX._method.add_sampler("v", v)

ocpX.set_value(x_ref, 1.0)
ocpX.set_value(y_ref, 1.0)

lambda_value = np.zeros([nx, Nhor+1])
ocpX.set_value(lambda_i, lambda_value)
copy_value = np.zeros([nx, Nhor+1])
ocpX.set_value(copy_i, copy_value)

lambda_values = np.zeros([nx*number_of_robots, Nhor+1])
ocpX.set_value(lambda_ji, lambda_values)
copy_values = np.zeros([nx*number_of_robots, Nhor+1])
ocpX.set_value(copy_ji, copy_values)

ocpX.set_value(X_0, [0, 0])

ocpX.solve()