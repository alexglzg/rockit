#
#     This file is part of rockit.
#
#     rockit -- Rapid Optimal Control Kit
#     Copyright (C) 2019 MECO, KU Leuven. All rights reserved.
#
#     Rockit is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     Rockit is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#

"""
Model Predictive Control Roboat
================================

"""

from rockit import *
from casadi import *

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Problem parameters
# -------------------------------
max_speed_limit = 0.04
boat_radius = 0.15**2
x0_1 = 1.1
y0_1 = 1.4
x0_2 = 1.8
y0_2 = 0.9
x0_3 = 0.7
y0_3 = 0.3
x0_4 = 3.1
y0_4 = 2.0

xd_1 = 2.9
yd_1 = 1.1
xd_2 = 3.1
yd_2 = 0.9
xd_3 = 2.9
yd_3 = 0.9
xd_4 = 3.1
yd_4 = 1.1

N_init = 200
mu = 1
N_mpc = 1

options = {"ipopt": {"print_level": 1}}
options["expand"] = True
options["print_time"] = False

nx    = 2                   # the system is composed of 2 states per robot
nu    = 2                   # the system has 2 inputs per robot
Tf    = 2                 # control horizon [s]
Nhor  = 20                  # number of control intervals
dt    = Tf/Nhor             # sample time

current_X1 = vertcat(x0_1, y0_1)  # initial state
current_X2 = vertcat(x0_2, y0_2)  # initial state
current_X3 = vertcat(x0_3, y0_3)  # initial state
current_X4 = vertcat(x0_4, y0_4)  # initial state

Nsim  = int(100 * Nhor / Tf)                 # how much samples to simulate

# -------------------------------
# Logging variables
# -------------------------------
x1_history     = np.zeros(Nsim+1)
y1_history   = np.zeros(Nsim+1)
x2_history     = np.zeros(Nsim+1)
y2_history   = np.zeros(Nsim+1)
x3_history     = np.zeros(Nsim+1)
y3_history   = np.zeros(Nsim+1)
x4_history     = np.zeros(Nsim+1)
y4_history   = np.zeros(Nsim+1)

# -------------------------------
# Set OCPs
# -------------------------------

'''
OcpX1
'''
ocpX1 = Ocp(T=Tf)
# States
x1 = ocpX1.state()
y1 = ocpX1.state()
X1 = vertcat(x1,y1)
# Controls
u1 = ocpX1.control()
v1 = ocpX1.control()
# Initial condition
X1_0 = ocpX1.parameter(nx)
# Parameters
X1_lambda_1 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_lambda_21 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_lambda_31 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_lambda_41 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_Z_1 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_Z_21 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_Z_31 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_Z_41 = ocpX1.parameter(nx, grid='control',include_last=True)
#ODE
ocpX1.set_der(x1, u1)
ocpX1.set_der(y1, v1)
# Lagrange objective
ocpX1.add_objective(ocpX1.integral((xd_1-x1)**2 + (yd_1-y1)**2))
ocpX1.add_objective(ocpX1.at_tf((xd_1-x1)**2 + (yd_1-y1)**2))
ocpX1.subject_to( (-max_speed_limit <= u1) <= max_speed_limit )
ocpX1.subject_to( (-max_speed_limit <= v1) <= max_speed_limit )
# Extended objective
X1_c1 = X1_Z_1 - X1
X1_term1 = dot(X1_lambda_1, X1_c1) + mu/2*sumsqr(X1_c1)
if ocpX1.is_signal(X1_term1):
    X1_term1 = ocpX1.sum(X1_term1,include_last=True)
ocpX1.add_objective(X1_term1)
#
X1_c2 = X1_Z_21 - X1
X1_term2 = dot(X1_lambda_21, X1_c2) + mu/2*sumsqr(X1_c2)
if ocpX1.is_signal(X1_term2):
    X1_term2 = ocpX1.sum(X1_term2,include_last=True)
ocpX1.add_objective(X1_term2)
#
X1_c3 = X1_Z_31 - X1
X1_term3 = dot(X1_lambda_31, X1_c3) + mu/2*sumsqr(X1_c3)
if ocpX1.is_signal(X1_term3):
    X1_term3 = ocpX1.sum(X1_term3,include_last=True)
ocpX1.add_objective(X1_term3)
#
X1_c4 = X1_Z_41 - X1
X1_term4 = dot(X1_lambda_41, X1_c4) + mu/2*sumsqr(X1_c4)
if ocpX1.is_signal(X1_term4):
    X1_term4 = ocpX1.sum(X1_term4,include_last=True)
ocpX1.add_objective(X1_term4)
#
# Initial constraints
ocpX1.subject_to(ocpX1.at_t0(X1)==X1_0)
# Pick a solution method
ocpX1.solver('ipopt',options)
# Make it concrete for this ocp
ocpX1.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

'''
OcpZ1
'''
ocpZ1 = Ocp(T=Tf)
# Variables
z1_x1 = ocpZ1.variable(grid='control',include_last=True)
z1_y1 = ocpZ1.variable(grid='control',include_last=True)
Z1 = vertcat(z1_x1, z1_y1)
z1_x2 = ocpZ1.variable(grid='control',include_last=True)
z1_y2 = ocpZ1.variable(grid='control',include_last=True)
Z1_Z_12 = vertcat(z1_x2, z1_y2)
z1_x3 = ocpZ1.variable(grid='control',include_last=True)
z1_y3 = ocpZ1.variable(grid='control',include_last=True)
Z1_Z_13 = vertcat(z1_x3, z1_y3)
z1_x4 = ocpZ1.variable(grid='control',include_last=True)
z1_y4 = ocpZ1.variable(grid='control',include_last=True)
Z1_Z_14 = vertcat(z1_x4, z1_y4)
# Parameters
Z1_lambda_1 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_lambda_12 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_lambda_13 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_lambda_14 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_X_1 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_X_2 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_X_3 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_X_4 = ocpZ1.parameter(nx, grid='control',include_last=True)
# Extended objective
Z1_c1 = Z1 - Z1_X_1
Z1_term1 = dot(Z1_lambda_1, Z1_c1) + mu/2*sumsqr(Z1_c1)
if ocpZ1.is_signal(Z1_term1):
    Z1_term1 = ocpZ1.sum(Z1_term1,include_last=True)
ocpZ1.add_objective(Z1_term1)
#
Z1_c2 = Z1_Z_12 - Z1_X_2
Z1_term2 = dot(Z1_lambda_12, Z1_c2) + mu/2*sumsqr(Z1_c2)
if ocpZ1.is_signal(Z1_term2):
    Z1_term2 = ocpZ1.sum(Z1_term2,include_last=True)
ocpZ1.add_objective(Z1_term2)
#
Z1_c3 = Z1_Z_13 - Z1_X_3
Z1_term3 = dot(Z1_lambda_13, Z1_c3) + mu/2*sumsqr(Z1_c3)
if ocpZ1.is_signal(Z1_term3):
    Z1_term3 = ocpZ1.sum(Z1_term3,include_last=True)
ocpZ1.add_objective(Z1_term3)
#
Z1_c4 = Z1_Z_14 - Z1_X_4
Z1_term4 = dot(Z1_lambda_14, Z1_c4) + mu/2*sumsqr(Z1_c4)
if ocpZ1.is_signal(Z1_term4):
    Z1_term4 = ocpZ1.sum(Z1_term4,include_last=True)
ocpZ1.add_objective(Z1_term4)
# Constraints
Z1_distance1 = (z1_x1-z1_x2)**2 + (z1_y1-z1_y2)**2
Z1_distance2 = (z1_x1-z1_x3)**2 + (z1_y1-z1_y3)**2
Z1_distance3 = (z1_x1-z1_x4)**2 + (z1_y1-z1_y4)**2
ocpZ1.subject_to( Z1_distance1 >= boat_radius )
ocpZ1.subject_to( Z1_distance2 >= boat_radius )
ocpZ1.subject_to( Z1_distance3 >= boat_radius )
# Pick a solution method
ocpZ1.solver('ipopt',options)
# Make it concrete for this ocp
ocpZ1.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

'''
OcpX2
'''
ocpX2 = Ocp(T=Tf)
# States
x2 = ocpX2.state()
y2 = ocpX2.state()
X2 = vertcat(x2,y2)
# Controls
u2 = ocpX2.control()
v2 = ocpX2.control()
# Initial condition
X2_0 = ocpX2.parameter(nx)
# Parameters
X2_lambda_2 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_lambda_12 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_lambda_32 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_lambda_42 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_Z_2 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_Z_12 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_Z_32 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_Z_42 = ocpX2.parameter(nx, grid='control',include_last=True)
#ODE
ocpX2.set_der(x2, u2)
ocpX2.set_der(y2, v2)
# Lagrange objective
ocpX2.add_objective(ocpX2.integral((xd_2-x2)**2 + (yd_2-y2)**2))
ocpX2.add_objective(ocpX2.at_tf((xd_2-x2)**2 + (yd_2-y2)**2))
ocpX2.subject_to( (-max_speed_limit <= u2) <= max_speed_limit )
ocpX2.subject_to( (-max_speed_limit <= v2) <= max_speed_limit )
# Extended objective
X2_c1 = X2_Z_2 - X2
X2_term1 = dot(X2_lambda_2, X2_c1) + mu/2*sumsqr(X2_c1)
if ocpX2.is_signal(X2_term1):
    X2_term1 = ocpX2.sum(X2_term1,include_last=True)
ocpX2.add_objective(X2_term1)
#
X2_c2 = X2_Z_12 - X2
X2_term2 = dot(X2_lambda_12, X2_c2) + mu/2*sumsqr(X2_c2)
if ocpX2.is_signal(X2_term2):
    X2_term2 = ocpX2.sum(X2_term2,include_last=True)
ocpX2.add_objective(X2_term2)
#
X2_c3 = X2_Z_32 - X2
X2_term3 = dot(X2_lambda_32, X2_c3) + mu/2*sumsqr(X2_c3)
if ocpX2.is_signal(X2_term3):
    X2_term3 = ocpX2.sum(X2_term3,include_last=True)
ocpX2.add_objective(X2_term3)
#
X2_c4 = X2_Z_42 - X2
X2_term4 = dot(X2_lambda_42, X2_c4) + mu/2*sumsqr(X2_c4)
if ocpX2.is_signal(X2_term4):
    X2_term4 = ocpX2.sum(X2_term4,include_last=True)
ocpX2.add_objective(X2_term4)
#
# Initial constraints
ocpX2.subject_to(ocpX2.at_t0(X2)==X2_0)
# Pick a solution method
ocpX2.solver('ipopt',options)
# Make it concrete for this ocp
ocpX2.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

'''
OcpZ2
'''
ocpZ2 = Ocp(T=Tf)
# Variables
z2_x2 = ocpZ2.variable(grid='control',include_last=True)
z2_y2 = ocpZ2.variable(grid='control',include_last=True)
Z2 = vertcat(z2_x2, z2_y2)
z2_x1 = ocpZ2.variable(grid='control',include_last=True)
z2_y1 = ocpZ2.variable(grid='control',include_last=True)
Z2_Z_21 = vertcat(z2_x1, z2_y1)
z2_x3 = ocpZ2.variable(grid='control',include_last=True)
z2_y3 = ocpZ2.variable(grid='control',include_last=True)
Z2_Z_23 = vertcat(z2_x3, z2_y3)
z2_x4 = ocpZ2.variable(grid='control',include_last=True)
z2_y4 = ocpZ2.variable(grid='control',include_last=True)
Z2_Z_24 = vertcat(z2_x4, z2_y4)
# Parameters
Z2_lambda_2 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_lambda_21 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_lambda_23 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_lambda_24 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_X_1 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_X_2 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_X_3 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_X_4 = ocpZ2.parameter(nx, grid='control',include_last=True)
# Extended objective
Z2_c1 = Z2 - Z2_X_2
Z2_term1 = dot(Z2_lambda_2, Z2_c1) + mu/2*sumsqr(Z2_c1)
if ocpZ2.is_signal(Z2_term1):
    Z2_term1 = ocpZ2.sum(Z2_term1,include_last=True)
ocpZ2.add_objective(Z2_term1)
#
Z2_c2 = Z2_Z_21 - Z2_X_1
Z2_term2 = dot(Z2_lambda_21, Z2_c2) + mu/2*sumsqr(Z2_c2)
if ocpZ2.is_signal(Z2_term2):
    Z2_term2 = ocpZ2.sum(Z2_term2,include_last=True)
ocpZ2.add_objective(Z2_term2)
#
Z2_c3 = Z2_Z_23 - Z2_X_3
Z2_term3 = dot(Z2_lambda_23, Z2_c3) + mu/2*sumsqr(Z2_c3)
if ocpZ2.is_signal(Z2_term3):
    Z2_term3 = ocpZ2.sum(Z2_term3,include_last=True)
ocpZ2.add_objective(Z2_term3)
#
Z2_c4 = Z2_Z_24 - Z2_X_4
Z2_term4 = dot(Z2_lambda_24, Z2_c4) + mu/2*sumsqr(Z2_c4)
if ocpZ2.is_signal(Z2_term4):
    Z2_term4 = ocpZ2.sum(Z2_term4,include_last=True)
ocpZ2.add_objective(Z2_term4)
# Constraints
Z2_distance1 = (z2_x1-z2_x2)**2 + (z2_y1-z2_y2)**2
Z2_distance4 = (z2_x2-z2_x3)**2 + (z2_y2-z2_y3)**2
Z2_distance5 = (z2_x2-z2_x4)**2 + (z2_y2-z2_y4)**2
ocpZ2.subject_to( Z2_distance1 >= boat_radius )
ocpZ2.subject_to( Z2_distance4 >= boat_radius )
ocpZ2.subject_to( Z2_distance5 >= boat_radius )
# Pick a solution method
ocpZ2.solver('ipopt',options)
# Make it concrete for this ocp
ocpZ2.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

'''
OcpX3
'''
ocpX3 = Ocp(T=Tf)
# States
x3 = ocpX3.state()
y3 = ocpX3.state()
X3 = vertcat(x3,y3)
# Controls
u3 = ocpX3.control()
v3 = ocpX3.control()
# Initial condition
X3_0 = ocpX3.parameter(nx)
# Parameters
X3_lambda_3 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_lambda_13 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_lambda_23 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_lambda_43 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_Z_3 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_Z_13 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_Z_23 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_Z_43 = ocpX3.parameter(nx, grid='control',include_last=True)
#ODE
ocpX3.set_der(x3, u3)
ocpX3.set_der(y3, v3)
# Lagrange objective
ocpX3.add_objective(ocpX3.integral((xd_3-x3)**2 + (yd_3-y3)**2))
ocpX3.add_objective(ocpX3.at_tf((xd_3-x3)**2 + (yd_3-y3)**2))
ocpX3.subject_to( (-max_speed_limit <= u3) <= max_speed_limit )
ocpX3.subject_to( (-max_speed_limit <= v3) <= max_speed_limit )
# Extended objective
X3_c1 = X3_Z_3 - X3
X3_term1 = dot(X3_lambda_3, X3_c1) + mu/2*sumsqr(X3_c1)
if ocpX3.is_signal(X3_term1):
    X3_term1 = ocpX3.sum(X3_term1,include_last=True)
ocpX3.add_objective(X3_term1)
#
X3_c2 = X3_Z_13 - X3
X3_term2 = dot(X3_lambda_13, X3_c2) + mu/2*sumsqr(X3_c2)
if ocpX3.is_signal(X3_term2):
    X3_term2 = ocpX3.sum(X3_term2,include_last=True)
ocpX3.add_objective(X3_term2)
#
X3_c3 = X3_Z_23 - X3
X3_term3 = dot(X3_lambda_23, X3_c3) + mu/2*sumsqr(X3_c3)
if ocpX3.is_signal(X3_term3):
    X3_term3 = ocpX3.sum(X3_term3,include_last=True)
ocpX3.add_objective(X3_term3)
#
X3_c4 = X3_Z_43 - X3
X3_term4 = dot(X3_lambda_43, X3_c4) + mu/2*sumsqr(X3_c4)
if ocpX3.is_signal(X3_term4):
    X3_term4 = ocpX3.sum(X3_term4,include_last=True)
ocpX3.add_objective(X3_term4)
#
# Initial constraints
ocpX3.subject_to(ocpX3.at_t0(X3)==X3_0)
# Pick a solution method
ocpX3.solver('ipopt',options)
# Make it concrete for this ocp
ocpX3.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

'''
OcpZ3
'''
ocpZ3 = Ocp(T=Tf)
# Variables
z3_x3 = ocpZ3.variable(grid='control',include_last=True)
z3_y3 = ocpZ3.variable(grid='control',include_last=True)
Z3 = vertcat(z3_x3, z3_y3)
z3_x1 = ocpZ3.variable(grid='control',include_last=True)
z3_y1 = ocpZ3.variable(grid='control',include_last=True)
Z3_Z_31 = vertcat(z3_x1, z3_y1)
z3_x2 = ocpZ3.variable(grid='control',include_last=True)
z3_y2 = ocpZ3.variable(grid='control',include_last=True)
Z3_Z_32 = vertcat(z3_x2, z3_y2)
z3_x4 = ocpZ3.variable(grid='control',include_last=True)
z3_y4 = ocpZ3.variable(grid='control',include_last=True)
Z3_Z_34 = vertcat(z3_x4, z3_y4)
# Parameters
Z3_lambda_3 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_lambda_31 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_lambda_32 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_lambda_34 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_X_1 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_X_2 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_X_3 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_X_4 = ocpZ3.parameter(nx, grid='control',include_last=True)
# Extended objective
Z3_c1 = Z3 - Z3_X_3
Z3_term1 = dot(Z3_lambda_3, Z3_c1) + mu/2*sumsqr(Z3_c1)
if ocpZ3.is_signal(Z3_term1):
    Z3_term1 = ocpZ3.sum(Z3_term1,include_last=True)
ocpZ3.add_objective(Z3_term1)
#
Z3_c2 = Z3_Z_31 - Z3_X_1
Z3_term2 = dot(Z3_lambda_31, Z3_c2) + mu/2*sumsqr(Z3_c2)
if ocpZ3.is_signal(Z3_term2):
    Z3_term2 = ocpZ3.sum(Z3_term2,include_last=True)
ocpZ3.add_objective(Z3_term2)
#
Z3_c3 = Z3_Z_32 - Z3_X_2
Z3_term3 = dot(Z3_lambda_32, Z3_c3) + mu/2*sumsqr(Z3_c3)
if ocpZ3.is_signal(Z3_term3):
    Z3_term3 = ocpZ3.sum(Z3_term3,include_last=True)
ocpZ3.add_objective(Z3_term3)
#
Z3_c4 = Z3_Z_34 - Z3_X_4
Z3_term4 = dot(Z3_lambda_34, Z3_c4) + mu/2*sumsqr(Z3_c4)
if ocpZ3.is_signal(Z3_term4):
    Z3_term4 = ocpZ3.sum(Z3_term4,include_last=True)
ocpZ3.add_objective(Z3_term4)
# Constraints
Z3_distance2 = (z3_x1-z3_x3)**2 + (z3_y1-z3_y3)**2
Z3_distance4 = (z3_x2-z3_x3)**2 + (z3_y2-z3_y3)**2
Z3_distance6 = (z3_x3-z3_x4)**2 + (z3_y3-z3_y4)**2
ocpZ3.subject_to( Z3_distance2 >= boat_radius )
ocpZ3.subject_to( Z3_distance4 >= boat_radius )
ocpZ3.subject_to( Z3_distance6 >= boat_radius )
# Pick a solution method
ocpZ3.solver('ipopt',options)
# Make it concrete for this ocp
ocpZ3.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

'''
OcpX4
'''
ocpX4 = Ocp(T=Tf)
# States
x4 = ocpX4.state()
y4 = ocpX4.state()
X4 = vertcat(x4,y4)
# Controls
u4 = ocpX4.control()
v4 = ocpX4.control()
# Initial condition
X4_0 = ocpX4.parameter(nx)
# Parameters
X4_lambda_4 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_lambda_14 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_lambda_24 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_lambda_34 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_Z_4 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_Z_14 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_Z_24 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_Z_34 = ocpX4.parameter(nx, grid='control',include_last=True)
#ODE
ocpX4.set_der(x4, u4)
ocpX4.set_der(y4, v4)
# Lagrange objective
ocpX4.add_objective(ocpX4.integral((xd_4-x4)**2 + (yd_4-y4)**2))
ocpX4.add_objective(ocpX4.at_tf((xd_4-x4)**2 + (yd_4-y4)**2))
ocpX4.subject_to( (-max_speed_limit <= u4) <= max_speed_limit )
ocpX4.subject_to( (-max_speed_limit <= v4) <= max_speed_limit )
# Extended objective
X4_c1 = X4_Z_4 - X4
X4_term1 = dot(X4_lambda_4, X4_c1) + mu/2*sumsqr(X4_c1)
if ocpX4.is_signal(X4_term1):
    X4_term1 = ocpX4.sum(X4_term1,include_last=True)
ocpX4.add_objective(X4_term1)
#
X4_c2 = X4_Z_14 - X4
X4_term2 = dot(X4_lambda_14, X4_c2) + mu/2*sumsqr(X4_c2)
if ocpX4.is_signal(X4_term2):
    X4_term2 = ocpX4.sum(X4_term2,include_last=True)
ocpX4.add_objective(X4_term2)
#
X4_c3 = X4_Z_24 - X4
X4_term3 = dot(X4_lambda_24, X4_c3) + mu/2*sumsqr(X4_c3)
if ocpX4.is_signal(X4_term3):
    X4_term3 = ocpX4.sum(X4_term3,include_last=True)
ocpX4.add_objective(X4_term3)
#
X4_c4 = X4_Z_34 - X4
X4_term4 = dot(X4_lambda_34, X4_c4) + mu/2*sumsqr(X4_c4)
if ocpX4.is_signal(X4_term4):
    X4_term4 = ocpX4.sum(X4_term4,include_last=True)
ocpX4.add_objective(X4_term4)
#
# Initial constraints
ocpX4.subject_to(ocpX4.at_t0(X4)==X4_0)
# Pick a solution method
ocpX4.solver('ipopt',options)
# Make it concrete for this ocp
ocpX4.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

'''
OcpZ4
'''
ocpZ4 = Ocp(T=Tf)
# Variables
z4_x4 = ocpZ4.variable(grid='control',include_last=True)
z4_y4 = ocpZ4.variable(grid='control',include_last=True)
Z4 = vertcat(z4_x4, z4_y4)
z4_x1 = ocpZ4.variable(grid='control',include_last=True)
z4_y1 = ocpZ4.variable(grid='control',include_last=True)
Z4_Z_41 = vertcat(z4_x1, z4_y1)
z4_x2 = ocpZ4.variable(grid='control',include_last=True)
z4_y2 = ocpZ4.variable(grid='control',include_last=True)
Z4_Z_42 = vertcat(z4_x2, z4_y2)
z4_x3 = ocpZ4.variable(grid='control',include_last=True)
z4_y3 = ocpZ4.variable(grid='control',include_last=True)
Z4_Z_43 = vertcat(z4_x3, z4_y3)
# Parameters
Z4_lambda_4 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_lambda_41 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_lambda_42 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_lambda_43 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_X_1 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_X_2 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_X_3 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_X_4 = ocpZ4.parameter(nx, grid='control',include_last=True)
# Extended objective
Z4_c1 = Z4 - Z4_X_4
Z4_term1 = dot(Z4_lambda_4, Z4_c1) + mu/2*sumsqr(Z4_c1)
if ocpZ4.is_signal(Z4_term1):
    Z4_term1 = ocpZ4.sum(Z4_term1,include_last=True)
ocpZ4.add_objective(Z4_term1)
#
Z4_c2 = Z4_Z_41 - Z4_X_1
Z4_term2 = dot(Z4_lambda_41, Z4_c2) + mu/2*sumsqr(Z4_c2)
if ocpZ4.is_signal(Z4_term2):
    Z4_term2 = ocpZ4.sum(Z4_term2,include_last=True)
ocpZ4.add_objective(Z4_term2)
#
Z4_c3 = Z4_Z_42 - Z4_X_2
Z4_term3 = dot(Z4_lambda_42, Z4_c3) + mu/2*sumsqr(Z4_c3)
if ocpZ4.is_signal(Z4_term3):
    Z4_term3 = ocpZ4.sum(Z4_term3,include_last=True)
ocpZ4.add_objective(Z4_term3)
#
Z4_c4 = Z4_Z_43 - Z4_X_3
Z4_term4 = dot(Z4_lambda_43, Z4_c4) + mu/2*sumsqr(Z4_c4)
if ocpZ4.is_signal(Z4_term4):
    Z4_term4 = ocpZ4.sum(Z4_term4,include_last=True)
ocpZ4.add_objective(Z4_term4)
# Constraints
Z4_distance3 = (z4_x1-z4_x4)**2 + (z4_y1-z4_y4)**2
Z4_distance5 = (z4_x2-z4_x4)**2 + (z4_y2-z4_y4)**2
Z4_distance6 = (z4_x3-z4_x4)**2 + (z4_y3-z4_y4)**2
ocpZ4.subject_to( Z4_distance3 >= boat_radius )
ocpZ4.subject_to( Z4_distance5 >= boat_radius )
ocpZ4.subject_to( Z4_distance6 >= boat_radius )
# Pick a solution method
ocpZ4.solver('ipopt',options)
# Make it concrete for this ocp
ocpZ4.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

# Initialize all values
l1 = np.zeros([nx, Nhor+1])
l2 = np.zeros([nx, Nhor+1])
l3 = np.zeros([nx, Nhor+1])
l4 = np.zeros([nx, Nhor+1])

l12 = np.zeros([nx, Nhor+1])
l13 = np.zeros([nx, Nhor+1])
l14 = np.zeros([nx, Nhor+1])
l21 = np.zeros([nx, Nhor+1])
l23 = np.zeros([nx, Nhor+1])
l24 = np.zeros([nx, Nhor+1])
l31 = np.zeros([nx, Nhor+1])
l32 = np.zeros([nx, Nhor+1])
l34 = np.zeros([nx, Nhor+1])
l41 = np.zeros([nx, Nhor+1])
l42 = np.zeros([nx, Nhor+1])
l43 = np.zeros([nx, Nhor+1])

z1 = np.zeros([nx, Nhor+1])
z2 = np.zeros([nx, Nhor+1])
z3 = np.zeros([nx, Nhor+1])
z4 = np.zeros([nx, Nhor+1])

z12 = np.zeros([nx, Nhor+1])
z13 = np.zeros([nx, Nhor+1])
z14 = np.zeros([nx, Nhor+1])
z21 = np.zeros([nx, Nhor+1])
z23 = np.zeros([nx, Nhor+1])
z24 = np.zeros([nx, Nhor+1])
z31 = np.zeros([nx, Nhor+1])
z32 = np.zeros([nx, Nhor+1])
z34 = np.zeros([nx, Nhor+1])
z41 = np.zeros([nx, Nhor+1])
z42 = np.zeros([nx, Nhor+1])
z43 = np.zeros([nx, Nhor+1])

X1p = np.zeros([nx, Nhor+1]) + [[x0_1],[y0_1]]
X2p = np.zeros([nx, Nhor+1]) + [[x0_2],[y0_2]]
X3p = np.zeros([nx, Nhor+1]) + [[x0_3],[y0_3]]
X4p = np.zeros([nx, Nhor+1]) + [[x0_4],[y0_4]]

# Dynamics declaration
Sim_asv_dyn = ocpX1._method.discrete_system(ocpX1)

# Log data for post-processing
x1_history[0]   = current_X1[0]
y1_history[0] = current_X1[1]
x2_history[0]   = current_X2[0]
y2_history[0] = current_X2[1]
x3_history[0]   = current_X3[0]
y3_history[0] = current_X3[1]
x4_history[0]   = current_X4[0]
y4_history[0] = current_X4[1]

#Initialization ADMM

for i in range(N_init):

    # Set values and solve for each agent ocpX

    ocpX1.set_value(X1_0, current_X1)
    ocpX1.set_value(X1_lambda_1, l1)
    ocpX1.set_value(X1_lambda_21, l21)
    ocpX1.set_value(X1_lambda_31, l31)
    ocpX1.set_value(X1_lambda_41, l41)
    ocpX1.set_value(X1_Z_1, z1)
    ocpX1.set_value(X1_Z_21, z21)
    ocpX1.set_value(X1_Z_31, z31)
    ocpX1.set_value(X1_Z_41, z41)

    solX1 = ocpX1.solve()

    ocpX2.set_value(X2_0, current_X2)
    ocpX2.set_value(X2_lambda_2, l2)
    ocpX2.set_value(X2_lambda_12, l12)
    ocpX2.set_value(X2_lambda_32, l32)
    ocpX2.set_value(X2_lambda_42, l42)
    ocpX2.set_value(X2_Z_2, z2)
    ocpX2.set_value(X2_Z_12, z12)
    ocpX2.set_value(X2_Z_32, z32)
    ocpX2.set_value(X2_Z_42, z42)

    solX2 = ocpX2.solve()

    ocpX3.set_value(X3_0, current_X3)
    ocpX3.set_value(X3_lambda_3, l3)
    ocpX3.set_value(X3_lambda_13, l13)
    ocpX3.set_value(X3_lambda_23, l23)
    ocpX3.set_value(X3_lambda_43, l43)
    ocpX3.set_value(X3_Z_3, z3)
    ocpX3.set_value(X3_Z_13, z13)
    ocpX3.set_value(X3_Z_23, z23)
    ocpX3.set_value(X3_Z_43, z43)

    solX3 = ocpX3.solve()

    ocpX4.set_value(X4_0, current_X4)
    ocpX4.set_value(X4_lambda_4, l4)
    ocpX4.set_value(X4_lambda_14, l14)
    ocpX4.set_value(X4_lambda_24, l24)
    ocpX4.set_value(X4_lambda_34, l34)
    ocpX4.set_value(X4_Z_4, z4)
    ocpX4.set_value(X4_Z_14, z14)
    ocpX4.set_value(X4_Z_24, z24)
    ocpX4.set_value(X4_Z_34, z34)

    solX4 = ocpX4.solve()

    # Save the information

    X1p = solX1.sample(X1, grid='control')[1].T
    X2p = solX2.sample(X2, grid='control')[1].T
    X3p = solX3.sample(X3, grid='control')[1].T
    X4p = solX4.sample(X4, grid='control')[1].T
    
    # Set values and solve for each agent ocpZ

    ocpZ1.set_value(Z1_lambda_1, l1)
    ocpZ1.set_value(Z1_lambda_12, l12)
    ocpZ1.set_value(Z1_lambda_13, l13)
    ocpZ1.set_value(Z1_lambda_14, l14)
    ocpZ1.set_value(Z1_X_1, X1p)
    ocpZ1.set_value(Z1_X_2, X2p)
    ocpZ1.set_value(Z1_X_3, X3p)
    ocpZ1.set_value(Z1_X_4, X4p)

    ocpZ2.set_value(Z2_lambda_2, l2)
    ocpZ2.set_value(Z2_lambda_21, l21)
    ocpZ2.set_value(Z2_lambda_23, l23)
    ocpZ2.set_value(Z2_lambda_24, l24)
    ocpZ2.set_value(Z2_X_1, X1p)
    ocpZ2.set_value(Z2_X_2, X2p)
    ocpZ2.set_value(Z2_X_3, X3p)
    ocpZ2.set_value(Z2_X_4, X4p)

    ocpZ3.set_value(Z3_lambda_3, l3)
    ocpZ3.set_value(Z3_lambda_31, l31)
    ocpZ3.set_value(Z3_lambda_32, l32)
    ocpZ3.set_value(Z3_lambda_34, l34)
    ocpZ3.set_value(Z3_X_1, X1p)
    ocpZ3.set_value(Z3_X_2, X2p)
    ocpZ3.set_value(Z3_X_3, X3p)
    ocpZ3.set_value(Z3_X_4, X4p)

    ocpZ4.set_value(Z4_lambda_4, l4)
    ocpZ4.set_value(Z4_lambda_41, l41)
    ocpZ4.set_value(Z4_lambda_42, l42)
    ocpZ4.set_value(Z4_lambda_43, l43)
    ocpZ4.set_value(Z4_X_1, X1p)
    ocpZ4.set_value(Z4_X_2, X2p)
    ocpZ4.set_value(Z4_X_3, X3p)
    ocpZ4.set_value(Z4_X_4, X4p)

    solZ1 = ocpZ1.solve()
    solZ2 = ocpZ2.solve()
    solZ3 = ocpZ3.solve()
    solZ4 = ocpZ4.solve()

    # Compute new Z parameters

    z1 = solZ1.sample(Z1, grid='control')[1].T
    z12 = solZ1.sample(Z1_Z_12, grid='control')[1].T
    z13 = solZ1.sample(Z1_Z_13, grid='control')[1].T
    z14 = solZ1.sample(Z1_Z_14, grid='control')[1].T

    z2 = solZ2.sample(Z2, grid='control')[1].T
    z21 = solZ2.sample(Z2_Z_21, grid='control')[1].T
    z23 = solZ2.sample(Z2_Z_23, grid='control')[1].T
    z24 = solZ2.sample(Z2_Z_24, grid='control')[1].T

    z3 = solZ3.sample(Z3, grid='control')[1].T
    z31 = solZ3.sample(Z3_Z_31, grid='control')[1].T
    z32 = solZ3.sample(Z3_Z_32, grid='control')[1].T
    z34 = solZ3.sample(Z3_Z_34, grid='control')[1].T

    z4 = solZ4.sample(Z4, grid='control')[1].T
    z41 = solZ4.sample(Z4_Z_41, grid='control')[1].T
    z42 = solZ4.sample(Z4_Z_42, grid='control')[1].T
    z43 = solZ4.sample(Z4_Z_43, grid='control')[1].T

    # Update lambda multipliers

    l1 = l1 + mu*(z1 - X1p)
    l12 = l12 + mu*(z12 - X2p)
    l13 = l13 + mu*(z13 - X3p)
    l14 = l14 + mu*(z14 - X4p)

    l2 = l2 + mu*(z2 - X2p)
    l21 = l21 + mu*(z21 - X1p)
    l23 = l23 + mu*(z23 - X3p)
    l24 = l24 + mu*(z24 - X4p)

    l3 = l3 + mu*(z3 - X3p)
    l31 = l31 + mu*(z31 - X1p)
    l32 = l32 + mu*(z32 - X2p)
    l34 = l34 + mu*(z34 - X4p)

    l4 = l4 + mu*(z4 - X4p)
    l41 = l41 + mu*(z41 - X1p)
    l42 = l42 + mu*(z42 - X2p)
    l43 = l43 + mu*(z43 - X3p)

    residuals = []
    residuals.append(norm_fro(z1-X1p))
    residuals.append(norm_fro(z21-X1p))
    residuals.append(norm_fro(z31-X1p))
    residuals.append(norm_fro(z41-X1p))
    residuals.append(norm_fro(z2-X2p))
    residuals.append(norm_fro(z12-X2p))
    residuals.append(norm_fro(z32-X2p))#
    residuals.append(norm_fro(z42-X2p))
    residuals.append(norm_fro(z3-X3p))
    residuals.append(norm_fro(z13-X3p))
    residuals.append(norm_fro(z23-X3p))
    residuals.append(norm_fro(z43-X3p))
    residuals.append(norm_fro(z4-X4p))
    residuals.append(norm_fro(z14-X4p))
    residuals.append(norm_fro(z24-X4p))
    residuals.append(norm_fro(z34-X4p))

    print("iteration", i+1, "of", N_init, residuals)

# MPC with ADMM

for j in range(Nsim):

    # Get the solution from sol
    tsx1, u1sol = solX1.sample(u1, grid='control')
    _, v1sol = solX1.sample(v1, grid='control')
    tsx2, u2sol = solX2.sample(u2, grid='control')
    _, v2sol = solX2.sample(v2, grid='control')
    tsx3, u3sol = solX3.sample(u3, grid='control')
    _, v3sol = solX3.sample(v3, grid='control')
    tsx4, u4sol = solX4.sample(u4, grid='control')
    _, v4sol = solX4.sample(v4, grid='control')
    # Simulate dynamics (applying the first control input) and update the current state
    current_X1 = Sim_asv_dyn(x0=current_X1, u=vertcat(u1sol[0],v1sol[0]), T=dt)["xf"]
    current_X2 = Sim_asv_dyn(x0=current_X2, u=vertcat(u2sol[0],v2sol[0]), T=dt)["xf"]
    current_X3 = Sim_asv_dyn(x0=current_X3, u=vertcat(u3sol[0],v3sol[0]), T=dt)["xf"]
    current_X4 = Sim_asv_dyn(x0=current_X4, u=vertcat(u4sol[0],v4sol[0]), T=dt)["xf"]

    # Log data for post-processing
    x1_history[j+1] = current_X1[0].full()
    y1_history[j+1] = current_X1[1].full()
    x2_history[j+1] = current_X2[0].full()
    y2_history[j+1] = current_X2[1].full()
    x3_history[j+1] = current_X3[0].full()
    y3_history[j+1] = current_X3[1].full()
    x4_history[j+1] = current_X4[0].full()
    y4_history[j+1] = current_X4[1].full()

    for i in range(N_mpc):

        # Set values and solve for each agent ocpX

        ocpX1.set_value(X1_0, current_X1)
        ocpX1.set_value(X1_lambda_1, l1)
        ocpX1.set_value(X1_lambda_21, l21)
        ocpX1.set_value(X1_lambda_31, l31)
        ocpX1.set_value(X1_lambda_41, l41)
        ocpX1.set_value(X1_Z_1, z1)
        ocpX1.set_value(X1_Z_21, z21)
        ocpX1.set_value(X1_Z_31, z31)
        ocpX1.set_value(X1_Z_41, z41)

        solX1 = ocpX1.solve()

        ocpX2.set_value(X2_0, current_X2)
        ocpX2.set_value(X2_lambda_2, l2)
        ocpX2.set_value(X2_lambda_12, l12)
        ocpX2.set_value(X2_lambda_32, l32)
        ocpX2.set_value(X2_lambda_42, l42)
        ocpX2.set_value(X2_Z_2, z2)
        ocpX2.set_value(X2_Z_12, z12)
        ocpX2.set_value(X2_Z_32, z32)
        ocpX2.set_value(X2_Z_42, z42)

        solX2 = ocpX2.solve()

        ocpX3.set_value(X3_0, current_X3)
        ocpX3.set_value(X3_lambda_3, l3)
        ocpX3.set_value(X3_lambda_13, l13)
        ocpX3.set_value(X3_lambda_23, l23)
        ocpX3.set_value(X3_lambda_43, l43)
        ocpX3.set_value(X3_Z_3, z3)
        ocpX3.set_value(X3_Z_13, z13)
        ocpX3.set_value(X3_Z_23, z23)
        ocpX3.set_value(X3_Z_43, z43)

        solX3 = ocpX3.solve()

        ocpX4.set_value(X4_0, current_X4)
        ocpX4.set_value(X4_lambda_4, l4)
        ocpX4.set_value(X4_lambda_14, l14)
        ocpX4.set_value(X4_lambda_24, l24)
        ocpX4.set_value(X4_lambda_34, l34)
        ocpX4.set_value(X4_Z_4, z4)
        ocpX4.set_value(X4_Z_14, z14)
        ocpX4.set_value(X4_Z_24, z24)
        ocpX4.set_value(X4_Z_34, z34)

        solX4 = ocpX4.solve()

        # Save the information

        X1p = solX1.sample(X1, grid='control')[1].T
        X2p = solX2.sample(X2, grid='control')[1].T
        X3p = solX3.sample(X3, grid='control')[1].T
        X4p = solX4.sample(X4, grid='control')[1].T
        
        # Set values and solve for each agent ocpZ

        ocpZ1.set_value(Z1_lambda_1, l1)
        ocpZ1.set_value(Z1_lambda_12, l12)
        ocpZ1.set_value(Z1_lambda_13, l13)
        ocpZ1.set_value(Z1_lambda_14, l14)
        ocpZ1.set_value(Z1_X_1, X1p)
        ocpZ1.set_value(Z1_X_2, X2p)
        ocpZ1.set_value(Z1_X_3, X3p)
        ocpZ1.set_value(Z1_X_4, X4p)

        ocpZ2.set_value(Z2_lambda_2, l2)
        ocpZ2.set_value(Z2_lambda_21, l21)
        ocpZ2.set_value(Z2_lambda_23, l23)
        ocpZ2.set_value(Z2_lambda_24, l24)
        ocpZ2.set_value(Z2_X_1, X1p)
        ocpZ2.set_value(Z2_X_2, X2p)
        ocpZ2.set_value(Z2_X_3, X3p)
        ocpZ2.set_value(Z2_X_4, X4p)

        ocpZ3.set_value(Z3_lambda_3, l3)
        ocpZ3.set_value(Z3_lambda_31, l31)
        ocpZ3.set_value(Z3_lambda_32, l32)
        ocpZ3.set_value(Z3_lambda_34, l34)
        ocpZ3.set_value(Z3_X_1, X1p)
        ocpZ3.set_value(Z3_X_2, X2p)
        ocpZ3.set_value(Z3_X_3, X3p)
        ocpZ3.set_value(Z3_X_4, X4p)

        ocpZ4.set_value(Z4_lambda_4, l4)
        ocpZ4.set_value(Z4_lambda_41, l41)
        ocpZ4.set_value(Z4_lambda_42, l42)
        ocpZ4.set_value(Z4_lambda_43, l43)
        ocpZ4.set_value(Z4_X_1, X1p)
        ocpZ4.set_value(Z4_X_2, X2p)
        ocpZ4.set_value(Z4_X_3, X3p)
        ocpZ4.set_value(Z4_X_4, X4p)

        solZ1 = ocpZ1.solve()
        solZ2 = ocpZ2.solve()
        solZ3 = ocpZ3.solve()
        solZ4 = ocpZ4.solve()

        # Compute new Z parameters

        z1 = solZ1.sample(Z1, grid='control')[1].T
        z12 = solZ1.sample(Z1_Z_12, grid='control')[1].T
        z13 = solZ1.sample(Z1_Z_13, grid='control')[1].T
        z14 = solZ1.sample(Z1_Z_14, grid='control')[1].T

        z2 = solZ2.sample(Z2, grid='control')[1].T
        z21 = solZ2.sample(Z2_Z_21, grid='control')[1].T
        z23 = solZ2.sample(Z2_Z_23, grid='control')[1].T
        z24 = solZ2.sample(Z2_Z_24, grid='control')[1].T

        z3 = solZ3.sample(Z3, grid='control')[1].T
        z31 = solZ3.sample(Z3_Z_31, grid='control')[1].T
        z32 = solZ3.sample(Z3_Z_32, grid='control')[1].T
        z34 = solZ3.sample(Z3_Z_34, grid='control')[1].T

        z4 = solZ4.sample(Z4, grid='control')[1].T
        z41 = solZ4.sample(Z4_Z_41, grid='control')[1].T
        z42 = solZ4.sample(Z4_Z_42, grid='control')[1].T
        z43 = solZ4.sample(Z4_Z_43, grid='control')[1].T

        # Update lambda multipliers

        l1 = l1 + mu*(z1 - X1p)
        l12 = l12 + mu*(z12 - X2p)
        l13 = l13 + mu*(z13 - X3p)
        l14 = l14 + mu*(z14 - X4p)

        l2 = l2 + mu*(z2 - X2p)
        l21 = l21 + mu*(z21 - X1p)
        l23 = l23 + mu*(z23 - X3p)
        l24 = l24 + mu*(z24 - X4p)

        l3 = l3 + mu*(z3 - X3p)
        l31 = l31 + mu*(z31 - X1p)
        l32 = l32 + mu*(z32 - X2p)
        l34 = l34 + mu*(z34 - X4p)

        l4 = l4 + mu*(z4 - X4p)
        l41 = l41 + mu*(z41 - X1p)
        l42 = l42 + mu*(z42 - X2p)
        l43 = l43 + mu*(z43 - X3p)

        residuals = []
        residuals.append(norm_fro(z1-X1p))
        residuals.append(norm_fro(z21-X1p))
        residuals.append(norm_fro(z31-X1p))
        residuals.append(norm_fro(z41-X1p))
        residuals.append(norm_fro(z2-X2p))
        residuals.append(norm_fro(z12-X2p))
        residuals.append(norm_fro(z32-X2p))#
        residuals.append(norm_fro(z42-X2p))
        residuals.append(norm_fro(z3-X3p))
        residuals.append(norm_fro(z13-X3p))
        residuals.append(norm_fro(z23-X3p))
        residuals.append(norm_fro(z43-X3p))
        residuals.append(norm_fro(z4-X4p))
        residuals.append(norm_fro(z14-X4p))
        residuals.append(norm_fro(z24-X4p))
        residuals.append(norm_fro(z34-X4p))

        
    print("timestep", j+1, "of", Nsim)


fig1, ax1 = plt.subplots()
ax1.plot(y1_history, x1_history, 'bo')
ax1.plot(y2_history, x2_history, 'r-')
ax1.plot(y3_history, x3_history, 'g--')
ax1.plot(y4_history, x4_history, 'c.-')
ax1.set_xlabel('Y [m]')
ax1.set_ylabel('X [m]')
plt.axis(equal=True)


plt.show()