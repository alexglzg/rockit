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
x0_1 = 0.3
y0_1 = 2.4
x0_2 = 1.8
y0_2 = 2.8
x0_3 = 0.7
y0_3 = 0.3
x0_4 = 1.1
y0_4 = 2.0
x0_5 = 1.9
y0_5 = 0.8
x0_6 = 0.5
y0_6 = 0.8
x0_7 = 1.3
y0_7 = 1.7
x0_8 = 2.1
y0_8 = 0.2

xd_1 = 2.32
yd_1 = 1.18
xd_2 = 1.68
yd_2 = 1.18
xd_3 = 1.68
yd_3 = 1.50
xd_4 = 2.00
yd_4 = 1.18
xd_5 = 2.00
yd_5 = 1.82
xd_6 = 2.32
yd_6 = 1.50
xd_7 = 2.32
yd_7 = 1.82
xd_8 = 1.68
yd_8 = 1.82

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
current_X5 = vertcat(x0_5, y0_5)  # initial state
current_X6 = vertcat(x0_6, y0_6)  # initial state
current_X7 = vertcat(x0_7, y0_7)  # initial state
current_X8 = vertcat(x0_8, y0_8)  # initial state

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
x5_history     = np.zeros(Nsim+1)
y5_history   = np.zeros(Nsim+1)
x6_history     = np.zeros(Nsim+1)
y6_history   = np.zeros(Nsim+1)
x7_history     = np.zeros(Nsim+1)
y7_history   = np.zeros(Nsim+1)
x8_history     = np.zeros(Nsim+1)
y8_history   = np.zeros(Nsim+1)

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
X1_lambda_11 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_lambda_21 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_lambda_31 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_lambda_41 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_lambda_51 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_lambda_61 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_lambda_71 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_lambda_81 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_Z_11 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_Z_21 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_Z_31 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_Z_41 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_Z_51 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_Z_61 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_Z_71 = ocpX1.parameter(nx, grid='control',include_last=True)
X1_Z_81 = ocpX1.parameter(nx, grid='control',include_last=True)
#ODE
ocpX1.set_der(x1, u1)
ocpX1.set_der(y1, v1)
# Lagrange objective
ocpX1.add_objective(ocpX1.integral((xd_1-x1)**2 + (yd_1-y1)**2))
ocpX1.add_objective(ocpX1.at_tf((xd_1-x1)**2 + (yd_1-y1)**2))
ocpX1.subject_to( (-max_speed_limit <= u1) <= max_speed_limit )
ocpX1.subject_to( (-max_speed_limit <= v1) <= max_speed_limit )
# Extended objective
X1_c1 = X1_Z_11 - X1
X1_term1 = dot(X1_lambda_11, X1_c1) + mu/2*sumsqr(X1_c1)
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
X1_c5 = X1_Z_51 - X1
X1_term5 = dot(X1_lambda_51, X1_c5) + mu/2*sumsqr(X1_c5)
if ocpX1.is_signal(X1_term5):
    X1_term5 = ocpX1.sum(X1_term5,include_last=True)
ocpX1.add_objective(X1_term5)
#
X1_c6 = X1_Z_61 - X1
X1_term6 = dot(X1_lambda_61, X1_c6) + mu/2*sumsqr(X1_c6)
if ocpX1.is_signal(X1_term6):
    X1_term6 = ocpX1.sum(X1_term6,include_last=True)
ocpX1.add_objective(X1_term6)
#
X1_c7 = X1_Z_71 - X1
X1_term7 = dot(X1_lambda_71, X1_c7) + mu/2*sumsqr(X1_c7)
if ocpX1.is_signal(X1_term7):
    X1_term7 = ocpX1.sum(X1_term7,include_last=True)
ocpX1.add_objective(X1_term7)
#
X1_c8 = X1_Z_81 - X1
X1_term8 = dot(X1_lambda_81, X1_c8) + mu/2*sumsqr(X1_c8)
if ocpX1.is_signal(X1_term8):
    X1_term8 = ocpX1.sum(X1_term8,include_last=True)
ocpX1.add_objective(X1_term8)
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
Z1_Z_11 = vertcat(z1_x1, z1_y1)
z1_x2 = ocpZ1.variable(grid='control',include_last=True)
z1_y2 = ocpZ1.variable(grid='control',include_last=True)
Z1_Z_12 = vertcat(z1_x2, z1_y2)
z1_x3 = ocpZ1.variable(grid='control',include_last=True)
z1_y3 = ocpZ1.variable(grid='control',include_last=True)
Z1_Z_13 = vertcat(z1_x3, z1_y3)
z1_x4 = ocpZ1.variable(grid='control',include_last=True)
z1_y4 = ocpZ1.variable(grid='control',include_last=True)
Z1_Z_14 = vertcat(z1_x4, z1_y4)
z1_x5 = ocpZ1.variable(grid='control',include_last=True)
z1_y5 = ocpZ1.variable(grid='control',include_last=True)
Z1_Z_15 = vertcat(z1_x5, z1_y5)
z1_x6 = ocpZ1.variable(grid='control',include_last=True)
z1_y6 = ocpZ1.variable(grid='control',include_last=True)
Z1_Z_16 = vertcat(z1_x6, z1_y6)
z1_x7 = ocpZ1.variable(grid='control',include_last=True)
z1_y7 = ocpZ1.variable(grid='control',include_last=True)
Z1_Z_17 = vertcat(z1_x7, z1_y7)
z1_x8 = ocpZ1.variable(grid='control',include_last=True)
z1_y8 = ocpZ1.variable(grid='control',include_last=True)
Z1_Z_18 = vertcat(z1_x8, z1_y8)
# Parameters
Z1_lambda_11 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_lambda_12 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_lambda_13 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_lambda_14 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_lambda_15 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_lambda_16 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_lambda_17 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_lambda_18 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_X_1 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_X_2 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_X_3 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_X_4 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_X_5 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_X_6 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_X_7 = ocpZ1.parameter(nx, grid='control',include_last=True)
Z1_X_8 = ocpZ1.parameter(nx, grid='control',include_last=True)
# Extended objective
Z1_c1 = Z1_Z_11 - Z1_X_1
Z1_term1 = dot(Z1_lambda_11, Z1_c1) + mu/2*sumsqr(Z1_c1)
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
#
Z1_c5 = Z1_Z_15 - Z1_X_5
Z1_term5 = dot(Z1_lambda_15, Z1_c5) + mu/2*sumsqr(Z1_c5)
if ocpZ1.is_signal(Z1_term5):
    Z1_term5 = ocpZ1.sum(Z1_term5,include_last=True)
ocpZ1.add_objective(Z1_term5)
#
Z1_c6 = Z1_Z_16 - Z1_X_6
Z1_term6 = dot(Z1_lambda_16, Z1_c6) + mu/2*sumsqr(Z1_c6)
if ocpZ1.is_signal(Z1_term6):
    Z1_term6 = ocpZ1.sum(Z1_term6,include_last=True)
ocpZ1.add_objective(Z1_term6)
#
Z1_c7 = Z1_Z_17 - Z1_X_7
Z1_term7 = dot(Z1_lambda_17, Z1_c7) + mu/2*sumsqr(Z1_c7)
if ocpZ1.is_signal(Z1_term7):
    Z1_term7 = ocpZ1.sum(Z1_term7,include_last=True)
ocpZ1.add_objective(Z1_term7)
#
Z1_c8 = Z1_Z_18 - Z1_X_8
Z1_term8 = dot(Z1_lambda_18, Z1_c8) + mu/2*sumsqr(Z1_c8)
if ocpZ1.is_signal(Z1_term8):
    Z1_term8 = ocpZ1.sum(Z1_term8,include_last=True)
ocpZ1.add_objective(Z1_term8)
# Constraints
Z1_distance1  = (z1_x1-z1_x2)**2 + (z1_y1-z1_y2)**2
Z1_distance2  = (z1_x1-z1_x3)**2 + (z1_y1-z1_y3)**2
Z1_distance3  = (z1_x1-z1_x4)**2 + (z1_y1-z1_y4)**2
Z1_distance4  = (z1_x1-z1_x5)**2 + (z1_y1-z1_y5)**2
Z1_distance5  = (z1_x1-z1_x6)**2 + (z1_y1-z1_y6)**2
Z1_distance6  = (z1_x1-z1_x7)**2 + (z1_y1-z1_y7)**2
Z1_distance7  = (z1_x1-z1_x8)**2 + (z1_y1-z1_y8)**2
ocpZ1.subject_to( Z1_distance1  >= boat_radius )
ocpZ1.subject_to( Z1_distance2  >= boat_radius )
ocpZ1.subject_to( Z1_distance3  >= boat_radius )
ocpZ1.subject_to( Z1_distance4  >= boat_radius )
ocpZ1.subject_to( Z1_distance5  >= boat_radius )
ocpZ1.subject_to( Z1_distance6  >= boat_radius )
ocpZ1.subject_to( Z1_distance7  >= boat_radius )
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
X2_lambda_12 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_lambda_22 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_lambda_32 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_lambda_42 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_lambda_52 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_lambda_62 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_lambda_72 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_lambda_82 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_Z_12 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_Z_22 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_Z_32 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_Z_42 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_Z_52 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_Z_62 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_Z_72 = ocpX2.parameter(nx, grid='control',include_last=True)
X2_Z_82 = ocpX2.parameter(nx, grid='control',include_last=True)
#ODE
ocpX2.set_der(x2, u2)
ocpX2.set_der(y2, v2)
# Lagrange objective
ocpX2.add_objective(ocpX2.integral((xd_2-x2)**2 + (yd_2-y2)**2))
ocpX2.add_objective(ocpX2.at_tf((xd_2-x2)**2 + (yd_2-y2)**2))
ocpX2.subject_to( (-max_speed_limit <= u2) <= max_speed_limit )
ocpX2.subject_to( (-max_speed_limit <= v2) <= max_speed_limit )
# Extended objective
X2_c1 = X2_Z_12 - X2
X2_term1 = dot(X2_lambda_12, X2_c1) + mu/2*sumsqr(X2_c1)
if ocpX2.is_signal(X2_term1):
    X2_term1 = ocpX2.sum(X2_term1,include_last=True)
ocpX2.add_objective(X2_term1)
#
X2_c2 = X2_Z_22 - X2
X2_term2 = dot(X2_lambda_22, X2_c2) + mu/2*sumsqr(X2_c2)
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
X2_c5 = X2_Z_52 - X2
X2_term5 = dot(X2_lambda_52, X2_c5) + mu/2*sumsqr(X2_c5)
if ocpX2.is_signal(X2_term5):
    X2_term5 = ocpX2.sum(X2_term5,include_last=True)
ocpX2.add_objective(X2_term5)
#
X2_c6 = X2_Z_62 - X2
X2_term6 = dot(X2_lambda_62, X2_c6) + mu/2*sumsqr(X2_c6)
if ocpX2.is_signal(X2_term6):
    X2_term6 = ocpX2.sum(X2_term6,include_last=True)
ocpX2.add_objective(X2_term6)
#
X2_c7 = X2_Z_72 - X2
X2_term7 = dot(X2_lambda_72, X2_c7) + mu/2*sumsqr(X2_c7)
if ocpX2.is_signal(X2_term7):
    X2_term7 = ocpX2.sum(X2_term7,include_last=True)
ocpX2.add_objective(X2_term7)
#
X2_c8 = X2_Z_82 - X2
X2_term8 = dot(X2_lambda_82, X2_c8) + mu/2*sumsqr(X2_c8)
if ocpX2.is_signal(X2_term8):
    X2_term8 = ocpX2.sum(X2_term8,include_last=True)
ocpX2.add_objective(X2_term8)
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
z2_x1 = ocpZ2.variable(grid='control',include_last=True)
z2_y1 = ocpZ2.variable(grid='control',include_last=True)
Z2_Z_21 = vertcat(z2_x1, z2_y1)
z2_x2 = ocpZ2.variable(grid='control',include_last=True)
z2_y2 = ocpZ2.variable(grid='control',include_last=True)
Z2_Z_22 = vertcat(z2_x2, z2_y2)
z2_x3 = ocpZ2.variable(grid='control',include_last=True)
z2_y3 = ocpZ2.variable(grid='control',include_last=True)
Z2_Z_23 = vertcat(z2_x3, z2_y3)
z2_x4 = ocpZ2.variable(grid='control',include_last=True)
z2_y4 = ocpZ2.variable(grid='control',include_last=True)
Z2_Z_24 = vertcat(z2_x4, z2_y4)
z2_x5 = ocpZ2.variable(grid='control',include_last=True)
z2_y5 = ocpZ2.variable(grid='control',include_last=True)
Z2_Z_25 = vertcat(z2_x5, z2_y5)
z2_x6 = ocpZ2.variable(grid='control',include_last=True)
z2_y6 = ocpZ2.variable(grid='control',include_last=True)
Z2_Z_26 = vertcat(z2_x6, z2_y6)
z2_x7 = ocpZ2.variable(grid='control',include_last=True)
z2_y7 = ocpZ2.variable(grid='control',include_last=True)
Z2_Z_27 = vertcat(z2_x7, z2_y7)
z2_x8 = ocpZ2.variable(grid='control',include_last=True)
z2_y8 = ocpZ2.variable(grid='control',include_last=True)
Z2_Z_28 = vertcat(z2_x8, z2_y8)
# Parameters
Z2_lambda_21 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_lambda_22 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_lambda_23 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_lambda_24 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_lambda_25 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_lambda_26 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_lambda_27 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_lambda_28 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_X_1 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_X_2 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_X_3 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_X_4 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_X_5 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_X_6 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_X_7 = ocpZ2.parameter(nx, grid='control',include_last=True)
Z2_X_8 = ocpZ2.parameter(nx, grid='control',include_last=True)
# Extended objective
Z2_c1 = Z2_Z_21 - Z2_X_1
Z2_term1 = dot(Z2_lambda_21, Z2_c1) + mu/2*sumsqr(Z2_c1)
if ocpZ2.is_signal(Z2_term1):
    Z2_term1 = ocpZ2.sum(Z2_term1,include_last=True)
ocpZ2.add_objective(Z2_term1)
#
Z2_c2 = Z2_Z_22 - Z2_X_2
Z2_term2 = dot(Z2_lambda_22, Z2_c2) + mu/2*sumsqr(Z2_c2)
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
#
Z2_c5 = Z2_Z_25 - Z2_X_5
Z2_term5 = dot(Z2_lambda_25, Z2_c5) + mu/2*sumsqr(Z2_c5)
if ocpZ2.is_signal(Z2_term5):
    Z2_term5 = ocpZ2.sum(Z2_term5,include_last=True)
ocpZ2.add_objective(Z2_term5)
#
Z2_c6 = Z2_Z_26 - Z2_X_6
Z2_term6 = dot(Z2_lambda_26, Z2_c6) + mu/2*sumsqr(Z2_c6)
if ocpZ2.is_signal(Z2_term6):
    Z2_term6 = ocpZ2.sum(Z2_term6,include_last=True)
ocpZ2.add_objective(Z2_term6)
#
Z2_c7 = Z2_Z_27 - Z2_X_7
Z2_term7 = dot(Z2_lambda_27, Z2_c7) + mu/2*sumsqr(Z2_c7)
if ocpZ2.is_signal(Z2_term7):
    Z2_term7 = ocpZ2.sum(Z2_term7,include_last=True)
ocpZ2.add_objective(Z2_term7)
#
Z2_c8 = Z2_Z_28 - Z2_X_8
Z2_term8 = dot(Z2_lambda_28, Z2_c8) + mu/2*sumsqr(Z2_c8)
if ocpZ2.is_signal(Z2_term8):
    Z2_term8 = ocpZ2.sum(Z2_term8,include_last=True)
ocpZ2.add_objective(Z2_term8)
# Constraints
Z2_distance1  = (z2_x2-z2_x1)**2 + (z2_y2-z2_y1)**2
Z2_distance2  = (z2_x2-z2_x3)**2 + (z2_y2-z2_y3)**2
Z2_distance3  = (z2_x2-z2_x4)**2 + (z2_y2-z2_y4)**2
Z2_distance4  = (z2_x2-z2_x5)**2 + (z2_y2-z2_y5)**2
Z2_distance5  = (z2_x2-z2_x6)**2 + (z2_y2-z2_y6)**2
Z2_distance6  = (z2_x2-z2_x7)**2 + (z2_y2-z2_y7)**2
Z2_distance7  = (z2_x2-z2_x8)**2 + (z2_y2-z2_y8)**2
ocpZ2.subject_to( Z2_distance1  >= boat_radius )
ocpZ2.subject_to( Z2_distance2  >= boat_radius )
ocpZ2.subject_to( Z2_distance3  >= boat_radius )
ocpZ2.subject_to( Z2_distance4  >= boat_radius )
ocpZ2.subject_to( Z2_distance5  >= boat_radius )
ocpZ2.subject_to( Z2_distance6  >= boat_radius )
ocpZ2.subject_to( Z2_distance7  >= boat_radius )
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
X3_lambda_13 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_lambda_23 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_lambda_33 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_lambda_43 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_lambda_53 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_lambda_63 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_lambda_73 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_lambda_83 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_Z_13 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_Z_23 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_Z_33 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_Z_43 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_Z_53 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_Z_63 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_Z_73 = ocpX3.parameter(nx, grid='control',include_last=True)
X3_Z_83 = ocpX3.parameter(nx, grid='control',include_last=True)
#ODE
ocpX3.set_der(x3, u3)
ocpX3.set_der(y3, v3)
# Lagrange objective
ocpX3.add_objective(ocpX3.integral((xd_3-x3)**2 + (yd_3-y3)**2))
ocpX3.add_objective(ocpX3.at_tf((xd_3-x3)**2 + (yd_3-y3)**2))
ocpX3.subject_to( (-max_speed_limit <= u3) <= max_speed_limit )
ocpX3.subject_to( (-max_speed_limit <= v3) <= max_speed_limit )
# Extended objective
X3_c1 = X3_Z_13 - X3
X3_term1 = dot(X3_lambda_13, X3_c1) + mu/2*sumsqr(X3_c1)
if ocpX3.is_signal(X3_term1):
    X3_term1 = ocpX3.sum(X3_term1,include_last=True)
ocpX3.add_objective(X3_term1)
#
X3_c2 = X3_Z_23 - X3
X3_term2 = dot(X3_lambda_23, X3_c2) + mu/2*sumsqr(X3_c2)
if ocpX3.is_signal(X3_term2):
    X3_term2 = ocpX3.sum(X3_term2,include_last=True)
ocpX3.add_objective(X3_term2)
#
X3_c3 = X3_Z_33 - X3
X3_term3 = dot(X3_lambda_33, X3_c3) + mu/2*sumsqr(X3_c3)
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
X3_c5 = X3_Z_53 - X3
X3_term5 = dot(X3_lambda_53, X3_c5) + mu/2*sumsqr(X3_c5)
if ocpX3.is_signal(X3_term5):
    X3_term5 = ocpX3.sum(X3_term5,include_last=True)
ocpX3.add_objective(X3_term5)
#
X3_c6 = X3_Z_63 - X3
X3_term6 = dot(X3_lambda_63, X3_c6) + mu/2*sumsqr(X3_c6)
if ocpX3.is_signal(X3_term6):
    X3_term6 = ocpX3.sum(X3_term6,include_last=True)
ocpX3.add_objective(X3_term6)
#
X3_c7 = X3_Z_73 - X3
X3_term7 = dot(X3_lambda_73, X3_c7) + mu/2*sumsqr(X3_c7)
if ocpX3.is_signal(X3_term7):
    X3_term7 = ocpX3.sum(X3_term7,include_last=True)
ocpX3.add_objective(X3_term7)
#
X3_c8 = X3_Z_83 - X3
X3_term8 = dot(X3_lambda_83, X3_c8) + mu/2*sumsqr(X3_c8)
if ocpX3.is_signal(X3_term8):
    X3_term8 = ocpX3.sum(X3_term8,include_last=True)
ocpX3.add_objective(X3_term8)
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
z3_x1 = ocpZ3.variable(grid='control',include_last=True)
z3_y1 = ocpZ3.variable(grid='control',include_last=True)
Z3_Z_31 = vertcat(z3_x1, z3_y1)
z3_x2 = ocpZ3.variable(grid='control',include_last=True)
z3_y2 = ocpZ3.variable(grid='control',include_last=True)
Z3_Z_32 = vertcat(z3_x2, z3_y2)
z3_x3 = ocpZ3.variable(grid='control',include_last=True)
z3_y3 = ocpZ3.variable(grid='control',include_last=True)
Z3_Z_33 = vertcat(z3_x3, z3_y3)
z3_x4 = ocpZ3.variable(grid='control',include_last=True)
z3_y4 = ocpZ3.variable(grid='control',include_last=True)
Z3_Z_34 = vertcat(z3_x4, z3_y4)
z3_x5 = ocpZ3.variable(grid='control',include_last=True)
z3_y5 = ocpZ3.variable(grid='control',include_last=True)
Z3_Z_35 = vertcat(z3_x5, z3_y5)
z3_x6 = ocpZ3.variable(grid='control',include_last=True)
z3_y6 = ocpZ3.variable(grid='control',include_last=True)
Z3_Z_36 = vertcat(z3_x6, z3_y6)
z3_x7 = ocpZ3.variable(grid='control',include_last=True)
z3_y7 = ocpZ3.variable(grid='control',include_last=True)
Z3_Z_37 = vertcat(z3_x7, z3_y7)
z3_x8 = ocpZ3.variable(grid='control',include_last=True)
z3_y8 = ocpZ3.variable(grid='control',include_last=True)
Z3_Z_38 = vertcat(z3_x8, z3_y8)
# Parameters
Z3_lambda_31 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_lambda_32 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_lambda_33 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_lambda_34 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_lambda_35 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_lambda_36 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_lambda_37 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_lambda_38 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_X_1 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_X_2 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_X_3 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_X_4 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_X_5 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_X_6 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_X_7 = ocpZ3.parameter(nx, grid='control',include_last=True)
Z3_X_8 = ocpZ3.parameter(nx, grid='control',include_last=True)
# Extended objective
Z3_c1 = Z3_Z_31 - Z3_X_1
Z3_term1 = dot(Z3_lambda_31, Z3_c1) + mu/2*sumsqr(Z3_c1)
if ocpZ3.is_signal(Z3_term1):
    Z3_term1 = ocpZ3.sum(Z3_term1,include_last=True)
ocpZ3.add_objective(Z3_term1)
#
Z3_c2 = Z3_Z_32 - Z3_X_2
Z3_term2 = dot(Z3_lambda_32, Z3_c2) + mu/2*sumsqr(Z3_c2)
if ocpZ3.is_signal(Z3_term2):
    Z3_term2 = ocpZ3.sum(Z3_term2,include_last=True)
ocpZ3.add_objective(Z3_term2)
#
Z3_c3 = Z3_Z_33 - Z3_X_3
Z3_term3 = dot(Z3_lambda_33, Z3_c3) + mu/2*sumsqr(Z3_c3)
if ocpZ3.is_signal(Z3_term3):
    Z3_term3 = ocpZ3.sum(Z3_term3,include_last=True)
ocpZ3.add_objective(Z3_term3)
#
Z3_c4 = Z3_Z_34 - Z3_X_4
Z3_term4 = dot(Z3_lambda_34, Z3_c4) + mu/2*sumsqr(Z3_c4)
if ocpZ3.is_signal(Z3_term4):
    Z3_term4 = ocpZ3.sum(Z3_term4,include_last=True)
ocpZ3.add_objective(Z3_term4)
#
Z3_c5 = Z3_Z_35 - Z3_X_5
Z3_term5 = dot(Z3_lambda_35, Z3_c5) + mu/2*sumsqr(Z3_c5)
if ocpZ3.is_signal(Z3_term5):
    Z3_term5 = ocpZ3.sum(Z3_term5,include_last=True)
ocpZ3.add_objective(Z3_term5)
#
Z3_c6 = Z3_Z_36 - Z3_X_6
Z3_term6 = dot(Z3_lambda_36, Z3_c6) + mu/2*sumsqr(Z3_c6)
if ocpZ3.is_signal(Z3_term6):
    Z3_term6 = ocpZ3.sum(Z3_term6,include_last=True)
ocpZ3.add_objective(Z3_term6)
#
Z3_c7 = Z3_Z_37 - Z3_X_7
Z3_term7 = dot(Z3_lambda_37, Z3_c7) + mu/2*sumsqr(Z3_c7)
if ocpZ3.is_signal(Z3_term7):
    Z3_term7 = ocpZ3.sum(Z3_term7,include_last=True)
ocpZ3.add_objective(Z3_term7)
#
Z3_c8 = Z3_Z_38 - Z3_X_8
Z3_term8 = dot(Z3_lambda_38, Z3_c8) + mu/2*sumsqr(Z3_c8)
if ocpZ3.is_signal(Z3_term8):
    Z3_term8 = ocpZ3.sum(Z3_term8,include_last=True)
ocpZ3.add_objective(Z3_term8)
# Constraints
Z3_distance1  = (z3_x3-z3_x1)**2 + (z3_y3-z3_y1)**2
Z3_distance2  = (z3_x3-z3_x2)**2 + (z3_y3-z3_y2)**2
Z3_distance3  = (z3_x3-z3_x4)**2 + (z3_y3-z3_y4)**2
Z3_distance4  = (z3_x3-z3_x5)**2 + (z3_y3-z3_y5)**2
Z3_distance5  = (z3_x3-z3_x6)**2 + (z3_y3-z3_y6)**2
Z3_distance6  = (z3_x3-z3_x7)**2 + (z3_y3-z3_y7)**2
Z3_distance7  = (z3_x3-z3_x8)**2 + (z3_y3-z3_y8)**2
ocpZ3.subject_to( Z3_distance1  >= boat_radius )
ocpZ3.subject_to( Z3_distance2  >= boat_radius )
ocpZ3.subject_to( Z3_distance3  >= boat_radius )
ocpZ3.subject_to( Z3_distance4  >= boat_radius )
ocpZ3.subject_to( Z3_distance5  >= boat_radius )
ocpZ3.subject_to( Z3_distance6  >= boat_radius )
ocpZ3.subject_to( Z3_distance7  >= boat_radius )
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
X4_lambda_14 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_lambda_24 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_lambda_34 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_lambda_44 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_lambda_54 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_lambda_64 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_lambda_74 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_lambda_84 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_Z_14 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_Z_24 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_Z_34 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_Z_44 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_Z_54 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_Z_64 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_Z_74 = ocpX4.parameter(nx, grid='control',include_last=True)
X4_Z_84 = ocpX4.parameter(nx, grid='control',include_last=True)
#ODE
ocpX4.set_der(x4, u4)
ocpX4.set_der(y4, v4)
# Lagrange objective
ocpX4.add_objective(ocpX4.integral((xd_4-x4)**2 + (yd_4-y4)**2))
ocpX4.add_objective(ocpX4.at_tf((xd_4-x4)**2 + (yd_4-y4)**2))
ocpX4.subject_to( (-max_speed_limit <= u4) <= max_speed_limit )
ocpX4.subject_to( (-max_speed_limit <= v4) <= max_speed_limit )
# Extended objective
X4_c1 = X4_Z_14 - X4
X4_term1 = dot(X4_lambda_14, X4_c1) + mu/2*sumsqr(X4_c1)
if ocpX4.is_signal(X4_term1):
    X4_term1 = ocpX4.sum(X4_term1,include_last=True)
ocpX4.add_objective(X4_term1)
#
X4_c2 = X4_Z_24 - X4
X4_term2 = dot(X4_lambda_24, X4_c2) + mu/2*sumsqr(X4_c2)
if ocpX4.is_signal(X4_term2):
    X4_term2 = ocpX4.sum(X4_term2,include_last=True)
ocpX4.add_objective(X4_term2)
#
X4_c3 = X4_Z_34 - X4
X4_term3 = dot(X4_lambda_34, X4_c3) + mu/2*sumsqr(X4_c3)
if ocpX4.is_signal(X4_term3):
    X4_term3 = ocpX4.sum(X4_term3,include_last=True)
ocpX4.add_objective(X4_term3)
#
X4_c4 = X4_Z_44 - X4
X4_term4 = dot(X4_lambda_44, X4_c4) + mu/2*sumsqr(X4_c4)
if ocpX4.is_signal(X4_term4):
    X4_term4 = ocpX4.sum(X4_term4,include_last=True)
ocpX4.add_objective(X4_term4)
#
X4_c5 = X4_Z_54 - X4
X4_term5 = dot(X4_lambda_54, X4_c5) + mu/2*sumsqr(X4_c5)
if ocpX4.is_signal(X4_term5):
    X4_term5 = ocpX4.sum(X4_term5,include_last=True)
ocpX4.add_objective(X4_term5)
#
X4_c6 = X4_Z_64 - X4
X4_term6 = dot(X4_lambda_64, X4_c6) + mu/2*sumsqr(X4_c6)
if ocpX4.is_signal(X4_term6):
    X4_term6 = ocpX4.sum(X4_term6,include_last=True)
ocpX4.add_objective(X4_term6)
#
X4_c7 = X4_Z_74 - X4
X4_term7 = dot(X4_lambda_74, X4_c7) + mu/2*sumsqr(X4_c7)
if ocpX4.is_signal(X4_term7):
    X4_term7 = ocpX4.sum(X4_term7,include_last=True)
ocpX4.add_objective(X4_term7)
#
X4_c8 = X4_Z_84 - X4
X4_term8 = dot(X4_lambda_84, X4_c8) + mu/2*sumsqr(X4_c8)
if ocpX4.is_signal(X4_term8):
    X4_term8 = ocpX4.sum(X4_term8,include_last=True)
ocpX4.add_objective(X4_term8)
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
z4_x1 = ocpZ4.variable(grid='control',include_last=True)
z4_y1 = ocpZ4.variable(grid='control',include_last=True)
Z4_Z_41 = vertcat(z4_x1, z4_y1)
z4_x2 = ocpZ4.variable(grid='control',include_last=True)
z4_y2 = ocpZ4.variable(grid='control',include_last=True)
Z4_Z_42 = vertcat(z4_x2, z4_y2)
z4_x3 = ocpZ4.variable(grid='control',include_last=True)
z4_y3 = ocpZ4.variable(grid='control',include_last=True)
Z4_Z_43 = vertcat(z4_x3, z4_y3)
z4_x4 = ocpZ4.variable(grid='control',include_last=True)
z4_y4 = ocpZ4.variable(grid='control',include_last=True)
Z4_Z_44 = vertcat(z4_x4, z4_y4)
z4_x5 = ocpZ4.variable(grid='control',include_last=True)
z4_y5 = ocpZ4.variable(grid='control',include_last=True)
Z4_Z_45 = vertcat(z4_x5, z4_y5)
z4_x6 = ocpZ4.variable(grid='control',include_last=True)
z4_y6 = ocpZ4.variable(grid='control',include_last=True)
Z4_Z_46 = vertcat(z4_x6, z4_y6)
z4_x7 = ocpZ4.variable(grid='control',include_last=True)
z4_y7 = ocpZ4.variable(grid='control',include_last=True)
Z4_Z_47 = vertcat(z4_x7, z4_y7)
z4_x8 = ocpZ4.variable(grid='control',include_last=True)
z4_y8 = ocpZ4.variable(grid='control',include_last=True)
Z4_Z_48 = vertcat(z4_x8, z4_y8)
# Parameters
Z4_lambda_41 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_lambda_42 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_lambda_43 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_lambda_44 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_lambda_45 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_lambda_46 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_lambda_47 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_lambda_48 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_X_1 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_X_2 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_X_3 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_X_4 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_X_5 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_X_6 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_X_7 = ocpZ4.parameter(nx, grid='control',include_last=True)
Z4_X_8 = ocpZ4.parameter(nx, grid='control',include_last=True)
# Extended objective
Z4_c1 = Z4_Z_41 - Z4_X_1
Z4_term1 = dot(Z4_lambda_41, Z4_c1) + mu/2*sumsqr(Z4_c1)
if ocpZ4.is_signal(Z4_term1):
    Z4_term1 = ocpZ4.sum(Z4_term1,include_last=True)
ocpZ4.add_objective(Z4_term1)
#
Z4_c2 = Z4_Z_42 - Z4_X_2
Z4_term2 = dot(Z4_lambda_42, Z4_c2) + mu/2*sumsqr(Z4_c2)
if ocpZ4.is_signal(Z4_term2):
    Z4_term2 = ocpZ4.sum(Z4_term2,include_last=True)
ocpZ4.add_objective(Z4_term2)
#
Z4_c3 = Z4_Z_43 - Z4_X_3
Z4_term3 = dot(Z4_lambda_43, Z4_c3) + mu/2*sumsqr(Z4_c3)
if ocpZ4.is_signal(Z4_term3):
    Z4_term3 = ocpZ4.sum(Z4_term3,include_last=True)
ocpZ4.add_objective(Z4_term3)
#
Z4_c4 = Z4_Z_44 - Z4_X_4
Z4_term4 = dot(Z4_lambda_44, Z4_c4) + mu/2*sumsqr(Z4_c4)
if ocpZ4.is_signal(Z4_term4):
    Z4_term4 = ocpZ4.sum(Z4_term4,include_last=True)
ocpZ4.add_objective(Z4_term4)
#
Z4_c5 = Z4_Z_45 - Z4_X_5
Z4_term5 = dot(Z4_lambda_45, Z4_c5) + mu/2*sumsqr(Z4_c5)
if ocpZ4.is_signal(Z4_term5):
    Z4_term5 = ocpZ4.sum(Z4_term5,include_last=True)
ocpZ4.add_objective(Z4_term5)
#
Z4_c6 = Z4_Z_46 - Z4_X_6
Z4_term6 = dot(Z4_lambda_46, Z4_c6) + mu/2*sumsqr(Z4_c6)
if ocpZ4.is_signal(Z4_term6):
    Z4_term6 = ocpZ4.sum(Z4_term6,include_last=True)
ocpZ4.add_objective(Z4_term6)
#
Z4_c7 = Z4_Z_47 - Z4_X_7
Z4_term7 = dot(Z4_lambda_47, Z4_c7) + mu/2*sumsqr(Z4_c7)
if ocpZ4.is_signal(Z4_term7):
    Z4_term7 = ocpZ4.sum(Z4_term7,include_last=True)
ocpZ4.add_objective(Z4_term7)
#
Z4_c8 = Z4_Z_48 - Z4_X_8
Z4_term8 = dot(Z4_lambda_48, Z4_c8) + mu/2*sumsqr(Z4_c8)
if ocpZ4.is_signal(Z4_term8):
    Z4_term8 = ocpZ4.sum(Z4_term8,include_last=True)
ocpZ4.add_objective(Z4_term8)
# Constraints
Z4_distance1  = (z4_x4-z4_x1)**2 + (z4_y4-z4_y1)**2
Z4_distance2  = (z4_x4-z4_x2)**2 + (z4_y4-z4_y2)**2
Z4_distance3  = (z4_x4-z4_x3)**2 + (z4_y4-z4_y3)**2
Z4_distance4  = (z4_x4-z4_x5)**2 + (z4_y4-z4_y5)**2
Z4_distance5  = (z4_x4-z4_x6)**2 + (z4_y4-z4_y6)**2
Z4_distance6  = (z4_x4-z4_x7)**2 + (z4_y4-z4_y7)**2
Z4_distance7  = (z4_x4-z4_x8)**2 + (z4_y4-z4_y8)**2
ocpZ4.subject_to( Z4_distance1  >= boat_radius )
ocpZ4.subject_to( Z4_distance2  >= boat_radius )
ocpZ4.subject_to( Z4_distance3  >= boat_radius )
ocpZ4.subject_to( Z4_distance4  >= boat_radius )
ocpZ4.subject_to( Z4_distance5  >= boat_radius )
ocpZ4.subject_to( Z4_distance6  >= boat_radius )
ocpZ4.subject_to( Z4_distance7  >= boat_radius )
# Pick a solution method
ocpZ4.solver('ipopt',options)
# Make it concrete for this ocp
ocpZ4.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

'''
OcpX5
'''
ocpX5 = Ocp(T=Tf)
# States
x5 = ocpX5.state()
y5 = ocpX5.state()
X5 = vertcat(x5,y5)
# Controls
u5 = ocpX5.control()
v5 = ocpX5.control()
# Initial condition
X5_0 = ocpX5.parameter(nx)
# Parameters
X5_lambda_15 = ocpX5.parameter(nx, grid='control',include_last=True)
X5_lambda_25 = ocpX5.parameter(nx, grid='control',include_last=True)
X5_lambda_35 = ocpX5.parameter(nx, grid='control',include_last=True)
X5_lambda_45 = ocpX5.parameter(nx, grid='control',include_last=True)
X5_lambda_55 = ocpX5.parameter(nx, grid='control',include_last=True)
X5_lambda_65 = ocpX5.parameter(nx, grid='control',include_last=True)
X5_lambda_75 = ocpX5.parameter(nx, grid='control',include_last=True)
X5_lambda_85 = ocpX5.parameter(nx, grid='control',include_last=True)
X5_Z_15 = ocpX5.parameter(nx, grid='control',include_last=True)
X5_Z_25 = ocpX5.parameter(nx, grid='control',include_last=True)
X5_Z_35 = ocpX5.parameter(nx, grid='control',include_last=True)
X5_Z_45 = ocpX5.parameter(nx, grid='control',include_last=True)
X5_Z_55 = ocpX5.parameter(nx, grid='control',include_last=True)
X5_Z_65 = ocpX5.parameter(nx, grid='control',include_last=True)
X5_Z_75 = ocpX5.parameter(nx, grid='control',include_last=True)
X5_Z_85 = ocpX5.parameter(nx, grid='control',include_last=True)
#ODE
ocpX5.set_der(x5, u5)
ocpX5.set_der(y5, v5)
# Lagrange objective
ocpX5.add_objective(ocpX5.integral((xd_5-x5)**2 + (yd_5-y5)**2))
ocpX5.add_objective(ocpX5.at_tf((xd_5-x5)**2 + (yd_5-y5)**2))
ocpX5.subject_to( (-max_speed_limit <= u5) <= max_speed_limit )
ocpX5.subject_to( (-max_speed_limit <= v5) <= max_speed_limit )
# Extended objective
X5_c1 = X5_Z_15 - X5
X5_term1 = dot(X5_lambda_15, X5_c1) + mu/2*sumsqr(X5_c1)
if ocpX5.is_signal(X5_term1):
    X5_term1 = ocpX5.sum(X5_term1,include_last=True)
ocpX5.add_objective(X5_term1)
#
X5_c2 = X5_Z_25 - X5
X5_term2 = dot(X5_lambda_25, X5_c2) + mu/2*sumsqr(X5_c2)
if ocpX5.is_signal(X5_term2):
    X5_term2 = ocpX5.sum(X5_term2,include_last=True)
ocpX5.add_objective(X5_term2)
#
X5_c3 = X5_Z_35 - X5
X5_term3 = dot(X5_lambda_35, X5_c3) + mu/2*sumsqr(X5_c3)
if ocpX5.is_signal(X5_term3):
    X5_term3 = ocpX5.sum(X5_term3,include_last=True)
ocpX5.add_objective(X5_term3)
#
X5_c4 = X5_Z_45 - X5
X5_term4 = dot(X5_lambda_45, X5_c4) + mu/2*sumsqr(X5_c4)
if ocpX5.is_signal(X5_term4):
    X5_term4 = ocpX5.sum(X5_term4,include_last=True)
ocpX5.add_objective(X5_term4)
#
X5_c5 = X5_Z_55 - X5
X5_term5 = dot(X5_lambda_55, X5_c5) + mu/2*sumsqr(X5_c5)
if ocpX5.is_signal(X5_term5):
    X5_term5 = ocpX5.sum(X5_term5,include_last=True)
ocpX5.add_objective(X5_term5)
#
X5_c6 = X5_Z_65 - X5
X5_term6 = dot(X5_lambda_65, X5_c6) + mu/2*sumsqr(X5_c6)
if ocpX5.is_signal(X5_term6):
    X5_term6 = ocpX5.sum(X5_term6,include_last=True)
ocpX5.add_objective(X5_term6)
#
X5_c7 = X5_Z_75 - X5
X5_term7 = dot(X5_lambda_75, X5_c7) + mu/2*sumsqr(X5_c7)
if ocpX5.is_signal(X5_term7):
    X5_term7 = ocpX5.sum(X5_term7,include_last=True)
ocpX5.add_objective(X5_term7)
#
X5_c8 = X5_Z_85 - X5
X5_term8 = dot(X5_lambda_85, X5_c8) + mu/2*sumsqr(X5_c8)
if ocpX5.is_signal(X5_term8):
    X5_term8 = ocpX5.sum(X5_term8,include_last=True)
ocpX5.add_objective(X5_term8)
#
# Initial constraints
ocpX5.subject_to(ocpX5.at_t0(X5)==X5_0)
# Pick a solution method
ocpX5.solver('ipopt',options)
# Make it concrete for this ocp
ocpX5.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

'''
OcpZ5
'''
ocpZ5 = Ocp(T=Tf)
# Variables
z5_x1 = ocpZ5.variable(grid='control',include_last=True)
z5_y1 = ocpZ5.variable(grid='control',include_last=True)
Z5_Z_51 = vertcat(z5_x1, z5_y1)
z5_x2 = ocpZ5.variable(grid='control',include_last=True)
z5_y2 = ocpZ5.variable(grid='control',include_last=True)
Z5_Z_52 = vertcat(z5_x2, z5_y2)
z5_x3 = ocpZ5.variable(grid='control',include_last=True)
z5_y3 = ocpZ5.variable(grid='control',include_last=True)
Z5_Z_53 = vertcat(z5_x3, z5_y3)
z5_x4 = ocpZ5.variable(grid='control',include_last=True)
z5_y4 = ocpZ5.variable(grid='control',include_last=True)
Z5_Z_54 = vertcat(z5_x4, z5_y4)
z5_x5 = ocpZ5.variable(grid='control',include_last=True)
z5_y5 = ocpZ5.variable(grid='control',include_last=True)
Z5_Z_55 = vertcat(z5_x5, z5_y5)
z5_x6 = ocpZ5.variable(grid='control',include_last=True)
z5_y6 = ocpZ5.variable(grid='control',include_last=True)
Z5_Z_56 = vertcat(z5_x6, z5_y6)
z5_x7 = ocpZ5.variable(grid='control',include_last=True)
z5_y7 = ocpZ5.variable(grid='control',include_last=True)
Z5_Z_57 = vertcat(z5_x7, z5_y7)
z5_x8 = ocpZ5.variable(grid='control',include_last=True)
z5_y8 = ocpZ5.variable(grid='control',include_last=True)
Z5_Z_58 = vertcat(z5_x8, z5_y8)
# Parameters
Z5_lambda_51 = ocpZ5.parameter(nx, grid='control',include_last=True)
Z5_lambda_52 = ocpZ5.parameter(nx, grid='control',include_last=True)
Z5_lambda_53 = ocpZ5.parameter(nx, grid='control',include_last=True)
Z5_lambda_54 = ocpZ5.parameter(nx, grid='control',include_last=True)
Z5_lambda_55 = ocpZ5.parameter(nx, grid='control',include_last=True)
Z5_lambda_56 = ocpZ5.parameter(nx, grid='control',include_last=True)
Z5_lambda_57 = ocpZ5.parameter(nx, grid='control',include_last=True)
Z5_lambda_58 = ocpZ5.parameter(nx, grid='control',include_last=True)
Z5_X_1 = ocpZ5.parameter(nx, grid='control',include_last=True)
Z5_X_2 = ocpZ5.parameter(nx, grid='control',include_last=True)
Z5_X_3 = ocpZ5.parameter(nx, grid='control',include_last=True)
Z5_X_4 = ocpZ5.parameter(nx, grid='control',include_last=True)
Z5_X_5 = ocpZ5.parameter(nx, grid='control',include_last=True)
Z5_X_6 = ocpZ5.parameter(nx, grid='control',include_last=True)
Z5_X_7 = ocpZ5.parameter(nx, grid='control',include_last=True)
Z5_X_8 = ocpZ5.parameter(nx, grid='control',include_last=True)
# Extended objective
Z5_c1 = Z5_Z_51 - Z5_X_1
Z5_term1 = dot(Z5_lambda_51, Z5_c1) + mu/2*sumsqr(Z5_c1)
if ocpZ5.is_signal(Z5_term1):
    Z5_term1 = ocpZ5.sum(Z5_term1,include_last=True)
ocpZ5.add_objective(Z5_term1)
#
Z5_c2 = Z5_Z_52 - Z5_X_2
Z5_term2 = dot(Z5_lambda_52, Z5_c2) + mu/2*sumsqr(Z5_c2)
if ocpZ5.is_signal(Z5_term2):
    Z5_term2 = ocpZ5.sum(Z5_term2,include_last=True)
ocpZ5.add_objective(Z5_term2)
#
Z5_c3 = Z5_Z_53 - Z5_X_3
Z5_term3 = dot(Z5_lambda_53, Z5_c3) + mu/2*sumsqr(Z5_c3)
if ocpZ5.is_signal(Z5_term3):
    Z5_term3 = ocpZ5.sum(Z5_term3,include_last=True)
ocpZ5.add_objective(Z5_term3)
#
Z5_c4 = Z5_Z_54 - Z5_X_4
Z5_term4 = dot(Z5_lambda_54, Z5_c4) + mu/2*sumsqr(Z5_c4)
if ocpZ5.is_signal(Z5_term4):
    Z5_term4 = ocpZ5.sum(Z5_term4,include_last=True)
ocpZ5.add_objective(Z5_term4)
#
Z5_c5 = Z5_Z_55 - Z5_X_5
Z5_term5 = dot(Z5_lambda_55, Z5_c5) + mu/2*sumsqr(Z5_c5)
if ocpZ5.is_signal(Z5_term5):
    Z5_term5 = ocpZ5.sum(Z5_term5,include_last=True)
ocpZ5.add_objective(Z5_term5)
#
Z5_c6 = Z5_Z_56 - Z5_X_6
Z5_term6 = dot(Z5_lambda_56, Z5_c6) + mu/2*sumsqr(Z5_c6)
if ocpZ5.is_signal(Z5_term6):
    Z5_term6 = ocpZ5.sum(Z5_term6,include_last=True)
ocpZ5.add_objective(Z5_term6)
#
Z5_c7 = Z5_Z_57 - Z5_X_7
Z5_term7 = dot(Z5_lambda_57, Z5_c7) + mu/2*sumsqr(Z5_c7)
if ocpZ5.is_signal(Z5_term7):
    Z5_term7 = ocpZ5.sum(Z5_term7,include_last=True)
ocpZ5.add_objective(Z5_term7)
#
Z5_c8 = Z5_Z_58 - Z5_X_8
Z5_term8 = dot(Z5_lambda_58, Z5_c8) + mu/2*sumsqr(Z5_c8)
if ocpZ5.is_signal(Z5_term8):
    Z5_term8 = ocpZ5.sum(Z5_term8,include_last=True)
ocpZ5.add_objective(Z5_term8)
# Constraints
Z5_distance1  = (z5_x5-z5_x1)**2 + (z5_y5-z5_y1)**2
Z5_distance2  = (z5_x5-z5_x2)**2 + (z5_y5-z5_y2)**2
Z5_distance3  = (z5_x5-z5_x3)**2 + (z5_y5-z5_y3)**2
Z5_distance4  = (z5_x5-z5_x4)**2 + (z5_y5-z5_y4)**2
Z5_distance5  = (z5_x5-z5_x6)**2 + (z5_y5-z5_y6)**2
Z5_distance6  = (z5_x5-z5_x7)**2 + (z5_y5-z5_y7)**2
Z5_distance7  = (z5_x5-z5_x8)**2 + (z5_y5-z5_y8)**2
ocpZ5.subject_to( Z5_distance1  >= boat_radius )
ocpZ5.subject_to( Z5_distance2  >= boat_radius )
ocpZ5.subject_to( Z5_distance3  >= boat_radius )
ocpZ5.subject_to( Z5_distance4  >= boat_radius )
ocpZ5.subject_to( Z5_distance5  >= boat_radius )
ocpZ5.subject_to( Z5_distance6  >= boat_radius )
ocpZ5.subject_to( Z5_distance7  >= boat_radius )
# Pick a solution method
ocpZ5.solver('ipopt',options)
# Make it concrete for this ocp
ocpZ5.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

'''
OcpX6
'''
ocpX6 = Ocp(T=Tf)
# States
x6 = ocpX6.state()
y6 = ocpX6.state()
X6 = vertcat(x6,y6)
# Controls
u6 = ocpX6.control()
v6 = ocpX6.control()
# Initial condition
X6_0 = ocpX6.parameter(nx)
# Parameters
X6_lambda_16 = ocpX6.parameter(nx, grid='control',include_last=True)
X6_lambda_26 = ocpX6.parameter(nx, grid='control',include_last=True)
X6_lambda_36 = ocpX6.parameter(nx, grid='control',include_last=True)
X6_lambda_46 = ocpX6.parameter(nx, grid='control',include_last=True)
X6_lambda_56 = ocpX6.parameter(nx, grid='control',include_last=True)
X6_lambda_66 = ocpX6.parameter(nx, grid='control',include_last=True)
X6_lambda_76 = ocpX6.parameter(nx, grid='control',include_last=True)
X6_lambda_86 = ocpX6.parameter(nx, grid='control',include_last=True)
X6_Z_16 = ocpX6.parameter(nx, grid='control',include_last=True)
X6_Z_26 = ocpX6.parameter(nx, grid='control',include_last=True)
X6_Z_36 = ocpX6.parameter(nx, grid='control',include_last=True)
X6_Z_46 = ocpX6.parameter(nx, grid='control',include_last=True)
X6_Z_56 = ocpX6.parameter(nx, grid='control',include_last=True)
X6_Z_66 = ocpX6.parameter(nx, grid='control',include_last=True)
X6_Z_76 = ocpX6.parameter(nx, grid='control',include_last=True)
X6_Z_86 = ocpX6.parameter(nx, grid='control',include_last=True)
#ODE
ocpX6.set_der(x6, u6)
ocpX6.set_der(y6, v6)
# Lagrange objective
ocpX6.add_objective(ocpX6.integral((xd_6-x6)**2 + (yd_6-y6)**2))
ocpX6.add_objective(ocpX6.at_tf((xd_6-x6)**2 + (yd_6-y6)**2))
ocpX6.subject_to( (-max_speed_limit <= u6) <= max_speed_limit )
ocpX6.subject_to( (-max_speed_limit <= v6) <= max_speed_limit )
# Extended objective
X6_c1 = X6_Z_16 - X6
X6_term1 = dot(X6_lambda_16, X6_c1) + mu/2*sumsqr(X6_c1)
if ocpX6.is_signal(X6_term1):
    X6_term1 = ocpX6.sum(X6_term1,include_last=True)
ocpX6.add_objective(X6_term1)
#
X6_c2 = X6_Z_26 - X6
X6_term2 = dot(X6_lambda_26, X6_c2) + mu/2*sumsqr(X6_c2)
if ocpX6.is_signal(X6_term2):
    X6_term2 = ocpX6.sum(X6_term2,include_last=True)
ocpX6.add_objective(X6_term2)
#
X6_c3 = X6_Z_36 - X6
X6_term3 = dot(X6_lambda_36, X6_c3) + mu/2*sumsqr(X6_c3)
if ocpX6.is_signal(X6_term3):
    X6_term3 = ocpX6.sum(X6_term3,include_last=True)
ocpX6.add_objective(X6_term3)
#
X6_c4 = X6_Z_46 - X6
X6_term4 = dot(X6_lambda_46, X6_c4) + mu/2*sumsqr(X6_c4)
if ocpX6.is_signal(X6_term4):
    X6_term4 = ocpX6.sum(X6_term4,include_last=True)
ocpX6.add_objective(X6_term4)
#
X6_c5 = X6_Z_56 - X6
X6_term5 = dot(X6_lambda_56, X6_c5) + mu/2*sumsqr(X6_c5)
if ocpX6.is_signal(X6_term5):
    X6_term5 = ocpX6.sum(X6_term5,include_last=True)
ocpX6.add_objective(X6_term5)
#
X6_c6 = X6_Z_66 - X6
X6_term6 = dot(X6_lambda_66, X6_c6) + mu/2*sumsqr(X6_c6)
if ocpX6.is_signal(X6_term6):
    X6_term6 = ocpX6.sum(X6_term6,include_last=True)
ocpX6.add_objective(X6_term6)
#
X6_c7 = X6_Z_76 - X6
X6_term7 = dot(X6_lambda_76, X6_c7) + mu/2*sumsqr(X6_c7)
if ocpX6.is_signal(X6_term7):
    X6_term7 = ocpX6.sum(X6_term7,include_last=True)
ocpX6.add_objective(X6_term7)
#
X6_c8 = X6_Z_86 - X6
X6_term8 = dot(X6_lambda_86, X6_c8) + mu/2*sumsqr(X6_c8)
if ocpX6.is_signal(X6_term8):
    X6_term8 = ocpX6.sum(X6_term8,include_last=True)
ocpX6.add_objective(X6_term8)
#
# Initial constraints
ocpX6.subject_to(ocpX6.at_t0(X6)==X6_0)
# Pick a solution method
ocpX6.solver('ipopt',options)
# Make it concrete for this ocp
ocpX6.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

'''
OcpZ6
'''
ocpZ6 = Ocp(T=Tf)
# Variables
z6_x1 = ocpZ6.variable(grid='control',include_last=True)
z6_y1 = ocpZ6.variable(grid='control',include_last=True)
Z6_Z_61 = vertcat(z6_x1, z6_y1)
z6_x2 = ocpZ6.variable(grid='control',include_last=True)
z6_y2 = ocpZ6.variable(grid='control',include_last=True)
Z6_Z_62 = vertcat(z6_x2, z6_y2)
z6_x3 = ocpZ6.variable(grid='control',include_last=True)
z6_y3 = ocpZ6.variable(grid='control',include_last=True)
Z6_Z_63 = vertcat(z6_x3, z6_y3)
z6_x4 = ocpZ6.variable(grid='control',include_last=True)
z6_y4 = ocpZ6.variable(grid='control',include_last=True)
Z6_Z_64 = vertcat(z6_x4, z6_y4)
z6_x5 = ocpZ6.variable(grid='control',include_last=True)
z6_y5 = ocpZ6.variable(grid='control',include_last=True)
Z6_Z_65 = vertcat(z6_x5, z6_y5)
z6_x6 = ocpZ6.variable(grid='control',include_last=True)
z6_y6 = ocpZ6.variable(grid='control',include_last=True)
Z6_Z_66 = vertcat(z6_x6, z6_y6)
z6_x7 = ocpZ6.variable(grid='control',include_last=True)
z6_y7 = ocpZ6.variable(grid='control',include_last=True)
Z6_Z_67 = vertcat(z6_x7, z6_y7)
z6_x8 = ocpZ6.variable(grid='control',include_last=True)
z6_y8 = ocpZ6.variable(grid='control',include_last=True)
Z6_Z_68 = vertcat(z6_x8, z6_y8)
# Parameters
Z6_lambda_61 = ocpZ6.parameter(nx, grid='control',include_last=True)
Z6_lambda_62 = ocpZ6.parameter(nx, grid='control',include_last=True)
Z6_lambda_63 = ocpZ6.parameter(nx, grid='control',include_last=True)
Z6_lambda_64 = ocpZ6.parameter(nx, grid='control',include_last=True)
Z6_lambda_65 = ocpZ6.parameter(nx, grid='control',include_last=True)
Z6_lambda_66 = ocpZ6.parameter(nx, grid='control',include_last=True)
Z6_lambda_67 = ocpZ6.parameter(nx, grid='control',include_last=True)
Z6_lambda_68 = ocpZ6.parameter(nx, grid='control',include_last=True)
Z6_X_1 = ocpZ6.parameter(nx, grid='control',include_last=True)
Z6_X_2 = ocpZ6.parameter(nx, grid='control',include_last=True)
Z6_X_3 = ocpZ6.parameter(nx, grid='control',include_last=True)
Z6_X_4 = ocpZ6.parameter(nx, grid='control',include_last=True)
Z6_X_5 = ocpZ6.parameter(nx, grid='control',include_last=True)
Z6_X_6 = ocpZ6.parameter(nx, grid='control',include_last=True)
Z6_X_7 = ocpZ6.parameter(nx, grid='control',include_last=True)
Z6_X_8 = ocpZ6.parameter(nx, grid='control',include_last=True)
# Extended objective
Z6_c1 = Z6_Z_61 - Z6_X_1
Z6_term1 = dot(Z6_lambda_61, Z6_c1) + mu/2*sumsqr(Z6_c1)
if ocpZ6.is_signal(Z6_term1):
    Z6_term1 = ocpZ6.sum(Z6_term1,include_last=True)
ocpZ6.add_objective(Z6_term1)
#
Z6_c2 = Z6_Z_62 - Z6_X_2
Z6_term2 = dot(Z6_lambda_62, Z6_c2) + mu/2*sumsqr(Z6_c2)
if ocpZ6.is_signal(Z6_term2):
    Z6_term2 = ocpZ6.sum(Z6_term2,include_last=True)
ocpZ6.add_objective(Z6_term2)
#
Z6_c3 = Z6_Z_63 - Z6_X_3
Z6_term3 = dot(Z6_lambda_63, Z6_c3) + mu/2*sumsqr(Z6_c3)
if ocpZ6.is_signal(Z6_term3):
    Z6_term3 = ocpZ6.sum(Z6_term3,include_last=True)
ocpZ6.add_objective(Z6_term3)
#
Z6_c4 = Z6_Z_64 - Z6_X_4
Z6_term4 = dot(Z6_lambda_64, Z6_c4) + mu/2*sumsqr(Z6_c4)
if ocpZ6.is_signal(Z6_term4):
    Z6_term4 = ocpZ6.sum(Z6_term4,include_last=True)
ocpZ6.add_objective(Z6_term4)
#
Z6_c5 = Z6_Z_65 - Z6_X_5
Z6_term5 = dot(Z6_lambda_65, Z6_c5) + mu/2*sumsqr(Z6_c5)
if ocpZ6.is_signal(Z6_term5):
    Z6_term5 = ocpZ6.sum(Z6_term5,include_last=True)
ocpZ6.add_objective(Z6_term5)
#
Z6_c6 = Z6_Z_66 - Z6_X_6
Z6_term6 = dot(Z6_lambda_66, Z6_c6) + mu/2*sumsqr(Z6_c6)
if ocpZ6.is_signal(Z6_term6):
    Z6_term6 = ocpZ6.sum(Z6_term6,include_last=True)
ocpZ6.add_objective(Z6_term6)
#
Z6_c7 = Z6_Z_67 - Z6_X_7
Z6_term7 = dot(Z6_lambda_67, Z6_c7) + mu/2*sumsqr(Z6_c7)
if ocpZ6.is_signal(Z6_term7):
    Z6_term7 = ocpZ6.sum(Z6_term7,include_last=True)
ocpZ6.add_objective(Z6_term7)
#
Z6_c8 = Z6_Z_68 - Z6_X_8
Z6_term8 = dot(Z6_lambda_68, Z6_c8) + mu/2*sumsqr(Z6_c8)
if ocpZ6.is_signal(Z6_term8):
    Z6_term8 = ocpZ6.sum(Z6_term8,include_last=True)
ocpZ6.add_objective(Z6_term8)
# Constraints
Z6_distance1  = (z6_x6-z6_x1)**2 + (z6_y6-z6_y1)**2
Z6_distance2  = (z6_x6-z6_x2)**2 + (z6_y6-z6_y2)**2
Z6_distance3  = (z6_x6-z6_x3)**2 + (z6_y6-z6_y3)**2
Z6_distance4  = (z6_x6-z6_x4)**2 + (z6_y6-z6_y4)**2
Z6_distance5  = (z6_x6-z6_x5)**2 + (z6_y6-z6_y5)**2
Z6_distance6  = (z6_x6-z6_x7)**2 + (z6_y6-z6_y7)**2
Z6_distance7  = (z6_x6-z6_x8)**2 + (z6_y6-z6_y8)**2
ocpZ6.subject_to( Z6_distance1  >= boat_radius )
ocpZ6.subject_to( Z6_distance2  >= boat_radius )
ocpZ6.subject_to( Z6_distance3  >= boat_radius )
ocpZ6.subject_to( Z6_distance4  >= boat_radius )
ocpZ6.subject_to( Z6_distance5  >= boat_radius )
ocpZ6.subject_to( Z6_distance6  >= boat_radius )
ocpZ6.subject_to( Z6_distance7  >= boat_radius )
# Pick a solution method
ocpZ6.solver('ipopt',options)
# Make it concrete for this ocp
ocpZ6.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

'''
OcpX7
'''
ocpX7 = Ocp(T=Tf)
# States
x7 = ocpX7.state()
y7 = ocpX7.state()
X7 = vertcat(x7,y7)
# Controls
u7 = ocpX7.control()
v7 = ocpX7.control()
# Initial condition
X7_0 = ocpX7.parameter(nx)
# Parameters
X7_lambda_17 = ocpX7.parameter(nx, grid='control',include_last=True)
X7_lambda_27 = ocpX7.parameter(nx, grid='control',include_last=True)
X7_lambda_37 = ocpX7.parameter(nx, grid='control',include_last=True)
X7_lambda_47 = ocpX7.parameter(nx, grid='control',include_last=True)
X7_lambda_57 = ocpX7.parameter(nx, grid='control',include_last=True)
X7_lambda_67 = ocpX7.parameter(nx, grid='control',include_last=True)
X7_lambda_77 = ocpX7.parameter(nx, grid='control',include_last=True)
X7_lambda_87 = ocpX7.parameter(nx, grid='control',include_last=True)
X7_Z_17 = ocpX7.parameter(nx, grid='control',include_last=True)
X7_Z_27 = ocpX7.parameter(nx, grid='control',include_last=True)
X7_Z_37 = ocpX7.parameter(nx, grid='control',include_last=True)
X7_Z_47 = ocpX7.parameter(nx, grid='control',include_last=True)
X7_Z_57 = ocpX7.parameter(nx, grid='control',include_last=True)
X7_Z_67 = ocpX7.parameter(nx, grid='control',include_last=True)
X7_Z_77 = ocpX7.parameter(nx, grid='control',include_last=True)
X7_Z_87 = ocpX7.parameter(nx, grid='control',include_last=True)
#ODE
ocpX7.set_der(x7, u7)
ocpX7.set_der(y7, v7)
# Lagrange objective
ocpX7.add_objective(ocpX7.integral((xd_7-x7)**2 + (yd_7-y7)**2))
ocpX7.add_objective(ocpX7.at_tf((xd_7-x7)**2 + (yd_7-y7)**2))
ocpX7.subject_to( (-max_speed_limit <= u7) <= max_speed_limit )
ocpX7.subject_to( (-max_speed_limit <= v7) <= max_speed_limit )
# Extended objective
X7_c1 = X7_Z_17 - X7
X7_term1 = dot(X7_lambda_17, X7_c1) + mu/2*sumsqr(X7_c1)
if ocpX7.is_signal(X7_term1):
    X7_term1 = ocpX7.sum(X7_term1,include_last=True)
ocpX7.add_objective(X7_term1)
#
X7_c2 = X7_Z_27 - X7
X7_term2 = dot(X7_lambda_27, X7_c2) + mu/2*sumsqr(X7_c2)
if ocpX7.is_signal(X7_term2):
    X7_term2 = ocpX7.sum(X7_term2,include_last=True)
ocpX7.add_objective(X7_term2)
#
X7_c3 = X7_Z_37 - X7
X7_term3 = dot(X7_lambda_37, X7_c3) + mu/2*sumsqr(X7_c3)
if ocpX7.is_signal(X7_term3):
    X7_term3 = ocpX7.sum(X7_term3,include_last=True)
ocpX7.add_objective(X7_term3)
#
X7_c4 = X7_Z_47 - X7
X7_term4 = dot(X7_lambda_47, X7_c4) + mu/2*sumsqr(X7_c4)
if ocpX7.is_signal(X7_term4):
    X7_term4 = ocpX7.sum(X7_term4,include_last=True)
ocpX7.add_objective(X7_term4)
#
X7_c5 = X7_Z_57 - X7
X7_term5 = dot(X7_lambda_57, X7_c5) + mu/2*sumsqr(X7_c5)
if ocpX7.is_signal(X7_term5):
    X7_term5 = ocpX7.sum(X7_term5,include_last=True)
ocpX7.add_objective(X7_term5)
#
X7_c6 = X7_Z_67 - X7
X7_term6 = dot(X7_lambda_67, X7_c6) + mu/2*sumsqr(X7_c6)
if ocpX7.is_signal(X7_term6):
    X7_term6 = ocpX7.sum(X7_term6,include_last=True)
ocpX7.add_objective(X7_term6)
#
X7_c7 = X7_Z_77 - X7
X7_term7 = dot(X7_lambda_77, X7_c7) + mu/2*sumsqr(X7_c7)
if ocpX7.is_signal(X7_term7):
    X7_term7 = ocpX7.sum(X7_term7,include_last=True)
ocpX7.add_objective(X7_term7)
#
X7_c8 = X7_Z_87 - X7
X7_term8 = dot(X7_lambda_87, X7_c8) + mu/2*sumsqr(X7_c8)
if ocpX7.is_signal(X7_term8):
    X7_term8 = ocpX7.sum(X7_term8,include_last=True)
ocpX7.add_objective(X7_term8)
#
# Initial constraints
ocpX7.subject_to(ocpX7.at_t0(X7)==X7_0)
# Pick a solution method
ocpX7.solver('ipopt',options)
# Make it concrete for this ocp
ocpX7.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

'''
OcpZ7
'''
ocpZ7 = Ocp(T=Tf)
# Variables
z7_x1 = ocpZ7.variable(grid='control',include_last=True)
z7_y1 = ocpZ7.variable(grid='control',include_last=True)
Z7_Z_71 = vertcat(z7_x1, z7_y1)
z7_x2 = ocpZ7.variable(grid='control',include_last=True)
z7_y2 = ocpZ7.variable(grid='control',include_last=True)
Z7_Z_72 = vertcat(z7_x2, z7_y2)
z7_x3 = ocpZ7.variable(grid='control',include_last=True)
z7_y3 = ocpZ7.variable(grid='control',include_last=True)
Z7_Z_73 = vertcat(z7_x3, z7_y3)
z7_x4 = ocpZ7.variable(grid='control',include_last=True)
z7_y4 = ocpZ7.variable(grid='control',include_last=True)
Z7_Z_74 = vertcat(z7_x4, z7_y4)
z7_x5 = ocpZ7.variable(grid='control',include_last=True)
z7_y5 = ocpZ7.variable(grid='control',include_last=True)
Z7_Z_75 = vertcat(z7_x5, z7_y5)
z7_x6 = ocpZ7.variable(grid='control',include_last=True)
z7_y6 = ocpZ7.variable(grid='control',include_last=True)
Z7_Z_76 = vertcat(z7_x6, z7_y6)
z7_x7 = ocpZ7.variable(grid='control',include_last=True)
z7_y7 = ocpZ7.variable(grid='control',include_last=True)
Z7_Z_77 = vertcat(z7_x7, z7_y7)
z7_x8 = ocpZ7.variable(grid='control',include_last=True)
z7_y8 = ocpZ7.variable(grid='control',include_last=True)
Z7_Z_78 = vertcat(z7_x8, z7_y8)
# Parameters
Z7_lambda_71 = ocpZ7.parameter(nx, grid='control',include_last=True)
Z7_lambda_72 = ocpZ7.parameter(nx, grid='control',include_last=True)
Z7_lambda_73 = ocpZ7.parameter(nx, grid='control',include_last=True)
Z7_lambda_74 = ocpZ7.parameter(nx, grid='control',include_last=True)
Z7_lambda_75 = ocpZ7.parameter(nx, grid='control',include_last=True)
Z7_lambda_76 = ocpZ7.parameter(nx, grid='control',include_last=True)
Z7_lambda_77 = ocpZ7.parameter(nx, grid='control',include_last=True)
Z7_lambda_78 = ocpZ7.parameter(nx, grid='control',include_last=True)
Z7_X_1 = ocpZ7.parameter(nx, grid='control',include_last=True)
Z7_X_2 = ocpZ7.parameter(nx, grid='control',include_last=True)
Z7_X_3 = ocpZ7.parameter(nx, grid='control',include_last=True)
Z7_X_4 = ocpZ7.parameter(nx, grid='control',include_last=True)
Z7_X_5 = ocpZ7.parameter(nx, grid='control',include_last=True)
Z7_X_6 = ocpZ7.parameter(nx, grid='control',include_last=True)
Z7_X_7 = ocpZ7.parameter(nx, grid='control',include_last=True)
Z7_X_8 = ocpZ7.parameter(nx, grid='control',include_last=True)
# Extended objective
Z7_c1 = Z7_Z_71 - Z7_X_1
Z7_term1 = dot(Z7_lambda_71, Z7_c1) + mu/2*sumsqr(Z7_c1)
if ocpZ7.is_signal(Z7_term1):
    Z7_term1 = ocpZ7.sum(Z7_term1,include_last=True)
ocpZ7.add_objective(Z7_term1)
#
Z7_c2 = Z7_Z_72 - Z7_X_2
Z7_term2 = dot(Z7_lambda_72, Z7_c2) + mu/2*sumsqr(Z7_c2)
if ocpZ7.is_signal(Z7_term2):
    Z7_term2 = ocpZ7.sum(Z7_term2,include_last=True)
ocpZ7.add_objective(Z7_term2)
#
Z7_c3 = Z7_Z_73 - Z7_X_3
Z7_term3 = dot(Z7_lambda_73, Z7_c3) + mu/2*sumsqr(Z7_c3)
if ocpZ7.is_signal(Z7_term3):
    Z7_term3 = ocpZ7.sum(Z7_term3,include_last=True)
ocpZ7.add_objective(Z7_term3)
#
Z7_c4 = Z7_Z_74 - Z7_X_4
Z7_term4 = dot(Z7_lambda_74, Z7_c4) + mu/2*sumsqr(Z7_c4)
if ocpZ7.is_signal(Z7_term4):
    Z7_term4 = ocpZ7.sum(Z7_term4,include_last=True)
ocpZ7.add_objective(Z7_term4)
#
Z7_c5 = Z7_Z_75 - Z7_X_5
Z7_term5 = dot(Z7_lambda_75, Z7_c5) + mu/2*sumsqr(Z7_c5)
if ocpZ7.is_signal(Z7_term5):
    Z7_term5 = ocpZ7.sum(Z7_term5,include_last=True)
ocpZ7.add_objective(Z7_term5)
#
Z7_c6 = Z7_Z_76 - Z7_X_6
Z7_term6 = dot(Z7_lambda_76, Z7_c6) + mu/2*sumsqr(Z7_c6)
if ocpZ7.is_signal(Z7_term6):
    Z7_term6 = ocpZ7.sum(Z7_term6,include_last=True)
ocpZ7.add_objective(Z7_term6)
#
Z7_c7 = Z7_Z_77 - Z7_X_7
Z7_term7 = dot(Z7_lambda_77, Z7_c7) + mu/2*sumsqr(Z7_c7)
if ocpZ7.is_signal(Z7_term7):
    Z7_term7 = ocpZ7.sum(Z7_term7,include_last=True)
ocpZ7.add_objective(Z7_term7)
#
Z7_c8 = Z7_Z_78 - Z7_X_8
Z7_term8 = dot(Z7_lambda_78, Z7_c8) + mu/2*sumsqr(Z7_c8)
if ocpZ7.is_signal(Z7_term8):
    Z7_term8 = ocpZ7.sum(Z7_term8,include_last=True)
ocpZ7.add_objective(Z7_term8)
# Constraints
Z7_distance1  = (z7_x7-z7_x1)**2 + (z7_y7-z7_y1)**2
Z7_distance2  = (z7_x7-z7_x2)**2 + (z7_y7-z7_y2)**2
Z7_distance3  = (z7_x7-z7_x3)**2 + (z7_y7-z7_y3)**2
Z7_distance4  = (z7_x7-z7_x4)**2 + (z7_y7-z7_y4)**2
Z7_distance5  = (z7_x7-z7_x5)**2 + (z7_y7-z7_y5)**2
Z7_distance6  = (z7_x7-z7_x6)**2 + (z7_y7-z7_y6)**2
Z7_distance7  = (z7_x7-z7_x8)**2 + (z7_y7-z7_y8)**2
ocpZ7.subject_to( Z7_distance1  >= boat_radius )
ocpZ7.subject_to( Z7_distance2  >= boat_radius )
ocpZ7.subject_to( Z7_distance3  >= boat_radius )
ocpZ7.subject_to( Z7_distance4  >= boat_radius )
ocpZ7.subject_to( Z7_distance5  >= boat_radius )
ocpZ7.subject_to( Z7_distance6  >= boat_radius )
ocpZ7.subject_to( Z7_distance7  >= boat_radius )
# Pick a solution method
ocpZ7.solver('ipopt',options)
# Make it concrete for this ocp
ocpZ7.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

'''
OcpX8
'''
ocpX8 = Ocp(T=Tf)
# States
x8 = ocpX8.state()
y8 = ocpX8.state()
X8 = vertcat(x8,y8)
# Controls
u8 = ocpX8.control()
v8 = ocpX8.control()
# Initial condition
X8_0 = ocpX8.parameter(nx)
# Parameters
X8_lambda_18 = ocpX8.parameter(nx, grid='control',include_last=True)
X8_lambda_28 = ocpX8.parameter(nx, grid='control',include_last=True)
X8_lambda_38 = ocpX8.parameter(nx, grid='control',include_last=True)
X8_lambda_48 = ocpX8.parameter(nx, grid='control',include_last=True)
X8_lambda_58 = ocpX8.parameter(nx, grid='control',include_last=True)
X8_lambda_68 = ocpX8.parameter(nx, grid='control',include_last=True)
X8_lambda_78 = ocpX8.parameter(nx, grid='control',include_last=True)
X8_lambda_88 = ocpX8.parameter(nx, grid='control',include_last=True)
X8_Z_18 = ocpX8.parameter(nx, grid='control',include_last=True)
X8_Z_28 = ocpX8.parameter(nx, grid='control',include_last=True)
X8_Z_38 = ocpX8.parameter(nx, grid='control',include_last=True)
X8_Z_48 = ocpX8.parameter(nx, grid='control',include_last=True)
X8_Z_58 = ocpX8.parameter(nx, grid='control',include_last=True)
X8_Z_68 = ocpX8.parameter(nx, grid='control',include_last=True)
X8_Z_78 = ocpX8.parameter(nx, grid='control',include_last=True)
X8_Z_88 = ocpX8.parameter(nx, grid='control',include_last=True)
#ODE
ocpX8.set_der(x8, u8)
ocpX8.set_der(y8, v8)
# Lagrange objective
ocpX8.add_objective(ocpX8.integral((xd_8-x8)**2 + (yd_8-y8)**2))
ocpX8.add_objective(ocpX8.at_tf((xd_8-x8)**2 + (yd_8-y8)**2))
ocpX8.subject_to( (-max_speed_limit <= u8) <= max_speed_limit )
ocpX8.subject_to( (-max_speed_limit <= v8) <= max_speed_limit )
# Extended objective
X8_c1 = X8_Z_18 - X8
X8_term1 = dot(X8_lambda_18, X8_c1) + mu/2*sumsqr(X8_c1)
if ocpX8.is_signal(X8_term1):
    X8_term1 = ocpX8.sum(X8_term1,include_last=True)
ocpX8.add_objective(X8_term1)
#
X8_c2 = X8_Z_28 - X8
X8_term2 = dot(X8_lambda_28, X8_c2) + mu/2*sumsqr(X8_c2)
if ocpX8.is_signal(X8_term2):
    X8_term2 = ocpX8.sum(X8_term2,include_last=True)
ocpX8.add_objective(X8_term2)
#
X8_c3 = X8_Z_38 - X8
X8_term3 = dot(X8_lambda_38, X8_c3) + mu/2*sumsqr(X8_c3)
if ocpX8.is_signal(X8_term3):
    X8_term3 = ocpX8.sum(X8_term3,include_last=True)
ocpX8.add_objective(X8_term3)
#
X8_c4 = X8_Z_48 - X8
X8_term4 = dot(X8_lambda_48, X8_c4) + mu/2*sumsqr(X8_c4)
if ocpX8.is_signal(X8_term4):
    X8_term4 = ocpX8.sum(X8_term4,include_last=True)
ocpX8.add_objective(X8_term4)
#
X8_c5 = X8_Z_58 - X8
X8_term5 = dot(X8_lambda_58, X8_c5) + mu/2*sumsqr(X8_c5)
if ocpX8.is_signal(X8_term5):
    X8_term5 = ocpX8.sum(X8_term5,include_last=True)
ocpX8.add_objective(X8_term5)
#
X8_c6 = X8_Z_68 - X8
X8_term6 = dot(X8_lambda_68, X8_c6) + mu/2*sumsqr(X8_c6)
if ocpX8.is_signal(X8_term6):
    X8_term6 = ocpX8.sum(X8_term6,include_last=True)
ocpX8.add_objective(X8_term6)
#
X8_c7 = X8_Z_78 - X8
X8_term7 = dot(X8_lambda_78, X8_c7) + mu/2*sumsqr(X8_c7)
if ocpX8.is_signal(X8_term7):
    X8_term7 = ocpX8.sum(X8_term7,include_last=True)
ocpX8.add_objective(X8_term7)
#
X8_c8 = X8_Z_88 - X8
X8_term8 = dot(X8_lambda_88, X8_c8) + mu/2*sumsqr(X8_c8)
if ocpX8.is_signal(X8_term8):
    X8_term8 = ocpX8.sum(X8_term8,include_last=True)
ocpX8.add_objective(X8_term8)
#
# Initial constraints
ocpX8.subject_to(ocpX8.at_t0(X8)==X8_0)
# Pick a solution method
ocpX8.solver('ipopt',options)
# Make it concrete for this ocp
ocpX8.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

'''
OcpZ8
'''
ocpZ8 = Ocp(T=Tf)
# Variables
z8_x1 = ocpZ8.variable(grid='control',include_last=True)
z8_y1 = ocpZ8.variable(grid='control',include_last=True)
Z8_Z_81 = vertcat(z8_x1, z8_y1)
z8_x2 = ocpZ8.variable(grid='control',include_last=True)
z8_y2 = ocpZ8.variable(grid='control',include_last=True)
Z8_Z_82 = vertcat(z8_x2, z8_y2)
z8_x3 = ocpZ8.variable(grid='control',include_last=True)
z8_y3 = ocpZ8.variable(grid='control',include_last=True)
Z8_Z_83 = vertcat(z8_x3, z8_y3)
z8_x4 = ocpZ8.variable(grid='control',include_last=True)
z8_y4 = ocpZ8.variable(grid='control',include_last=True)
Z8_Z_84 = vertcat(z8_x4, z8_y4)
z8_x5 = ocpZ8.variable(grid='control',include_last=True)
z8_y5 = ocpZ8.variable(grid='control',include_last=True)
Z8_Z_85 = vertcat(z8_x5, z8_y5)
z8_x6 = ocpZ8.variable(grid='control',include_last=True)
z8_y6 = ocpZ8.variable(grid='control',include_last=True)
Z8_Z_86 = vertcat(z8_x6, z8_y6)
z8_x7 = ocpZ8.variable(grid='control',include_last=True)
z8_y7 = ocpZ8.variable(grid='control',include_last=True)
Z8_Z_87 = vertcat(z8_x7, z8_y7)
z8_x8 = ocpZ8.variable(grid='control',include_last=True)
z8_y8 = ocpZ8.variable(grid='control',include_last=True)
Z8_Z_88 = vertcat(z8_x8, z8_y8)
# Parameters
Z8_lambda_81 = ocpZ8.parameter(nx, grid='control',include_last=True)
Z8_lambda_82 = ocpZ8.parameter(nx, grid='control',include_last=True)
Z8_lambda_83 = ocpZ8.parameter(nx, grid='control',include_last=True)
Z8_lambda_84 = ocpZ8.parameter(nx, grid='control',include_last=True)
Z8_lambda_85 = ocpZ8.parameter(nx, grid='control',include_last=True)
Z8_lambda_86 = ocpZ8.parameter(nx, grid='control',include_last=True)
Z8_lambda_87 = ocpZ8.parameter(nx, grid='control',include_last=True)
Z8_lambda_88 = ocpZ8.parameter(nx, grid='control',include_last=True)
Z8_X_1 = ocpZ8.parameter(nx, grid='control',include_last=True)
Z8_X_2 = ocpZ8.parameter(nx, grid='control',include_last=True)
Z8_X_3 = ocpZ8.parameter(nx, grid='control',include_last=True)
Z8_X_4 = ocpZ8.parameter(nx, grid='control',include_last=True)
Z8_X_5 = ocpZ8.parameter(nx, grid='control',include_last=True)
Z8_X_6 = ocpZ8.parameter(nx, grid='control',include_last=True)
Z8_X_7 = ocpZ8.parameter(nx, grid='control',include_last=True)
Z8_X_8 = ocpZ8.parameter(nx, grid='control',include_last=True)
# Extended objective
Z8_c1 = Z8_Z_81 - Z8_X_1
Z8_term1 = dot(Z8_lambda_81, Z8_c1) + mu/2*sumsqr(Z8_c1)
if ocpZ8.is_signal(Z8_term1):
    Z8_term1 = ocpZ8.sum(Z8_term1,include_last=True)
ocpZ8.add_objective(Z8_term1)
#
Z8_c2 = Z8_Z_82 - Z8_X_2
Z8_term2 = dot(Z8_lambda_82, Z8_c2) + mu/2*sumsqr(Z8_c2)
if ocpZ8.is_signal(Z8_term2):
    Z8_term2 = ocpZ8.sum(Z8_term2,include_last=True)
ocpZ8.add_objective(Z8_term2)
#
Z8_c3 = Z8_Z_83 - Z8_X_3
Z8_term3 = dot(Z8_lambda_83, Z8_c3) + mu/2*sumsqr(Z8_c3)
if ocpZ8.is_signal(Z8_term3):
    Z8_term3 = ocpZ8.sum(Z8_term3,include_last=True)
ocpZ8.add_objective(Z8_term3)
#
Z8_c4 = Z8_Z_84 - Z8_X_4
Z8_term4 = dot(Z8_lambda_84, Z8_c4) + mu/2*sumsqr(Z8_c4)
if ocpZ8.is_signal(Z8_term4):
    Z8_term4 = ocpZ8.sum(Z8_term4,include_last=True)
ocpZ8.add_objective(Z8_term4)
#
Z8_c5 = Z8_Z_85 - Z8_X_5
Z8_term5 = dot(Z8_lambda_85, Z8_c5) + mu/2*sumsqr(Z8_c5)
if ocpZ8.is_signal(Z8_term5):
    Z8_term5 = ocpZ8.sum(Z8_term5,include_last=True)
ocpZ8.add_objective(Z8_term5)
#
Z8_c6 = Z8_Z_86 - Z8_X_6
Z8_term6 = dot(Z8_lambda_86, Z8_c6) + mu/2*sumsqr(Z8_c6)
if ocpZ8.is_signal(Z8_term6):
    Z8_term6 = ocpZ8.sum(Z8_term6,include_last=True)
ocpZ8.add_objective(Z8_term6)
#
Z8_c7 = Z8_Z_87 - Z8_X_7
Z8_term7 = dot(Z8_lambda_87, Z8_c7) + mu/2*sumsqr(Z8_c7)
if ocpZ8.is_signal(Z8_term7):
    Z8_term7 = ocpZ8.sum(Z8_term7,include_last=True)
ocpZ8.add_objective(Z8_term7)
#
Z8_c8 = Z8_Z_88 - Z8_X_8
Z8_term8 = dot(Z8_lambda_88, Z8_c8) + mu/2*sumsqr(Z8_c8)
if ocpZ8.is_signal(Z8_term8):
    Z8_term8 = ocpZ8.sum(Z8_term8,include_last=True)
ocpZ8.add_objective(Z8_term8)
# Constraints
Z8_distance1  = (z8_x8-z8_x1)**2 + (z8_y8-z8_y1)**2
Z8_distance2  = (z8_x8-z8_x2)**2 + (z8_y8-z8_y2)**2
Z8_distance3  = (z8_x8-z8_x3)**2 + (z8_y8-z8_y3)**2
Z8_distance4  = (z8_x8-z8_x4)**2 + (z8_y8-z8_y4)**2
Z8_distance5  = (z8_x8-z8_x5)**2 + (z8_y8-z8_y5)**2
Z8_distance6  = (z8_x8-z8_x6)**2 + (z8_y8-z8_y6)**2
Z8_distance7  = (z8_x8-z8_x7)**2 + (z8_y8-z8_y7)**2
ocpZ8.subject_to( Z8_distance1  >= boat_radius )
ocpZ8.subject_to( Z8_distance2  >= boat_radius )
ocpZ8.subject_to( Z8_distance3  >= boat_radius )
ocpZ8.subject_to( Z8_distance4  >= boat_radius )
ocpZ8.subject_to( Z8_distance5  >= boat_radius )
ocpZ8.subject_to( Z8_distance6  >= boat_radius )
ocpZ8.subject_to( Z8_distance7  >= boat_radius )
# Pick a solution method
ocpZ8.solver('ipopt',options)
# Make it concrete for this ocp
ocpZ8.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

# Initialize all values

l11 = np.zeros([nx, Nhor+1])
l12 = np.zeros([nx, Nhor+1])
l13 = np.zeros([nx, Nhor+1])
l14 = np.zeros([nx, Nhor+1])
l15 = np.zeros([nx, Nhor+1])
l16 = np.zeros([nx, Nhor+1])
l17 = np.zeros([nx, Nhor+1])
l18 = np.zeros([nx, Nhor+1])

l21 = np.zeros([nx, Nhor+1])
l22 = np.zeros([nx, Nhor+1])
l23 = np.zeros([nx, Nhor+1])
l24 = np.zeros([nx, Nhor+1])
l25 = np.zeros([nx, Nhor+1])
l26 = np.zeros([nx, Nhor+1])
l27 = np.zeros([nx, Nhor+1])
l28 = np.zeros([nx, Nhor+1])

l31 = np.zeros([nx, Nhor+1])
l32 = np.zeros([nx, Nhor+1])
l33 = np.zeros([nx, Nhor+1])
l34 = np.zeros([nx, Nhor+1])
l35 = np.zeros([nx, Nhor+1])
l36 = np.zeros([nx, Nhor+1])
l37 = np.zeros([nx, Nhor+1])
l38 = np.zeros([nx, Nhor+1])

l41 = np.zeros([nx, Nhor+1])
l42 = np.zeros([nx, Nhor+1])
l43 = np.zeros([nx, Nhor+1])
l44 = np.zeros([nx, Nhor+1])
l45 = np.zeros([nx, Nhor+1])
l46 = np.zeros([nx, Nhor+1])
l47 = np.zeros([nx, Nhor+1])
l48 = np.zeros([nx, Nhor+1])

l51 = np.zeros([nx, Nhor+1])
l52 = np.zeros([nx, Nhor+1])
l53 = np.zeros([nx, Nhor+1])
l54 = np.zeros([nx, Nhor+1])
l55 = np.zeros([nx, Nhor+1])
l56 = np.zeros([nx, Nhor+1])
l57 = np.zeros([nx, Nhor+1])
l58 = np.zeros([nx, Nhor+1])

l61 = np.zeros([nx, Nhor+1])
l62 = np.zeros([nx, Nhor+1])
l63 = np.zeros([nx, Nhor+1])
l64 = np.zeros([nx, Nhor+1])
l65 = np.zeros([nx, Nhor+1])
l66 = np.zeros([nx, Nhor+1])
l67 = np.zeros([nx, Nhor+1])
l68 = np.zeros([nx, Nhor+1])

l71 = np.zeros([nx, Nhor+1])
l72 = np.zeros([nx, Nhor+1])
l73 = np.zeros([nx, Nhor+1])
l74 = np.zeros([nx, Nhor+1])
l75 = np.zeros([nx, Nhor+1])
l76 = np.zeros([nx, Nhor+1])
l77 = np.zeros([nx, Nhor+1])
l78 = np.zeros([nx, Nhor+1])

l81 = np.zeros([nx, Nhor+1])
l82 = np.zeros([nx, Nhor+1])
l83 = np.zeros([nx, Nhor+1])
l84 = np.zeros([nx, Nhor+1])
l85 = np.zeros([nx, Nhor+1])
l86 = np.zeros([nx, Nhor+1])
l87 = np.zeros([nx, Nhor+1])
l88 = np.zeros([nx, Nhor+1])

z11 = np.zeros([nx, Nhor+1])
z12 = np.zeros([nx, Nhor+1])
z13 = np.zeros([nx, Nhor+1])
z14 = np.zeros([nx, Nhor+1])
z15 = np.zeros([nx, Nhor+1])
z16 = np.zeros([nx, Nhor+1])
z17 = np.zeros([nx, Nhor+1])
z18 = np.zeros([nx, Nhor+1])

z21 = np.zeros([nx, Nhor+1])
z22 = np.zeros([nx, Nhor+1])
z23 = np.zeros([nx, Nhor+1])
z24 = np.zeros([nx, Nhor+1])
z25 = np.zeros([nx, Nhor+1])
z26 = np.zeros([nx, Nhor+1])
z27 = np.zeros([nx, Nhor+1])
z28 = np.zeros([nx, Nhor+1])

z31 = np.zeros([nx, Nhor+1])
z32 = np.zeros([nx, Nhor+1])
z33 = np.zeros([nx, Nhor+1])
z34 = np.zeros([nx, Nhor+1])
z35 = np.zeros([nx, Nhor+1])
z36 = np.zeros([nx, Nhor+1])
z37 = np.zeros([nx, Nhor+1])
z38 = np.zeros([nx, Nhor+1])

z41 = np.zeros([nx, Nhor+1])
z42 = np.zeros([nx, Nhor+1])
z43 = np.zeros([nx, Nhor+1])
z44 = np.zeros([nx, Nhor+1])
z45 = np.zeros([nx, Nhor+1])
z46 = np.zeros([nx, Nhor+1])
z47 = np.zeros([nx, Nhor+1])
z48 = np.zeros([nx, Nhor+1])

z51 = np.zeros([nx, Nhor+1])
z52 = np.zeros([nx, Nhor+1])
z53 = np.zeros([nx, Nhor+1])
z54 = np.zeros([nx, Nhor+1])
z55 = np.zeros([nx, Nhor+1])
z56 = np.zeros([nx, Nhor+1])
z57 = np.zeros([nx, Nhor+1])
z58 = np.zeros([nx, Nhor+1])

z61 = np.zeros([nx, Nhor+1])
z62 = np.zeros([nx, Nhor+1])
z63 = np.zeros([nx, Nhor+1])
z64 = np.zeros([nx, Nhor+1])
z65 = np.zeros([nx, Nhor+1])
z66 = np.zeros([nx, Nhor+1])
z67 = np.zeros([nx, Nhor+1])
z68 = np.zeros([nx, Nhor+1])

z71 = np.zeros([nx, Nhor+1])
z72 = np.zeros([nx, Nhor+1])
z73 = np.zeros([nx, Nhor+1])
z74 = np.zeros([nx, Nhor+1])
z75 = np.zeros([nx, Nhor+1])
z76 = np.zeros([nx, Nhor+1])
z77 = np.zeros([nx, Nhor+1])
z78 = np.zeros([nx, Nhor+1])

z81 = np.zeros([nx, Nhor+1])
z82 = np.zeros([nx, Nhor+1])
z83 = np.zeros([nx, Nhor+1])
z84 = np.zeros([nx, Nhor+1])
z85 = np.zeros([nx, Nhor+1])
z86 = np.zeros([nx, Nhor+1])
z87 = np.zeros([nx, Nhor+1])
z88 = np.zeros([nx, Nhor+1])

X1p = np.zeros([nx, Nhor+1]) + [[x0_1],[y0_1]]
X2p = np.zeros([nx, Nhor+1]) + [[x0_2],[y0_2]]
X3p = np.zeros([nx, Nhor+1]) + [[x0_3],[y0_3]]
X4p = np.zeros([nx, Nhor+1]) + [[x0_4],[y0_4]]
X5p = np.zeros([nx, Nhor+1]) + [[x0_5],[y0_5]]
X6p = np.zeros([nx, Nhor+1]) + [[x0_6],[y0_6]]
X7p = np.zeros([nx, Nhor+1]) + [[x0_7],[y0_7]]
X8p = np.zeros([nx, Nhor+1]) + [[x0_8],[y0_8]]

# Dynamics declaration
Sim_asv_dyn = ocpX1._method.discrete_system(ocpX1)

# Log data for post-processing
x1_history[0] = current_X1[0]
y1_history[0] = current_X1[1]
x2_history[0] = current_X2[0]
y2_history[0] = current_X2[1]
x3_history[0] = current_X3[0]
y3_history[0] = current_X3[1]
x4_history[0] = current_X4[0]
y4_history[0] = current_X4[1]
x5_history[0] = current_X5[0]
y5_history[0] = current_X5[1]
x6_history[0] = current_X6[0]
y6_history[0] = current_X6[1]
x7_history[0] = current_X7[0]
y7_history[0] = current_X7[1]
x8_history[0] = current_X8[0]
y8_history[0] = current_X8[1]

#Initialization ADMM

for i in range(N_init):

    # Set values and solve for each agent ocpX

    ocpX1.set_value(X1_0, current_X1)
    ocpX1.set_value(X1_lambda_11, l11)
    ocpX1.set_value(X1_lambda_21, l21)
    ocpX1.set_value(X1_lambda_31, l31)
    ocpX1.set_value(X1_lambda_41, l41)
    ocpX1.set_value(X1_lambda_51, l51)
    ocpX1.set_value(X1_lambda_61, l61)
    ocpX1.set_value(X1_lambda_71, l71)
    ocpX1.set_value(X1_lambda_81, l81)
    ocpX1.set_value(X1_Z_11, z11)
    ocpX1.set_value(X1_Z_21, z21)
    ocpX1.set_value(X1_Z_31, z31)
    ocpX1.set_value(X1_Z_41, z41)
    ocpX1.set_value(X1_Z_51, z51)
    ocpX1.set_value(X1_Z_61, z61)
    ocpX1.set_value(X1_Z_71, z71)
    ocpX1.set_value(X1_Z_81, z81)

    solX1 = ocpX1.solve()

    ocpX2.set_value(X2_0, current_X2)
    ocpX2.set_value(X2_lambda_12, l12)
    ocpX2.set_value(X2_lambda_22, l22)
    ocpX2.set_value(X2_lambda_32, l32)
    ocpX2.set_value(X2_lambda_42, l42)
    ocpX2.set_value(X2_lambda_52, l52)
    ocpX2.set_value(X2_lambda_62, l62)
    ocpX2.set_value(X2_lambda_72, l72)
    ocpX2.set_value(X2_lambda_82, l82)
    ocpX2.set_value(X2_Z_12, z12)
    ocpX2.set_value(X2_Z_22, z22)
    ocpX2.set_value(X2_Z_32, z32)
    ocpX2.set_value(X2_Z_42, z42)
    ocpX2.set_value(X2_Z_52, z52)
    ocpX2.set_value(X2_Z_62, z62)
    ocpX2.set_value(X2_Z_72, z72)
    ocpX2.set_value(X2_Z_82, z82)

    solX2 = ocpX2.solve()

    ocpX3.set_value(X3_0, current_X3)
    ocpX3.set_value(X3_lambda_13, l13)
    ocpX3.set_value(X3_lambda_23, l23)
    ocpX3.set_value(X3_lambda_33, l33)
    ocpX3.set_value(X3_lambda_43, l43)
    ocpX3.set_value(X3_lambda_53, l53)
    ocpX3.set_value(X3_lambda_63, l63)
    ocpX3.set_value(X3_lambda_73, l73)
    ocpX3.set_value(X3_lambda_83, l83)
    ocpX3.set_value(X3_Z_13, z13)
    ocpX3.set_value(X3_Z_23, z23)
    ocpX3.set_value(X3_Z_33, z33)
    ocpX3.set_value(X3_Z_43, z43)
    ocpX3.set_value(X3_Z_53, z53)
    ocpX3.set_value(X3_Z_63, z63)
    ocpX3.set_value(X3_Z_73, z73)
    ocpX3.set_value(X3_Z_83, z83)

    solX3 = ocpX3.solve()

    ocpX4.set_value(X4_0, current_X4)
    ocpX4.set_value(X4_lambda_14, l14)
    ocpX4.set_value(X4_lambda_24, l24)
    ocpX4.set_value(X4_lambda_34, l34)
    ocpX4.set_value(X4_lambda_44, l44)
    ocpX4.set_value(X4_lambda_54, l54)
    ocpX4.set_value(X4_lambda_64, l64)
    ocpX4.set_value(X4_lambda_74, l74)
    ocpX4.set_value(X4_lambda_84, l84)
    ocpX4.set_value(X4_Z_14, z14)
    ocpX4.set_value(X4_Z_24, z24)
    ocpX4.set_value(X4_Z_34, z34)
    ocpX4.set_value(X4_Z_44, z44)
    ocpX4.set_value(X4_Z_54, z54)
    ocpX4.set_value(X4_Z_64, z64)
    ocpX4.set_value(X4_Z_74, z74)
    ocpX4.set_value(X4_Z_84, z84)

    solX4 = ocpX4.solve()

    ocpX5.set_value(X5_0, current_X5)
    ocpX5.set_value(X5_lambda_15, l15)
    ocpX5.set_value(X5_lambda_25, l25)
    ocpX5.set_value(X5_lambda_35, l35)
    ocpX5.set_value(X5_lambda_45, l45)
    ocpX5.set_value(X5_lambda_55, l55)
    ocpX5.set_value(X5_lambda_65, l65)
    ocpX5.set_value(X5_lambda_75, l75)
    ocpX5.set_value(X5_lambda_85, l85)
    ocpX5.set_value(X5_Z_15, z15)
    ocpX5.set_value(X5_Z_25, z25)
    ocpX5.set_value(X5_Z_35, z35)
    ocpX5.set_value(X5_Z_45, z45)
    ocpX5.set_value(X5_Z_55, z55)
    ocpX5.set_value(X5_Z_65, z65)
    ocpX5.set_value(X5_Z_75, z75)
    ocpX5.set_value(X5_Z_85, z85)

    solX5 = ocpX5.solve()

    ocpX6.set_value(X6_0, current_X6)
    ocpX6.set_value(X6_lambda_16, l16)
    ocpX6.set_value(X6_lambda_26, l26)
    ocpX6.set_value(X6_lambda_36, l36)
    ocpX6.set_value(X6_lambda_46, l46)
    ocpX6.set_value(X6_lambda_56, l56)
    ocpX6.set_value(X6_lambda_66, l66)
    ocpX6.set_value(X6_lambda_76, l76)
    ocpX6.set_value(X6_lambda_86, l86)
    ocpX6.set_value(X6_Z_16, z16)
    ocpX6.set_value(X6_Z_26, z26)
    ocpX6.set_value(X6_Z_36, z36)
    ocpX6.set_value(X6_Z_46, z46)
    ocpX6.set_value(X6_Z_56, z56)
    ocpX6.set_value(X6_Z_66, z66)
    ocpX6.set_value(X6_Z_76, z76)
    ocpX6.set_value(X6_Z_86, z86)

    solX6 = ocpX6.solve()

    ocpX7.set_value(X7_0, current_X7)
    ocpX7.set_value(X7_lambda_17, l17)
    ocpX7.set_value(X7_lambda_27, l27)
    ocpX7.set_value(X7_lambda_37, l37)
    ocpX7.set_value(X7_lambda_47, l47)
    ocpX7.set_value(X7_lambda_57, l57)
    ocpX7.set_value(X7_lambda_67, l67)
    ocpX7.set_value(X7_lambda_77, l77)
    ocpX7.set_value(X7_lambda_87, l87)
    ocpX7.set_value(X7_Z_17, z17)
    ocpX7.set_value(X7_Z_27, z27)
    ocpX7.set_value(X7_Z_37, z37)
    ocpX7.set_value(X7_Z_47, z47)
    ocpX7.set_value(X7_Z_57, z57)
    ocpX7.set_value(X7_Z_67, z67)
    ocpX7.set_value(X7_Z_77, z77)
    ocpX7.set_value(X7_Z_87, z87)

    solX7 = ocpX7.solve()

    ocpX8.set_value(X8_0, current_X8)
    ocpX8.set_value(X8_lambda_18, l18)
    ocpX8.set_value(X8_lambda_28, l28)
    ocpX8.set_value(X8_lambda_38, l38)
    ocpX8.set_value(X8_lambda_48, l48)
    ocpX8.set_value(X8_lambda_58, l58)
    ocpX8.set_value(X8_lambda_68, l68)
    ocpX8.set_value(X8_lambda_78, l78)
    ocpX8.set_value(X8_lambda_88, l88)
    ocpX8.set_value(X8_Z_18, z18)
    ocpX8.set_value(X8_Z_28, z28)
    ocpX8.set_value(X8_Z_38, z38)
    ocpX8.set_value(X8_Z_48, z48)
    ocpX8.set_value(X8_Z_58, z58)
    ocpX8.set_value(X8_Z_68, z68)
    ocpX8.set_value(X8_Z_78, z78)
    ocpX8.set_value(X8_Z_88, z88)

    solX8 = ocpX8.solve()

    # Save the information

    X1p = solX1.sample(X1, grid='control')[1].T
    X2p = solX2.sample(X2, grid='control')[1].T
    X3p = solX3.sample(X3, grid='control')[1].T
    X4p = solX4.sample(X4, grid='control')[1].T
    X5p = solX5.sample(X5, grid='control')[1].T
    X6p = solX6.sample(X6, grid='control')[1].T
    X7p = solX7.sample(X7, grid='control')[1].T
    X8p = solX8.sample(X8, grid='control')[1].T
    
    # Set values and solve for each agent ocpZ

    ocpZ1.set_value(Z1_lambda_11, l11)
    ocpZ1.set_value(Z1_lambda_12, l12)
    ocpZ1.set_value(Z1_lambda_13, l13)
    ocpZ1.set_value(Z1_lambda_14, l14)
    ocpZ1.set_value(Z1_lambda_15, l15)
    ocpZ1.set_value(Z1_lambda_16, l16)
    ocpZ1.set_value(Z1_lambda_17, l17)
    ocpZ1.set_value(Z1_lambda_18, l18)
    ocpZ1.set_value(Z1_X_1, X1p)
    ocpZ1.set_value(Z1_X_2, X2p)
    ocpZ1.set_value(Z1_X_3, X3p)
    ocpZ1.set_value(Z1_X_4, X4p)
    ocpZ1.set_value(Z1_X_5, X5p)
    ocpZ1.set_value(Z1_X_6, X6p)
    ocpZ1.set_value(Z1_X_7, X7p)
    ocpZ1.set_value(Z1_X_8, X8p)

    ocpZ2.set_value(Z2_lambda_21, l21)
    ocpZ2.set_value(Z2_lambda_22, l22)
    ocpZ2.set_value(Z2_lambda_23, l23)
    ocpZ2.set_value(Z2_lambda_24, l24)
    ocpZ2.set_value(Z2_lambda_25, l25)
    ocpZ2.set_value(Z2_lambda_26, l26)
    ocpZ2.set_value(Z2_lambda_27, l27)
    ocpZ2.set_value(Z2_lambda_28, l28)
    ocpZ2.set_value(Z2_X_1, X1p)
    ocpZ2.set_value(Z2_X_2, X2p)
    ocpZ2.set_value(Z2_X_3, X3p)
    ocpZ2.set_value(Z2_X_4, X4p)
    ocpZ2.set_value(Z2_X_5, X5p)
    ocpZ2.set_value(Z2_X_6, X6p)
    ocpZ2.set_value(Z2_X_7, X7p)
    ocpZ2.set_value(Z2_X_8, X8p)

    ocpZ3.set_value(Z3_lambda_31, l31)
    ocpZ3.set_value(Z3_lambda_32, l32)
    ocpZ3.set_value(Z3_lambda_33, l33)
    ocpZ3.set_value(Z3_lambda_34, l34)
    ocpZ3.set_value(Z3_lambda_35, l35)
    ocpZ3.set_value(Z3_lambda_36, l36)
    ocpZ3.set_value(Z3_lambda_37, l37)
    ocpZ3.set_value(Z3_lambda_38, l38)
    ocpZ3.set_value(Z3_X_1, X1p)
    ocpZ3.set_value(Z3_X_2, X2p)
    ocpZ3.set_value(Z3_X_3, X3p)
    ocpZ3.set_value(Z3_X_4, X4p)
    ocpZ3.set_value(Z3_X_5, X5p)
    ocpZ3.set_value(Z3_X_6, X6p)
    ocpZ3.set_value(Z3_X_7, X7p)
    ocpZ3.set_value(Z3_X_8, X8p)

    ocpZ4.set_value(Z4_lambda_41, l41)
    ocpZ4.set_value(Z4_lambda_42, l42)
    ocpZ4.set_value(Z4_lambda_43, l43)
    ocpZ4.set_value(Z4_lambda_44, l44)
    ocpZ4.set_value(Z4_lambda_45, l45)
    ocpZ4.set_value(Z4_lambda_46, l46)
    ocpZ4.set_value(Z4_lambda_47, l47)
    ocpZ4.set_value(Z4_lambda_48, l48)
    ocpZ4.set_value(Z4_X_1, X1p)
    ocpZ4.set_value(Z4_X_2, X2p)
    ocpZ4.set_value(Z4_X_3, X3p)
    ocpZ4.set_value(Z4_X_4, X4p)
    ocpZ4.set_value(Z4_X_5, X5p)
    ocpZ4.set_value(Z4_X_6, X6p)
    ocpZ4.set_value(Z4_X_7, X7p)
    ocpZ4.set_value(Z4_X_8, X8p)

    ocpZ5.set_value(Z5_lambda_51, l51)
    ocpZ5.set_value(Z5_lambda_52, l52)
    ocpZ5.set_value(Z5_lambda_53, l53)
    ocpZ5.set_value(Z5_lambda_54, l54)
    ocpZ5.set_value(Z5_lambda_55, l55)
    ocpZ5.set_value(Z5_lambda_56, l56)
    ocpZ5.set_value(Z5_lambda_57, l57)
    ocpZ5.set_value(Z5_lambda_58, l58)
    ocpZ5.set_value(Z5_X_1, X1p)
    ocpZ5.set_value(Z5_X_2, X2p)
    ocpZ5.set_value(Z5_X_3, X3p)
    ocpZ5.set_value(Z5_X_4, X4p)
    ocpZ5.set_value(Z5_X_5, X5p)
    ocpZ5.set_value(Z5_X_6, X6p)
    ocpZ5.set_value(Z5_X_7, X7p)
    ocpZ5.set_value(Z5_X_8, X8p)

    ocpZ6.set_value(Z6_lambda_61, l61)
    ocpZ6.set_value(Z6_lambda_62, l62)
    ocpZ6.set_value(Z6_lambda_63, l63)
    ocpZ6.set_value(Z6_lambda_64, l64)
    ocpZ6.set_value(Z6_lambda_65, l65)
    ocpZ6.set_value(Z6_lambda_66, l66)
    ocpZ6.set_value(Z6_lambda_67, l67)
    ocpZ6.set_value(Z6_lambda_68, l68)
    ocpZ6.set_value(Z6_X_1, X1p)
    ocpZ6.set_value(Z6_X_2, X2p)
    ocpZ6.set_value(Z6_X_3, X3p)
    ocpZ6.set_value(Z6_X_4, X4p)
    ocpZ6.set_value(Z6_X_5, X5p)
    ocpZ6.set_value(Z6_X_6, X6p)
    ocpZ6.set_value(Z6_X_7, X7p)
    ocpZ6.set_value(Z6_X_8, X8p)

    ocpZ7.set_value(Z7_lambda_71, l71)
    ocpZ7.set_value(Z7_lambda_72, l72)
    ocpZ7.set_value(Z7_lambda_73, l73)
    ocpZ7.set_value(Z7_lambda_74, l74)
    ocpZ7.set_value(Z7_lambda_75, l75)
    ocpZ7.set_value(Z7_lambda_76, l76)
    ocpZ7.set_value(Z7_lambda_77, l77)
    ocpZ7.set_value(Z7_lambda_78, l78)
    ocpZ7.set_value(Z7_X_1, X1p)
    ocpZ7.set_value(Z7_X_2, X2p)
    ocpZ7.set_value(Z7_X_3, X3p)
    ocpZ7.set_value(Z7_X_4, X4p)
    ocpZ7.set_value(Z7_X_5, X5p)
    ocpZ7.set_value(Z7_X_6, X6p)
    ocpZ7.set_value(Z7_X_7, X7p)
    ocpZ7.set_value(Z7_X_8, X8p)

    ocpZ8.set_value(Z8_lambda_81, l81)
    ocpZ8.set_value(Z8_lambda_82, l82)
    ocpZ8.set_value(Z8_lambda_83, l83)
    ocpZ8.set_value(Z8_lambda_84, l84)
    ocpZ8.set_value(Z8_lambda_85, l85)
    ocpZ8.set_value(Z8_lambda_86, l86)
    ocpZ8.set_value(Z8_lambda_87, l87)
    ocpZ8.set_value(Z8_lambda_88, l88)
    ocpZ8.set_value(Z8_X_1, X1p)
    ocpZ8.set_value(Z8_X_2, X2p)
    ocpZ8.set_value(Z8_X_3, X3p)
    ocpZ8.set_value(Z8_X_4, X4p)
    ocpZ8.set_value(Z8_X_5, X5p)
    ocpZ8.set_value(Z8_X_6, X6p)
    ocpZ8.set_value(Z8_X_7, X7p)
    ocpZ8.set_value(Z8_X_8, X8p)

    solZ1 = ocpZ1.solve()
    solZ2 = ocpZ2.solve()
    solZ3 = ocpZ3.solve()
    solZ4 = ocpZ4.solve()
    solZ5 = ocpZ5.solve()
    solZ6 = ocpZ6.solve()
    solZ7 = ocpZ7.solve()
    solZ8 = ocpZ8.solve()

    # Compute new Z parameters

    z11 = solZ1.sample(Z1_Z_11, grid='control')[1].T
    z12 = solZ1.sample(Z1_Z_12, grid='control')[1].T
    z13 = solZ1.sample(Z1_Z_13, grid='control')[1].T
    z14 = solZ1.sample(Z1_Z_14, grid='control')[1].T
    z15 = solZ1.sample(Z1_Z_15, grid='control')[1].T
    z16 = solZ1.sample(Z1_Z_16, grid='control')[1].T
    z17 = solZ1.sample(Z1_Z_17, grid='control')[1].T
    z18 = solZ1.sample(Z1_Z_18, grid='control')[1].T

    z21 = solZ2.sample(Z2_Z_21, grid='control')[1].T
    z22 = solZ2.sample(Z2_Z_22, grid='control')[1].T
    z23 = solZ2.sample(Z2_Z_23, grid='control')[1].T
    z24 = solZ2.sample(Z2_Z_24, grid='control')[1].T
    z25 = solZ2.sample(Z2_Z_25, grid='control')[1].T
    z26 = solZ2.sample(Z2_Z_26, grid='control')[1].T
    z27 = solZ2.sample(Z2_Z_27, grid='control')[1].T
    z28 = solZ2.sample(Z2_Z_28, grid='control')[1].T

    z31 = solZ3.sample(Z3_Z_31, grid='control')[1].T
    z32 = solZ3.sample(Z3_Z_32, grid='control')[1].T
    z33 = solZ3.sample(Z3_Z_33, grid='control')[1].T
    z34 = solZ3.sample(Z3_Z_34, grid='control')[1].T
    z35 = solZ3.sample(Z3_Z_35, grid='control')[1].T
    z38 = solZ3.sample(Z3_Z_38, grid='control')[1].T
    z36 = solZ3.sample(Z3_Z_36, grid='control')[1].T
    z37 = solZ3.sample(Z3_Z_37, grid='control')[1].T

    z41 = solZ4.sample(Z4_Z_41, grid='control')[1].T
    z42 = solZ4.sample(Z4_Z_42, grid='control')[1].T
    z43 = solZ4.sample(Z4_Z_43, grid='control')[1].T
    z44 = solZ4.sample(Z4_Z_44, grid='control')[1].T
    z45 = solZ4.sample(Z4_Z_45, grid='control')[1].T
    z46 = solZ4.sample(Z4_Z_46, grid='control')[1].T
    z47 = solZ4.sample(Z4_Z_47, grid='control')[1].T
    z48 = solZ4.sample(Z4_Z_48, grid='control')[1].T

    z51 = solZ5.sample(Z5_Z_51, grid='control')[1].T
    z52 = solZ5.sample(Z5_Z_52, grid='control')[1].T
    z53 = solZ5.sample(Z5_Z_53, grid='control')[1].T
    z54 = solZ5.sample(Z5_Z_54, grid='control')[1].T
    z55 = solZ5.sample(Z5_Z_55, grid='control')[1].T
    z56 = solZ5.sample(Z5_Z_56, grid='control')[1].T
    z57 = solZ5.sample(Z5_Z_57, grid='control')[1].T
    z58 = solZ5.sample(Z5_Z_58, grid='control')[1].T

    z61 = solZ6.sample(Z6_Z_61, grid='control')[1].T
    z62 = solZ6.sample(Z6_Z_62, grid='control')[1].T
    z63 = solZ6.sample(Z6_Z_63, grid='control')[1].T
    z64 = solZ6.sample(Z6_Z_64, grid='control')[1].T
    z65 = solZ6.sample(Z6_Z_65, grid='control')[1].T
    z66 = solZ6.sample(Z6_Z_66, grid='control')[1].T
    z67 = solZ6.sample(Z6_Z_67, grid='control')[1].T
    z68 = solZ6.sample(Z6_Z_68, grid='control')[1].T

    z71 = solZ7.sample(Z7_Z_71, grid='control')[1].T
    z72 = solZ7.sample(Z7_Z_72, grid='control')[1].T
    z73 = solZ7.sample(Z7_Z_73, grid='control')[1].T
    z74 = solZ7.sample(Z7_Z_74, grid='control')[1].T
    z75 = solZ7.sample(Z7_Z_75, grid='control')[1].T
    z76 = solZ7.sample(Z7_Z_76, grid='control')[1].T
    z77 = solZ7.sample(Z7_Z_77, grid='control')[1].T
    z78 = solZ7.sample(Z7_Z_78, grid='control')[1].T

    z81 = solZ8.sample(Z8_Z_81, grid='control')[1].T
    z82 = solZ8.sample(Z8_Z_82, grid='control')[1].T
    z83 = solZ8.sample(Z8_Z_83, grid='control')[1].T
    z84 = solZ8.sample(Z8_Z_84, grid='control')[1].T
    z85 = solZ8.sample(Z8_Z_85, grid='control')[1].T
    z86 = solZ8.sample(Z8_Z_86, grid='control')[1].T
    z87 = solZ8.sample(Z8_Z_87, grid='control')[1].T
    z88 = solZ8.sample(Z8_Z_88, grid='control')[1].T

    # Update lambda multipliers

    l11 = l11 + mu*(z11 - X1p)
    l12 = l12 + mu*(z12 - X2p)
    l13 = l13 + mu*(z13 - X3p)
    l14 = l14 + mu*(z14 - X4p)
    l15 = l15 + mu*(z15 - X5p)
    l16 = l16 + mu*(z16 - X6p)
    l17 = l17 + mu*(z17 - X7p)
    l18 = l18 + mu*(z18 - X8p)

    l21 = l21 + mu*(z21 - X1p)
    l22 = l22 + mu*(z22 - X2p)
    l23 = l23 + mu*(z23 - X3p)
    l24 = l24 + mu*(z24 - X4p)
    l25 = l25 + mu*(z25 - X5p)
    l26 = l26 + mu*(z26 - X6p)
    l27 = l27 + mu*(z27 - X7p)
    l28 = l28 + mu*(z28 - X8p)

    l31 = l31 + mu*(z31 - X1p)
    l32 = l32 + mu*(z32 - X2p)
    l33 = l33 + mu*(z33 - X3p)
    l34 = l34 + mu*(z34 - X4p)
    l35 = l35 + mu*(z35 - X5p)
    l36 = l36 + mu*(z36 - X6p)
    l37 = l37 + mu*(z37 - X7p)
    l38 = l38 + mu*(z38 - X8p)

    l41 = l41 + mu*(z41 - X1p)
    l42 = l42 + mu*(z42 - X2p)
    l43 = l43 + mu*(z43 - X3p)
    l44 = l44 + mu*(z44 - X4p)
    l45 = l45 + mu*(z45 - X5p)
    l46 = l46 + mu*(z46 - X6p)
    l47 = l47 + mu*(z47 - X7p)
    l48 = l48 + mu*(z48 - X8p)

    l51 = l51 + mu*(z51 - X1p)
    l52 = l52 + mu*(z52 - X2p)
    l53 = l53 + mu*(z53 - X3p)
    l54 = l54 + mu*(z54 - X4p)
    l55 = l55 + mu*(z55 - X5p)
    l56 = l56 + mu*(z56 - X6p)
    l57 = l57 + mu*(z57 - X7p)
    l58 = l58 + mu*(z58 - X8p)

    l61 = l61 + mu*(z61 - X1p)
    l62 = l62 + mu*(z62 - X2p)
    l63 = l63 + mu*(z63 - X3p)
    l64 = l64 + mu*(z64 - X4p)
    l65 = l65 + mu*(z65 - X5p)
    l66 = l66 + mu*(z66 - X6p)
    l67 = l67 + mu*(z67 - X7p)
    l68 = l68 + mu*(z68 - X8p)

    l71 = l71 + mu*(z71 - X1p)
    l72 = l72 + mu*(z72 - X2p)
    l73 = l73 + mu*(z73 - X3p)
    l74 = l74 + mu*(z74 - X4p)
    l75 = l75 + mu*(z75 - X5p)
    l76 = l76 + mu*(z76 - X6p)
    l77 = l77 + mu*(z77 - X7p)
    l78 = l78 + mu*(z78 - X8p)

    l81 = l81 + mu*(z81 - X1p)
    l82 = l82 + mu*(z82 - X2p)
    l83 = l83 + mu*(z83 - X3p)
    l84 = l84 + mu*(z84 - X4p)
    l85 = l85 + mu*(z85 - X5p)
    l86 = l86 + mu*(z86 - X6p)
    l87 = l87 + mu*(z87 - X7p)
    l88 = l88 + mu*(z88 - X8p)

    residuals = []
    residuals.append(norm_fro(z11-X1p))
    residuals.append(norm_fro(z21-X1p))
    residuals.append(norm_fro(z31-X1p))
    residuals.append(norm_fro(z41-X1p))
    residuals.append(norm_fro(z51-X1p))
    residuals.append(norm_fro(z61-X1p))
    residuals.append(norm_fro(z71-X1p))
    residuals.append(norm_fro(z81-X1p))

    residuals.append(norm_fro(z12-X2p))
    residuals.append(norm_fro(z22-X2p))
    residuals.append(norm_fro(z32-X2p))
    residuals.append(norm_fro(z42-X2p))
    residuals.append(norm_fro(z52-X2p))
    residuals.append(norm_fro(z62-X2p))
    residuals.append(norm_fro(z72-X2p))
    residuals.append(norm_fro(z82-X2p))

    residuals.append(norm_fro(z13-X3p))
    residuals.append(norm_fro(z23-X3p))
    residuals.append(norm_fro(z33-X3p))
    residuals.append(norm_fro(z43-X3p))
    residuals.append(norm_fro(z53-X3p))
    residuals.append(norm_fro(z63-X3p))
    residuals.append(norm_fro(z73-X3p))
    residuals.append(norm_fro(z83-X3p))

    residuals.append(norm_fro(z14-X4p))
    residuals.append(norm_fro(z24-X4p))
    residuals.append(norm_fro(z34-X4p))
    residuals.append(norm_fro(z44-X4p))
    residuals.append(norm_fro(z54-X4p))
    residuals.append(norm_fro(z64-X4p))
    residuals.append(norm_fro(z74-X4p))
    residuals.append(norm_fro(z84-X4p))

    residuals.append(norm_fro(z15-X5p))
    residuals.append(norm_fro(z25-X5p))
    residuals.append(norm_fro(z35-X5p))
    residuals.append(norm_fro(z45-X5p))
    residuals.append(norm_fro(z55-X5p))
    residuals.append(norm_fro(z65-X5p))
    residuals.append(norm_fro(z75-X5p))
    residuals.append(norm_fro(z85-X5p))

    residuals.append(norm_fro(z16-X6p))
    residuals.append(norm_fro(z26-X6p))
    residuals.append(norm_fro(z36-X6p))
    residuals.append(norm_fro(z46-X6p))
    residuals.append(norm_fro(z56-X6p))
    residuals.append(norm_fro(z66-X6p))
    residuals.append(norm_fro(z76-X6p))
    residuals.append(norm_fro(z86-X6p))

    residuals.append(norm_fro(z17-X7p))
    residuals.append(norm_fro(z27-X7p))
    residuals.append(norm_fro(z37-X7p))
    residuals.append(norm_fro(z47-X7p))
    residuals.append(norm_fro(z57-X7p))
    residuals.append(norm_fro(z67-X7p))
    residuals.append(norm_fro(z77-X7p))
    residuals.append(norm_fro(z87-X7p))

    residuals.append(norm_fro(z18-X8p))
    residuals.append(norm_fro(z28-X8p))
    residuals.append(norm_fro(z38-X8p))
    residuals.append(norm_fro(z48-X8p))
    residuals.append(norm_fro(z58-X8p))
    residuals.append(norm_fro(z68-X8p))
    residuals.append(norm_fro(z78-X8p))
    residuals.append(norm_fro(z88-X8p))

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
    tsx5, u5sol = solX5.sample(u5, grid='control')
    _, v5sol = solX5.sample(v5, grid='control')
    tsx6, u6sol = solX6.sample(u6, grid='control')
    _, v6sol = solX6.sample(v6, grid='control')
    tsx7, u7sol = solX7.sample(u7, grid='control')
    _, v7sol = solX7.sample(v7, grid='control')
    tsx8, u8sol = solX8.sample(u8, grid='control')
    _, v8sol = solX8.sample(v8, grid='control')
    # Simulate dynamics (applying the first control input) and update the current state
    current_X1 = Sim_asv_dyn(x0=current_X1, u=vertcat(u1sol[0],v1sol[0]), T=dt)["xf"]
    current_X2 = Sim_asv_dyn(x0=current_X2, u=vertcat(u2sol[0],v2sol[0]), T=dt)["xf"]
    current_X3 = Sim_asv_dyn(x0=current_X3, u=vertcat(u3sol[0],v3sol[0]), T=dt)["xf"]
    current_X4 = Sim_asv_dyn(x0=current_X4, u=vertcat(u4sol[0],v4sol[0]), T=dt)["xf"]
    current_X5 = Sim_asv_dyn(x0=current_X5, u=vertcat(u5sol[0],v5sol[0]), T=dt)["xf"]
    current_X6 = Sim_asv_dyn(x0=current_X6, u=vertcat(u6sol[0],v6sol[0]), T=dt)["xf"]
    current_X7 = Sim_asv_dyn(x0=current_X7, u=vertcat(u7sol[0],v7sol[0]), T=dt)["xf"]
    current_X8 = Sim_asv_dyn(x0=current_X8, u=vertcat(u8sol[0],v8sol[0]), T=dt)["xf"]

    # Log data for post-processing
    x1_history[j+1] = current_X1[0].full()
    y1_history[j+1] = current_X1[1].full()
    x2_history[j+1] = current_X2[0].full()
    y2_history[j+1] = current_X2[1].full()
    x3_history[j+1] = current_X3[0].full()
    y3_history[j+1] = current_X3[1].full()
    x4_history[j+1] = current_X4[0].full()
    y4_history[j+1] = current_X4[1].full()
    x5_history[j+1] = current_X5[0].full()
    y5_history[j+1] = current_X5[1].full()
    x6_history[j+1] = current_X6[0].full()
    y6_history[j+1] = current_X6[1].full()
    x7_history[j+1] = current_X7[0].full()
    y7_history[j+1] = current_X7[1].full()
    x8_history[j+1] = current_X8[0].full()
    y8_history[j+1] = current_X8[1].full()

    for i in range(N_mpc):

        # Set values and solve for each agent ocpX

        ocpX1.set_value(X1_0, current_X1)
        ocpX1.set_value(X1_lambda_11, l11)
        ocpX1.set_value(X1_lambda_21, l21)
        ocpX1.set_value(X1_lambda_31, l31)
        ocpX1.set_value(X1_lambda_41, l41)
        ocpX1.set_value(X1_lambda_51, l51)
        ocpX1.set_value(X1_lambda_61, l61)
        ocpX1.set_value(X1_lambda_71, l71)
        ocpX1.set_value(X1_lambda_81, l81)
        ocpX1.set_value(X1_Z_11, z11)
        ocpX1.set_value(X1_Z_21, z21)
        ocpX1.set_value(X1_Z_31, z31)
        ocpX1.set_value(X1_Z_41, z41)
        ocpX1.set_value(X1_Z_51, z51)
        ocpX1.set_value(X1_Z_61, z61)
        ocpX1.set_value(X1_Z_71, z71)
        ocpX1.set_value(X1_Z_81, z81)

        solX1 = ocpX1.solve()

        ocpX2.set_value(X2_0, current_X2)
        ocpX2.set_value(X2_lambda_12, l12)
        ocpX2.set_value(X2_lambda_22, l22)
        ocpX2.set_value(X2_lambda_32, l32)
        ocpX2.set_value(X2_lambda_42, l42)
        ocpX2.set_value(X2_lambda_52, l52)
        ocpX2.set_value(X2_lambda_62, l62)
        ocpX2.set_value(X2_lambda_72, l72)
        ocpX2.set_value(X2_lambda_82, l82)
        ocpX2.set_value(X2_Z_12, z12)
        ocpX2.set_value(X2_Z_22, z22)
        ocpX2.set_value(X2_Z_32, z32)
        ocpX2.set_value(X2_Z_42, z42)
        ocpX2.set_value(X2_Z_52, z52)
        ocpX2.set_value(X2_Z_62, z62)
        ocpX2.set_value(X2_Z_72, z72)
        ocpX2.set_value(X2_Z_82, z82)

        solX2 = ocpX2.solve()

        ocpX3.set_value(X3_0, current_X3)
        ocpX3.set_value(X3_lambda_13, l13)
        ocpX3.set_value(X3_lambda_23, l23)
        ocpX3.set_value(X3_lambda_33, l33)
        ocpX3.set_value(X3_lambda_43, l43)
        ocpX3.set_value(X3_lambda_53, l53)
        ocpX3.set_value(X3_lambda_63, l63)
        ocpX3.set_value(X3_lambda_73, l73)
        ocpX3.set_value(X3_lambda_83, l83)
        ocpX3.set_value(X3_Z_13, z13)
        ocpX3.set_value(X3_Z_23, z23)
        ocpX3.set_value(X3_Z_33, z33)
        ocpX3.set_value(X3_Z_43, z43)
        ocpX3.set_value(X3_Z_53, z53)
        ocpX3.set_value(X3_Z_63, z63)
        ocpX3.set_value(X3_Z_73, z73)
        ocpX3.set_value(X3_Z_83, z83)

        solX3 = ocpX3.solve()

        ocpX4.set_value(X4_0, current_X4)
        ocpX4.set_value(X4_lambda_14, l14)
        ocpX4.set_value(X4_lambda_24, l24)
        ocpX4.set_value(X4_lambda_34, l34)
        ocpX4.set_value(X4_lambda_44, l44)
        ocpX4.set_value(X4_lambda_54, l54)
        ocpX4.set_value(X4_lambda_64, l64)
        ocpX4.set_value(X4_lambda_74, l74)
        ocpX4.set_value(X4_lambda_84, l84)
        ocpX4.set_value(X4_Z_14, z14)
        ocpX4.set_value(X4_Z_24, z24)
        ocpX4.set_value(X4_Z_34, z34)
        ocpX4.set_value(X4_Z_44, z44)
        ocpX4.set_value(X4_Z_54, z54)
        ocpX4.set_value(X4_Z_64, z64)
        ocpX4.set_value(X4_Z_74, z74)
        ocpX4.set_value(X4_Z_84, z84)

        solX4 = ocpX4.solve()

        ocpX5.set_value(X5_0, current_X5)
        ocpX5.set_value(X5_lambda_15, l15)
        ocpX5.set_value(X5_lambda_25, l25)
        ocpX5.set_value(X5_lambda_35, l35)
        ocpX5.set_value(X5_lambda_45, l45)
        ocpX5.set_value(X5_lambda_55, l55)
        ocpX5.set_value(X5_lambda_65, l65)
        ocpX5.set_value(X5_lambda_75, l75)
        ocpX5.set_value(X5_lambda_85, l85)
        ocpX5.set_value(X5_Z_15, z15)
        ocpX5.set_value(X5_Z_25, z25)
        ocpX5.set_value(X5_Z_35, z35)
        ocpX5.set_value(X5_Z_45, z45)
        ocpX5.set_value(X5_Z_55, z55)
        ocpX5.set_value(X5_Z_65, z65)
        ocpX5.set_value(X5_Z_75, z75)
        ocpX5.set_value(X5_Z_85, z85)

        solX5 = ocpX5.solve()

        ocpX6.set_value(X6_0, current_X6)
        ocpX6.set_value(X6_lambda_16, l16)
        ocpX6.set_value(X6_lambda_26, l26)
        ocpX6.set_value(X6_lambda_36, l36)
        ocpX6.set_value(X6_lambda_46, l46)
        ocpX6.set_value(X6_lambda_56, l56)
        ocpX6.set_value(X6_lambda_66, l66)
        ocpX6.set_value(X6_lambda_76, l76)
        ocpX6.set_value(X6_lambda_86, l86)
        ocpX6.set_value(X6_Z_16, z16)
        ocpX6.set_value(X6_Z_26, z26)
        ocpX6.set_value(X6_Z_36, z36)
        ocpX6.set_value(X6_Z_46, z46)
        ocpX6.set_value(X6_Z_56, z56)
        ocpX6.set_value(X6_Z_66, z66)
        ocpX6.set_value(X6_Z_76, z76)
        ocpX6.set_value(X6_Z_86, z86)

        solX6 = ocpX6.solve()

        ocpX7.set_value(X7_0, current_X7)
        ocpX7.set_value(X7_lambda_17, l17)
        ocpX7.set_value(X7_lambda_27, l27)
        ocpX7.set_value(X7_lambda_37, l37)
        ocpX7.set_value(X7_lambda_47, l47)
        ocpX7.set_value(X7_lambda_57, l57)
        ocpX7.set_value(X7_lambda_67, l67)
        ocpX7.set_value(X7_lambda_77, l77)
        ocpX7.set_value(X7_lambda_87, l87)
        ocpX7.set_value(X7_Z_17, z17)
        ocpX7.set_value(X7_Z_27, z27)
        ocpX7.set_value(X7_Z_37, z37)
        ocpX7.set_value(X7_Z_47, z47)
        ocpX7.set_value(X7_Z_57, z57)
        ocpX7.set_value(X7_Z_67, z67)
        ocpX7.set_value(X7_Z_77, z77)
        ocpX7.set_value(X7_Z_87, z87)

        solX7 = ocpX7.solve()

        ocpX8.set_value(X8_0, current_X8)
        ocpX8.set_value(X8_lambda_18, l18)
        ocpX8.set_value(X8_lambda_28, l28)
        ocpX8.set_value(X8_lambda_38, l38)
        ocpX8.set_value(X8_lambda_48, l48)
        ocpX8.set_value(X8_lambda_58, l58)
        ocpX8.set_value(X8_lambda_68, l68)
        ocpX8.set_value(X8_lambda_78, l78)
        ocpX8.set_value(X8_lambda_88, l88)
        ocpX8.set_value(X8_Z_18, z18)
        ocpX8.set_value(X8_Z_28, z28)
        ocpX8.set_value(X8_Z_38, z38)
        ocpX8.set_value(X8_Z_48, z48)
        ocpX8.set_value(X8_Z_58, z58)
        ocpX8.set_value(X8_Z_68, z68)
        ocpX8.set_value(X8_Z_78, z78)
        ocpX8.set_value(X8_Z_88, z88)

        solX8 = ocpX8.solve()

        # Save the information

        X1p = solX1.sample(X1, grid='control')[1].T
        X2p = solX2.sample(X2, grid='control')[1].T
        X3p = solX3.sample(X3, grid='control')[1].T
        X4p = solX4.sample(X4, grid='control')[1].T
        X5p = solX5.sample(X5, grid='control')[1].T
        X6p = solX6.sample(X6, grid='control')[1].T
        X7p = solX7.sample(X7, grid='control')[1].T
        X8p = solX8.sample(X8, grid='control')[1].T
        
        # Set values and solve for each agent ocpZ

        ocpZ1.set_value(Z1_lambda_11, l11)
        ocpZ1.set_value(Z1_lambda_12, l12)
        ocpZ1.set_value(Z1_lambda_13, l13)
        ocpZ1.set_value(Z1_lambda_14, l14)
        ocpZ1.set_value(Z1_lambda_15, l15)
        ocpZ1.set_value(Z1_lambda_16, l16)
        ocpZ1.set_value(Z1_lambda_17, l17)
        ocpZ1.set_value(Z1_lambda_18, l18)
        ocpZ1.set_value(Z1_X_1, X1p)
        ocpZ1.set_value(Z1_X_2, X2p)
        ocpZ1.set_value(Z1_X_3, X3p)
        ocpZ1.set_value(Z1_X_4, X4p)
        ocpZ1.set_value(Z1_X_5, X5p)
        ocpZ1.set_value(Z1_X_6, X6p)
        ocpZ1.set_value(Z1_X_7, X7p)
        ocpZ1.set_value(Z1_X_8, X8p)

        ocpZ2.set_value(Z2_lambda_21, l21)
        ocpZ2.set_value(Z2_lambda_22, l22)
        ocpZ2.set_value(Z2_lambda_23, l23)
        ocpZ2.set_value(Z2_lambda_24, l24)
        ocpZ2.set_value(Z2_lambda_25, l25)
        ocpZ2.set_value(Z2_lambda_26, l26)
        ocpZ2.set_value(Z2_lambda_27, l27)
        ocpZ2.set_value(Z2_lambda_28, l28)
        ocpZ2.set_value(Z2_X_1, X1p)
        ocpZ2.set_value(Z2_X_2, X2p)
        ocpZ2.set_value(Z2_X_3, X3p)
        ocpZ2.set_value(Z2_X_4, X4p)
        ocpZ2.set_value(Z2_X_5, X5p)
        ocpZ2.set_value(Z2_X_6, X6p)
        ocpZ2.set_value(Z2_X_7, X7p)
        ocpZ2.set_value(Z2_X_8, X8p)

        ocpZ3.set_value(Z3_lambda_31, l31)
        ocpZ3.set_value(Z3_lambda_32, l32)
        ocpZ3.set_value(Z3_lambda_33, l33)
        ocpZ3.set_value(Z3_lambda_34, l34)
        ocpZ3.set_value(Z3_lambda_35, l35)
        ocpZ3.set_value(Z3_lambda_36, l36)
        ocpZ3.set_value(Z3_lambda_37, l37)
        ocpZ3.set_value(Z3_lambda_38, l38)
        ocpZ3.set_value(Z3_X_1, X1p)
        ocpZ3.set_value(Z3_X_2, X2p)
        ocpZ3.set_value(Z3_X_3, X3p)
        ocpZ3.set_value(Z3_X_4, X4p)
        ocpZ3.set_value(Z3_X_5, X5p)
        ocpZ3.set_value(Z3_X_6, X6p)
        ocpZ3.set_value(Z3_X_7, X7p)
        ocpZ3.set_value(Z3_X_8, X8p)

        ocpZ4.set_value(Z4_lambda_41, l41)
        ocpZ4.set_value(Z4_lambda_42, l42)
        ocpZ4.set_value(Z4_lambda_43, l43)
        ocpZ4.set_value(Z4_lambda_44, l44)
        ocpZ4.set_value(Z4_lambda_45, l45)
        ocpZ4.set_value(Z4_lambda_46, l46)
        ocpZ4.set_value(Z4_lambda_47, l47)
        ocpZ4.set_value(Z4_lambda_48, l48)
        ocpZ4.set_value(Z4_X_1, X1p)
        ocpZ4.set_value(Z4_X_2, X2p)
        ocpZ4.set_value(Z4_X_3, X3p)
        ocpZ4.set_value(Z4_X_4, X4p)
        ocpZ4.set_value(Z4_X_5, X5p)
        ocpZ4.set_value(Z4_X_6, X6p)
        ocpZ4.set_value(Z4_X_7, X7p)
        ocpZ4.set_value(Z4_X_8, X8p)

        ocpZ5.set_value(Z5_lambda_51, l51)
        ocpZ5.set_value(Z5_lambda_52, l52)
        ocpZ5.set_value(Z5_lambda_53, l53)
        ocpZ5.set_value(Z5_lambda_54, l54)
        ocpZ5.set_value(Z5_lambda_55, l55)
        ocpZ5.set_value(Z5_lambda_56, l56)
        ocpZ5.set_value(Z5_lambda_57, l57)
        ocpZ5.set_value(Z5_lambda_58, l58)
        ocpZ5.set_value(Z5_X_1, X1p)
        ocpZ5.set_value(Z5_X_2, X2p)
        ocpZ5.set_value(Z5_X_3, X3p)
        ocpZ5.set_value(Z5_X_4, X4p)
        ocpZ5.set_value(Z5_X_5, X5p)
        ocpZ5.set_value(Z5_X_6, X6p)
        ocpZ5.set_value(Z5_X_7, X7p)
        ocpZ5.set_value(Z5_X_8, X8p)

        ocpZ6.set_value(Z6_lambda_61, l61)
        ocpZ6.set_value(Z6_lambda_62, l62)
        ocpZ6.set_value(Z6_lambda_63, l63)
        ocpZ6.set_value(Z6_lambda_64, l64)
        ocpZ6.set_value(Z6_lambda_65, l65)
        ocpZ6.set_value(Z6_lambda_66, l66)
        ocpZ6.set_value(Z6_lambda_67, l67)
        ocpZ6.set_value(Z6_lambda_68, l68)
        ocpZ6.set_value(Z6_X_1, X1p)
        ocpZ6.set_value(Z6_X_2, X2p)
        ocpZ6.set_value(Z6_X_3, X3p)
        ocpZ6.set_value(Z6_X_4, X4p)
        ocpZ6.set_value(Z6_X_5, X5p)
        ocpZ6.set_value(Z6_X_6, X6p)
        ocpZ6.set_value(Z6_X_7, X7p)
        ocpZ6.set_value(Z6_X_8, X8p)

        ocpZ7.set_value(Z7_lambda_71, l71)
        ocpZ7.set_value(Z7_lambda_72, l72)
        ocpZ7.set_value(Z7_lambda_73, l73)
        ocpZ7.set_value(Z7_lambda_74, l74)
        ocpZ7.set_value(Z7_lambda_75, l75)
        ocpZ7.set_value(Z7_lambda_76, l76)
        ocpZ7.set_value(Z7_lambda_77, l77)
        ocpZ7.set_value(Z7_lambda_78, l78)
        ocpZ7.set_value(Z7_X_1, X1p)
        ocpZ7.set_value(Z7_X_2, X2p)
        ocpZ7.set_value(Z7_X_3, X3p)
        ocpZ7.set_value(Z7_X_4, X4p)
        ocpZ7.set_value(Z7_X_5, X5p)
        ocpZ7.set_value(Z7_X_6, X6p)
        ocpZ7.set_value(Z7_X_7, X7p)
        ocpZ7.set_value(Z7_X_8, X8p)

        ocpZ8.set_value(Z8_lambda_81, l81)
        ocpZ8.set_value(Z8_lambda_82, l82)
        ocpZ8.set_value(Z8_lambda_83, l83)
        ocpZ8.set_value(Z8_lambda_84, l84)
        ocpZ8.set_value(Z8_lambda_85, l85)
        ocpZ8.set_value(Z8_lambda_86, l86)
        ocpZ8.set_value(Z8_lambda_87, l87)
        ocpZ8.set_value(Z8_lambda_88, l88)
        ocpZ8.set_value(Z8_X_1, X1p)
        ocpZ8.set_value(Z8_X_2, X2p)
        ocpZ8.set_value(Z8_X_3, X3p)
        ocpZ8.set_value(Z8_X_4, X4p)
        ocpZ8.set_value(Z8_X_5, X5p)
        ocpZ8.set_value(Z8_X_6, X6p)
        ocpZ8.set_value(Z8_X_7, X7p)
        ocpZ8.set_value(Z8_X_8, X8p)

        solZ1 = ocpZ1.solve()
        solZ2 = ocpZ2.solve()
        solZ3 = ocpZ3.solve()
        solZ4 = ocpZ4.solve()
        solZ5 = ocpZ5.solve()
        solZ6 = ocpZ6.solve()
        solZ7 = ocpZ7.solve()
        solZ8 = ocpZ8.solve()

        # Compute new Z parameters

        z11 = solZ1.sample(Z1_Z_11, grid='control')[1].T
        z12 = solZ1.sample(Z1_Z_12, grid='control')[1].T
        z13 = solZ1.sample(Z1_Z_13, grid='control')[1].T
        z14 = solZ1.sample(Z1_Z_14, grid='control')[1].T
        z15 = solZ1.sample(Z1_Z_15, grid='control')[1].T
        z16 = solZ1.sample(Z1_Z_16, grid='control')[1].T
        z17 = solZ1.sample(Z1_Z_17, grid='control')[1].T
        z18 = solZ1.sample(Z1_Z_18, grid='control')[1].T

        z21 = solZ2.sample(Z2_Z_21, grid='control')[1].T
        z22 = solZ2.sample(Z2_Z_22, grid='control')[1].T
        z23 = solZ2.sample(Z2_Z_23, grid='control')[1].T
        z24 = solZ2.sample(Z2_Z_24, grid='control')[1].T
        z25 = solZ2.sample(Z2_Z_25, grid='control')[1].T
        z26 = solZ2.sample(Z2_Z_26, grid='control')[1].T
        z27 = solZ2.sample(Z2_Z_27, grid='control')[1].T
        z28 = solZ2.sample(Z2_Z_28, grid='control')[1].T

        z31 = solZ3.sample(Z3_Z_31, grid='control')[1].T
        z32 = solZ3.sample(Z3_Z_32, grid='control')[1].T
        z33 = solZ3.sample(Z3_Z_33, grid='control')[1].T
        z34 = solZ3.sample(Z3_Z_34, grid='control')[1].T
        z35 = solZ3.sample(Z3_Z_35, grid='control')[1].T
        z38 = solZ3.sample(Z3_Z_38, grid='control')[1].T
        z36 = solZ3.sample(Z3_Z_36, grid='control')[1].T
        z37 = solZ3.sample(Z3_Z_37, grid='control')[1].T

        z41 = solZ4.sample(Z4_Z_41, grid='control')[1].T
        z42 = solZ4.sample(Z4_Z_42, grid='control')[1].T
        z43 = solZ4.sample(Z4_Z_43, grid='control')[1].T
        z44 = solZ4.sample(Z4_Z_44, grid='control')[1].T
        z45 = solZ4.sample(Z4_Z_45, grid='control')[1].T
        z46 = solZ4.sample(Z4_Z_46, grid='control')[1].T
        z47 = solZ4.sample(Z4_Z_47, grid='control')[1].T
        z48 = solZ4.sample(Z4_Z_48, grid='control')[1].T

        z51 = solZ5.sample(Z5_Z_51, grid='control')[1].T
        z52 = solZ5.sample(Z5_Z_52, grid='control')[1].T
        z53 = solZ5.sample(Z5_Z_53, grid='control')[1].T
        z54 = solZ5.sample(Z5_Z_54, grid='control')[1].T
        z55 = solZ5.sample(Z5_Z_55, grid='control')[1].T
        z56 = solZ5.sample(Z5_Z_56, grid='control')[1].T
        z57 = solZ5.sample(Z5_Z_57, grid='control')[1].T
        z58 = solZ5.sample(Z5_Z_58, grid='control')[1].T

        z61 = solZ6.sample(Z6_Z_61, grid='control')[1].T
        z62 = solZ6.sample(Z6_Z_62, grid='control')[1].T
        z63 = solZ6.sample(Z6_Z_63, grid='control')[1].T
        z64 = solZ6.sample(Z6_Z_64, grid='control')[1].T
        z65 = solZ6.sample(Z6_Z_65, grid='control')[1].T
        z66 = solZ6.sample(Z6_Z_66, grid='control')[1].T
        z67 = solZ6.sample(Z6_Z_67, grid='control')[1].T
        z68 = solZ6.sample(Z6_Z_68, grid='control')[1].T

        z71 = solZ7.sample(Z7_Z_71, grid='control')[1].T
        z72 = solZ7.sample(Z7_Z_72, grid='control')[1].T
        z73 = solZ7.sample(Z7_Z_73, grid='control')[1].T
        z74 = solZ7.sample(Z7_Z_74, grid='control')[1].T
        z75 = solZ7.sample(Z7_Z_75, grid='control')[1].T
        z76 = solZ7.sample(Z7_Z_76, grid='control')[1].T
        z77 = solZ7.sample(Z7_Z_77, grid='control')[1].T
        z78 = solZ7.sample(Z7_Z_78, grid='control')[1].T

        z81 = solZ8.sample(Z8_Z_81, grid='control')[1].T
        z82 = solZ8.sample(Z8_Z_82, grid='control')[1].T
        z83 = solZ8.sample(Z8_Z_83, grid='control')[1].T
        z84 = solZ8.sample(Z8_Z_84, grid='control')[1].T
        z85 = solZ8.sample(Z8_Z_85, grid='control')[1].T
        z86 = solZ8.sample(Z8_Z_86, grid='control')[1].T
        z87 = solZ8.sample(Z8_Z_87, grid='control')[1].T
        z88 = solZ8.sample(Z8_Z_88, grid='control')[1].T

        # Update lambda multipliers

        l11 = l11 + mu*(z11 - X1p)
        l12 = l12 + mu*(z12 - X2p)
        l13 = l13 + mu*(z13 - X3p)
        l14 = l14 + mu*(z14 - X4p)
        l15 = l15 + mu*(z15 - X5p)
        l16 = l16 + mu*(z16 - X6p)
        l17 = l17 + mu*(z17 - X7p)
        l18 = l18 + mu*(z18 - X8p)

        l21 = l21 + mu*(z21 - X1p)
        l22 = l22 + mu*(z22 - X2p)
        l23 = l23 + mu*(z23 - X3p)
        l24 = l24 + mu*(z24 - X4p)
        l25 = l25 + mu*(z25 - X5p)
        l26 = l26 + mu*(z26 - X6p)
        l27 = l27 + mu*(z27 - X7p)
        l28 = l28 + mu*(z28 - X8p)

        l31 = l31 + mu*(z31 - X1p)
        l32 = l32 + mu*(z32 - X2p)
        l33 = l33 + mu*(z33 - X3p)
        l34 = l34 + mu*(z34 - X4p)
        l35 = l35 + mu*(z35 - X5p)
        l36 = l36 + mu*(z36 - X6p)
        l37 = l37 + mu*(z37 - X7p)
        l38 = l38 + mu*(z38 - X8p)

        l41 = l41 + mu*(z41 - X1p)
        l42 = l42 + mu*(z42 - X2p)
        l43 = l43 + mu*(z43 - X3p)
        l44 = l44 + mu*(z44 - X4p)
        l45 = l45 + mu*(z45 - X5p)
        l46 = l46 + mu*(z46 - X6p)
        l47 = l47 + mu*(z47 - X7p)
        l48 = l48 + mu*(z48 - X8p)

        l51 = l51 + mu*(z51 - X1p)
        l52 = l52 + mu*(z52 - X2p)
        l53 = l53 + mu*(z53 - X3p)
        l54 = l54 + mu*(z54 - X4p)
        l55 = l55 + mu*(z55 - X5p)
        l56 = l56 + mu*(z56 - X6p)
        l57 = l57 + mu*(z57 - X7p)
        l58 = l58 + mu*(z58 - X8p)

        l61 = l61 + mu*(z61 - X1p)
        l62 = l62 + mu*(z62 - X2p)
        l63 = l63 + mu*(z63 - X3p)
        l64 = l64 + mu*(z64 - X4p)
        l65 = l65 + mu*(z65 - X5p)
        l66 = l66 + mu*(z66 - X6p)
        l67 = l67 + mu*(z67 - X7p)
        l68 = l68 + mu*(z68 - X8p)

        l71 = l71 + mu*(z71 - X1p)
        l72 = l72 + mu*(z72 - X2p)
        l73 = l73 + mu*(z73 - X3p)
        l74 = l74 + mu*(z74 - X4p)
        l75 = l75 + mu*(z75 - X5p)
        l76 = l76 + mu*(z76 - X6p)
        l77 = l77 + mu*(z77 - X7p)
        l78 = l78 + mu*(z78 - X8p)

        l81 = l81 + mu*(z81 - X1p)
        l82 = l82 + mu*(z82 - X2p)
        l83 = l83 + mu*(z83 - X3p)
        l84 = l84 + mu*(z84 - X4p)
        l85 = l85 + mu*(z85 - X5p)
        l86 = l86 + mu*(z86 - X6p)
        l87 = l87 + mu*(z87 - X7p)
        l88 = l88 + mu*(z88 - X8p)
        
    print("timestep", j+1, "of", Nsim)


fig1, ax1 = plt.subplots()
ax1.plot(y1_history, x1_history, 'bo')
ax1.plot(y2_history, x2_history, 'r-')
ax1.plot(y3_history, x3_history, 'g--')
ax1.plot(y4_history, x4_history, 'c.-')
ax1.plot(y5_history, x5_history, 'y.')
ax1.plot(y6_history, x6_history, 'mo')
ax1.plot(y7_history, x7_history, 'kv')
ax1.plot(y8_history, x8_history, 'bs')
ax1.set_xlabel('Y [m]')
ax1.set_ylabel('X [m]')
plt.axis(equal=True)

plt.show()

np.savez('mpc_decentralized_arrays', x1=x1_history, y1=y1_history, x2=x2_history, y2=y2_history, 
         x3=x3_history, y3=y3_history, x4=x4_history, y4=y4_history,
         x5=x5_history, y5=y5_history, x6=x6_history, y6=y6_history,
         x7=x7_history, y7=y7_history, x8=x8_history, y8=y8_history)