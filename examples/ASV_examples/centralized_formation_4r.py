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

nx    = 8                   # the system is composed of 2 states per robot
nu    = 8                   # the system has 2 inputs per robot
Tf    = 2                 # control horizon [s]
Nhor  = 20                  # number of control intervals
dt    = Tf/Nhor             # sample time

current_X = vertcat(x0_1, y0_1, x0_2, y0_2, x0_3, y0_3, x0_4, y0_4)  # initial state

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
# Set OCP
# -------------------------------
ocp = Ocp(T=Tf)

# Define states
x1 = ocp.state()
y1 = ocp.state()
x2 = ocp.state()
y2 = ocp.state()
x3 = ocp.state()
y3 = ocp.state()
x4 = ocp.state()
y4 = ocp.state()

# Defince controls
u1 = ocp.control()
v1 = ocp.control()
u2 = ocp.control()
v2 = ocp.control()
u3 = ocp.control()
v3 = ocp.control()
u4 = ocp.control()
v4 = ocp.control()

# Define parameter
X_0 = ocp.parameter(nx)

# Specify ODE
distance1 = (x1-x2)**2 + (y1-y2)**2
distance2 = (x1-x3)**2 + (y1-y3)**2
distance3 = (x1-x4)**2 + (y1-y4)**2
distance4 = (x2-x3)**2 + (y2-y3)**2
distance5 = (x2-x4)**2 + (y2-y4)**2
distance6 = (x3-x4)**2 + (y3-y4)**2
ocp.set_der(x1, u1)
ocp.set_der(y1, v1)
ocp.set_der(x2, u2)
ocp.set_der(y2, v2)
ocp.set_der(x3, u3)
ocp.set_der(y3, v3)
ocp.set_der(x4, u4)
ocp.set_der(y4, v4)

# Lagrange objective
ocp.add_objective(ocp.integral((xd_1-x1)**2 + (yd_1-y1)**2 + (xd_2-x2)**2 + (yd_2-y2)**2 
                                + (xd_3-x3)**2 + (yd_3-y3)**2 + (xd_4-x4)**2 + (yd_4-y4)**2))
ocp.add_objective(ocp.at_tf((xd_1-x1)**2 + (yd_1-y1)**2 + (xd_2-x2)**2 + (yd_2-y2)**2 
                                + (xd_3-x3)**2 + (yd_3-y3)**2 + (xd_4-x4)**2 + (yd_4-y4)**2))

# Path constraints
ocp.subject_to( (-max_speed_limit <= u1) <= max_speed_limit )
ocp.subject_to( (-max_speed_limit <= v1) <= max_speed_limit )
ocp.subject_to( (-max_speed_limit <= u2) <= max_speed_limit )
ocp.subject_to( (-max_speed_limit <= v2) <= max_speed_limit )
ocp.subject_to( (-max_speed_limit <= u3) <= max_speed_limit )
ocp.subject_to( (-max_speed_limit <= v3) <= max_speed_limit )
ocp.subject_to( (-max_speed_limit <= u4) <= max_speed_limit )
ocp.subject_to( (-max_speed_limit <= v4) <= max_speed_limit )
ocp.subject_to( distance1 >= boat_radius )
ocp.subject_to( distance2 >= boat_radius )
ocp.subject_to( distance3 >= boat_radius )
ocp.subject_to( distance4 >= boat_radius )
ocp.subject_to( distance5 >= boat_radius )
ocp.subject_to( distance6 >= boat_radius )

# Initial constraints
X = vertcat(x1, y1, x2, y2, x3, y3, x4, y4)
ocp.subject_to(ocp.at_t0(X)==X_0)

# Pick a solution method
options = {"ipopt": {"print_level": 1}}
options["expand"] = True
options["print_time"] = False
ocp.solver('ipopt',options)

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

# -------------------------------
# Solve the OCP wrt a parameter value (for the first time)
# -------------------------------
# Set initial value for parameters
ocp.set_value(X_0, current_X)
# Solve
sol = ocp.solve()

# Get discretisd dynamics as CasADi function
#Sim_pendulum_dyn = ocp._method.discrete_system(ocp)
Sim_asv_dyn = ocp._method.discrete_system(ocp)

# Log data for post-processing
x1_history[0]   = current_X[0]
y1_history[0] = current_X[1]
x2_history[0]   = current_X[2]
y2_history[0] = current_X[3]
x3_history[0]   = current_X[4]
y3_history[0] = current_X[5]
x4_history[0]   = current_X[6]
y4_history[0] = current_X[7]

# -------------------------------
# Simulate the MPC solving the OCP (with the updated state) several times
# -------------------------------

for i in range(Nsim):
    print("timestep", i+1, "of", Nsim)
    # Get the solution from sol
    tsa, u1sol = sol.sample(u1, grid='control')
    _, v1sol = sol.sample(v1, grid='control')
    _, u2sol = sol.sample(u2, grid='control')
    _, v2sol = sol.sample(v2, grid='control')
    _, u3sol = sol.sample(u3, grid='control')
    _, v3sol = sol.sample(v3, grid='control')
    _, u4sol = sol.sample(u4, grid='control')
    _, v4sol = sol.sample(v4, grid='control')
    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_asv_dyn(x0=current_X, u=vertcat(u1sol[0],v1sol[0],u2sol[0],v2sol[0],u3sol[0],v3sol[0],u4sol[0],v4sol[0]), T=dt)["xf"]
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X[:8])
    # Solve the optimization problem
    sol = ocp.solve()
    ocp._method.opti.set_initial(ocp._method.opti.x, ocp._method.opti.value(ocp._method.opti.x))

    # Log data for post-processing
    x1_history[i+1]   = current_X[0].full()
    y1_history[i+1] = current_X[1].full()
    x2_history[i+1]   = current_X[2].full()
    y2_history[i+1] = current_X[3].full()
    x3_history[i+1]   = current_X[4].full()
    y3_history[i+1] = current_X[5].full()
    x4_history[i+1]   = current_X[6].full()
    y4_history[i+1] = current_X[7].full()

# -------------------------------
# Plot the results
# -------------------------------
time_sim = np.linspace(0, dt*Nsim, Nsim+1)

fig1, ax1 = plt.subplots()
ax1.plot(y1_history, x1_history, 'bo')
ax1.plot(y2_history, x2_history, 'r-')
ax1.plot(y3_history, x3_history, 'g--')
ax1.plot(y4_history, x4_history, 'c.-')
ax1.set_xlabel('Y [m]')
ax1.set_ylabel('X [m]')
plt.axis(equal=True)


plt.show()