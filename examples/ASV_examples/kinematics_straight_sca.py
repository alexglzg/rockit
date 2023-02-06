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
Model Predictive Control example
================================

"""

from rockit import *
from casadi import *

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Problem parameters
# -------------------------------
T1 = 1.0

nx    = 8                   # the system is composed of 5 states
nu    = 1                   # the system has 1 input
Tf    = 5                   # control horizon [s]
Nhor  = 100                  # number of control intervals
dt    = Tf/Nhor             # sample time

starting_angle = 0.0
x_1 = 4.0
y_1 = -5.0
x2 = 4.0
y2 = 25.0
a_k = np.math.atan2(y2-y_1, x2-x_1)
ned_x = 0.0
ned_y = 0.0
y_e = -(ned_x-x_1)*np.sin(a_k)+(ned_y-y_1)*np.cos(a_k)
psi_e = starting_angle - a_k
u_ref = 0.7
current_X = vertcat(u_ref,0,y_e,psi_e,psi_e, ned_x, ned_y, starting_angle)  # initial state

Nsim  = int(30 * Nhor / Tf)                 # how much samples to simulate

# -------------------------------
# Logging variables
# -------------------------------
ye_history     = np.zeros(Nsim+1)
chie_history   = np.zeros(Nsim+1)
psied_history  = np.zeros(Nsim+1)
x_history     = np.zeros(Nsim+1)
y_history   = np.zeros(Nsim+1)

# -------------------------------
# Set OCP
# -------------------------------
ocp = Ocp(T=Tf)

# Define states
u = ocp.state()
v = ocp.state()
ye = ocp.state()
chie = ocp.state()
psied = ocp.state()
nedx = ocp.state()
nedy = ocp.state()
psi = ocp.state()

# Defince controls
Upsieddot = ocp.control()
slack1u = ocp.control()
slack2u = ocp.control()
slack3u = ocp.control()
slack4u = ocp.control()
slack5u = ocp.control()
slack6u = ocp.control()
slack7u = ocp.control()
slack8u = ocp.control()
slack1l = ocp.control()
slack2l = ocp.control()
slack3l = ocp.control()
slack4l = ocp.control()
slack5l = ocp.control()
slack6l = ocp.control()
slack7l = ocp.control()
slack8l = ocp.control()

# Define parameter
X_0 = ocp.parameter(nx)
obs_pos = ocp.parameter(16)
obs_rad = ocp.parameter(8)

# Specify ODE
beta = atan2(v,u+.001)
psie = chie + beta
distance1 = sqrt((nedx-obs_pos[0])*(nedx-obs_pos[0]) + (nedy-obs_pos[1])*(nedy-obs_pos[1]))
distance2 = sqrt((nedx-obs_pos[2])*(nedx-obs_pos[2]) + (nedy-obs_pos[3])*(nedy-obs_pos[3]))
distance3 = sqrt((nedx-obs_pos[4])*(nedx-obs_pos[4]) + (nedy-obs_pos[5])*(nedy-obs_pos[5]))
distance4 = sqrt((nedx-obs_pos[6])*(nedx-obs_pos[6]) + (nedy-obs_pos[7])*(nedy-obs_pos[7]))
distance5 = sqrt((nedx-obs_pos[8])*(nedx-obs_pos[8]) + (nedy-obs_pos[9])*(nedy-obs_pos[9]))
distance6 = sqrt((nedx-obs_pos[10])*(nedx-obs_pos[10]) + (nedy-obs_pos[11])*(nedy-obs_pos[11]))
distance7 = sqrt((nedx-obs_pos[12])*(nedx-obs_pos[12]) + (nedy-obs_pos[13])*(nedy-obs_pos[13]))
distance8 = sqrt((nedx-obs_pos[14])*(nedx-obs_pos[14]) + (nedy-obs_pos[15])*(nedy-obs_pos[15]))
ocp.set_der(u, 0)
ocp.set_der(v, 0)
ocp.set_der(ye, (u*sin(psie) + v*cos(psie)))
ocp.set_der(chie, ((psied-psie)/T1))
ocp.set_der(psied, Upsieddot)
ocp.set_der(nedx, (u*cos(psi) - v*sin(psi)))
ocp.set_der(nedy, (u*sin(psi) + v*cos(psi)))
ocp.set_der(psi, ((psied-psie)/T1))

danger_zone = 0.2
Qye = 0.025
Qchie = 0.005
R = 0.1
QNye = 0.05
QNchie = 0.025
zul = 5

# Lagrange objective
ocp.add_objective(ocp.integral(Qye*(ye**2) + Qchie*(chie**2) + R*(Upsieddot**2) + zul*slack1u + zul*slack1l + zul*slack2u + zul*slack2l 
                                + zul*slack3u + zul*slack3l + zul*slack4u + zul*slack4l + zul*slack5u + zul*slack5l + zul*slack6u + zul*slack6l 
                                + zul*slack7u + zul*slack7l + zul*slack8u + zul*slack8l ))
ocp.add_objective(ocp.at_tf(QNye*(ye**2) + QNchie*(chie**2)))

# Path constraints
psieddot_max = 0.5
ocp.subject_to( (-psieddot_max <= Upsieddot) <= psieddot_max )
ocp.subject_to( obs_rad[0] <= (distance1 + slack1l) )
ocp.subject_to( obs_rad[1] <= (distance2 + slack2l) )
ocp.subject_to( obs_rad[2] <= (distance3 + slack3l) )
ocp.subject_to( obs_rad[3] <= (distance4 + slack4l) )
ocp.subject_to( obs_rad[4] <= (distance5 + slack5l) )
ocp.subject_to( obs_rad[5] <= (distance6 + slack6l) )
ocp.subject_to( obs_rad[6] <= (distance7 + slack7l) )
ocp.subject_to( obs_rad[7] <= (distance8 + slack8l) )
ocp.subject_to( (distance1 - slack1u) <=  1000000 )
ocp.subject_to( (distance2 - slack2u) <=  1000000 )
ocp.subject_to( (distance3 - slack3u) <=  1000000 )
ocp.subject_to( (distance4 - slack4u) <=  1000000 )
ocp.subject_to( (distance5 - slack5u) <=  1000000 )
ocp.subject_to( (distance6 - slack6u) <=  1000000 )
ocp.subject_to( (distance7 - slack7u) <=  1000000 )
ocp.subject_to( (distance8 - slack8u) <=  1000000 )
ocp.subject_to( ( 0 >= slack1l) >= -danger_zone )
ocp.subject_to( ( 0 >= slack2l) >= -danger_zone )
ocp.subject_to( ( 0 >= slack3l) >= -danger_zone )
ocp.subject_to( ( 0 >= slack4l) >= -danger_zone )
ocp.subject_to( ( 0 >= slack5l) >= -danger_zone )
ocp.subject_to( ( 0 >= slack6l) >= -danger_zone )
ocp.subject_to( ( 0 >= slack7l) >= -danger_zone )
ocp.subject_to( ( 0 >= slack8l) >= -danger_zone )
ocp.subject_to( slack1u >= 0 )
ocp.subject_to( slack2u >= 0 )
ocp.subject_to( slack3u >= 0 )
ocp.subject_to( slack4u >= 0 )
ocp.subject_to( slack5u >= 0 )
ocp.subject_to( slack6u >= 0 )
ocp.subject_to( slack7u >= 0 )
ocp.subject_to( slack8u >= 0 )

# Initial constraints
X = vertcat(u,v,ye,chie,psied,nedx,nedy,psi)
ocp.subject_to(ocp.at_t0(X)==X_0)

# Pick a solution method
options = {"ipopt": {"print_level": 0}}
options["expand"] = True
options["print_time"] = False
ocp.solver('ipopt',options)

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

# -------------------------------
# Solve the OCP wrt a parameter value (for the first time)
# -------------------------------
# Set initial value for parameters
obstacle_radius = 1.0
ocp.set_value(X_0, current_X)
obstacles = vertcat(4,6,6,8,2,8,4,10,100,100,100,100,100,100,100,100)
radius = vertcat(obstacle_radius, obstacle_radius, obstacle_radius, obstacle_radius, 0, 0, 0, 0)
ocp.set_value(obs_pos, obstacles)
ocp.set_value(obs_rad, radius)
#ocp.set_value(ak, a_k)
# Solve
sol = ocp.solve()

# Get discretisd dynamics as CasADi function
Sim_asv_dyn = ocp._method.discrete_system(ocp)

# Log data for post-processing
ye_history[0]   = current_X[2]
chie_history[0] = current_X[3]
psied_history[0] = current_X[4]
x_history[0]   = current_X[5]
y_history[0] = current_X[6]

# -------------------------------
# Simulate the MPC solving the OCP (with the updated state) several times
# -------------------------------

for i in range(Nsim):
    print("timestep", i+1, "of", Nsim)
    # Get the solution from sol
    #tsa, Fsol = sol.sample(F, grid='control')
    tsa, Usol = sol.sample(Upsieddot, grid='control')
    _, S1usol = sol.sample(slack1u, grid='control')
    _, S2usol = sol.sample(slack2u, grid='control')
    _, S3usol = sol.sample(slack3u, grid='control')
    _, S4usol = sol.sample(slack4u, grid='control')
    _, S5usol = sol.sample(slack5u, grid='control')
    _, S6usol = sol.sample(slack6u, grid='control')
    _, S7usol = sol.sample(slack7u, grid='control')
    _, S8usol = sol.sample(slack8u, grid='control')
    _, S1lsol = sol.sample(slack1l, grid='control')
    _, S2lsol = sol.sample(slack2l, grid='control')
    _, S3lsol = sol.sample(slack3l, grid='control')
    _, S4lsol = sol.sample(slack4l, grid='control')
    _, S5lsol = sol.sample(slack5l, grid='control')
    _, S6lsol = sol.sample(slack6l, grid='control')
    _, S7lsol = sol.sample(slack7l, grid='control')
    _, S8lsol = sol.sample(slack8l, grid='control')
    # Simulate dynamics (applying the first control input) and update the current state
    #current_X = Sim_pendulum_dyn(x0=current_X, u=Fsol[0], T=dt)["xf"]
    current_X = Sim_asv_dyn(x0=current_X, u=vertcat(Usol[0],S1usol[0],S2usol[0],S3usol[0],S4usol[0],S5usol[0],S6usol[0],S7usol[0],S8usol[0],
                                                    S1lsol[0],S2lsol[0],S3lsol[0],S4lsol[0],S5lsol[0],S6lsol[0],S7lsol[0],S8lsol[0]), T=dt)["xf"]
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X[:8])
    # Solve the optimization problem
    sol = ocp.solve()
    ocp._method.opti.set_initial(ocp._method.opti.x, ocp._method.opti.value(ocp._method.opti.x))

    # Log data for post-processing
    ye_history[i+1]   = current_X[2].full()
    chie_history[i+1] = current_X[3].full()
    psied_history[i+1] = current_X[4].full()
    x_history[i+1]   = current_X[5].full()
    y_history[i+1] = current_X[6].full()

# -------------------------------
# Plot the results
# -------------------------------
time_sim = np.linspace(0, dt*Nsim, Nsim+1)

fig, ax1 = plt.subplots()
ax1.plot(time_sim, ye_history, 'b-')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Cross-track error [m]')
fig.tight_layout()

fig3, ax4 = plt.subplots()
ax4.plot(time_sim, chie_history, 'b-')
ax4.plot(time_sim, psied_history, 'r--')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Angle [rad]')
fig3.tight_layout()

fig2, ax3 = plt.subplots()
ax3.plot(y_history, x_history, 'r-')
ax3.set_xlabel('Y [m]')
ax3.set_ylabel('X [m]')
obstacle_array = np.array([4,6,6,8,2,8,4,10])
for j in range(4):
    c = plt.Circle((obstacle_array[2*j+1],obstacle_array[2*j]),obstacle_radius)
    ax3.add_patch(c)
plt.show()