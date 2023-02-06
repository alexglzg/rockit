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
from scipy.optimize import minimize

# -------------------------------
# Problem parameters
# -------------------------------
T1 = 1.0

nx    = 7                   # the system is composed of 5 states
nu    = 1                   # the system has 1 input
Tf    = 1                   # control horizon [s]
Nhor  = 20                  # number of control intervals
dt    = Tf/Nhor             # sample time

starting_angle = 0.0
ned_x = 0.0
ned_y = 0.0
u_ref = 0.5

x_multiplier = 0.2
y_amplitude = 1.0
y_freq = 3*np.pi/40
'''def path(s):
    return ((ned_x-x_multiplier*s)**2 + (ned_y-y_amplitude*np.sin(s*y_freq))**2)
res = minimize(path, 0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})'''

def desired_x(s_var):
    return x_multiplier*s_var

def desired_y(s_var):
    return y_amplitude*np.sin(s_var*y_freq)

def path_w_args(s_var, xpos, ypos):
    return ((xpos-x_multiplier*s_var)**2 + (ypos-y_amplitude*np.sin(s_var*y_freq))**2)
s_0 = minimize(path_w_args, 0, method='nelder-mead', args=(ned_x, ned_y), options={'xatol': 1e-8, 'disp': True})
s_0 = s_0.x

current_X = vertcat(ned_x,ned_y,starting_angle,u_ref,0,0,s_0)  # initial state

Nsim  = int(40 * Nhor / Tf)#200                 # how much samples to simulate

# -------------------------------
# Logging variables
# -------------------------------
#ye_history     = np.zeros(Nsim+1)
r_history   = np.zeros(Nsim+1)
x_history     = np.zeros(Nsim+1)
y_history   = np.zeros(Nsim+1)
xd_history     = np.zeros(Nsim+1)
yd_history   = np.zeros(Nsim+1)

# -------------------------------
# Set OCP
# -------------------------------
ocp = Ocp(T=Tf)
# Define states
nedx = ocp.state()
nedy = ocp.state()
psi = ocp.state()
u = ocp.state()
v = ocp.state()
r = ocp.state()
s = ocp.state()

# Defince controls
Urdot = ocp.control()
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
x_d = x_multiplier*s
y_d = y_amplitude*np.sin(s*y_freq)
xdot_d = x_multiplier
ydot_d = y_amplitude*y_freq*cos((y_freq) * s)
gamma_p = atan2(ydot_d, xdot_d)
ye = -(nedx-x_d)*sin(gamma_p)+(nedy-y_d)*cos(gamma_p)
distance1 = sqrt((nedx-obs_pos[0])*(nedx-obs_pos[0]) + (nedy-obs_pos[1])*(nedy-obs_pos[1]))
distance2 = sqrt((nedx-obs_pos[2])*(nedx-obs_pos[2]) + (nedy-obs_pos[3])*(nedy-obs_pos[3]))
distance3 = sqrt((nedx-obs_pos[4])*(nedx-obs_pos[4]) + (nedy-obs_pos[5])*(nedy-obs_pos[5]))
distance4 = sqrt((nedx-obs_pos[6])*(nedx-obs_pos[6]) + (nedy-obs_pos[7])*(nedy-obs_pos[7]))
distance5 = sqrt((nedx-obs_pos[8])*(nedx-obs_pos[8]) + (nedy-obs_pos[9])*(nedy-obs_pos[9]))
distance6 = sqrt((nedx-obs_pos[10])*(nedx-obs_pos[10]) + (nedy-obs_pos[11])*(nedy-obs_pos[11]))
distance7 = sqrt((nedx-obs_pos[12])*(nedx-obs_pos[12]) + (nedy-obs_pos[13])*(nedy-obs_pos[13]))
distance8 = sqrt((nedx-obs_pos[14])*(nedx-obs_pos[14]) + (nedy-obs_pos[15])*(nedy-obs_pos[15]))
ocp.set_der(nedx, (u*cos(psi) - v*sin(psi)))
ocp.set_der(nedy, (u*sin(psi) + v*cos(psi)))
ocp.set_der(psi, r)
ocp.set_der(u, 0)
ocp.set_der(v, 0)
ocp.set_der(r, Urdot)
ocp.set_der(s, u)

danger_zone = 0.2
Qye = 5.0
Qr = 0.1
R = 0.005
QNye = 5.0
QNr = 0.1
zul = 50.0

# Lagrange objective
ocp.add_objective(ocp.integral(Qye*(ye**2) + Qr*(r**2) + R*(Urdot**2) + zul*slack1l + zul*slack2l 
                                + zul*slack3l + zul*slack4l + zul*slack5l + zul*slack6l 
                                + zul*slack7l + zul*slack8l ))
ocp.add_objective(ocp.at_tf(QNye*(ye**2) + QNr*(r**2)))

# Path constraints
'''r_max = 10.0
ocp.subject_to( (-r_max <= r) <= r_max )'''
ocp.subject_to( s >= 0)
ocp.subject_to( obs_rad[0] <= (distance1 + slack1l) )
ocp.subject_to( obs_rad[1] <= (distance2 + slack2l) )
ocp.subject_to( obs_rad[2] <= (distance3 + slack3l) )
ocp.subject_to( obs_rad[3] <= (distance4 + slack4l) )
ocp.subject_to( obs_rad[4] <= (distance5 + slack5l) )
ocp.subject_to( obs_rad[5] <= (distance6 + slack6l) )
ocp.subject_to( obs_rad[6] <= (distance7 + slack7l) )
ocp.subject_to( obs_rad[7] <= (distance8 + slack8l) )
ocp.subject_to( ( 0 >= slack1l) >= -danger_zone )
ocp.subject_to( ( 0 >= slack2l) >= -danger_zone )
ocp.subject_to( ( 0 >= slack3l) >= -danger_zone )
ocp.subject_to( ( 0 >= slack4l) >= -danger_zone )
ocp.subject_to( ( 0 >= slack5l) >= -danger_zone )
ocp.subject_to( ( 0 >= slack6l) >= -danger_zone )
ocp.subject_to( ( 0 >= slack7l) >= -danger_zone )
ocp.subject_to( ( 0 >= slack8l) >= -danger_zone )


# Initial constraints
X = vertcat(nedx,nedy,psi,u,v,r,s)
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
ocp.set_value(X_0, current_X)
obstacle_radius = 0.3
obstacles = vertcat(4,-1,6.65,1,8,0,10,-0.4,100,100,100,100,100,100,100,100)
radius = vertcat(obstacle_radius, obstacle_radius, obstacle_radius, obstacle_radius, 0, 0, 0, 0)
ocp.set_value(obs_pos, obstacles)
ocp.set_value(obs_rad, radius)

# Solve
sol = ocp.solve()

# Get discretisd dynamics as CasADi function
Sim_asv_dyn = ocp._method.discrete_system(ocp)

# Log data for post-processing
r_history[0]   = current_X[5]
x_history[0] = current_X[0]
y_history[0] = current_X[1]
xd_history[0]   = desired_x(s_0)
yd_history[0] = desired_y(s_0)

# -------------------------------
# Simulate the MPC solving the OCP (with the updated state) several times
# -------------------------------

for i in range(Nsim):
    print("timestep", i+1, "of", Nsim)
    # Get the solution from sol
    tsa, Usol = sol.sample(Urdot, grid='control')
    _, S1lsol = sol.sample(slack1l, grid='control')
    _, S2lsol = sol.sample(slack2l, grid='control')
    _, S3lsol = sol.sample(slack3l, grid='control')
    _, S4lsol = sol.sample(slack4l, grid='control')
    _, S5lsol = sol.sample(slack5l, grid='control')
    _, S6lsol = sol.sample(slack6l, grid='control')
    _, S7lsol = sol.sample(slack7l, grid='control')
    _, S8lsol = sol.sample(slack8l, grid='control')
    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_asv_dyn(x0=current_X, u=vertcat(Usol[0],S1lsol[0],S2lsol[0],S3lsol[0],S4lsol[0],S5lsol[0],S6lsol[0],S7lsol[0],S8lsol[0]), T=dt)["xf"]
    # Compute new starting s0
    s_0 = minimize(path_w_args, 0, method='nelder-mead', args=(current_X[0], current_X[1]), options={'xatol': 1e-8, 'disp': True})
    s_0 = s_0.x
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, vertcat(current_X[:6],s_0))
    #ocp.set_value(X_0, current_X[:7])
    # Solve the optimization problem
    sol = ocp.solve()
    ocp._method.opti.set_initial(ocp._method.opti.x, ocp._method.opti.value(ocp._method.opti.x))

    # Log data for post-processing
    x_history[i+1]   = current_X[0].full()
    y_history[i+1] = current_X[1].full()
    r_history[i+1] = current_X[5].full()
    xd_history[i+1]   = desired_x(s_0)
    yd_history[i+1] = desired_y(s_0)
    #xd_history[i+1]   = desired_x(current_X[6])
    #yd_history[i+1] = desired_y(current_X[6])
# -------------------------------
# Plot the results
# -------------------------------
time_sim = np.linspace(0, dt*Nsim, Nsim+1)

fig, ax1 = plt.subplots()
ax1.plot(time_sim, r_history, 'b-')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Angular velocity [rad/s]')
fig.tight_layout()

fig2, ax3 = plt.subplots()
ax3.plot(yd_history, xd_history, 'r-')
ax3.plot(y_history, x_history, 'b--')
ax3.set_xlabel('Y [m]')
ax3.set_ylabel('X [m]')
obstacle_array = np.array([4,-1,6.65,1,8,0,10,-0.4])
for j in range(4):
    c = plt.Circle((obstacle_array[2*j+1],obstacle_array[2*j]),obstacle_radius)
    ax3.add_patch(c)

plt.show()