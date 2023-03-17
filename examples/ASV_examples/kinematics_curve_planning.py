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
Tr = 1.0
Tu = 3.0

nx    = 7                   # the system is composed of 5 states
nu    = 2                   # the system has 1 input
Tf    = 1                   # control horizon [s]
Nhor  = 20                  # number of control intervals
dt    = Tf/Nhor             # sample time

starting_angle = 0.0
ned_x = 0.0
ned_y = 0.0
u_ref = 1.0

x_multiplier = 0.5
y_amplitude = 2.5
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

current_X = vertcat(ned_x,ned_y,starting_angle,0,0,0,s_0)  # initial state

Nsim  = int(15 * Nhor / Tf)#200                 # how much samples to simulate

# -------------------------------
# Logging variables
# -------------------------------
#ye_history     = np.zeros(Nsim+1)
r_history   = np.zeros(Nsim+1)
u_history   = np.zeros(Nsim+1)
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
Uudot = ocp.control()

# Define parameter
X_0 = ocp.parameter(nx)

# Specify ODE
x_d = x_multiplier*s
y_d = y_amplitude*np.sin(s*y_freq)
xdot_d = x_multiplier
ydot_d = y_amplitude*y_freq*cos((y_freq) * s)
gamma_p = atan2(ydot_d, xdot_d)
ye = -(nedx-x_d)*sin(gamma_p)+(nedy-y_d)*cos(gamma_p)
ocp.set_der(nedx, (u*cos(psi) - v*sin(psi)))
ocp.set_der(nedy, (u*sin(psi) + v*cos(psi)))
ocp.set_der(psi, r)
ocp.set_der(u, Uudot)
ocp.set_der(v, 0)
ocp.set_der(r, Urdot)
ocp.set_der(s, u)

Qye = 50.0
Qr = 0.025
Qu = 5.0
R = 0.005
#QNye = 50.1
#QNr = 0.05

# Lagrange objective
ocp.add_objective(ocp.integral(Qye*(ye**2) + Qr*(r**2) + Qu*((u-u_ref)**2) + R*(Urdot**2)))
ocp.add_objective(ocp.at_tf(Qye*(ye**2) + Qr*(r**2) + Qu*((u-u_ref)**2)))

# Path constraints
u_max = 1.5
r_max = 5.0
ocp.subject_to( (-u_max <= u) <= u_max )
ocp.subject_to( (-r_max <= r) <= r_max )
ocp.subject_to( s >= 0)

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
# Solve
sol = ocp.solve()

# Get discretisd dynamics as CasADi function
Sim_asv_dyn = ocp._method.discrete_system(ocp)

# Log data for post-processing
r_history[0]   = current_X[5]
u_history[0]   = current_X[3]
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
    tsa, Ursol = sol.sample(Urdot, grid='control')
    _, Uusol = sol.sample(Uudot, grid='control')
    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_asv_dyn(x0=current_X, u=vertcat(Ursol[0],Uusol[0]), T=dt)["xf"]
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
    u_history[i+1] = current_X[3].full()
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
ax1.plot(time_sim, u_history, 'r-')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Surge speed [m/s]', color='r')
ax1.tick_params('y', colors='r')
ax2 = ax1.twinx()
ax2.plot(time_sim, r_history, 'b-')
ax2.set_ylabel('Angular velocity [rad/s]', color='b')
ax2.tick_params('y', colors='b')
fig.tight_layout()

fig2, ax3 = plt.subplots()
ax3.plot(yd_history, xd_history, 'r-')
ax3.plot(y_history, x_history, 'b--')
ax3.set_xlabel('Y [m]')
ax3.set_ylabel('X [m]')

plt.show()
plt.show()