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
X_u_dot = -2.25
Y_v_dot = -23.13
Y_r_dot = -1.31
N_v_dot = -16.41
N_r_dot = -2.79
Yvv = -99.99
Yvr = -5.49
Yrv = -5.49
Yrr = -8.8
Nvv = -5.49
Nvr = -8.8
Nrv = -8.8
Nrr = -3.49
m = 30
Iz = 4.1
B = 0.41
c = 1.0

nx    = 9                   # the system is composed of 7 states
nu    = 2                   # the system has 2 inputs
Tf    = 1                   # control horizon [s]
Nhor  = 20                  # number of control intervals
dt    = Tf/Nhor             # sample time

starting_angle = 0.0
ned_x = 0.0
ned_y = 0.0
u_ref = 1.0

x_multiplier = 0.5
y_amplitude = 2.0
y_freq = 3*np.pi/40

def desired_x(s_var):
    return x_multiplier*s_var

def desired_y(s_var):
    return y_amplitude*np.sin(s_var*y_freq)

def path_w_args(s_var, xpos, ypos):
    return ((xpos-desired_x(s_var))**2 + (ypos-desired_y(s_var))**2)
s_0 = minimize(path_w_args, 0, method='nelder-mead', args=(ned_x, ned_y), options={'xatol': 1e-8, 'disp': True})
s_0 = s_0.x

current_X = vertcat(ned_x,ned_y,starting_angle,0,0,0,0,0,s_0)  # initial state

Nsim  = int(40 * Nhor / Tf)                 # how much samples to simulate

# -------------------------------
# Logging variables
# -------------------------------
x_history     = np.zeros(Nsim+1)
y_history   = np.zeros(Nsim+1)
yaw_history     = np.zeros(Nsim+1)
xd_history     = np.zeros(Nsim+1)
yd_history   = np.zeros(Nsim+1)
u_history     = np.zeros(Nsim+1)
r_history   = np.zeros(Nsim+1)
Tport_history       = np.zeros(Nsim+1)
Tstbd_history       = np.zeros(Nsim+1)

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
Tport = ocp.state()
Tstbd = ocp.state()
s = ocp.state()
ocp.set_initial(u,0.001)

# Defince controls
UTportdot = ocp.control()
UTstbddot = ocp.control()

# Define parameter
X_0 = ocp.parameter(nx)

# Specify ODE
x_d = x_multiplier*s
y_d = y_amplitude*np.sin(s*y_freq)
xdot_d = x_multiplier
ydot_d = y_amplitude*y_freq*cos((y_freq) * s)
gamma_p = atan2(ydot_d, xdot_d)
ye = -(nedx-x_d)*sin(gamma_p)+(nedy-y_d)*cos(gamma_p)
Xu = if_else(u > 1.25, 64.55, -25)
Xuu = if_else(u > 1.25, -70.92, 0)
Yv = 0.5*(-40*1000*fabs(v))*(1.1+0.0045*(1.01/0.09)-0.1*(0.27/0.09)+0.016*((0.27/0.09)*(0.27/0.09)))
Nr = (-0.52)*sqrt(u*u + v*v)
Tu = Tport + c * Tstbd
Tr = (Tport - c * Tstbd) * B / 2
beta = atan2(v,u+.001)
chi = psi + beta
ocp.set_der(nedx, (u*cos(psi) - v*sin(psi)))
ocp.set_der(nedy, (u*sin(psi) + v*cos(psi)))
ocp.set_der(psi, r)
ocp.set_der(u, ((Tu - (-m + 2 * Y_v_dot)*v - (Y_r_dot + N_v_dot)*r*r - (-Xu*u - Xuu*fabs(u)*u)) / (m - X_u_dot)))
ocp.set_der(v, ((-(m - X_u_dot)*u*r - (- Yv - Yvv*fabs(v) - Yvr*fabs(r))*v) / (m - Y_v_dot)))
ocp.set_der(r, ((Tr - (-2*Y_v_dot*u*v - (Y_r_dot + N_v_dot)*r*u + X_u_dot*u*r) - (-Nr*r - Nrv*fabs(v)*r - Nrr*fabs(r)*r)) / (Iz - N_r_dot)))
ocp.set_der(Tport, UTportdot)
ocp.set_der(Tstbd, (UTstbddot/c))
ocp.set_der(s, u)

Qye = 100.0
Qr = 0.025
Qpsi = 50.0
Qu = 10.0
R = 0.005
#QNye = 50.1
#QNr = 0.05

# Lagrange objective
ocp.add_objective(ocp.integral(Qye*(ye**2) + Qpsi*(psi-gamma_p)**2 + Qu*(u-u_ref)**2))
ocp.add_objective(ocp.at_tf(Qye*(ye**2) + Qpsi*(psi-gamma_p)**2 + Qu*(u-u_ref)**2))

# Path constraints
ocp.subject_to( (-2.0 <= u) <= 2.0 )
ocp.subject_to( (-30.0 <= Tport) <= 35.0 )
ocp.subject_to( (-30.0 <= Tstbd) <= 35.0 )
ocp.subject_to( (-30.0 <= UTportdot) <= 30.0 )
ocp.subject_to( (-30.0 <= UTstbddot) <= 30.0 )
ocp.subject_to( s >= 0)

# Initial constraints
X = vertcat(nedx,nedy,psi,u,v,r,Tport,Tstbd,s)
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
x_history[0]   = current_X[0]
y_history[0] = current_X[1]
yaw_history[0]   = current_X[2]
xd_history[0]   = desired_x(s_0)
yd_history[0] = desired_y(s_0)
u_history[0]   = current_X[3]
r_history[0] = current_X[5]
Tport_history[0] = current_X[6]
Tstbd_history[0] = current_X[7]

# -------------------------------
# Simulate the MPC solving the OCP (with the updated state) several times
# -------------------------------

for i in range(Nsim):
    print("timestep", i+1, "of", Nsim)
    # Get the solution from sol
    tsa, Tpsol = sol.sample(UTportdot, grid='control')
    _, Tssol = sol.sample(UTstbddot, grid='control')
    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_asv_dyn(x0=current_X, u=vertcat(Tpsol[0],Tssol[0]), T=dt)["xf"]
    # Compute new starting s0
    s_0 = minimize(path_w_args, 0, method='nelder-mead', args=(current_X[0], current_X[1]), options={'xatol': 1e-8, 'disp': True})
    s_0 = s_0.x
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, vertcat(current_X[:8],s_0))
    # Solve the optimization problem
    sol = ocp.solve()
    ocp._method.opti.set_initial(ocp._method.opti.x, ocp._method.opti.value(ocp._method.opti.x))

    # Log data for post-processing
    x_history[i+1]   = current_X[0].full()
    y_history[i+1] = current_X[1].full()
    yaw_history[i+1]   = current_X[2].full()
    u_history[i+1]   = current_X[3].full()
    r_history[i+1] = current_X[5].full()
    Tport_history[i+1] = current_X[6].full()
    Tstbd_history[i+1] = current_X[7].full()
    xd_history[i+1]   = desired_x(s_0)
    yd_history[i+1] = desired_y(s_0)

# -------------------------------
# Plot the results
# -------------------------------
time_sim = np.linspace(0, dt*Nsim, Nsim+1)
time_sim2 = np.linspace(0, dt*Nsim, Nsim)

fig2, ax3 = plt.subplots()
ax3.plot(yd_history, xd_history, 'r-')
ax3.plot(y_history, x_history, 'b--')
ax3.set_xlabel('Y [m]')
ax3.set_ylabel('X [m]')

fig3, ax4 = plt.subplots()
ax4.plot(time_sim, Tport_history, 'r-')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Tport [N]', color='r')
ax4.tick_params('y', colors='r')
ax5 = ax4.twinx()
ax5.plot(time_sim, Tstbd_history, 'b-')
ax5.set_ylabel('Tstbd [N]', color='b')
ax5.tick_params('y', colors='b')
fig3.tight_layout()

fig4, ax6 = plt.subplots()
ax6.plot(time_sim, yaw_history, 'b-')
ax6.set_xlabel('Time [s]')
ax6.set_ylabel('yaw [rad]')
fig4.tight_layout()

fig5, ax7 = plt.subplots()
ax7.plot(time_sim, u_history, 'r-')
ax7.set_xlabel('Time [s]')
ax7.set_ylabel('u [m/s]', color='r')
ax7.tick_params('y', colors='r')
ax8 = ax7.twinx()
ax8.plot(time_sim, r_history, 'b-')
ax8.set_ylabel('r [rad/s]', color='b')
ax8.tick_params('y', colors='b')
fig5.tight_layout()

plt.show()