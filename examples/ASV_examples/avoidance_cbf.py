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
m11 = 12.0 
m22 = 16.0 
m33 = 3.0
d11 = 6.0
d22 = 8.0
d33 = 0.6
aa = 0.45
bb = 0.9
max_force_limit = 6

nx    = 7                   # the system is composed of 7 states
nu    = 4                   # the system has 4 inputs
Tf    = .5                   # control horizon [s]
Nhor  = 5                  # number of control intervals
dt    = Tf/Nhor             # sample time

starting_angle = 0.0
ned_x = 0.2
ned_y = 3.2
u_ref = 0.3

x_multiplier = 0.2
#y_amplitude = 1.0
#y_freq = 3*np.pi/40
'''def path(s):
    return ((ned_x-x_multiplier*s)**2 + (ned_y-y_amplitude*np.sin(s*y_freq))**2)
res = minimize(path, 0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})'''

def desired_x(s_var):
    return x_multiplier*s_var

def desired_y(s_var):
    #return y_amplitude*np.sin(s_var*y_freq)
    return (np.sin(x_multiplier*s_var)+3)

def path_w_args(s_var, xpos, ypos):
    #return ((xpos-x_multiplier*s_var)**2 + (ypos-y_amplitude*np.sin(s_var*y_freq))**2)
    return ((xpos-(x_multiplier*s_var))**2 + (ypos-(np.sin(x_multiplier*s_var)+3))**2)
s_0 = minimize(path_w_args, 0, method='nelder-mead', args=(ned_x, ned_y), options={'xatol': 1e-8, 'disp': True})
s_0 = s_0.x

current_X = vertcat(ned_x,ned_y,starting_angle,0,0,0,s_0)  # initial state

Nsim  = int(30 * Nhor / Tf)                 # how much samples to simulate

# -------------------------------
# Logging variables
# -------------------------------
x_history     = np.zeros(Nsim+1)
y_history   = np.zeros(Nsim+1)
yaw_history     = np.zeros(Nsim+1)
u_history     = np.zeros(Nsim+1)
xd_history     = np.zeros(Nsim+1)
yd_history   = np.zeros(Nsim+1)
f1_history       = np.zeros(Nsim)
f2_history       = np.zeros(Nsim)
f3_history       = np.zeros(Nsim)
f4_history       = np.zeros(Nsim)

exec_time = np.zeros(Nsim)

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
u1 = ocp.control()
u2 = ocp.control()
u3 = ocp.control()
u4 = ocp.control()

# Define parameter
X_0 = ocp.register_parameter(MX.sym('X_0', nx))

# Specify ODE
x_d = x_multiplier*s
#y_d = y_amplitude*np.sin(s*y_freq)
y_d = sin(s*x_multiplier)+3
xdot_d = x_multiplier
#ydot_d = y_amplitude*y_freq*cos((y_freq) * s)
ydot_d = x_multiplier*cos(x_multiplier * s)
gamma_p = atan2(ydot_d, xdot_d)
ye = -(nedx-x_d)*sin(gamma_p)+(nedy-y_d)*cos(gamma_p)
ocp.set_der(nedx, (u*cos(psi) - v*sin(psi)))
ocp.set_der(nedy, (u*sin(psi) + v*cos(psi)))
ocp.set_der(psi, r)
ocp.set_der(u, (-d11/m11*u+u1/(m11)+u2/(m11)))
ocp.set_der(v, (-d22/m22*v+u3/(m22)+u4/(m22)))
ocp.set_der(r, (-d33/m33*r+aa/(2*(m33))*u1-aa/(2*(m33))*u2+bb/(2*(m33))*u3-bb/(2*(m33))*u4))
ocp.set_der(s, u)

Qye = 40
Qr = 0.1
Qpsi = 10.0
Qu = 200.0
#R = 0.005
#QNye = 50.1
#QNr = 0.05

# Lagrange objective
ocp.add_objective(ocp.sum(Qye*(ye**2) + Qr*(r**2) + Qpsi*(sin(psi)-sin(gamma_p))**2 + Qpsi*(cos(psi)-cos(gamma_p))**2 + Qu*(u-u_ref)**2 + u1**2 + u2**2 + u3**2 + u4**2))
ocp.add_objective(ocp.at_tf(Qye*(ye**2) + Qr*(r**2) + Qpsi*(sin(psi)-sin(gamma_p))**2 + Qpsi*(cos(psi)-cos(gamma_p))**2 + Qu*(u-u_ref)**2))
#ocp.add_objective(ocp.sum(Qye*((nedx - 5)**2) + Qye*((nedy - 3)**2) + Qr*(r**2) + Qu*(u-u_ref)**2 + u1**2 + u2**2 + u3**2 + u4**2))
#ocp.add_objective(ocp.at_tf(Qye*((nedx - 5)**2) + Qye*((nedy - 3)**2) + Qr*(r**2) + Qu*(u-u_ref)**2))

# Path constraints
ocp.subject_to( (-max_force_limit <= u1) <= max_force_limit )
ocp.subject_to( (-max_force_limit <= u2) <= max_force_limit )
ocp.subject_to( (-max_force_limit <= u3) <= max_force_limit )
ocp.subject_to( (-max_force_limit <= u4) <= max_force_limit )

obstacle_x = 3.0
obstacle_y = 3.0
obstacle_a = 0.6
obstacle_b = 0.4
obstacle_theta = 0.0
x_diff = nedx - obstacle_x
y_diff = nedy - obstacle_y
x_prime = x_diff * cos(obstacle_theta) + y_diff * sin(obstacle_theta)
y_prime = -x_diff * sin(obstacle_theta) + y_diff * cos(obstacle_theta)
bsafe = (x_prime*x_prime)/(obstacle_a*obstacle_a) + (y_prime*y_prime)/(obstacle_b*obstacle_b)

x_dot = (u*cos(psi) - v*sin(psi))
y_dot = (u*sin(psi) + v*cos(psi))
x_prime_dot = x_dot*cos(obstacle_theta) + y_dot*sin(obstacle_theta)
y_prime_dot = -x_dot*sin(obstacle_theta) + y_dot*cos(obstacle_theta)
bsafe_dot = (2/(obstacle_a*obstacle_a))*(x_prime*x_prime_dot) + (2/(obstacle_b*obstacle_b))*(y_prime*y_prime_dot)

#Tu = u1 + u2
#Tv = u3 + u4
u_dot = (-d11/m11*u+u1/(m11)+u2/(m11))
v_dot = (-d22/m22*v+u3/(m22)+u4/(m22))
x_ddot = u_dot*cos(psi) - u*r*sin(psi) - v_dot*sin(psi) - v*r*cos(psi)
y_ddot = u_dot*sin(psi) + u*r*cos(psi) + v_dot*cos(psi) - v*r*sin(psi)
x_prime_ddot = x_ddot*cos(obstacle_theta) + y_ddot*sin(obstacle_theta)
y_prime_ddot = -x_ddot*sin(obstacle_theta) + y_ddot*cos(obstacle_theta)
bsafe_ddot = (2/(obstacle_a*obstacle_a))*(x_prime_dot*x_prime_dot + x_prime*x_prime_ddot) + (2/(obstacle_b*obstacle_b))*(y_prime_dot*y_prime_dot + y_prime*y_prime_ddot)

param_1 = 1.0
param_2 = 1.0
cbf = bsafe_ddot + (param_1+param_2)*bsafe_dot + param_1*param_2*bsafe
ocp.subject_to(cbf >= 0, include_last=False)

#ocp.subject_to( s >= 0)

# Initial constraints
X = vertcat(nedx,nedy,psi,u,v,r,s)
ocp.subject_to(ocp.at_t0(X)==X_0)

# -------------------------------
# Solve the OCP wrt a parameter value (for the first time)
# -------------------------------
# Set initial value for parameters
ocp.set_value(X_0, current_X)

# Pick a solution method
options = {"ipopt": {"print_level": 0}}
options["expand"] = False
options["print_time"] = True
ocp.solver('ipopt',options)

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

# Get discretisd dynamics as CasADi function
Sim_asv_dyn = ocp._method.discrete_system(ocp)

# Solve
sol = ocp.solve()

ocpf = ocp

# fatrop
method = external_method('fatrop',N=Nhor, intg='rk')
ocpf.method(method)    

ocpf._method.set_name("avoidance_cbf")

#ocpf._method.add_sampler('Urdot', Urdot)
ocpf._method.add_sampler('f1', u1)
ocpf._method.add_sampler('f2', u2)
ocpf._method.add_sampler('f3', u3)
ocpf._method.add_sampler('f4', u4)
ocpf._method.add_sampler('ye', ye)
ocpf.set_value(X_0, current_X)
# Solve
sol = ocpf.solve()

print(sol.stats)

# OCP for initial s computation
ocp_s = Ocp(T=1)
x_0 = ocp_s.state()
y_0 = ocp_s.state()
s_min = ocp_s.control()
xy_0 = ocp_s.register_parameter(MX.sym('xy_0', 2))
ocp_s.set_der(x_0, 0)
ocp_s.set_der(y_0, 0)
ocp_s.add_objective(ocp_s.at_t0(path_w_args(s_min, x_0, y_0)))
#ocp_s.subject_to(s_min >= 0)
xy = vertcat(x_0, y_0)
ocp_s.subject_to(ocp_s.at_t0(xy)==xy_0)
ocp_s.set_value(xy_0, [ned_x, ned_y])
# fatrop
method_2 = external_method('fatrop',N=1, intg='rk')
ocp_s.method(method_2)
ocp_s._method.set_name("initial_s")
ocp_s._method.add_sampler('s_min', s_min)
# Solve
sol_s = ocp_s.solve()


# Log data for post-processing
x_history[0]   = current_X[0]
y_history[0] = current_X[1]
yaw_history[0]   = current_X[2]
u_history[0]   = current_X[3]
xd_history[0]   = desired_x(s_0)
yd_history[0] = desired_y(s_0)

# -------------------------------
# Simulate the MPC solving the OCP (with the updated state) several times
# -------------------------------

for i in range(Nsim):
    print("timestep", i+1, "of", Nsim)
    # Get the solution from sol
    tsa, f1sol = sol.sample(u1, grid='control')
    _, f2sol = sol.sample(u2, grid='control')
    _, f3sol = sol.sample(u3, grid='control')
    _, f4sol = sol.sample(u4, grid='control')
    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_asv_dyn(x0=current_X, u=vertcat(f1sol[0],f2sol[0],f3sol[0],f4sol[0]), T=dt)["xf"]
    # Compute new starting s0
    s_0 = minimize(path_w_args, 0, method='nelder-mead', args=(current_X[0], current_X[1]), options={'xatol': 1e-8, 'disp': True})
    s_0 = s_0.x
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, vertcat(current_X[:6],s_0))
    #ocp.set_value(X_0, current_X[:7])
    # Solve the optimization problem
    sol = ocp.solve()

    # Log data for post-processing
    x_history[i+1]   = current_X[0].full()
    y_history[i+1] = current_X[1].full()
    yaw_history[i+1]   = current_X[2].full()
    u_history[i+1]   = current_X[3].full()
    f1_history[i]       = f1sol[0]
    f2_history[i]       = f2sol[0]
    f3_history[i]       = f3sol[0]
    f4_history[i]       = f4sol[0]
    xd_history[i+1]   = desired_x(s_0)
    yd_history[i+1] = desired_y(s_0)
    exec_time[0] = sol.stats

#Ellipse

def tilted_ellipse(h, k, a, b, theta, num_points=100):
    phi = np.linspace(0, 2*np.pi, num_points)
    x_e = h + a * np.cos(phi) * np.cos(theta) - b * np.sin(phi) * np.sin(theta)
    y_e = k + a * np.cos(phi) * np.sin(theta) + b * np.sin(phi) * np.cos(theta)
    return x_e, y_e

# -------------------------------
# Plot the results
# -------------------------------
time_sim = np.linspace(0, dt*Nsim, Nsim+1)
time_sim2 = np.linspace(0, dt*Nsim, Nsim)

x_e, y_e = tilted_ellipse(obstacle_x, obstacle_y, obstacle_a, obstacle_b, obstacle_theta)
fig2, ax3 = plt.subplots()
ax3.plot(yd_history, xd_history, 'r-')
ax3.plot(y_history, x_history, 'b--')
ax3.plot(y_e, x_e)
ax3.set_xlabel('Y [m]')
ax3.set_ylabel('X [m]')

fig3, ax4 = plt.subplots()
ax4.plot(time_sim2, f1_history, 'r-')
ax4.plot(time_sim2, f2_history, 'b-')
ax4.plot(time_sim2, f3_history, 'g-')
ax4.plot(time_sim2, f4_history, 'y-')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('f [N]')
fig3.tight_layout()

fig4, ax6 = plt.subplots()
ax6.plot(time_sim, yaw_history, 'b-')
ax6.set_xlabel('Time [s]')
ax6.set_ylabel('yaw [rad]')
fig4.tight_layout()

fig5, ax7 = plt.subplots()
ax7.plot(time_sim, u_history, 'b-')
ax7.set_xlabel('Time [s]')
ax7.set_ylabel('u [m/s]')
fig5.tight_layout()


plt.show()