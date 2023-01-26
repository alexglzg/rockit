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
m11 = 26.34 
m22 = 27.41 
m33 = 1.67
d11 = 6.70
d22 = 13.31
d33 = 0.67
aa = 0.45
bb = 0.9
max_force_limit = 6

Q0 = 20
Q1 = 20
Q2 = 5
Q3 = 10
Q4 = 10
Q5 = 5

nx    = 6                   # the system is composed of 6 states
nu    = 4                   # the system has 4 input
Tf    = 4                 # control horizon [s]
Nhor  = 40                  # number of control intervals
dt    = Tf/Nhor             # sample time

current_X = vertcat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # initial state

Nsim  = int(60 * Nhor / Tf)#200                 # how much samples to simulate

# -------------------------------
# Logging variables
# -------------------------------
x_history     = np.zeros(Nsim+1)
y_history   = np.zeros(Nsim+1)
yaw_history     = np.zeros(Nsim+1)
xd_history     = np.zeros(Nsim+1)
yd_history   = np.zeros(Nsim+1)
yawd_history     = np.zeros(Nsim+1)
f1_history       = np.zeros(Nsim)
f2_history       = np.zeros(Nsim)
f3_history       = np.zeros(Nsim)
f4_history       = np.zeros(Nsim)

# -------------------------------
# Set OCP
# -------------------------------
ocp = Ocp(T=Tf)

# Define states
nedx = ocp.state()
nedy = ocp.state()
psi = ocp.state()
#sinpsi = ocp.state() 
#cospsi = ocp.state() 
u = ocp.state()
v = ocp.state()
r = ocp.state()


# Defince controls
u1 = ocp.control()
u2 = ocp.control()
u3 = ocp.control()
u4 = ocp.control()

# Define parameter
X_0 = ocp.parameter(nx)

# Specify ODE
ocp.set_der(nedx, (u*cos(psi) - v*sin(psi)))
ocp.set_der(nedy, (u*sin(psi) + v*cos(psi)))
ocp.set_der(psi, r)
#ocp.set_der(sinpsi, (cos(chi)*r))
#ocp.set_der(cospsi, (-sin(chi)*r))
ocp.set_der(u, (-d11/m11*u+u1/(m11)+u2/(m11)))
ocp.set_der(v, (-d22/m22*v+u3/(m22)+u4/(m22)))
ocp.set_der(r, (-d33/m33*r+aa/(2*(m33))*u1-aa/(2*(m33))*u2+bb/(2*(m33))*u3-bb/(2*(m33))*u4))

# Define a placeholder for concrete waypoints to be defined on edges of the control grid
trajectory = ocp.parameter(6, grid='control')

# Lagrange objective
ocp.add_objective(ocp.integral(Q0*(nedx-trajectory[0])**2 + Q1*(nedy-trajectory[1])**2 + Q2*(psi-trajectory[2])**2 
                                + Q3*(u-trajectory[3])**2 + Q4*(v-trajectory[4])**2 + Q5*(r-trajectory[5])**2))
ocp.add_objective(ocp.at_tf(Q0*(nedx-trajectory[0])**2 + Q1*(nedy-trajectory[1])**2 + Q2*(psi-trajectory[2])**2 
                                + Q3*(u-trajectory[3])**2 + Q4*(v-trajectory[4])**2 + Q5*(r-trajectory[5])**2))

# Path constraints
ocp.subject_to( (-max_force_limit <= u1) <= max_force_limit )
ocp.subject_to( (-max_force_limit <= u2) <= max_force_limit )
ocp.subject_to( (-max_force_limit <= u3) <= max_force_limit )
ocp.subject_to( (-max_force_limit <= u4) <= max_force_limit )

# Initial constraints
X = vertcat(nedx, nedy, psi, u, v, r)
ocp.subject_to(ocp.at_t0(X)==X_0)
#ocp.subject_to(ocp.at_tf(X)==final_X)

trajectory_N = np.zeros([6,40])
x_multiplier = 0.2
y_amplitude = 1.0
y_freq = 3*pi/40
for i in range(Nhor):
    timer = i * dt
    x_d = x_multiplier * timer
    xdot_d = x_multiplier
    xddot_d = 0
    y_d = y_amplitude*sin((y_freq) * timer)
    ydot_d = y_amplitude*y_freq*cos((y_freq) * timer)
    yddot_d = -y_amplitude*y_freq*y_freq*sin((y_freq) * timer)
    psi_d = atan2(ydot_d,xdot_d)
    r_d = (xdot_d*yddot_d - xddot_d*ydot_d)/(xdot_d*xdot_d + ydot_d*ydot_d)
    u_d = xdot_d*cos(psi_d) + ydot_d*sin(psi_d)
    v_d = -xdot_d*sin(psi_d) + ydot_d*cos(psi_d)
    trajectory_N[0,i] = x_d
    trajectory_N[1,i] = y_d
    trajectory_N[2,i] = psi_d
    trajectory_N[3,i] = u_d
    trajectory_N[4,i] = v_d
    trajectory_N[5,i] = r_d    

ocp.set_value(trajectory, trajectory_N)

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
#Sim_pendulum_dyn = ocp._method.discrete_system(ocp)
Sim_asv_dyn = ocp._method.discrete_system(ocp)

# Log data for post-processing
x_history[0]   = current_X[0]
y_history[0] = current_X[1]
yaw_history[0]   = current_X[2]
xd_history[0]   = trajectory_N[0,0]
yd_history[0] = trajectory_N[1,0]
yawd_history[0]   = trajectory_N[2,0]

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
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X[:6])
    # Set the new trajectory
    for j in range(Nhor):
        timer = ((j+1)*dt) + (i*dt)
        x_d = x_multiplier * timer
        xdot_d = x_multiplier
        xddot_d = 0
        y_d = y_amplitude*sin((y_freq) * timer)
        ydot_d = y_amplitude*y_freq*cos((y_freq) * timer)
        yddot_d = -y_amplitude*y_freq*y_freq*sin((y_freq) * timer)
        psi_d = atan2(ydot_d,xdot_d)
        r_d = (xdot_d*yddot_d - xddot_d*ydot_d)/(xdot_d*xdot_d + ydot_d*ydot_d)
        u_d = xdot_d*cos(psi_d) + ydot_d*sin(psi_d)
        v_d = -xdot_d*sin(psi_d) + ydot_d*cos(psi_d)
        trajectory_N[0,j] = x_d
        trajectory_N[1,j] = y_d
        trajectory_N[2,j] = psi_d
        trajectory_N[3,j] = u_d
        trajectory_N[4,j] = v_d
        trajectory_N[5,j] = r_d
    ocp.set_value(trajectory, trajectory_N)
    # Solve the optimization problem
    sol = ocp.solve()
    ocp._method.opti.set_initial(ocp._method.opti.x, ocp._method.opti.value(ocp._method.opti.x))

    # Log data for post-processing
    x_history[i+1]   = current_X[0].full()
    y_history[i+1] = current_X[1].full()
    yaw_history[i+1]   = current_X[2].full()
    f1_history[i]       = f1sol[0]
    f2_history[i]       = f2sol[0]
    f3_history[i]       = f3sol[0]
    f4_history[i]       = f4sol[0]
    xd_history[i+1]   = trajectory_N[0,0]
    yd_history[i+1] = trajectory_N[1,0]
    yawd_history[i+1]   = trajectory_N[2,0]

# -------------------------------
# Plot the results
# -------------------------------
time_sim = np.linspace(0, dt*Nsim, Nsim+1)
time_sim2 = np.linspace(0, dt*Nsim, Nsim)


fig2, ax3 = plt.subplots()
ax3.plot(y_history, x_history, 'r-')
ax3.plot(y_history, x_history, 'b--')
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
ax6.plot(time_sim, yawd_history, 'r-')
ax6.plot(time_sim, yaw_history, 'b--')
ax6.set_xlabel('Time [s]')
ax6.set_ylabel('yaw [rad]')
fig4.tight_layout()

plt.show()

# -------------------------------
# Animate results
# -------------------------------
'''if plt.isinteractive():
  fig2, ax3 = plt.subplots(1, 1)
  plt.ion()
  ax3.set_xlabel("X [m]")
  ax3.set_ylabel("Y [m]")
  for k in range(Nsim+1):
      cart_pos_k      = ye_history[k]
      theta_k         = theta_history[k]
      pendulum_pos_k  = vertcat(horzcat(cart_pos_k,0), vertcat(cart_pos_k-L*sin(theta_k),L*cos(theta_k)).T)
      color_k     = 3*[0.95*(1-float(k)/Nsim)]
      ax3.plot(pendulum_pos_k[0,0], pendulum_pos_k[0,1], "s", markersize = 15, color = color_k)
      ax3.plot(pendulum_pos_k[:,0], pendulum_pos_k[:,1], "-", linewidth = 1.5, color = color_k)
      ax3.plot(pendulum_pos_k[1,0], pendulum_pos_k[1,1], "o", markersize = 10, color = color_k)
      plt.pause(dt)
plt.show(block=True)'''
