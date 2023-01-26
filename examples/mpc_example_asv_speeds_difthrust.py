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
'''mcart = 0.5                 # cart mass [kg]
m     = 1                   # pendulum mass [kg]
L     = 2                   # pendulum length [m]
g     = 9.81                # gravitation [m/s^2]'''
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

'''nx    = 4                   # the system is composed of 4 states
nu    = 1                   # the system has 1 input
Tf    = 2                   # control horizon [s]
Nhor  = 50                  # number of control intervals
dt    = Tf/Nhor             # sample time'''

nx    = 5                   # the system is composed of 5 states
nu    = 2                   # the system has 2 input
Tf    = 3                   # control horizon [s]
Nhor  = 30                  # number of control intervals
dt    = Tf/Nhor             # sample time

current_X = vertcat(0.001, 0.0, 0.05, 0, 0)  # initial state

u_ref = 0.7
r_ref = 0.2
final_X   = vertcat(u_ref, 0, r_ref, 0, 0)    # desired terminal state

Nsim  = int(10 * Nhor / Tf)        # how much samples to simulate
add_noise = False #True            # enable/disable the measurement noise addition in simulation
add_disturbance = False #True      # enable/disable the disturbance addition in simulation

# -------------------------------
# Logging variables
# -------------------------------
'''pos_history     = np.zeros(Nsim+1)
theta_history   = np.zeros(Nsim+1)
F_history       = np.zeros(Nsim)'''
u_history     = np.zeros(Nsim+1)
r_history   = np.zeros(Nsim+1)
Tport_history       = np.zeros(Nsim+1)
Tstbd_history       = np.zeros(Nsim+1)
UTport_history       = np.zeros(Nsim)
UTstbd_history       = np.zeros(Nsim)

# -------------------------------
# Set OCP
# -------------------------------
ocp = Ocp(T=Tf)

# Define states
'''pos    = ocp.state()  # [m]
theta  = ocp.state()  # [rad]
dpos   = ocp.state()  # [m/s]
dtheta = ocp.state()  # [rad/s]'''
#x = ocp.state(10)
u = ocp.state()
v = ocp.state()
r = ocp.state()
Tport = ocp.state()
Tstbd = ocp.state()
ocp.set_initial(u,0.001)
#ocp.set_initial(v,0.001)
#ocp.set_initial(r,0.1)

# Defince controls
#F = ocp.control(nu, order=0)
Tportdot = ocp.control()
Tstbddot = ocp.control()

# Define parameter
#X_0 = ocp.parameter(nx);
X_0 = ocp.parameter(nx)

# Specify ODE
Xu = if_else(u > 1.25, 64.55, -25)
Xuu = if_else(u > 1.25, -70.92, 0)
Yv = 0.5*(-40*1000*fabs(v))*(1.1+0.0045*(1.01/0.09)-0.1*(0.27/0.09)+0.016*((0.27/0.09)*(0.27/0.09)))
Nr = (-0.52)*sqrt(u*u + v*v)
Tu = Tport + c * Tstbd
Tr = (Tport - c * Tstbd) * B / 2
ocp.set_der(u, ((Tu - (-m + 2 * Y_v_dot)*v - (Y_r_dot + N_v_dot)*r*r - (-Xu*u - Xuu*fabs(u)*u)) / (m - X_u_dot)))
ocp.set_der(v, ((-(m - X_u_dot)*u*r - (- Yv - Yvv*fabs(v) - Yvr*fabs(r))*v) / (m - Y_v_dot)))
ocp.set_der(r, ((Tr - (-2*Y_v_dot*u*v - (Y_r_dot + N_v_dot)*r*u + X_u_dot*u*r) - (-Nr*r - Nrv*fabs(v)*r - Nrr*fabs(r)*r)) / (Iz - N_r_dot)))
ocp.set_der(Tport, Tportdot)
ocp.set_der(Tstbd, Tstbddot)

# Lagrange objective
#ocp.add_objective(ocp.integral(F*2 + 100*pos**2))
#ocp.add_objective(ocp.integral((ye-ye_ref)**2 + (sinpsi-sinpsi_ref)**2 + (cospsi-cospsi_ref)**2 + (u-u_ref)**2 + Tstbd**2 + Tport**2))
X = vertcat(u, v, r, Tport, Tstbd)
ocp.add_objective(ocp.integral((u-u_ref)**2))

# Path constraints
'''ocp.subject_to(      F <= 2  )
ocp.subject_to(-2 <= F       )
ocp.subject_to(-2 <= pos     )
ocp.subject_to(      pos <= 2)'''
ocp.subject_to( -1.5 <= u )
ocp.subject_to( u <= 1.5)
ocp.subject_to( -1.0 <= r )
ocp.subject_to( r <= 1.0)
ocp.subject_to( -30.0 <= Tport )
ocp.subject_to( Tport <= 35.0)
ocp.subject_to( -30.0 <= Tstbd )
ocp.subject_to( Tstbd <= 35.0)
'''ocp.subject_to( -90.0 <= Tportdot )
ocp.subject_to( Tportdot <= 90.0)
ocp.subject_to( -90.0 <= Tstbddot )
ocp.subject_to( Tstbddot <= 90.0)'''

# Initial constraints
ocp.subject_to(ocp.at_t0(X)==X_0)
#ocp.subject_to(ocp.at_tf(X)==final_X)
ocp.subject_to(ocp.at_tf(u)==u_ref)
ocp.subject_to(ocp.at_tf(r)==r_ref)

# Pick a solution method
options = {"ipopt": {"print_level": 0, "max_iter": 100000}}
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
u_history[0]   = current_X[0]
r_history[0] = current_X[2]
Tport_history[0] = current_X[3]
Tstbd_history[0] = current_X[4]

# -------------------------------
# Simulate the MPC solving the OCP (with the updated state) several times
# -------------------------------

for i in range(Nsim):
    print("timestep", i+1, "of", Nsim)
    # Get the solution from sol
    #tsa, Fsol = sol.sample(F, grid='control')
    tsa, Tpsol = sol.sample(Tport, grid='control')
    _, Tssol = sol.sample(Tstbd, grid='control')
    # Simulate dynamics (applying the first control input) and update the current state
    #current_X = Sim_pendulum_dyn(x0=current_X, u=Fsol[0], T=dt)["xf"]
    current_X = Sim_asv_dyn(x0=current_X, u=vertcat(Tpsol[0],Tssol[0]), T=dt)["xf"]
    # Add disturbance at t = 2*Tf
    '''if add_disturbance:
        if i == round(2*Nhor)-1:
            disturbance = vertcat(0,0,-1e-1,0)
            current_X = current_X + disturbance
    # Add measurement noise
    if add_noise:
        meas_noise = 5e-4*(vertcat(np.random.rand(nx,1))-vertcat(1,1,1,1)) # 4x1 vector with values in [-1e-3, 1e-3]
        current_X = current_X + meas_noise'''
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X[:5])
    # Solve the optimization problem
    sol = ocp.solve()

    # Log data for post-processing
    u_history[i+1]   = current_X[0].full()
    r_history[i+1] = current_X[2].full()
    Tport_history[i+1] = current_X[3].full()
    Tstbd_history[i+1] = current_X[4].full()
    UTport_history[i] = Tpsol[0]
    UTstbd_history[i] = Tssol[0]
    #F_history[i]       = Fsol[0]

# -------------------------------
# Plot the results
# -------------------------------
time_sim = np.linspace(0, dt*Nsim, Nsim+1)
time_sim2 = np.linspace(0, dt*Nsim, Nsim)

fig, ax1 = plt.subplots()
ax1.plot(time_sim, u_history, 'r-')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('u [m/s]', color='r')
ax1.tick_params('y', colors='r')
ax2 = ax1.twinx()
ax2.plot(time_sim, r_history, 'b-')
ax2.set_ylabel('r [rad/s]', color='b')
ax2.tick_params('y', colors='b')
#ax2.axvline(x=2*Tf, color='k', linestyle='--')
#ax2.text(2*Tf+0.1,0.025,'disturbance applied',rotation=90)
fig.tight_layout()

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

fig5, ax8 = plt.subplots()
ax8.plot(time_sim2, UTport_history, 'r-')
ax8.set_xlabel('Time [s]')
ax8.set_ylabel('Tportdot [N/s]', color='r')
ax8.tick_params('y', colors='r')
ax9 = ax8.twinx()
ax9.plot(time_sim2, UTstbd_history, 'b-')
ax9.set_ylabel('Tstbddot [N/s]', color='b')
ax9.tick_params('y', colors='b')
fig5.tight_layout()


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
