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

nx    = 14                   # the system is composed of 14 states
nu    = 2                   # the system has 2 input
Tf    = 1                   # control horizon [s]
Nhor  = 20                  # number of control intervals
dt    = Tf/Nhor             # sample time

starting_angle = 0.00
x_1 = 4.0
y_1 = -5.0
x2 = 4.0
y2 = 25.0
a_k = np.math.atan2(y2-y_1, x2-x_1)
ned_x = 0.1
ned_y = 0.1
y_e = -(ned_x-x_1)*np.sin(a_k)+(ned_y-y_1)*np.cos(a_k)
current_X = vertcat(starting_angle, np.sin(starting_angle), np.cos(starting_angle), 0.001, 0.00, 0.00, y_e, x_1, y_1, a_k, ned_x, ned_y, 0.00, 0.00)  # initial state

u_ref = 1.4
ak_ref = a_k
sinpsi_ref = np.sin(ak_ref)
cospsi_ref = np.cos(ak_ref)
ye_ref = 0.0
final_X   = vertcat(0, sinpsi_ref, cospsi_ref, u_ref, 0, 0, ye_ref, 0, 0, 0, 0, 0, 0, 0)    # desired terminal state

Nsim  = int(20 * Nhor / Tf)#200                 # how much samples to simulate
add_noise = False #True            # enable/disable the measurement noise addition in simulation
add_disturbance = False #True      # enable/disable the disturbance addition in simulation

# -------------------------------
# Logging variables
# -------------------------------
'''pos_history     = np.zeros(Nsim+1)
theta_history   = np.zeros(Nsim+1)
F_history       = np.zeros(Nsim)'''
ye_history     = np.zeros(Nsim+1)
theta_history   = np.zeros(Nsim+1)
x_history     = np.zeros(Nsim+1)
y_history   = np.zeros(Nsim+1)
u_history     = np.zeros(Nsim+1)
r_history   = np.zeros(Nsim+1)
Tport_history       = np.zeros(Nsim+1)
Tstbd_history       = np.zeros(Nsim+1)

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
psi = ocp.state()
sinpsi = ocp.state() 
cospsi = ocp.state() 
u = ocp.state()
v = ocp.state()
r = ocp.state()
ye = ocp.state()
x1 = ocp.state()
y1 = ocp.state()
ak = ocp.state()
nedx = ocp.state()
nedy = ocp.state()
Tport = ocp.state()
Tstbd = ocp.state()
#ocp.set_initial(psi,0.0)
#ocp.set_initial(sinpsi,0.0)
#ocp.set_initial(cospsi,1.0)
ocp.set_initial(u,0.001)
#ocp.set_initial(v,0.001)
#ocp.set_initial(r,0.001)
#ocp.set_initial(nedx,0.001)
#ocp.set_initial(nedy,0.001)
#ocp.set_initial(Tport,0.001)
#ocp.set_initial(Tstbd,0.001)


# Defince controls
#F = ocp.control(nu, order=0)
UTportdot = ocp.control()
UTstbddot = ocp.control()
#ocp.set_initial(UTportdot,30)
#ocp.set_initial(UTstbddot,30)

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
beta = atan2(v,u+.001)
chi = psi + beta
'''ocp.set_der(pos, dpos)
ocp.set_der(theta, dtheta)
ocp.set_der(dpos, (-m*L*sin(theta)*dtheta*dtheta + m*g*cos(theta)*sin(theta)+F)/(mcart + m - m*cos(theta)*cos(theta)) )
ocp.set_der(dtheta, (-m*L*cos(theta)*sin(theta)*dtheta*dtheta + F*cos(theta)+(mcart+m)*g*sin(theta))/(L*(mcart + m - m*cos(theta)*cos(theta))))'''
ocp.set_der(psi, r)
ocp.set_der(sinpsi, (cos(chi)*r))
ocp.set_der(cospsi, (-sin(chi)*r))
ocp.set_der(u, ((Tu - (-m + 2 * Y_v_dot)*v - (Y_r_dot + N_v_dot)*r*r - (-Xu*u - Xuu*fabs(u)*u)) / (m - X_u_dot)))
ocp.set_der(v, ((-(m - X_u_dot)*u*r - (- Yv - Yvv*fabs(v) - Yvr*fabs(r))*v) / (m - Y_v_dot)))
ocp.set_der(r, ((Tr - (-2*Y_v_dot*u*v - (Y_r_dot + N_v_dot)*r*u + X_u_dot*u*r) - (-Nr*r - Nrv*fabs(v)*r - Nrr*fabs(r)*r)) / (Iz - N_r_dot)))
ocp.set_der(ye, (-(u*cos(psi) - v*sin(psi))*sin(ak) + (u*sin(psi) + v*cos(psi))*cos(ak)))
ocp.set_der(x1, 0)
ocp.set_der(y1, 0)
ocp.set_der(ak, 0)
ocp.set_der(nedx, (u*cos(psi) - v*sin(psi)))
ocp.set_der(nedy, (u*sin(psi) + v*cos(psi)))
ocp.set_der(Tport, UTportdot)
ocp.set_der(Tstbd, (UTstbddot/c))

# Lagrange objective
#ocp.add_objective(ocp.integral(F*2 + 100*pos**2))
ocp.add_objective(ocp.integral(0.4*(ye-ye_ref)**2 + 0.15*(sinpsi-sinpsi_ref)**2 + 0.15*(cospsi-cospsi_ref)**2 + 40*(u-u_ref)**2 + 0.00005*Tstbd**2 + 0.00005*Tport**2))
ocp.add_objective(ocp.at_tf(0.5*(ye-ye_ref)**2 + 0.25*(sinpsi-sinpsi_ref)**2 + 0.25*(cospsi-cospsi_ref)**2 + 0*(u-u_ref)**2 + 0.00025*Tstbd**2 + 0.00025*Tport**2))
#ocp.add_objective(ocp.integral(1*(ye-ye_ref)**2))

# Path constraints
'''ocp.subject_to(      F <= 2  )
ocp.subject_to(-2 <= F       )
ocp.subject_to(-2 <= pos     )
ocp.subject_to(      pos <= 2)'''
ocp.subject_to( (-2.0 <= u) <= 2.0 )
#ocp.subject_to( u <= 1.5)
ocp.subject_to( (-10.0 <= r) <= 10.0 )
#ocp.subject_to( r <= 1.0)
ocp.subject_to( (-30.0 <= Tport) <= 35.0 )
#ocp.subject_to( Tport <= 35.0)
ocp.subject_to( (-30.0 <= Tstbd) <= 35.0 )
#ocp.subject_to( Tstbd <= 35.0)
ocp.subject_to( (-30.0 <= UTportdot) <= 30.0 )
#ocp.subject_to( UTportdot <= 35.0)
ocp.subject_to( (-30.0 <= UTstbddot) <= 30.0 )
#ocp.subject_to( UTstbddot <= 35.0)

# Initial constraints
X = vertcat(psi, sinpsi, cospsi, u, v, r, ye, x1, y1, ak, nedx, nedy, Tport, Tstbd)
ocp.subject_to(ocp.at_t0(X)==X_0)
#ocp.subject_to(ocp.at_tf(X)==final_X)

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
#ye_history[0]   = current_X[6]
ye_history[0]   = current_X[4]
theta_history[0] = current_X[0]
x_history[0]   = current_X[10]
y_history[0] = current_X[11]
u_history[0]   = current_X[3]
r_history[0] = current_X[5]
Tport_history[0] = current_X[12]
Tstbd_history[0] = current_X[13]

# -------------------------------
# Simulate the MPC solving the OCP (with the updated state) several times
# -------------------------------

for i in range(Nsim):
    print("timestep", i+1, "of", Nsim)
    # Get the solution from sol
    #tsa, Fsol = sol.sample(F, grid='control')
    tsa, Tpsol = sol.sample(UTportdot, grid='control')
    _, Tssol = sol.sample(UTstbddot, grid='control')
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
    ocp.set_value(X_0, current_X[:14])
    # Solve the optimization problem
    sol = ocp.solve()
    ocp._method.opti.set_initial(ocp._method.opti.x, ocp._method.opti.value(ocp._method.opti.x))

    # Log data for post-processing
    ye_history[i+1]   = current_X[6].full()
    theta_history[i+1] = current_X[0].full()
    x_history[i+1]   = current_X[10].full()
    y_history[i+1] = current_X[11].full()
    u_history[i+1]   = current_X[3].full()
    r_history[i+1] = current_X[5].full()
    Tport_history[i+1] = current_X[12].full()
    Tstbd_history[i+1] = current_X[13].full()
    #F_history[i]       = Fsol[0]

# -------------------------------
# Plot the results
# -------------------------------
time_sim = np.linspace(0, dt*Nsim, Nsim+1)

fig, ax1 = plt.subplots()
ax1.plot(time_sim, ye_history, 'r-')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Cross-track error [m]', color='r')
ax1.tick_params('y', colors='r')
ax2 = ax1.twinx()
ax2.plot(time_sim, theta_history, 'b-')
ax2.set_ylabel('Heading angle [rad]', color='b')
ax2.tick_params('y', colors='b')
fig.tight_layout()

fig2, ax3 = plt.subplots()
ax3.plot(y_history, x_history, 'r-')
ax3.set_xlabel('Y [m]')
ax3.set_ylabel('X [m]', color='r')
ax3.tick_params('y', colors='r')

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
ax6.plot(time_sim, u_history, 'r-')
ax6.set_xlabel('Time [s]')
ax6.set_ylabel('u [m/s]', color='r')
ax6.tick_params('y', colors='r')
ax7 = ax6.twinx()
ax7.plot(time_sim, r_history, 'b-')
ax7.set_ylabel('r [rad/s]', color='b')
ax7.tick_params('y', colors='b')
#ax2.axvline(x=2*Tf, color='k', linestyle='--')
#ax2.text(2*Tf+0.1,0.025,'disturbance applied',rotation=90)
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
