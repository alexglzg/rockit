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

nx    = 12                   # the system is composed of 12 states
nu    = 2                   # the system has 2 input
Tf    = 4                   # control horizon [s]
Nhor  = 40                  # number of control intervals
dt    = Tf/Nhor             # sample time

starting_angle = 0.0
x_1 = 3.0
y_1 = -5.0
x2 = 3.0
y2 = 25.0
a_k = np.math.atan2(y2-y_1, x2-x_1)
ned_x = 0.0
ned_y = 0.0
y_e = -(ned_x-x_1)*np.sin(a_k)+(ned_y-y_1)*np.cos(a_k)
current_X = vertcat(starting_angle, np.sin(starting_angle), np.cos(starting_angle), 0.001, 0.00, 0.00, y_e, a_k, ned_x, ned_y, 0.00, 0.00)  # initial state

u_ref = 1.2
ak_ref = a_k
sinpsi_ref = np.sin(ak_ref)
cospsi_ref = np.cos(ak_ref)
ye_ref = 0.0
final_X   = vertcat(0, sinpsi_ref, cospsi_ref, u_ref, 0, 0, ye_ref, 0, 0, 0, 0, 0, 0, 0)    # desired terminal state

Nsim  = int(30 * Nhor / Tf)                 # how much samples to simulate
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
theta  = ocp.state()  # [rad]2.0,
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
#ocp.set_initial(UTportdot,30)
#ocp.set_initial(UTstbddot,30)

# Define parameter
#X_0 = ocp.parameter(nx);
X_0 = ocp.parameter(nx)
obs_pos = ocp.parameter(16)
obs_rad = ocp.parameter(8)

# Specify ODE
Xu = if_else(u > 1.2627, 64.55, -25.0)
Xuu = if_else(u > 1.2627, -70.92, 0.0)
Yv = 0.5*(-40*1000*fabs(v))*(1.1+0.0045*(1.01/0.09)-0.1*(0.27/0.09)+0.016*((0.27/0.09)*(0.27/0.09)))
Nr = -3#(-0.52)*sqrt(u*u + v*v)
Tu = Tport + c * Tstbd
Tr = (Tport - c * Tstbd) * B / 2
beta = 0#atan2(v,u+.001)
chi = psi + beta
distance1 = sqrt((nedx-obs_pos[0])*(nedx-obs_pos[0]) + (nedy-obs_pos[1])*(nedy-obs_pos[1]))
distance2 = sqrt((nedx-obs_pos[2])*(nedx-obs_pos[2]) + (nedy-obs_pos[3])*(nedy-obs_pos[3]))
distance3 = sqrt((nedx-obs_pos[4])*(nedx-obs_pos[4]) + (nedy-obs_pos[5])*(nedy-obs_pos[5]))
distance4 = sqrt((nedx-obs_pos[6])*(nedx-obs_pos[6]) + (nedy-obs_pos[7])*(nedy-obs_pos[7]))
distance5 = sqrt((nedx-obs_pos[8])*(nedx-obs_pos[8]) + (nedy-obs_pos[9])*(nedy-obs_pos[9]))
distance6 = sqrt((nedx-obs_pos[10])*(nedx-obs_pos[10]) + (nedy-obs_pos[11])*(nedy-obs_pos[11]))
distance7 = sqrt((nedx-obs_pos[12])*(nedx-obs_pos[12]) + (nedy-obs_pos[13])*(nedy-obs_pos[13]))
distance8 = sqrt((nedx-obs_pos[14])*(nedx-obs_pos[14]) + (nedy-obs_pos[15])*(nedy-obs_pos[15]))
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
ocp.set_der(ak, 0)
ocp.set_der(nedx, (u*cos(psi) - v*sin(psi)))
ocp.set_der(nedy, (u*sin(psi) + v*cos(psi)))
ocp.set_der(Tport, UTportdot)
ocp.set_der(Tstbd, (UTstbddot/c))

danger_zone = 0.2
# Lagrange objective
#ocp.add_objective(ocp.integral(F*2 + 100*pos**2))
#ocp.add_objective(ocp.integral(0.5*(ye-ye_ref)**2 + 2.0*(sinpsi-sinpsi_ref)**2 + 2.0*(cospsi-cospsi_ref)**2 + 30*(u-u_ref)**2 + 0.1*r**2 + 0.001*Tstbd**2 + 0.001*Tport**2))
ocp.add_objective(ocp.integral(5.0*(ye-ye_ref)**2 + 0.5*(sinpsi-sinpsi_ref)**2 + 0.5*(cospsi-cospsi_ref)**2 + 40*(u-u_ref)**2 
                                + 0.05*r**2 + 0.0005*Tstbd**2 + 0.0005*Tport**2 + 50*slack1u + 50*slack1l + 50*slack2u + 50*slack2l 
                                + 50*slack3u + 50*slack3l + 50*slack4u + 50*slack4l + 50*slack5u + 50*slack5l + 50*slack6u + 50*slack6l 
                                + 50*slack7u + 50*slack7l + 50*slack8u + 50*slack8l))
'''ocp.add_objective(ocp.integral(0.5*(ye-ye_ref)**2 + 2.0*(sinpsi-sinpsi_ref)**2 + 2.0*(cospsi-cospsi_ref)**2 + 30*(u-u_ref)**2 
                                + 0.1*r**2 + 0.001*Tstbd**2 + 0.001*Tport**2 + 50*slack1l + 50*slack2l + 50*slack3l + 50*slack4l 
                                + 50*slack5l + 50*slack6l + 50*slack7l + 50*slack8l))'''
ocp.add_objective(ocp.at_tf(10.0*(ye-ye_ref)**2 + 1.0*(sinpsi-sinpsi_ref)**2 + 1.0*(cospsi-cospsi_ref)**2 + 80*(u-u_ref)**2 
                                + 0.1*r**2 + 0.001*Tstbd**2 + 0.001*Tport**2))
#ocp.add_objective(ocp.integral(1*(ye-ye_ref)**2))

# Path constraints
'''ocp.subject_to(      F <= 2  )
ocp.subject_to(-2 <= F       )
ocp.subject_to(-2 <= pos     )
ocp.subject_to(      pos <= 2)'''
ocp.subject_to( (-1.5 <= u) <= 1.5 )
ocp.subject_to( (-5.0 <= r) <= 5.0 )
ocp.subject_to( (-30.0 <= Tport) <= 36.5 )
ocp.subject_to( (-30.0 <= Tstbd) <= 36.5 )
ocp.subject_to( (-90.0 <= UTportdot) <= 90.0 )
ocp.subject_to( (-90.0 <= UTstbddot) <= 90.0 )
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
ocp.subject_to( slack1l >= -danger_zone )
ocp.subject_to( slack2l >= -danger_zone )
ocp.subject_to( slack3l >= -danger_zone )
ocp.subject_to( slack4l >= -danger_zone )
ocp.subject_to( slack5l >= -danger_zone )
ocp.subject_to( slack6l >= -danger_zone )
ocp.subject_to( slack7l >= -danger_zone )
ocp.subject_to( slack8l >= -danger_zone )
ocp.subject_to( slack1u >= 0 )
ocp.subject_to( slack2u >= 0 )
ocp.subject_to( slack3u >= 0 )
ocp.subject_to( slack4u >= 0 )
ocp.subject_to( slack5u >= 0 )
ocp.subject_to( slack6u >= 0 )
ocp.subject_to( slack7u >= 0 )
ocp.subject_to( slack8u >= 0 )

# Initial constraints
X = vertcat(psi, sinpsi, cospsi, u, v, r, ye, ak, nedx, nedy, Tport, Tstbd)
ocp.subject_to(ocp.at_t0(X)==X_0)
#ocp.subject_to(ocp.at_tf(X)==final_X)

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
obstacle_radius = 0.5
ocp.set_value(X_0, current_X)
obstacles = vertcat(3,6,5,8,1,8,3,10,100,100,100,100,100,100,100,100)
radius = vertcat(obstacle_radius, obstacle_radius, obstacle_radius, obstacle_radius, 0, 0, 0, 0)
ocp.set_value(obs_pos, obstacles)
ocp.set_value(obs_rad, radius)
# Solve
sol = ocp.solve()

# Get discretisd dynamics as CasADi function
#Sim_pendulum_dyn = ocp._method.discrete_system(ocp)
Sim_asv_dyn = ocp._method.discrete_system(ocp)

# Log data for post-processing
ye_history[0]   = current_X[6]
theta_history[0] = current_X[0]
x_history[0]   = current_X[8]
y_history[0] = current_X[9]
u_history[0]   = current_X[3]
r_history[0] = current_X[5]
Tport_history[0] = current_X[10]
Tstbd_history[0] = current_X[11]

# -------------------------------
# Simulate the MPC solving the OCP (with the updated state) several times
# -------------------------------

for i in range(Nsim):
    print("timestep", i+1, "of", Nsim)
    # Get the solution from sol
    #tsa, Fsol = sol.sample(F, grid='control')
    tsa, Tpsol = sol.sample(UTportdot, grid='control')
    _, Tssol = sol.sample(UTstbddot, grid='control')
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
    # print(S4 sol)
    # Simulate dynamics (applying the first control input) and update the current state
    #current_X = Sim_pendulum_dyn(x0=current_X, u=Fsol[0], T=dt)["xf"]
    current_X = Sim_asv_dyn(x0=current_X, u=vertcat(Tpsol[0],Tssol[0],S1usol[0],S2usol[0],S3usol[0],S4usol[0],S5usol[0],S6usol[0],S7usol[0],S8usol[0],
                                                    S1lsol[0],S2lsol[0],S3lsol[0],S4lsol[0],S5lsol[0],S6lsol[0],S7lsol[0],S8lsol[0]), T=dt)["xf"]
    #current_X = Sim_asv_dyn(x0=current_X, u=vertcat(Tpsol[0],Tssol[0],S1lsol[0],S2lsol[0],S3lsol[0],S4lsol[0],S5lsol[0],S6lsol[0],S7lsol[0],S8lsol[0]), T=dt)["xf"]
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
    ocp.set_value(X_0, current_X[:12])
    ocp.set_value(obs_pos, obstacles)
    ocp.set_value(obs_rad, radius)
    # Solve the optimization problem
    # import pdb; pdb.set_trace()
    sol = ocp.solve()
    ocp._method.opti.set_initial(ocp._method.opti.x, ocp._method.opti.value(ocp._method.opti.x))
    # ocp._method.opti.set_initial(ocp._method.opti.lam, ocp._method.opti.value(ocp._method.opti.lam))

    # Log data for post-processing
    ye_history[i+1]   = current_X[6].full()
    theta_history[i+1] = current_X[0].full()
    x_history[i+1]   = current_X[8].full()
    y_history[i+1] = current_X[9].full()
    u_history[i+1]   = current_X[3].full()
    r_history[i+1] = current_X[5].full()
    Tport_history[i+1] = current_X[10].full()
    Tstbd_history[i+1] = current_X[11].full()
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
ax3.set_ylabel('X [m]')
#ax3.tick_params('y', colors='r')
#obstacles = vertcat(4,4,6,6,2,6,4,8)
#radius = vertcat(0.5, 0.5, 0.5, 0.5)
obstacle_array = np.array([3,6,5,8,1,8,3,10])
for j in range(4):
    c = plt.Circle((obstacle_array[2*j+1],obstacle_array[2*j]),obstacle_radius)
    ax3.add_patch(c)

fig3, ax4 = plt.subplots()
ax4.plot(time_sim, Tport_history, 'r-')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Thrust [N]')
#ax4.tick_params('y', colors='r')
ax4.plot(time_sim, Tstbd_history, 'b-')
#ax5.set_ylabel('Tstbd [N]', color='b')
#ax5.tick_params('y', colors='b')
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
