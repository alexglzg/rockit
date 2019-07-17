ocp = Ocp(t0=0,tf=10) # tf or T deltaT or T


x = ocp.state() # Framehack to get 'x'
y = ocp.state()
u = ocp.control()
u = ocp.control(order=2)

q = ocp.state(7)
dq = ocp.state(7)


ocp.subject_to(x<=inf) # path constraint
ocp.subject_to(x>=-0.25)
ocp.subject_to(y<=inf)
ocp.subject_to(y>=-inf)
ocp.subject_to(u<=1)
ocp.subject_to(u>=-1)

ocp.add_objective(ocp.integral(x**2 + y**2 + u**2)) # Lagrange term
ocp.add_objective(ocp.sum(x**2 + y**2 + u**2))      # discrete ?
ocp.add_objective(ocp.at_tf(x**2))                  # Mayer term


ocp.add_objective(ocp.sum(ocp.ConvexOverNonlinear('norm_2',sin(x)))

l=ocp.variables('control')

ocp.subject_to(ocp.ConvexOverNonlinear('norm_2',sin(x))<=rho+l)

# Sub-sampling of constraints!!

# todo: slack variables

ocp.set_der(x, (1-y**2)*x-y+u+ocp.t) # Time-dependant ODE
ocp.set_der(y, x)


ocp.set_der(q, (pinocchio), jac=ddyn/dq)

eT = Function('e_T',{"custom_jacobian":...})

ocp.subject_to(e_T(x,u)>=0, jac=de_T/dsdasd) # syntax error if >=x

#
#ocp.set_intg(x, 'rk4') #?

# accuracy may not be needed for some states

#ocp.set_intg([x,y], 'rk4') #?
#ocp.set_intg(ocp.all_states, 'rk4') #?

ocp.subject_to(ocp.der(ocp.der(x+y**2))>=0) # cache
ocp.subject_to(ocp.der(u)>=0)

# At time t=0
ocp.subject_to(ocp.at_t0(x)==0) # Should we automatically recognize path_constraints? [Armin: yes ]
ocp.subject_to(ocp.at_t0(y)==0) # pointwise_constraint
ocp.subject_to(ocp.at_t0(y)==ocp.at_tf(y)) # periodic



ocp.solver('ipopt',algo='ms',intg='rk4',N=25,M=4,casadi_options=dict()) # 4 integration steps per control  # auto-detect solver (QP)?

# ordering can be preserved

# ms - multiple-shooting
# ss - single-shooting
# dc - direct collocation
#      pseudo-spectral
# spline
# spline-middle



# different integrators for different subsystem; in linear subsystem -> matrix exponential?

ocp.set_initial(x, cos(ocp.t))

ocp.set_initial(x, (ts, xs))

ocp.solver('ipopt')

sol = ocp.solve()


# what to do with controls? drop the last.
ts, xsol = sol.sample_intg(x) # Sample at the integrator boundaries (0.1s)
ts, xsol = sol.sample_control(x) # Sample at the control boundaries (0.4s)
ts, xsol = sol.sample_intg_fine(x,N=10) # Refine integration output locally
ts, xsol = sol.sample_sim(x,N=1000) # Simulate feed-forward

# Automatic plot funcitonality -