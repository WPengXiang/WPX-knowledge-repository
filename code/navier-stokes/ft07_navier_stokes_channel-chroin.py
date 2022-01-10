"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for channel flow (Poisseuille) on the unit square using the
Incremental Pressure Correction Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""

from __future__ import print_function
from fenics import *
import numpy as np

T = 10.0           # final time
num_steps = 50    # number of time steps
dt = T / num_steps # time step size
mu = 1             # kinematic viscosity
rho = 1            # density
ns = 16
# Create mesh and define function spaces
mesh = UnitSquareMesh(ns, ns)
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundaries
inflow  = 'near(x[0], 0)'
outflow = 'near(x[0], 1)'
walls   = 'near(x[1], 0) || near(x[1], 1)'

# Define boundary conditions
bcu_noslip  = DirichletBC(V, Constant((0, 0)), walls)
bcp_inflow  = DirichletBC(Q, Constant(8), inflow)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_noslip]
bcp = [bcp_inflow, bcp_outflow]

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)

# Define expressions used in variational forms
n   = FacetNormal(mesh)
f   = Constant((0, 0))
k   = Constant(dt)
mu  = Constant(mu)
rho = Constant(rho)

# Define strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Define variational problem for step 1
'''
F1 = rho*dot((u - u_n) / k, v)*dx + \
     rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + mu*inner(nabla_grad(u), nabla_grad(v))*dx \
   - dot(f, v)*dx
'''
'''
F1 = (1/k)*inner(v,u-u_n)*dx + \
    inner(v,grad(u_n)*u_n)*dx \
   + mu*inner(grad(v),grad(u))*dx \
   - inner(v,f)*dx\
   - dot(mu*nabla_grad(u)*n,v)*ds
'''

F1 = (1/k)*inner(u - u_n,v)*dx + inner(grad(u_n)*u_n,v)*dx \
    + mu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = inner(grad(q), grad(p))*dx
L2 = -(1/k)*q*div(u_)*dx

# Define variational problem for step 3
a3 = inner(u,v)*dx
L3 = inner(u_,v)*dx - k*inner(grad(p_),v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(A1,b1) for bc in bcu]
    solve(A1, u_.vector(), b1,"gmres","default")

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(A2,b2) for bc in bcp]
    solve(A2, p_.vector(), b2,"gmres","default")

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    [bc.apply(A3,b3) for bc in bcu]
    solve(A3, u_.vector(), b3,"gmres","default")

    # Plot solution
    plot(u_)
    #print(u_.vector().get_local())
    # Compute error
    u_e = Expression(('4*x[1]*(1.0 - x[1])', '0'), degree=2)
    u_e = interpolate(u_e, V)
    error = np.abs(u_e.vector() - u_.vector()).max()
    print('t = %.2f: error = %.3g' % (t, error))
    #print('max u:', u_.vector().max())

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)

# Hold plot
import matplotlib.pyplot as plt
plt.show()
