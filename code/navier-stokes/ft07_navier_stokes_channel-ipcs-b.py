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

T = 1.0           # final time
num_steps = 50    # number of time steps
dt = T / num_steps # time step size
mu = 1             # kinematic viscosity
rho = 1            # density
ns = 8
degree = 1
# Create mesh and define function spaces
mesh = UnitSquareMesh(ns, ns)
V = VectorFunctionSpace(mesh, 'P', degree)
Q = FunctionSpace(mesh, 'P', degree)

# Define boundaries

ud = Expression(('4*x[1]*(1.0 - x[1])', '0'), degree=2)

# Define boundary conditions
bcu  = DirichletBC(V, ud , "on_boundary")
bcu = [bcu]
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
U   = 0.5*(u_n + u)
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
F1 = rho*dot((u - u_n) / k, v)*dx + \
     rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   - dot(f, v)*dx

a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(grad(p_ - p_n), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

#print("\n",np.abs(A1.array()).sum())
# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]

#print("\n",np.abs(A1.array()).sum())
# Time-stepping
t = 0
for n in range(num_steps):
    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1)
    #print(np.abs(A1.array()).sum())
    #print("\n",np.abs(b1.get_local()).sum())
    
    # Step 2: Pressure correction step
    b2 = assemble(L2)
    solve(A2, p_.vector(), b2)
    #print(np.abs(A2.array()).sum())
    #print("\n",np.abs(b2.get_local()).sum())
    
    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3)
    print(np.abs(b3.get_local()).sum())
    #print(np.abs(A3.array()).sum())
    #print("t=",t)
    #print("\n",np.abs(u_.vector().get_local()).sum())
    
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
'''
import matplotlib.pyplot as plt
plt.show()
'''
