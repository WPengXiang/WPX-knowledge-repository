from dolfin import *
from fenics import *
import numpy as np
# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

# Load mesh from file
ns = 16
mesh = UnitSquareMesh(ns,ns)

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

# Set parameter values
dt = 0.2
T = 10.0
nu = 1

# Define boundary conditions

pinflow  = 'near(x[0], 0)'
poutflow = 'near(x[0], 1)'
uwalls   = 'near(x[1], 0) || near(x[1], 1)'
noslip  = DirichletBC(V, Constant((0,0)),uwalls)
inflow  = DirichletBC(Q, Constant(8), pinflow)
outflow = DirichletBC(Q, Constant(0), poutflow)
bcu = [noslip]
bcp = [inflow, outflow]

# Create functions
u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)

# Define coefficients
k = Constant(dt)
f = Constant((0, 0))

# Tentative velocity step
F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx + \
     nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
a2 = inner(grad(p), grad(q))*dx
L2 = -(1/k)*div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx
L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Create files for storing solution
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")

# Time-stepping
t = dt
while t < T + DOLFIN_EPS:

    # Compute tentative velocity step
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "gmres", "default")

    # Pressure correction
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    solve(A2, p1.vector(), b2, "gmres", prec)

    # Velocity correction
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "gmres", "default")

    plot(u1)
    
    u_e = Expression(('4*x[1]*(1.0 - x[1])', '0'), degree=2)
    u_e = interpolate(u_e, V)
    error = np.abs(u_e.vector() - u1.vector()).max()
    print('t = %.2f: error = %.3g' % (t, error))
    #print('max u:', u_.vector().max())

    # Move to next time step
    u0.assign(u1)
    t += dt
    print ("t =", t)

import matplotlib.pyplot as plt
plt.show()
