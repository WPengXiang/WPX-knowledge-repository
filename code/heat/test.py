import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory as MF


points = np.array([[0.0, 0.0], [2.2, 0.0], [2.2, 0.41], [0.0, 0.41]],
        dtype=np.float64)
facets = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int_)


p, f = MF.circle_interval_mesh([0.2, 0.2], 0.1, 0.01) 

points = np.append(points, p, axis=0)
facets = np.append(facets, f+4, axis=0)


fm = np.array([0, 1, 2, 3])

mesh = MF.meshpy2d(points, facets, 0.01, hole_points=[[0.2, 0.2]], facet_markers=fm, meshtype='tri')

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()

