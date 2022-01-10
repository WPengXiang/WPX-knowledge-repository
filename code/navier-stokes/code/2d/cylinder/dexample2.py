import numpy as np
import matplotlib.pyplot as plt
import argparse

from fealpy.mesh.simple_mesh_generator import distmesh2d,unitcircledomainmesh
from fealpy.geometry import huniform
from fealpy.geometry import dcircle,drectangle,ddiff,dmin
from fealpy.geometry import DistDomain2d

from fealpy.mesh import DistMesh2d
from fealpy.mesh import PolygonMesh
from fealpy.mesh import TriangleMeshWithInfinityNode

parser = argparse.ArgumentParser(description='复杂二维区域distmesh网格生成示例')
parser.add_argument('--domain', default='circle',type=str,help = '区域类型, 默认是circle, 还可以选择:circle_h,square_h,adaptive_geo,superellipse')
parser.add_argument('--h0',default=0.1,type = float,help = '网格尺寸, 默认为0.1')

args = parser.parse_args()
domain = args.domain
h0 = 0.01
fd = lambda p:ddiff(drectangle(p,[-0.2,2,-0.2,0.21]),dcircle(p,[0.0,0.0],0.05))
fh = huniform


def fh(p):
    h = np.sqrt(p[:,0]*p[:,0]+p[:,1]*p[:,1])-0.048
    h[h>0.003] = 0.005
    return h


bbox = [-0.4,2.2,-0.22,0.23]
pfix = np.array([(-0.2,-0.2),(2,-0.2),(2,0.21),(-0.2,0.21)],dtype=np.float)
domain = DistDomain2d(fd,fh,bbox,pfix)
distmesh2d = DistMesh2d(domain,h0)
distmesh2d.run()


fig = plt.figure()
axes = fig.gca()
distmesh2d.mesh.add_plot(axes)
plt.show()
