import numpy as np
from ..quadrature import TetrahedronQuadrature

from .Mesh3d import Mesh3d, Mesh3dDataStructure

from .TetrahedronMesh import TetrahedronMesh

from fealpy.mesh.multi_index import multi_index_matrix1d
from fealpy.mesh.multi_index import multi_index_matrix2d
from fealpy.mesh.multi_index import multi_index_matrix3d

class LagrangeTetrahedronMesh(Mesh3d):
    def __init__(self, node, cell, p=1, surface=None):

        mesh = TetrahedronMesh(node, cell) 
        dof = LagrangeTetrahedronDof3d(mesh, p)

        self.p = p
        self.node = dof.interpolation_points()

        if surface is not None:
            self.node, _ = surface.project(self.node)
   
        self.ds = LagrangeTetrahedronMeshDataStructure(dof)

        self.TD = 3
        self.GD = node.shape[1]

        self.meshtype = 'ltet'
        self.ftype = mesh.ftype
        self.itype = mesh.itype
        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}

        self.multi_index_matrix = [multi_index_matrix1d, multi_index_matrix2d, multi_index_matrix3d]


    def vtk_cell_type(self, etype='cell'):
        """

        Notes
        -----
            返回网格单元对应的 vtk类型。
        """
        if etype in {'cell', 2}:
            VTK_LAGRANGE_TETRAHEDRON = 71
            return VTK_LAGRANGE_TETRAHEDRON 
        elif etype in {'face', 'edge', 1}:
            VTK_LAGRANGE_CURVE = 68
            return VTK_LAGRANGE_CURVE

    def to_vtk(self, etype='cell', index=np.s_[:], fname=None):
        """
        Parameters
        ----------

        Notes
        -----
        把网格转化为 VTK 的格式
        """
        from .vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = np.concatenate((node, np.zeros((node.shape[0], 1), dtype=self.ftype)), axis=1)

        cell = self.entity(etype)[index]
        cellType = self.vtk_cell_type(etype)
        index = vtk_cell_index(self.p, cellType)
        NV = cell.shape[-1]

        cell = np.r_['1', np.zeros((len(cell), 1), dtype=cell.dtype), cell[:, index]]
        cell[:, 0] = NV

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC 
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)

     def bc_to_point(self, bc, etype='cell', index=np.s_[:]):
        node = self.node
        entity = self.entity(etype) # default  cell
        phi = self.lagrange_basis(bc, etype=etype) # (NQ, 1, ldof)
        p = np.einsum('ijk, jkn->ijn', phi, node[entity[index]])
        return p

    def lagrange_basis(self, bc, etype='cell'):
        p = self.p   # the degree of lagrange basis function

        if etype in {'cell', 3}:
            TD = 3
        elif etype in {'edge', 2}:
            TD = 2
        elif etype in {'face' , 1}:
            TD = 1

        multiIndex = self.multi_index_matrix[TD-1](p)

        c = np.arange(1, p+1, dtype=np.int)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., multiIndex, idx], axis=-1)
        return phi[..., np.newaxis, :] # (..., 1, ldof)

    def print(self):

        """

        Notes
        -----
            打印网格信息， 用于调试。
        """

        node = self.entity('node')
        print('node:')
        for i, v in enumerate(node):
            print(i, ": ", v)

        edge = self.entity('edge')
        print('edge:')
        for i, e in enumerate(edge):
            print(i, ": ", e)

        cell = self.entity('cell')
        print('cell:')
        for i, c in enumerate(cell):
            print(i, ": ", c)

        edge2cell = self.ds.edge_to_cell()
        print('edge2cell:')
        for i, ec in enumerate(edge2cell):
            print(i, ": ", ec)



class LagrangeTetrahedronMeshDataStructure(Mesh3dDataStructure):
    def __init__(self, dof):
        self.cell = dof.cell_to_dof()
        self.edge = dof.edge_to_dof()
        self.edge2cell = dof.mesh.ds.edge_to_cell()

        self.NN = dof.number_of_global_dofs() 
        self.NE = len(self.edge)
        self.NC = len(self.cell)

        self.V = dof.number_of_local_dofs() 
        self.E = 6

class LagrangeTetrahedronDof3d():
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = multi_index_matrix3d(p)
        self.multiIndex2d = multi_index_matrix2d(p)
        self.cell2dof = self.cell_to_dof()

    def is_on_node_local_dof(self):
        p = self.p
        isNodeDof = np.sum(self.multiIndex == p, axis=-1) == 1
        return isNodeDof

    def is_on_edge_local_dof(self):
        p =self.p
        ldof = self.number_of_local_dofs()
        localEdge = self.mesh.ds.localEdge
        isEdgeDof = np.zeros((ldof, 6), dtype=np.bool)
        for i in range(6):
            isEdgeDof[:, i] = (self.multiIndex[:, localEdge[-(i+1), 0]] == 0) & (self.multiIndex[:, localEdge[-(i+1), 1]] == 0 )
        return isEdgeDof

    def is_on_face_local_dof(self):
        p = self.p
        ldof = self.number_of_local_dofs()
        isFaceDof = (self.multiIndex == 0)
        return isFaceDof

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh

        N = mesh.number_of_nodes()
        NE = mesh.number_of_edges()

        base = N
        edge = mesh.ds.edge
        edge2dof = np.zeros((NE, p+1), dtype=np.int)
        edge2dof[:, [0, -1]] = edge
        if p > 1:
            edge2dof[:,1:-1] = base + np.arange(NE*(p-1)).reshape(NE, p-1)
        return edge2dof

    def face_to_dof(self):
        p = self.p
        fdof = (p+1)*(p+2)//2

        edgeIdx = np.zeros((2, p+1), dtype=np.int)
        edgeIdx[0, :] = range(p+1)
        edgeIdx[1, :] = edgeIdx[0, -1::-1]

        mesh = self.mesh

        N = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()

        face = mesh.ds.face
        edge = mesh.ds.edge
        face2edge = mesh.ds.face_to_edge()

        edge2dof = self.edge_to_dof()

        face2dof = np.zeros((NF, fdof), dtype=np.int)
        faceIdx = self.multiIndex2d
        isEdgeDof = (faceIdx == 0)

        fe = np.array([1, 0, 0])
        for i in range(3):
            I = np.ones(NF, dtype=np.int)
            sign = (face[:, fe[i]] == edge[face2edge[:, i], 0])
            I[sign] = 0
            face2dof[:, isEdgeDof[:, i]] = edge2dof[face2edge[:, [i]], edgeIdx[I]]

        if p > 2:
            base = N + (p-1)*NE
            isInFaceDof = ~(isEdgeDof[:, 0] | isEdgeDof[:, 1] | isEdgeDof[:, 2])
            fidof = fdof - 3*p
            face2dof[:, isInFaceDof] = base + np.arange(NF*fidof).reshape(NF, fidof)

        return face2dof

    def boundary_dof(self, threshold=None):

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        face2dof = self.face_to_dof()
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool)
        isBdDof[face2dof[index]] = True
        return isBdDof

    def is_boundary_dof(self, threshold=None):

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        face2dof = self.face_to_dof()
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool)
        isBdDof[face2dof[index]] = True
        return isBdDof

    def cell_to_dof(self):
        p = self.p
        fdof = (p+1)*(p+2)//2
        ldof = self.number_of_local_dofs()

        localFace = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])

        mesh = self.mesh

        N = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        face = mesh.ds.face
        cell = mesh.ds.cell

        cell2face = mesh.ds.cell_to_face()

        cell2dof = np.zeros((NC, ldof), dtype=np.int)

        face2dof = self.face_to_dof()
        isFaceDof = self.is_on_face_local_dof()
        faceIdx = self.multiIndex2d.T

        for i in range(4):
            fi = face[cell2face[:, i]]
            fj = cell[:, localFace[i]]
            idxj = np.argsort(fj, axis=1)
            idxjr = np.argsort(idxj, axis=1)
            idxi = np.argsort(fi, axis=1)
            idx = idxi[np.arange(NC).reshape(-1, 1), idxjr]
            isCase0 = (np.sum(idx == np.array([1, 2, 0]), axis=1) == 3)
            isCase1 = (np.sum(idx == np.array([2, 0, 1]), axis=1) == 3)
            idx[isCase0, :] = [2, 0, 1]
            idx[isCase1, :] = [1, 2, 0]
            k = faceIdx[idx[:, 1], :] + faceIdx[idx[:, 2], :]
            a = k*(k+1)//2 + faceIdx[idx[:, 2], :]
            cell2dof[:, isFaceDof[:, i]] = face2dof[cell2face[:, [i]], a]

        if p > 3:
            base = N + (p-1)*NE + (fdof - 3*p)*NF
            idof = ldof - 4 - 6*(p - 1) - 4*(fdof - 3*p)
            isInCellDof = ~(isFaceDof[:, 0] | isFaceDof[:, 1] | isFaceDof[:, 2] | isFaceDof[:, 3])
            cell2dof[:, isInCellDof] = base + np.arange(NC*idof).reshape(NC, idof)

        return cell2dof

    def cell_to_dof_1(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')

        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs()
        cell2dof = np.zeros((NC, ldof), dtype=np.int)

        idx = np.array([
            0,
            ldof - (p+1)*(p+2)//2 - 1,
            ldof - p -1,
            ldof - 1], dtype=np.int)

        cell2dof[:, idx] = cell

        if p == 1:
            return cell2dof
        if p == 2:
            cell2edge = mesh.ds.cell_to_edge()
            idx = np.array([1, 2, 3, 5, 6, 8], dtype=np.int)
            cel2dof[:, idx] = cell2edge + NN
            return cell2dof
        else:
            w = self.multiIndex

            flag = (w != 0)
            isCellIDof = (flag.sum(axis=-1) == 4)
            isNodeDof = (flag.sum(axis=-1) == 1)
            isNewBdDof = ~(isCellIDof | isNodeDof)

            nd = isNewBdDof.sum()
            ps = np.einsum('im, km->ikm', cell + NN + NC, w[isNewBdDof])
            ps.sort()
            _, i0, j = np.unique(
                    ps.reshape(-1, 4),
                    return_index=True,
                    return_inverse=True,
                    axis=0)
            cell2dof[:, isNewBdDof] = j.reshape(-1, nd) + NN

            NB = len(i0)
            nd = isCellIDof.sum()
            if nd > 0:
                cell2dof[:, isCellIDof] = NB + NN + nd*np.arange(NC).reshape(-1, 1) \
                        + np.arange(nd)
            return cell2dof

    def cell_to_dof_2(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')

        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs()
        cell2dof = np.zeros((NC, ldof), dtype=np.int)

        idx = np.array([
            0,
            ldof - (p+1)*(p+2)//2 - 1,
            ldof - p -1,
            ldof - 1], dtype=np.int)

        cell2dof[:, idx] = cell

        if p == 1:
            return cell2dof
        if p == 2:
            cell2edge = mesh.ds.cell_to_edge()
            idx = np.array([1, 2, 3, 5, 6, 8], dtype=np.int)
            cel2dof[:, idx] = cell2edge + NN
            return cell2dof
        else:
            w = self.multiIndex
            flag = (w != 0)
            isCellIDof = (flag.sum(axis=-1) == 4)
            nd = isCellIDof.sum()
            if nd > 0:
                cell2dof[:, isCellIDof] = (
                        NB + NN +
                        nd*np.arange(NC).reshape(-1, 1) + np.arange(nd)
                    )

            isNodeDof = (flag.sum(axis=-1) == 1)
            isNewBdDof = ~(isCellIDof | isNodeDof)
            # 边内部自由度编码
            m1 = multi_index_matrix(p, 1)
            edge = mesh.entity('edge')
            # 面内部自由度编码 
            m2 = multi_index_matrix(p, 2)
            face = mesh.entity('face')
            # 单元内部自由编码
            m3 = multi_index_matrix(p, 3)
            pass


    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        N = mesh.number_of_nodes()
        gdof = N

        if p > 1:
            NE = mesh.number_of_edges()
            edof = p - 1
            gdof += edof*NE

        if p > 2:
            NF = mesh.number_of_faces()
            fdof = (p+1)*(p+2)//2 - 3*p
            gdof += fdof*NF

        if p > 3:
            NC = mesh.number_of_cells()
            ldof = self.number_of_local_dofs()
            cdof = ldof - 6*edof - 4*fdof - 4
            gdof += cdof*NC

        return gdof


    def number_of_local_dofs(self, doftype='cell'):
        p = self.p
        if doftype in {'cell', 3}:
            return (p+1)*(p+2)*(p+3)//6
        elif doftype in {'face', 2}:
            return (p+1)*(p+2)//2
        elif doftype in {'edge', 1}:
            return p + 1
        elif doftype in {'node', 0}:
            return 1

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell
        node = mesh.node

        if p == 1:
            return node

        N = node.shape[0]
        dim = node.shape[1]
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()
        ipoint = np.zeros((gdof, dim), dtype=np.float)
        ipoint[:N, :] = node
        if p > 1:
            NE = mesh.number_of_edges()
            edge = mesh.ds.edge
            w = np.zeros((p-1,2), dtype=np.float)
            w[:,0] = np.arange(p-1, 0, -1)/p
            w[:,1] = w[-1::-1, 0]
            ipoint[N:N+(p-1)*NE, :] = np.einsum('ij, kj...->ki...', w, node[edge,:]).reshape(-1, dim) 
        if p > 2:
            NF = mesh.number_of_faces()
            fidof = (p+1)*(p+2)//2 - 3*p
            face = mesh.ds.face
            isEdgeDof = (self.multiIndex2d == 0)
            isInFaceDof = ~(isEdgeDof[:, 0] | isEdgeDof[:, 1] | isEdgeDof[:, 2])
            w = self.multiIndex2d[isInFaceDof, :]/p
            ipoint[N+(p-1)*NE:N+(p-1)*NE+fidof*NF, :] = np.einsum('ij, kj...->ki...', w, node[face,:]).reshape(-1, dim)

        if p > 3:
            isFaceDof = self.is_on_face_local_dof()
            isInCellDof = ~(isFaceDof[:,0] | isFaceDof[:,1] | isFaceDof[:,2] | isFaceDof[:, 3])
            w = self.multiIndex[isInCellDof, :]/p
            ipoint[N+(p-1)*NE+fidof*NF:, :] = np.einsum('ij, kj...->ki...', w,
                    node[cell,:]).reshape(-1, dim)
        return ipoint


