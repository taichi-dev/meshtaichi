import taichi as ti
from mpm import *
from utils import *

@ti.data_oriented
class FEM():
    def __init__(self, metadata):
        self.rho = 1e3
        self.E = 7e4 # Young's modulus
        self.nu = 0.3 # Poisson's ratio: nu \in [0, 0.5)
        self.mu = self.E / (2 * (1 + self.nu))
        self.la = self.E * self.nu / ((1 + self.nu) * (1 -2 * self.nu))

        self.mesh_builder = ti.TetMesh()
        self.mesh_builder.verts.place({'x' : ti.math.vec3,
                  'v' : ti.math.vec3,
                  'color' : ti.i32,
                  'f' : ti.math.vec3,
                  'm' : ti.f32,
                  'C' : ti.math.mat3})
        self.mesh_builder.cells.place({'W' : ti.f32,
                                       'B' : ti.math.mat3})
        self.model = self.mesh_builder.build(metadata)

        self.grid = ti.root.pointer(ti.ijk, tuple([x // grid_block_size for x in n_grid]))
        self.block = self.grid.pointer(ti.ijk, grid_block_size // leaf_block_size)
        self.pid = ti.field(ti.i32)
        self.block.dynamic(ti.axes(3), 1024 * 1024, chunk_size=leaf_block_size**3 * 8).place(self.pid)

        self.precomputeTetMat(self.model)
        self.initColor(self.model)
    
    @ti.kernel
    def initColor(self, model : ti.template()):
        for v in model.verts:
            v.color = get_color(v.x[2] / 2.0) # Red -> Blue
    
    @ti.kernel
    def precomputeTetMat(self, model : ti.template()):
        for t in model.cells:
            M = ti.Matrix.zero(ti.f32, 3, 3)
            for ii in ti.static(range(3)):
                for jj in ti.static(range(3)):
                    M[jj, ii] = t.verts[ii].x[jj] - t.verts[3].x[jj]
            t.B = M.inverse()
            vol = -(1.0 / 6.0) * M.determinant()
            t.W = vol

            for ii in ti.static(range(4)):
                t.verts[ii].m += 0.25 * self.rho * vol

    @ti.func
    def PK1(self, u, l, F):
        U, sig, V = ti.svd(F, ti.f32)
        R = U @ V.transpose()
        J = F.determinant()
        return 2 * u * (F - R) + l * (J - 1) * J * F.inverse().transpose()

    @ti.kernel
    def computeForce(self, model : ti.template()):
        ti.mesh_local(model.verts.f, model.verts.x)
        for e in model.cells:
            Ds = ti.Matrix.zero(ti.f32, 3, 3)
            for ii in ti.static(range(3)):
                for jj in ti.static(range(3)):
                    Ds[jj, ii] = e.verts[ii].x[jj] - e.verts[3].x[jj]
            
            F = Ds @ e.B
            P = self.PK1(self.mu, self.la, F)
            H = -e.W * P @ e.B.transpose()
            for ii in ti.static(range(3)):
                fi = ti.Vector([H[0, ii], H[1, ii], H[2, ii]])
                e.verts[ii].f += fi
                e.verts[3].f += -fi

        for v in model.verts:
            v.f += ti.Vector([0, 0, -gravity]) * v.m