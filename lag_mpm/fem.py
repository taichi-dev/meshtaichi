import taichi as ti
from mpm import *
from utils import *

@ti.data_oriented
class FEM():
    def __init__(self, mesh):
        self.rho = 1e3
        self.E = 7e4 # Young's modulus
        self.nu = 0.3 # Poisson's ratio: nu \in [0, 0.5)
        self.mu = self.E / (2 * (1 + self.nu))
        self.la = self.E * self.nu / ((1 + self.nu) * (1 -2 * self.nu))

        self.x = ti.Vector.field(3, ti.f32, shape=len(mesh.verts))
        self.v = ti.Vector.field(3, ti.f32, shape=len(mesh.verts))
        self.color = ti.field(ti.i32, shape=len(mesh.verts))
        self.f = ti.Vector.field(3, ti.f32, shape=len(mesh.verts))
        self.m = ti.field(ti.f32, shape=len(mesh.verts))
        self.C = ti.Matrix.field(3, 3, ti.f32, shape=len(mesh.verts))

        self.W = ti.field(ti.f32, shape=len(mesh.cells))
        self.B = ti.Matrix.field(3, 3, ti.f32, shape=len(mesh.cells))

        self.mesh = mesh
        self.x.from_numpy(mesh.get_position_as_numpy())

        self.grid = ti.root.pointer(ti.ijk, tuple([x // grid_block_size for x in n_grid]))
        self.block = self.grid.pointer(ti.ijk, grid_block_size // leaf_block_size)
        self.pid = ti.field(ti.i32)
        self.block.dynamic(ti.axes(3), 1024 * 1024, chunk_size=leaf_block_size**3 * 8).place(self.pid)

        self.precomputeTetMat(self.mesh)
        self.initColor(self.mesh)
    
    @ti.kernel
    def initColor(self, mesh : ti.template()):
        for v in mesh.verts:
            self.color[v.id] = get_color(self.x[v.id][2] / 2.0) # Red -> Blue
    
    @ti.kernel
    def precomputeTetMat(self, mesh : ti.template()):
        for t in mesh.cells:
            M = ti.Matrix.zero(ti.f32, 3, 3)
            for ii in ti.static(range(3)):
                for jj in ti.static(range(3)):
                    M[jj, ii] = self.x[t.verts[ii].id][jj] - self.x[t.verts[3].id][jj]
            self.B[t.id] = M.inverse()
            vol = -(1.0 / 6.0) * M.determinant()
            self.W[t.id] = vol

            for ii in ti.static(range(4)):
                self.m[t.verts[ii].id] += 0.25 * self.rho * vol

    @ti.func
    def PK1(self, u, l, F):
        U, sig, V = ti.svd(F, ti.f32)
        R = U @ V.transpose()
        J = F.determinant()
        return 2 * u * (F - R) + l * (J - 1) * J * F.inverse().transpose()

    @ti.kernel
    def computeForce(self, mesh : ti.template()):
        ti.mesh_local(self.f, self.x)
        for e in mesh.cells:
            Ds = ti.Matrix.zero(ti.f32, 3, 3)
            for ii in ti.static(range(3)):
                for jj in ti.static(range(3)):
                    Ds[jj, ii] = self.x[e.verts[ii].id][jj] - self.x[e.verts[3].id][jj]
            
            F = Ds @ self.B[e.id]
            P = self.PK1(self.mu, self.la, F)
            H = -self.W[e.id] * P @ self.B[e.id].transpose()
            for ii in ti.static(range(3)):
                fi = ti.Vector([H[0, ii], H[1, ii], H[2, ii]])
                self.f[e.verts[ii].id] += fi
                self.f[e.verts[3].id] += -fi

        for v in mesh.verts:
            self.f[v.id] += ti.Vector([0, 0, -gravity]) * self.m[v.id]
