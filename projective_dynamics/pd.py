import taichi as ti
import meshtaichi_patcher as Patcher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="models/deer.1.node")
parser.add_argument('--arch', default='gpu')
args = parser.parse_args()

ti.init(arch=getattr(ti, args.arch), dynamic_index=True)

E, nu = 5e5, 0.0
mu, la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
density = 100.0
dt = 2e-2

mesh = Patcher.load_mesh(args.model, relations=["CE", "CV", "EV"])
mesh.verts.place({'x' :         ti.math.vec3, 
                  'v' :         ti.math.vec3,
                  'mul_ans' :   ti.math.vec3,
                  'f' :         ti.math.vec3,
                  'hessian':    ti.f32,
                  'm' :         ti.f32})

mesh.edges.place({'hessian':    ti.f32})
mesh.cells.place({'B' :         ti.math.mat3, 
                  'W' :         ti.f32})
mesh.verts.x.from_numpy(mesh.get_position_as_numpy())

x = mesh.verts.x
v = mesh.verts.v
f = mesh.verts.f
m = mesh.verts.m
mul_ans = mesh.verts.mul_ans

@ti.func
def ssvd(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)): U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)): V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V

@ti.kernel
def get_force():
    ti.mesh_local(mesh.verts.f)
    for c in mesh.cells:
        Ds = ti.Matrix.cols([c.verts[i].x - c.verts[3].x for i in ti.static(range(3))])
        F = Ds @ c.B
        U, sig, V = ssvd(F)
        P = 2 * mu * (F - U @ V.transpose())
        H = -c.W * P @ c.B.transpose()
        for i in ti.static(range(3)):
            Hx = ti.Vector([H[j, i] for j in ti.static(range(3))])
            c.verts[i].f += Hx
            c.verts[3].f -= Hx

@ti.kernel
def get_matrix():
    for c in mesh.cells:
        hes = ti.Matrix.zero(ti.f32, 4, 4)
        for u in range(4):
            dD = ti.Matrix.zero(ti.f32, 3, 3)
            if u == 3:
                for j in ti.static(range(3)):
                    dD[0, j] = -1
            else: dD[0, u] = 1
            dF = dD @ c.B
            dP = 2.0 * mu * dF
            dH = -c.W * dP @ c.B.transpose()
            for i in ti.static(range(3)):
                for j in ti.static(range(1)):
                    hes[i, u] = -dt**2 * dH[j, i]
                    hes[3, u] += dt**2 * dH[j, i]

        for z in range(c.edges.size):
            e = c.edges[z]
            u = ti.Vector([0, 0])
            for i in ti.static(range(2)):
                for j in ti.static(range(4)):
                    if e.verts[i].id == c.verts[j].id:
                        u[i] = j
            e.hessian += hes[u[0], u[1]]

        for z in range(c.verts.size):
            v = c.verts[z]
            v.hessian += hes[z, z]

@ti.kernel
def mul_kernel(ret: ti.template(), vel: ti.template()):
    for v in mesh.verts:
        ret[v.id] = vel[v.id] * m[v.id] + v.hessian * vel[v.id]

    ti.mesh_local(ret, vel)
    for e in mesh.edges:
        u = e.verts[0].id
        v = e.verts[1].id
        ret[u] += e.hessian * vel[v]
        ret[v] += e.hessian * vel[u]

def mul(x):
    mul_kernel(mul_ans, x)
    return mul_ans

@ti.kernel
def add(ans: ti.template(), a: ti.template(), k: ti.f32, b: ti.template()):
    for i in ans:
        ans[i] = a[i] + k * b[i]
    
@ti.kernel
def dot(a: ti.template(), b: ti.template()) -> ti.f32:
    ans = 0.0
    ti.loop_config(block_dim=32)
    for i in a: ans += a[i].dot(b[i])
    return ans

b = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.verts))
r0 = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.verts))
p0 = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.verts))
y = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.verts))
x0 = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.verts))

@ti.kernel
def get_b():
    for i in b:
        b[i] = m[i] * (x[i] - y[i]) - (dt ** 2) * f[i]

@ti.kernel
def update_v():
    for i in v:
        v[i] = (x[i] - x0[i]) / dt

def newton():
    x0.copy_from(x)
    add(y, x, dt, v)
    x.copy_from(y)
    for i in range(5): # PD iterations
        f.fill(0.0)
        get_force()
        get_b()

        v.fill(0.0) # reuse `v` here as dx
        r0.copy_from(b)

        d = p0
        d.copy_from(r0)
        r_2 = dot(r0, r0)
        n_iter = 30 # CG iterations
        epsilon = 1e-5
        r_2_init = r_2
        r_2_new = r_2
        for iter in range(n_iter):
            q = mul(d)
            alpha = r_2_new / dot(d, q)
            add(v, v, alpha, d) 
            add(r0, r0, -alpha, q)
            r_2 = r_2_new
            r_2_new = dot(r0, r0)
            if r_2_new <= r_2_init * epsilon**2: break
            beta = r_2_new / r_2
            add(d, r0, beta, d)

        add(x, x, -1, v)

    update_v()

indices = ti.field(ti.u32, shape = len(mesh.cells) * 4 * 3)
@ti.kernel
def init():
    for c in mesh.cells:
        Ds = ti.Matrix.cols([c.verts[i].x - c.verts[3].x for i in ti.static(range(3))])
        c.B = Ds.inverse()
        c.W = ti.abs(Ds.determinant()) / 6
        ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
        for i in ti.static(range(4)):
            c.verts[i].m += density * c.W / 4
            for j in ti.static(range(3)):
                indices[(c.id * 4 + i) * 3 + j] = c.verts[ind[i][j]].id
    for u in mesh.verts:
        for i in ti.static(range(3)):
            u.x[i] = ti.random()

init()
get_matrix()

window = ti.ui.Window("Projective Dynamics", (1024, 768))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(1, 1.5, 0)
camera.up(0, 1, 0)
camera.lookat(0, 0, 0)
camera.fov(75)

while window.running:
    newton()
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.mesh(mesh.verts.x, indices, color = (0.5, 0.5, 0.5))

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    scene.ambient_light((1, 1, 1))

    canvas.scene(scene)

    window.show()
