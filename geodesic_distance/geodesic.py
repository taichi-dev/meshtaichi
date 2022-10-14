import taichi as ti, argparse, numpy as np, pymeshlab
import meshtaichi_patcher as Patcher

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="models/yog.obj")
parser.add_argument('--test', action='store_true')
parser.add_argument('--arch', default='gpu')
parser.add_argument('--output', default='colored.obj')
args = parser.parse_args()

ti.init(arch=getattr(ti, args.arch))

count = ti.field(dtype=ti.i32, shape=())
error_tol = 1e-3
src = 233 # Point-0

mesh = Patcher.load_mesh(args.model, relations=['VV', 'VF', 'FV'])
mesh.verts.place({'x' : ti.math.vec3, 
                  'level' : ti.i32, 
                  'd' : ti.f32, 
                  'new_d' : ti.f32})
mesh.verts.x.from_numpy(mesh.get_position_as_numpy())

x = mesh.verts.x
d = mesh.verts.d
level = mesh.verts.level
new_d = mesh.verts.new_d

@ti.func
def update_step(v0, v1, v2):
    xs = [x[v1] - x[v0], x[v2] - x[v0]]
    X = ti.Matrix.cols(xs)
    t = ti.Vector([d[v1], d[v2]])
    l = ti.Vector([1.0] * 2)
    lt = l.transpose()
    q = ti.Matrix([[xs[i].dot(xs[j]) for j in ti.static(range(2))] for i in ti.static(range(2))])
    Q = q.inverse()
    p = ((lt @ Q @ t + ((lt @ Q @ t)**2 - lt @ Q @ l * (t.transpose() @ Q @ t - 1))**0.5) / (lt @ Q @ l))[0]
    n = X @ Q @ (t - 1)
    cond = Q @ X.transpose() @ n
    if cond[0] >= 0 or cond[1] >= 0 or d[v1] > 1e9 or d[v2] > 1e9:
        p = min(d[v1] + xs[0].norm(), d[v2] + xs[1].norm())
    return p

@ti.kernel
def ptp(l: ti.i32, r: ti.i32):
    for u in mesh.verts:
        if l <= u.level <= r:
            dist = u.d
            for f in u.faces:
                j = 0
                for i in ti.static(range(3)):
                    if f.verts[i].id == u.id:
                        j = i
                dist = min(dist, update_step(*[f.verts[(j + i) % 3].id for i in ti.static(range(3))]))
            u.new_d = dist
            if u.level == l and abs(dist - u.d) / u.d < error_tol:
                count[None] += 1

@ti.kernel
def get_level(l: ti.i32):
    for u in mesh.verts:
        if u.level == l:
            count[None] += 1
            for v in u.verts:
                if v.level == -1:
                    v.level = l + 1

level.fill(-1)
level[src] = 0

levels = []
for i in range(10000):
    count[None] = 0
    get_level(i)
    if count[None] == 0: break
    levels.append(count[None])

d.fill(1e10)
d[src] = 0
new_d.copy_from(d)

i = 1
for k in range(1, 10000):
    count[None] = 0
    ptp(i, k)
    if count[None] == levels[i]: i += 1
    if i >= len(levels): break
    d.copy_from(new_d)

arr = d.to_numpy()
print("mean =", arr.mean())

if args.test:
    assert int(arr.mean()) == 17
    exit(0)

def FF_to_rgb(x):
    r = (x & 0xFF0000) >> 16
    g = (x & 0x00FF00) >> 8
    b = (x & 0x0000FF) >> 0
    return (r / 255.0, g / 255.0, b / 255.0)

def sample(x):
    assert 0 <= x <= 1
    x = 1 - x
    palette = [
        0x750e13, 0xa2191f, 0xda1e28,
        0xfa4d46, 0xff8389, 0xffb3b8,
        0xffd7d9, 0xbae6ff,0x82cfff,
        0x33b1ff, 0x1192e8, 0x0072c3,
        0x00539a, 0x003a6d
    ]
    x *= len(palette)
    x0 = min(int(x), len(palette) - 1)
    x1 = min(x0 + 1, len(palette) - 1)
    c0 = FF_to_rgb(palette[x0])
    c1 = FF_to_rgb(palette[x1])
    p = x - x0
    return [p * c1[i] + (1 - p) * c0[i] for i in range(3)]

indices = ti.Vector.field(3, dtype=ti.u32, shape=len(mesh.faces))
@ti.kernel
def get_vertices():
    for f in mesh.faces:
        indices[f.id] = [f.verts[i].id for i in ti.static(range(3))]
get_vertices()

vert_colors = np.zeros(dtype=np.float32, shape=(len(mesh.verts), 4))
vert_colors[:, 3] = 1
ma = arr.max()
for i in range(vert_colors.shape[0]):
    vert_colors[i, :3] = sample(arr[i] / ma)

ms = pymeshlab.MeshSet()
ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.verts.x.to_numpy(), 
                           face_matrix=indices.to_numpy(), 
                           v_color_matrix=vert_colors))
ms.save_current_mesh(args.output)
