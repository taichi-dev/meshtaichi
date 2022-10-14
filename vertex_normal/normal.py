import taichi as ti
import meshtaichi_patcher as Patcher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="./models/bunny.obj")
parser.add_argument('--test', action='store_true')
parser.add_argument('--arch', default='gpu')
args = parser.parse_args()

ti.init(arch=getattr(ti, args.arch))

mesh = Patcher.load_mesh(args.model, relations=['FV'])
mesh.verts.place({'x' : ti.math.vec3, 
                  'normal' : ti.math.vec3})
mesh.verts.x.from_numpy(mesh.get_position_as_numpy())

@ti.kernel
def vertex_normal():
    ti.mesh_local(mesh.verts.x, mesh.verts.normal)
    for f in mesh.faces:
        v0 = f.verts[0]
        v1 = f.verts[1]
        v2 = f.verts[2]

        n = (v0.x - v2.x).cross(v1.x - v2.x)
        l = [(v0.x - v1.x).norm_sqr(),
             (v1.x - v2.x).norm_sqr(),
             (v2.x - v0.x).norm_sqr()]

        for i in ti.static(range(3)):
            f.verts[i].normal += n / (l[i] + l[(i + 2) % 3])

vertex_normal()

n = mesh.verts.normal.to_numpy()
if args.test:
    assert int(n.sum()) == 4617
    assert int((n**2).sum() + 0.5) == 173790
else:
    print(n.shape, n.sum(), (n**2).sum())
