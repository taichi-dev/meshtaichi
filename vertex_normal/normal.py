import taichi as ti
import meshtaichi_patcher as Patcher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="./models/bunny.obj")
args = parser.parse_args()

ti.init(arch=ti.gpu)

mesh = ti.TriMesh()
mesh.verts.place({'x' : ti.math.vec3, 
                  'normal' : ti.math.vec3})

model = mesh.build(Patcher.mesh2meta(args.model, relations=['fv']))

@ti.kernel
def vertex_normal():
    ti.mesh_local(model.verts.x, model.verts.normal)
    for f in model.faces:
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

n = model.verts.normal.to_numpy()
print(n.shape, n.sum(), (n**2).sum())
