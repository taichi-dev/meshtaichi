import os
import taichi as ti
import meshtaichi_patcher as Patcher
ti.init(arch=ti.cuda, device_memory_GB=4.0, packed=True)
from fem import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default="./results/",
                        help='Output Path')
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    return args


args = parse_args()

os.makedirs(args.output + "/armadillo", exist_ok=True)
os.makedirs(args.output + "/particles", exist_ok=True)

model_size = 0.10

fems, meshes = [], []

def transform(verts, scale, offset): return verts / max(verts.max(0) - verts.min(0)) * scale + offset
def init(x, y, i):
    mesh = Patcher.load_mesh("./models/armadillo0/armadillo0.1.node")
    mesh[0] = transform(mesh[0], model_size, [x, y, 0.05 + (model_size / 2 + 0.012) * i])
    meshes.append(mesh)

for i in range(30):
    init(0.5, 0.5, i)
    init(0.77, 0.22, i)
    init(0.22, 0.77, i)
    init(0.22, 0.27, i)
    init(0.77, 0.77, i)

fems.append(FEM(Patcher.mesh2meta(meshes)))

window = ti.ui.Window("MPM", (1920, 1080), show_window=False)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(-1.0, -0.8, 0.80)
camera.up(0, 0, 1.0)
camera.lookat(0.5, 0.5, 0.37)
camera.fov(39)

for frame in range(100):
    print(f"frame: {frame}")

    solve(1, fems)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.7, 0.2, 1.7), color=(0.55, 0.55, 0.55))
    scene.point_light(pos=(0.2, 0.7, 1.7), color=(0.55, 0.55, 0.55))

    for i in range(1):
        scene.particles(fems[i].model.verts.x, 1e-3, color = (0.5, 0.5, 0.5))

    canvas.set_background_color(color = (1.0, 1.0, 1.0))
    canvas.scene(scene)
    window.save_image(f"results/armadillo/{frame:06d}.jpg")
    write_particles(1, fems, f'results/particles/{frame:05d}.npz')
