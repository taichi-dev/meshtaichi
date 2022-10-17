import taichi as ti
import numpy as np
from render import *
from solver import PositionBasedDynamics
from sdf import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='gpu')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

ti.init(arch=getattr(ti, args.arch))

sdf = HangSdfModel(np.array([[0.25, 0.75, 0.75], 
                             [0.75, 0.75, 0.75], 
                             [0.25, 0.25, 0.75], 
                             [0.75, 0.25, 0.75]]))

@ti.func
def stretch_callback(id): return 1e-7

@ti.func
def bending_callback(id): return 1e-6

solver = PositionBasedDynamics(rest_pose = "models/cloth.obj",
                          sdf = sdf,
                          stretch_compliance_callback = stretch_callback,
                          bending_compliance_callback = bending_callback,
                          frame_dt = 1e-2, 
                          dt = 5e-4,
                          rest_iter= 5,
                          reorder_all=False)

if args.test:
    for frame in range(100):
        solver.solve()
    arr = solver.mesh.verts.x.to_numpy()
    assert '%.3f' % arr.mean() == '0.579'
    assert '%.3f' % (arr**2).mean() == '0.363'
    exit(0)

initScene(position=(0.5, -0.3, 0.95), 
          lookat=(0.52, 0.52, 0.4), 
          show_window=True)

frame = 0
running = True
while running:
    solver.solve()
    running = renderScene(solver, frame)
    frame += 1