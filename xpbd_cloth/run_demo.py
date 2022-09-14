import taichi as ti
import numpy as np
from render import *
from solver import PositionBasedDynamics
from sdf import *
import meshtaichi_patcher as Patcher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

ti.init(arch=ti.cpu if args.cpu else ti.cuda)

sdf = HangSdfModel(np.array([[0.25, 0.75, 0.75], 
                             [0.75, 0.75, 0.75], 
                             [0.25, 0.25, 0.75], 
                             [0.75, 0.25, 0.75]]))

@ti.func
def stretch_callback(id): return 1e-7

@ti.func
def bending_callback(id): return 1e-6

solver = PositionBasedDynamics(rest_pose = Patcher.mesh2meta("models/cloth.obj", relations=["ev", "fv", "ef"]),
                          sdf = sdf,
                          stretch_compliance_callback = stretch_callback,
                          bending_compliance_callback = bending_callback,
                          frame_dt = 1e-2, 
                          dt = 5e-4,
                          rest_iter= 5,
                          reorder_all=False)

initScene(position=(0.5, -0.3, 0.95), 
          lookat=(0.52, 0.52, 0.4), 
          show_window=True)

frame = 0
while True:
    solver.solve()
    renderScene(solver, frame)
    frame += 1