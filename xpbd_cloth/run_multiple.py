import taichi as ti
import numpy as np
import time
from render import *
from solver import PositionBasedDynamics
from sdf import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default="./results/",
                        help='Output Path')
    parser.add_argument('--arch', default='gpu')
    
    args = parser.parse_args()
    return args


args = parse_args()

ti.init(arch=getattr(ti, args.arch), device_memory_GB=4)

N = 10
points = []
for i in range(N+1):
    for j in range(N+1):
        points.append([0.25 + 0.5 * i, 0.25 + 0.5 * j, 0.75])

sdf = HangSdfModel(np.array(points))

@ti.func
def stretch_callback(id):
    block = id // (100 * 100)
    stretch_compliance = 1e-8 * (1.668**float(block % N))
    return stretch_compliance

@ti.func
def bending_callback(id):
    block = id // (100 * 100)
    x = block // N

    bending_compliance = 0.0
    if x == 0: bending_compliance = 1e-4
    if x == 1: bending_compliance = 1e-3
    if x == 2: bending_compliance = 1e-2
    if x == 3: bending_compliance = 1e-1
    if x == 4: bending_compliance = 1
    if x == 5: bending_compliance = 0.5e1
    if x == 6: bending_compliance = 1e1
    if x == 7: bending_compliance = 0.5e2
    if x == 8: bending_compliance = 1e2
    if x == 9: bending_compliance = 0.5e3
    return bending_compliance

solver = PositionBasedDynamics(rest_pose = "models/cloth10x10.obj",
                          sdf = sdf,
                          stretch_compliance_callback = stretch_callback,
                          bending_compliance_callback = bending_callback,
                          frame_dt = 1 / 24, 
                          dt = 2.5e-4,
                          rest_iter = 5,
                          reorder_all=False)

for frame in range(240):
    start_time = time.time()
    solver.solve()
    end_time = time.time()
    print(f"frame: {frame}, time={(end_time - start_time):03f}s")
    exportScene(solver, frame, args.output)