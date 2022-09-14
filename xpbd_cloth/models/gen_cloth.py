import numpy as np
from Delaunator import Delaunator
import time
import pymeshlab

start_x = 0.25
start_y = 0.25
z = 0.75
N = 100
cloth_num = 10
length = 0.50
dx = length / (N-1)

points = []
faces = []

for h_i in range(cloth_num):
  for h_j in range(cloth_num):
      sx = start_x + length * h_i
      sy = start_y + length * h_j

      start_time = time.process_time()
      np.random.seed(h_i*N+h_j)
      p = []
      for i in range(N):
        p.append([i * dx, 0])
        p.append([i * dx, length])
      for i in range(N-2):
        p.append([0, (i+1) * dx])
        p.append([length, (i+1) * dx])

      _points = np.random.random(size=(N**2-N*2-(N-2)*2,2)) * length
      _points = np.append(np.array(p), _points, axis=0)
      _triangles = Delaunator(_points).triangles
      triangulation_time = time.process_time() - start_time
      print(str(triangulation_time))

      _triangles = np.array(_triangles).reshape(-1, 3)
      
      off = len(points)
      for i in range(_points.shape[0]):
        points.append([_points[i][0]+sx, _points[i][1]+sy, z])
      for i in range(_triangles.shape[0]):
        faces.append([_triangles[i][0]+off, _triangles[i][1]+off, _triangles[i][2]+off])

ms = pymeshlab.MeshSet()
ms.add_mesh(pymeshlab.Mesh(points, faces))
ms.meshing_remove_unreferenced_vertices()
ms.save_current_mesh("models/cloth10x10.obj")
ms.clear()