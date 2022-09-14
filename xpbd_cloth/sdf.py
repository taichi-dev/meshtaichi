import taichi as ti
import numpy as np

PI = 3.14159265

@ti.data_oriented
class SdfModel:
    def __init__(self, fixed):
        self.vel = ti.Vector.field(3, float, shape=())
        self.fixed = ti.field(int, shape=())
        self.fixed[None] = fixed

    @ti.func
    def check(self, pos, vel):
        phi = self.dist(pos)
        inside = False
        dotnv = 0.0
        diff_vel = ti.Vector.zero(ti.f32, 3)
        n = ti.Vector.zero(ti.f32, 3)
        if phi < 0.0:
            n = self.normal(pos)
            diff_vel = self.vel[None] - vel
            dotnv = n.dot(diff_vel)
            if dotnv > 0.0 or self.fixed[None]:
                inside = True
        
        return self.fixed[None], inside, dotnv, diff_vel, n

@ti.data_oriented
class HangSdfModel(SdfModel):
    def __init__(self, pos):
        super().__init__(fixed=True)
        self.sphere_pos = ti.Vector.field(3, float, shape=pos.shape[0])
        self.sphere_pos.from_numpy(pos.astype(np.float32))
        self.sphere_radius = 0.013

    @ti.func
    def dist(self, pos): # Function computing the signed distance field
        dist = 1e5
        for i in range(self.sphere_pos.shape[0]):
            dist = min((pos - self.sphere_pos[i]).norm(1e-9) - self.sphere_radius, dist)
        return dist

    @ti.func
    def normal(self, pos): # Function computing the gradient of signed distance field
        dist = 1e5
        normal = ti.Vector.zero(ti.f32, 3)
        for i in range(self.sphere_pos.shape[0]):
            dist0 = (pos - self.sphere_pos[i]).norm(1e-9) - self.sphere_radius
            if dist0 < dist:
                dist = dist0
                normal = (pos - self.sphere_pos[0]).normalized(1e-9)
        return normal
    
    def render(self, scene):
        scene.particles(self.sphere_pos, self.sphere_radius, color = (1, 0, 0))