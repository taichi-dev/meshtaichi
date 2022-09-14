import taichi as ti

@ti.data_oriented
class PositionBasedDynamics:
    def __init__(self,
    rest_pose,
    sdf,
    bending_compliance_callback,
    stretch_compliance_callback,
    scale=1.0,
    offset=(0.0, 0.0, 0.0),
    frame_dt=1e-2,
    dt=1e-2,
    rest_iter=1000,
    XPBD=True,
    reorder_all=False,
    block_size=128,
    stretch_relaxation=0.3,
    bending_relaxation=0.2):

        # constants
        self.time = 0.0
        self.frame_dt = frame_dt
        self.dt = dt
        self.gravity = ti.Vector.field(3, ti.f32, shape=())
        self.gravity[None] = ti.Vector([0.0, 0.0, -15.0])

        # Callback functions used to define stiffness
        self.bending_compliance_callback = bending_compliance_callback
        self.stretch_compliance_callback = stretch_compliance_callback
        
        # pbd parameters
        self.rest_iter = rest_iter
        self.stretch_relaxation = stretch_relaxation
        self.bending_relaxation = bending_relaxation

        # profiling parameter(s)
        self.block_size = block_size

        self.mass = 1.0

        # Xpbd parameters
        self.XPBD = XPBD

        # mesh
        self.mesh_builder = ti.Mesh.Tri()
        self.mesh_builder.verts.place({
            'x' : ti.math.vec3,
            'new_x' : ti.math.vec3,
            'v' : ti.math.vec3,
            'invM' : ti.f32,
            'dp' : ti.math.vec3
        }, reorder = reorder_all)
        self.mesh_builder.edges.place({
            'rest_len' : ti.f32,
            'la_s' : ti.f32,    # total Lagrange multiplier for stretch constraint (at the current iteration)
            'la_b' : ti.f32, # multiplier for bending constraint
            # Ref: http://blog.mmacklin.com/2016/10/12/xpbd-slides-and-stiffness/
            'stretch_compliance' : ti.f32,
            'bending_compliance' : ti.f32
        }, reorder = reorder_all)
        self.model = self.mesh_builder.build(rest_pose)
        self.indices = ti.field(dtype = ti.u32, shape = len(self.model.faces) * 3)

        self.initIndices()

        # collision
        self.sdf = sdf
        self.initialize(scale, offset)

    @ti.kernel
    def initialize(self, scale : ti.f32, offset : ti.template()):
        for v in self.model.verts:
            v.x = v.x * scale + ti.Vector(offset)
            v.invM = self.mass
            fixed, inside, dotnv, diff_vel, n = self.sdf.check(v.x, v.v)
            if inside:
                v.invM = 0.0
        
        for e in self.model.edges:
            e.rest_len = (e.verts[0].x - e.verts[1].x).norm()
            e.stretch_compliance = self.stretch_compliance_callback(e.verts[0].id)
            e.bending_compliance = self.bending_compliance_callback(e.verts[0].id)

    @ti.kernel
    def initIndices(self):
        for f in self.model.faces:
            self.indices[f.id * 3 + 0] = f.verts[0].id
            self.indices[f.id * 3 + 1] = f.verts[1].id
            self.indices[f.id * 3 + 2] = f.verts[2].id

    @ti.kernel
    def applyExtForce(self, dt : ti.f32):
        for v0 in self.model.verts:
            if v0.invM > 0.0:
                v0.v += self.gravity[None] * dt
            v0.new_x = v0.x + v0.v * dt
    
    @ti.kernel
    def update(self, dt : ti.f32):
        for v0 in self.model.verts:
            if v0.invM <= 0.0:
                v0.new_x = v0.x
            else:
                v0.v = (v0.new_x - v0.x) / dt
                v0.x = v0.new_x

    @ti.kernel
    def preSolve(self):
        for v in self.model.verts:
            v.dp.fill(0.0)

    @ti.kernel
    def postSolve(self, sc : ti.template()):
        for v in self.model.verts:
            v.new_x += v.dp * sc
    
    @ti.kernel
    def solveStretch(self, dt : ti.f32):
        ti.loop_config(block_dim=self.block_size)
        ti.mesh_local(self.model.verts.dp, self.model.verts.invM, self.model.verts.new_x)
        for e in self.model.edges:
            v0, v1 = e.verts[0], e.verts[1]
            w1, w2 = v0.invM, v1.invM
            if w1 + w2 > 0.:
                n = v0.new_x - v1.new_x
                d = n.norm()
                dp = ti.zero(n)
                constraint = (d - e.rest_len)
                if ti.static(self.XPBD): # https://matthias-research.github.io/pages/publications/XPBD.pdf
                    compliance = e.stretch_compliance / (dt**2)
                    d_lambda = -(constraint + compliance * e.la_s) / (w1 + w2 + compliance) * self.stretch_relaxation # eq. (18)
                    dp = d_lambda * n.normalized(1e-12) # eq. (17)
                    e.la_s += d_lambda
                else: # https://matthias-research.github.io/pages/publications/posBasedDyn.pdf
                    dp = -constraint / (w1 + w2) * n.normalized(1e-12) * self.stretch_relaxation # eq. (1)
                v0.dp += dp * w1
                v1.dp -= dp * w2

    @ti.kernel
    def solveBending(self, dt : ti.f32):
        ti.loop_config(block_dim=self.block_size)
        ti.mesh_local(self.model.verts.dp, self.model.verts.invM, self.model.verts.new_x)
        for e in self.model.edges:
            if e.faces.size == 2:
                v1, v2 = e.verts[0], e.verts[1]
                k, l = 0, 0
                for i in range(3):
                    if e.faces[0].verts[i].id != v1.id and \
                       e.faces[0].verts[i].id != v2.id: k = i
                v3 = e.faces[0].verts[k]
                for i in range(3):
                    if e.faces[1].verts[i].id != v1.id and \
                       e.faces[1].verts[i].id != v2.id: l = i
                v4 = e.faces[1].verts[l]
                w1, w2, w3, w4 = v1.invM, v2.invM, v3.invM, v4.invM
                if w1 + w2 + w3 + w4 > 0.:
                    # Appendix A: Bending Constraint Projection
                    p2 = v2.new_x - v1.new_x
                    p3 = v3.new_x - v1.new_x
                    p4 = v4.new_x - v1.new_x
                    l23 = p2.cross(p3).norm()
                    l24 = p2.cross(p4).norm()
                    if l23 < 1e-8: l23 = 1.
                    if l24 < 1e-8: l24 = 1.
                    n1 = p2.cross(p3) / l23
                    n2 = p2.cross(p4) / l24
                    d = ti.math.clamp(n1.dot(n2), -1., 1.)
                    
                    q3 = (p2.cross(n2) + n1.cross(p2) * d) / l23 # eq. (25)
                    q4 = (p2.cross(n1) + n2.cross(p2) * d) / l24 # eq. (26)
                    q2 = -(p3.cross(n2) + n1.cross(p3) * d) / l23 \
                         -(p4.cross(n1) + n2.cross(p4) * d) / l24 # eq. (27)
                    q1 = -q2 - q3 - q4
                    # eq. (29)
                    sum_wq = w1 * q1.norm_sqr() + \
                             w2 * q2.norm_sqr() + \
                             w3 * q3.norm_sqr() + \
                             w4 * q4.norm_sqr()
                    constraint = (ti.acos(d) - ti.acos(-1.))
                    if 1: # for perf comp
                        if ti.static(self.XPBD):
                            compliance = e.bending_compliance / (dt**2)
                            d_lambda = -(constraint + compliance * e.la_b) / (sum_wq + compliance) * self.bending_relaxation # eq. (18)
                            constraint = ti.sqrt(1 - d ** 2) * d_lambda
                            e.la_b += d_lambda
                        else:
                            constraint = -ti.sqrt(1 - d ** 2) * constraint / (sum_wq + 1e-7) * self.bending_relaxation
                        v1.dp += w1 * constraint * q1
                        v2.dp += w2 * constraint * q2
                        v3.dp += w3 * constraint * q3
                        v4.dp += w4 * constraint * q4

    def solve(self):
        frame_time_left = self.frame_dt
        substep = 0
        while frame_time_left > 0.0:
            substep += 1
            dt0 = min(self.dt, frame_time_left)
            frame_time_left -= dt0

            self.applyExtForce(dt0)
            if self.XPBD:
                self.model.edges.la_s.fill(0.)
                self.model.edges.la_b.fill(0.)
            for iter in range(self.rest_iter):
                    self.preSolve()
                    self.solveStretch(dt0)
                    self.solveBending(dt0)
                    self.postSolve(1.0)

            self.update(dt0)
            self.time += dt0
