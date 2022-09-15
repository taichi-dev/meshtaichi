import taichi as ti
import numpy as np
import time

@ti.func
def lerp(x, y, val):
    return x * (1.0 - val) + y * val

__palette__ = [
    0x750e13, 0xa2191f, 0xda1e28,
    0xfa4d46, 0xff8389, 0xffb3b8,
    0xffd7d9, 0xbae6ff,0x82cfff,
    0x33b1ff, 0x1192e8, 0x0072c3,
    0x00539a, 0x003a6d
]
palette = ti.field(ti.i32, shape=len(__palette__))
palette.from_numpy(np.array(__palette__, dtype=np.int32))

@ti.func
def FF_to_rgb(x):
    r = (x & 0xFF0000) >> 16
    g = (x & 0x00FF00) >> 8
    b = (x & 0x0000FF) >> 0
    return ti.Vector([r / 255.0, g / 255.0, b / 255.0])

@ti.func
def rgb_to_FF(c):
    return int((int(c[0] * 255.0) << 16) | (int(c[1] * 255.0) << 8) | (int(c[2] * 255.0) << 0))

@ti.func
def get_color(x):
    r_s = min(int(x * (len(__palette__) - 1)), len(__palette__) - 2)
    r_e = r_s + 1
    val = x * (len(__palette__) - 1) - r_s
    s_c = FF_to_rgb(palette[r_s])
    s_e = FF_to_rgb(palette[r_e])
    c = lerp(s_c, s_e, val)
    return rgb_to_FF(c)

v_bits = 8
x_bits = 32 - v_bits

@ti.kernel
def copy_ranged(np_x: ti.ext_arr(), input_x: ti.template(), begin: ti.i32, end: ti.i32):
    for i in range(begin, end):
        np_x[i - begin] = input_x[i]

def write_particles(cnt, fems, fn, slice_size=1000000):
    t = time.time()
    output_fn = fn

    n_particles = 0
    for i in range(cnt):
      n_particles += len(fems[i].model.verts)

    x_and_v = np.ndarray((n_particles, 3), dtype=np.uint32)
    # Value ranges of x and v components, for quantization
    ranges = np.ndarray((2, 3, 2), dtype=np.float32)

    def rotate_axis(x):
        if x == 0: return 0
        if x == 1: return 2
        if x == 2: return 1
        assert False

    for d in range(3):
        np_x = np.ndarray((n_particles, ), dtype=np.float32)
        np_v = np.ndarray((n_particles, ), dtype=np.float32)

        np_x_slice = np.ndarray((slice_size, ), dtype=np.float32)
        np_v_slice = np.ndarray((slice_size, ), dtype=np.float32)

        for i in range(cnt):
          # Fetch data slice after slice since we don't have the GPU memory to fetch them channel after channel...
          group_size = len(fems[i].model.verts)
          num_slices = (group_size + slice_size - 1) // slice_size
          for s in range(num_slices):
              begin = slice_size * s
              end = min(slice_size * (s + 1), group_size)
              copy_ranged(np_x_slice, fems[i].model.verts.x.get_scalar_field(rotate_axis(d)),
                          begin, end)
              copy_ranged(np_v_slice, fems[i].model.verts.v.get_scalar_field(rotate_axis(d)),
                          begin, end)

              np_x[begin:end] = np_x_slice[:end - begin]
              np_v[begin:end] = np_v_slice[:end - begin]

          ranges[0, d] = [np.min(np_x), np.max(np_x)]
          ranges[1, d] = [np.min(np_v), np.max(np_v)]

        # Avoid too narrow ranges
        for c in range(2):
            ranges[c, d, 1] = max(ranges[c, d, 0] + 1e-5, ranges[c, d, 1])
        np_x = ((np_x - ranges[0, d, 0]) *
                (1 / (ranges[0, d, 1] - ranges[0, d, 0])) *
                (2**x_bits - 1) + 0.499).astype(np.uint32)
        np_v = ((np_v - ranges[1, d, 0]) *
                (1 / (ranges[1, d, 1] - ranges[1, d, 0])) *
                (2**v_bits - 1) + 0.499).astype(np.uint32)
        x_and_v[:, d] = (np_x << v_bits) + np_v
        del np_x, np_v

    color = np.ndarray((n_particles, 3), dtype=np.uint8)
    np_color = np.ndarray((n_particles, ), dtype=np.uint32)

    np_color_slice = np.ndarray((slice_size, ), dtype=np.float32)

    for i in range(cnt):
        # Fetch data slice after slice since we don't have the GPU memory to fetch them channel after channel...
        group_size = len(fems[i].model.verts)
        num_slices = (group_size + slice_size - 1) // slice_size
        for s in range(num_slices):
            begin = slice_size * s
            end = min(slice_size * (s + 1), group_size)

            copy_ranged(np_color_slice, fems[i].model.verts.color, begin, end)
            np_color[begin:end] = np_color_slice[:end - begin]

    for c in range(3):
        color[:, c] = (np_color >> (8 * (2 - c))) & 255

    np.savez(output_fn, ranges=ranges, x_and_v=x_and_v, color=color)

    print(f'Writing to disk: {time.time() - t:.3f} s')