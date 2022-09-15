import taichi as ti
import os
import time
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b',
                        '--begin',
                        type=int,
                        default=0,
                        help='Beginning frame')
    parser.add_argument('-e',
                        '--end',
                        type=int,
                        default=10000,
                        help='Ending frame')
    parser.add_argument('-s', '--step', type=int, default=1, help='Frame step')
    parser.add_argument('-r',
                        '--res',
                        type=int,
                        default=512,
                        help='Grid resolution')
    parser.add_argument('-g', '--gui', action='store_true', help='Show GUI')
    parser.add_argument('-o', '--out-dir', default="./results/output", type=str, help='Output folder')
    parser.add_argument('-i', '--in-dir', default="./results/particles", type=str, help='Input folder')
    parser.add_argument(
        '-t',
        '--shutter-time',
        type=float,
        default=2e-3,
        help=
        'Shutter time, which affects motion blur. Note that memory usage will increase when '
        'shutter time increases')
    parser.add_argument('-f',
                        '--force',
                        action='store_true',
                        help='Overwrite existing outputs')
    parser.add_argument('-m',
                        '--gpu-memory',
                        type=float,
                        default=4,
                        help='GPU memory')
    parser.add_argument('-M',
                        '--max-particles',
                        type=int,
                        default=128,
                        help='Max num particles (million)')
    args = parser.parse_args()
    print(args)
    return args


args = parse_args()

ti.init(arch=ti.cuda, device_memory_GB=args.gpu_memory)

output_folder = args.out_dir
os.makedirs(output_folder, exist_ok=True)

from engine.renderer import Renderer

model_size = 0.10
res = args.res
renderer = Renderer(dx=1 / res,
                    sphere_radius= 3.0 * model_size / res,
                    shutter_time=args.shutter_time,
                    max_num_particles_million=args.max_particles,
                    taichi_logo=False)

with_gui = args.gui
if with_gui:
    gui = ti.GUI('Particle Renderer', (1920, 1080))

spp = 200

# 0.23, (0.0, 0.8, 5.5)


def main():
    for f in range(args.begin, args.end, args.step):
        print('frame', f, end=' ')
        output_fn = f'{output_folder}/{f:05d}.png'
        if os.path.exists(output_fn) and not args.force:
            print('skip.')
            continue
        else:
            print('rendering...')

        t = time.time()

        renderer.set_camera_pos(1.3, 0.8, -1.0)
        renderer.set_look_at(0.5, 0.16, 0.5)
        renderer.floor_height[None] = 0.009

        cur_render_input = f'{args.in_dir}/{f:05d}.npz'
        if not os.path.exists(cur_render_input):
            print(f'warning, {cur_render_input} not existed, skip!')
            continue
        Path(output_fn).touch()
        renderer.initialize_particles_from_taichi_elements(cur_render_input)
        img = renderer.render_frame(spp=spp)

        if with_gui:
            gui.set_image(img)
            gui.show(output_fn)
        else:
            ti.imwrite(img, output_fn)
        ti.print_memory_profile_info()
        print(f'Frame rendered. {spp} take {time.time() - t} s.')


# if __name__ == '__main__':
main()
