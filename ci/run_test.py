import subprocess, argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='gpu')
args = parser.parse_args()

paths = os.path.split(os.path.abspath(__file__))
normal_path = os.path.join(paths[0], '..', 'vertex_normal')

subprocess.run(f'cd {normal_path}; python3 normal.py --test --arch {args.arch}', shell=True, check=True)
