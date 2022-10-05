import subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='gpu')
args = parser.parse_args()

subprocess.run(f'cd vertex_normal; python3 normal.py --test --arch {args.arch}', shell=True, check=True)
