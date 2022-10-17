import subprocess, argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='gpu')
args = parser.parse_args()

fd = os.path.split(os.path.abspath(__file__))[0]

def run_test(testfn):
    td, tn = os.path.split(testfn)
    test_folder = os.path.join(fd, td)
    subprocess.run(f'cd {test_folder}; python3 {tn} --test --arch {args.arch}', shell=True, check=True)
    print(f'{testfn} test passed.')

run_test('../geodesic_distance/geodesic.py')
run_test('../lag_mpm/run.py')
run_test('../mass_spring/ms.py')
run_test('../projective_dynamics/pd.py')
run_test('../vertex_normal/normal.py')
run_test('../xpbd_cloth/run_demo.py')
