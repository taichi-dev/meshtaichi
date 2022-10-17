import subprocess, argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='gpu')
parser.add_argument('--quiet', '-q', action='store_true')
args = parser.parse_args()

fd = os.path.split(os.path.abspath(__file__))[0]

def run_test(testfn):
    td, tn = os.path.split(testfn)
    test_folder = os.path.join(fd, td)
    res = subprocess.run(f'cd {test_folder}; python3 {tn} --test --arch {args.arch}', shell=True, capture_output=args.quiet)
    if res.returncode:
        if args.quiet:
            print(res.stdout.decode('utf-8'))
            print(res.stderr.decode('utf-8'))
        raise Exception(f'{testfn} test failed.') 
    print(f'{testfn} test passed.')

run_test('../geodesic_distance/geodesic.py')
run_test('../lag_mpm/run.py')
run_test('../mass_spring/ms.py')
run_test('../projective_dynamics/pd.py')
run_test('../vertex_normal/normal.py')
run_test('../xpbd_cloth/run_demo.py')
