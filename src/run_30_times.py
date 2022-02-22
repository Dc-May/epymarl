import subprocess
# import shlex
from multiprocessing import Pool

# command_line =  "python main.py --config=qmix --env-config=gymma with env_args.time_limit=50 env_args.key='Foraging-8x8-2p-3f-v1'"

# args = shlex.split(command_line)
# print(args)
arguments = [ 'C:/Users/molly/.virtualenvs/seac-q3LC12Ko/Scripts/python.exe', 'src/main.py', '--config=qmix', '--env-config=gymma', 'with', 'env_args.time_limit=50', "env_args.key='Foraging-8x8-2p-3f-v1'"]

# for index in range(2):
#     with subprocess.Popen(arguments) as process:
#         print('WE GOOD BRUH', process.pid)
    

def run_experiment(args):
    with subprocess.Popen(args) as process: 
        print("starting process", process.pid)

if __name__ == '__main__':
    arglist = arguments*5
    with Pool(5) as p:
        p.map(run_experiment, arglist)