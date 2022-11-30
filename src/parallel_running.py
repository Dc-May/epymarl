import multiprocessing
import subprocess
import sys
_CPU_COUNT = multiprocessing.cpu_count() - 1 


def work(cmd): 
	cmd = cmd.split(" ")
	return subprocess.run(cmd, shell=False)


def _main():
    pool = multiprocessing.Pool(processes= _CPU_COUNT)
    list_of_algs = [
         'maa2c_best_config_1',
         'maa2c_best_config_2'
    ]
    list_of_envs = [
             'Foraging-2s-8x8-2p-2f-coop-v2',
             'Foraging-8x8-2p-2f-coop-v2',
             'Foraging-10x10-3p-3f-v2',
             'Foraging-2s-10x10-3p-3f-v2'
         ]
    seed = 3
    configs = build_configs(list_of_algs, list_of_envs, seed)
    # print(configs)
    print(pool.map(work, configs))

    
def build_configs(algorithms, environments, seed=1):
    """
    algorithms = []
    environments = []
    seed = int 
    _exec main.py --config=algorithm --env-config=gymcoop with env_args.time_limit=25 env_args.key=environment 
    """
    _exec = sys.executable 
    configs = []
    for algorithm in algorithms: 
        for environment in environments: 
            for i in range(seed): 
                configs.append(_exec + ' main.py --config=' + algorithm + ' --env-config=gymcoop with env_args.time_limit=25 env_args.key=' + environment)

    return configs

if __name__ == '__main__':
    _main()
