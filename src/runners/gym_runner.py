from envs import REGISTRY as env_REGISTRY
import numpy as np 
from functools import partial
from components.episode_buffer import EpisodeBatch

from pathlib import Path
import subprocess
# TODO: Oct 27, 2021; need to set up the imports here to import the TREX package here 

class GymRunner:
    """
    This is the runner object for the TrexEnv object.

    """
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run 
        assert self.batch_size == 1 # make sure batches are one
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0 
        # Log the first run
        self.log_train_stats_t = -1000000
    
    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac


    def start_trex(self):
        """
        This method uses subprocesses to launch the TREX-core package
        trex is launched using
        """
        # Setting up the path to the environments
        #Args for the launch
        #Fixme: Nov 29, 2021; At the moment there are no args to the launch, but I will be changing that.

        # Call the subprocess
        # python main.py
        subprocess.call()


    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
        
    def run(self, test_mode=False):
        pass 

    def _log(self, returns, stats, prefix):
        pass 



def __main():
    """
    This is used for debugging purposes only
    """
    runner = GymRunner()
    runner.start_trex()

if __name__ == '__main__':
    __main()