import gym
from envs.multiagentenv import MultiAgentEnv
from gym import error, spaces
import numpy as np
import TREX_Core._utils.runner
import os

class TrexEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.terminated = False
        self.launch_TREX()
        if kwargs['using_batteries']:
            self.using_batteries = True
        else:
            self.using_batteries = False  # Todo: Jan 24, 2022: this needs to be able to be set based on the trex config in main

        if kwargs['n_agents']:
            self.n_agents = kwargs['n_agents']
        else:
            self.n_agents = 0  # Todo: Jan 24, 2022: this needs to be able to be set based on the trex config in main

        self.episode_limit = 0  # Todo Jan 24, 2022: this needs to be able to be set based on generation length in trex config

        #Items pulled from the trexGymEnv(gym.Env) class
        # self.action_space = tuple()
        self.action_space = spaces.Tuple(tuple([spaces.Box(low=np.array([0.07, -17.0, 0.0]),
                                                           high=np.array([0.0, 17, 100.0]))] * self.n_agents))
        # self.observation_space = tuple()
        self.observation_space = spaces.Tuple(tuple([spaces.Box(low=0.0, high=1.0, shape=(5,))] * self.n_agents))
        self._seed = 0

    def run_subprocess(self, args: list, delay=0):
        import subprocess
        import time

        time.sleep(delay)
        try:
            subprocess.run(['venv/bin/python', args[0], *args[1]])
        except:
            subprocess.run(['C:/Users/molly/.virtualenvs/atpeterpymarl-qnIKOvrx/Scripts/python', args[0], *args[1]])
        finally:
            subprocess.run(['python', args[0], *args[1]])


    def read_trex_configs(self, path=None):
        print("reading trex config file")
        n_agents = 0
        action_space = []
        generation_size = 0
        return n_agents, action_space, generation_size

    def launch_TREX(self):
        """
        this method launches trex as a process
        :return:
        """
        import sys
        print("launching TREX")
        simulations = [
            # {'simulation_type': 'baseline'},
            {'simulation_type': 'training'}
            # {'simulation_type': 'validation'}
        ]
        runner = TREX_Core._utils.runner.Runner("TB8", resume=False, purge=False)
        launch_list = runner.run(simulations, run=False)


        '''
        This is where I set up the envController client
        ( Absolute path , args list [] )

        '''
        path_to_env_client = os.getcwd() + '/src/envs/env_controller/sio_client.py'
        # FIXME: March 8 2022, hardcode below. Plz fix market id
        args_list = ['--port=3501', '--market_id=training']

        launch_list.append((path_to_env_client, args_list))

        from multiprocessing import Pool
        pool_size = len(launch_list)
        pool = Pool(pool_size)
        pool.map(self.run_subprocess, launch_list)
        pool.close()


    def connect_envController(self):
        """"
        This method creates a envController and connect it to TREX
        """
        self.envController = envController({"sim": "information"})


    def get_state_size(self):
        """
        This method gets the size of the state global state used by get_env_info
        Global state is the individual agents state multiplied by the number of agents.
        :return:
        The size of the state information is a single int
        """
        state_size = int(self.observation_space[-1].shape[0]) * self.n_agents
        print("State size ", state_size)
        return state_size

    def get_obs(self, timestep):
        """
        This gets the full observation state for that timestep
        :return:
        """
        print("not done yet, go check the database yourself.")
        return 0

    def get_obs_agent(self, agent_id):
        agent_obs = []
        return agent_obs

    def get_obs_size(self):
        """
        This method returns the size of each individual agents
        """
        if self.using_batteries:
            obs_size = 5
        else:
            obs_size = 3


        return obs_size

    def get_total_actions(self):
        """
        Returns the total number of actions that an agent could ever take:
        :return:
        """
        if self.using_batteries:
            tot_actions = 3
        else:
            tot_actions = 2

        return tot_actions

    def render(self, mode="human"):
        print('This is not done yet, just read the matrix ')
        return 0

    def step(self, actions):
        # terminated:
        # this will need to be able to get set on the end of each generation
        terminated = self.envController.terminated()

        # Reward:
        # To get the reward I will have to get the matched reward, same way that steven gets it in the current way of
        # training Deep neural networks This can be found in stevens repo under bess test branch in agent.learn()
        reward = self.envController.get_reward()

        # info:
        # Imma keep it as a open dictionary for now:
        info = {}

        return float(reward), all(terminated), info


