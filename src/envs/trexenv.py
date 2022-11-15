import gym
from envs.multiagentenv import MultiAgentEnv
from gym import error, spaces
import numpy as np
import TREX_Core._utils.runner
import os


class TrexEnv(MultiAgentEnv):
    """

    """
    def __init__(self, **kwargs):
        self.terminated = False

        #TODO: this is the code for making the runnner and getting the parameters of the sim
        # self.config_name = kwargs['trex_config'] -> this should be passable through cli env_args
        self.config_name = 'CLI_agent_testig'
        TREX_path = 'C:/source/TREX-Core/TREX_Core/'
        self.runner = TREX_Core._utils.runner.Runner(self.config_name, resume=False, purge=False, path=TREX_path)
        self.config = self.runner.configs

        mem_lists = self.setup_interprocess_memory()


        self.launch_TREX() #FIXME: August 16 2022, this is where TREX is launched, no where else.



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

    def setup_interprocess_memory(self):
        from multiprocessing import shared_memory
        for ident in self.config['participants']:
            print(ident)
            print()
            if self.config['participants'][ident]['trader']['type'] == 'gym_agent':
                # this is where we set up the shared memory object, each agent needs 2 objects actions, observations
                actions_name = ident+'_actions'
                obs_name = ident+'_obs'
                # actions [flag,
                actions_list = shared_memory.ShareableList([0.0,0.0,0.0,0.0], name=actions_name)
                obs_list = shared_memory.ShareableList([0.0,0.0,0.0,0.0], name= obs_name)




        agent_dict = {}
        return agent_dict

    def run_subprocess(self, args: list, delay=0):
        import subprocess
        import time

        time.sleep(delay)
        print("the trex args", args)
        try:
            print('Trying to find the venv python worked')
            subprocess.run(['venv/Scripts/python', args[0], *args[1]]) #FIXME: August 29 2022,  this is coded for postix system
        except:
            print('Excepting: using the atpetepymarl python')
            subprocess.run(['C:/Users/molly/.virtualenvs/atpeterpymarl-qnIKOvrx/Scripts/python', args[0], *args[1]])




    def launch_TREX(self):
        """
        this method launches trex as a process
        :return:
        """

        # simulations = [
            # {'simulation_type': 'baseline'},
            # {'simulation_type': 'training'}
            # {'simulation_type': 'validation'}
        # ]
        # runner = TREX_Core._utils.runner.Runner("TB8", resume=False, purge=False)
        # launch_list = runner.run(simulations, run=False) #running with run=False just creates a launch list

        # path_to_env_client = os.getcwd() + '/envs/env_controller/sio_client.py'

        # FIXME: March 8 2022, hardcode below. Plz fix market id
        # args_list = ['--port=3500', '--market_id=training']
        # launch_list.append((path_to_env_client, args_list))
        # launch_list = self.create_trex_launchlist()
        launch_list = self.runner.make_launch_list(self.runner.configs)
        #launch the TREX launchlist from inside EPYMARL
        from multiprocessing import Pool
        pool_size = len(launch_list)
        pool = Pool(pool_size)
        pool.map(self.run_subprocess, launch_list) #FIXME: July 2022: this map seems to be giving me a error to do with the launch list.
        pool.close()

    def create_trex_launchlist(self,
                               config_name='TB8',
                               simulations=[{'simulation_type': 'training'}],
                               port=3500):
        """
        This function creates the lanchlist
        I should be able to specify the config name and simulations from the command line for hyperparameter search
        TODO:  I should probably be able to specify the port from the command line for hyperparameter search
        """
        runner = TREX_Core._utils.runner.Runner(config_name, resume=False, purge=False)
        # launch_list = runner.run(simulations, run=False)
        launch_list = runner.make_launch_list(runner.configs)

        # need to add the env controller here
        path_to_envcontroller_sioclient = os.getcwd() + '/envs/env_controller/sio_client.py'
        args_list = ['--port='+str(port), '--market_id=training']
        launch_list.append((path_to_envcontroller_sioclient, args_list))

        return launch_list


    def get_state_size(self):
        """
        This method is required for gym
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
        #TODO: As of August 16 2022, this is not implemented
        :return:
        """
        raise NotImplementedError
        return 0

    def get_obs_agent(self, agent_id):
        '''
        #todo: as of august 16, 2022 this is not yet implemented
        '''
        agent_obs = []
        raise NotImplementedError
        return agent_obs

    def get_obs_size(self):
        """
        THIS METHOD IS REQUIRED FOR GYM
        This method returns the size of each individual agents observation space.
        """
        if self.using_batteries:
            obs_size = 5
        else:
            obs_size = 3


        return obs_size

    def get_total_actions(self):
        """
        THIS IS REQUIRED FFOR GYM
        Returns the total number of actions that an agent could ever take:
        :return:
        """
        if self.using_batteries:
            tot_actions = 3
        else:
            tot_actions = 2

        return tot_actions

    def render(self, mode="human"):
        '''
        #TODO: August 16, 2022: make this compatible with the browser code that steven has finished.
        '''
        return 0

    def step(self, actions):
        '''
        #TODO: August 16 2022, as of this moment this is still just pseudocode.
        '''
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

    def reset(self):
        '''
        This method resets the trex environment.
        TODO:  figure out what this needs to do
        '''
        return None

    def get_state(self):
        '''
        This method gets called for the pretransition data; this is the state data from the envcontroller
        '''
        state = None
        return state