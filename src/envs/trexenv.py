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
        self.n_agents = 0

        self.config_name = kwargs['TREX_config']
        TREX_path = kwargs['TREX_path']

        # changes where python is looking to open the right config
        cur_dir = os.getcwd()
        os.chdir(TREX_path)
        self.runner = TREX_Core._utils.runner.Runner(self.config_name, resume=False, purge=False, path=TREX_path)
        os.chdir(cur_dir)

        self.config = self.runner.configs

        # get the n agents from the config:
        for ident in self.config['participants']:
            if self.config['participants'][ident]['trader']['type'] == 'gym_agent':
                self.n_agents = self.n_agents + 1

        # setup the memory lists
        self.setup_spaces()
        self.mem_lists = self.setup_interprocess_memory()

        # #########################################################################
        # catch if there are no gym traders. This is probably not needed.
        if not self.n_agents:
            self.n_agents = 0

        '''
        TODO: November 19 2022
        Calculation here is in seconds because steven made it so.
        Number of seconds in a day: 86400
        number of days in sim can be pulled from cofig 
        timestep size in seconds is also in config 
        episode limit = 86400 * days / time_step_size 
        EPISODE LIMIT IS AN INTEGER -- No weird half timesteps 
        '''
        self.episode_limit = int(86400 * self.config['study']['days'] / self.config['study']['time_step_size'])
        self._seed = 0

        ####### TREX GETS LAUNCHED HERE #########
        # self.launch_TREX()

    def setup_spaces(self):
        '''
        This method sets up the action and observation spaces based on the values that are in the config
        For now, agents are assumed to be homogenous in the
        '''

        # Bring up the values in the config

        for ident in self.config['participants']:
            if self.config['participants'][ident]['trader']['type'] == 'gym_agent':
                try:
                    obs = self.config['participants'][ident]['trader']['observations']
                except:
                    obs_len = 5


                try:
                    actions = self.config['participants'][ident]['trader']['actions']
                except:
                    actions_len = 7

        # FIXME: November 22 2022; this works but still needs to be further defined in the trex config

        self.action_space = spaces.Tuple(tuple([spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                                           high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf,
                                                                          np.inf]))] * self.n_agents))
        self.observation_space = spaces.Tuple(tuple([spaces.Box(low=0.0, high=np.inf, shape=(5,))] * self.n_agents))

    def setup_interprocess_memory(self):
        """
        This method sets up the interprocess Shareable lists in memory for all the agents that have the
        designation gym_agent.
        Takes in nothing
        Returns: Dictionary {agent_identification_from_config : { obs_list :obs_list_object, action_list :action_list_object
        """
        from multiprocessing import shared_memory
        agent_dict = {}
        for ident in self.config['participants']:
            if self.config['participants'][ident]['trader']['type'] == 'gym_agent':
                # this is where we set up the shared memory object, each agent needs 2 objects actions, observations
                # todo: November 21 2022; for parallel runner there will need to be extra identifiers for sharelists to remain unique
                actions_name = ident+'_actions'
                obs_name = ident+'_obs'

                # TODO: November 19, 2022: see if you can put objects into here like full gym spaces, or at least
                # Flattened gym spaces. Actions are like this:
                # [bid price, bid quantity, solar ask price, solar ask quantity, bess ask price, bess ask quantity]
                #TODO: Novemeber 22, 2022: these two should be created from the lenght of actionspaces and observation spaces.
                actions_list = shared_memory.ShareableList([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], name=actions_name)
                obs_list = shared_memory.ShareableList([0.0, 0.0, 0.0, 0.0], name=obs_name)
                agent_dict[ident] = {
                    'obs':  obs_list,
                    'actions': actions_list
                }
        return agent_dict

    def run_subprocess(self, args: list, delay=0):
        """
        This method runs the venv interpreter
        """
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

    def get_obs(self):

        """
        This gets the full observation state for that timestep

        :return: a single list that contains all the agents individual observations as lists:
        [agent1_obs_list, agent2_obs_list, ..., agentn_obs_list

        """

        # in the lbf this simply returns self._obs
        # self._obs is populated in env.step, but the values are pulled before the next
        # steps

        return

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
        from gym.spaces import flatdim
        obs_size = flatdim(self.observation_space[-1])

        return obs_size

    def get_total_actions(self):
        """
        THIS IS REQUIRED FFOR GYM
        Returns the total number of actions that an agent could ever take:
        :return:
        """
        # if self.using_batteries:
        #     tot_actions = 3
        # else:
        #     tot_actions = 2
        from gym.spaces import flatdim
        action_size = flatdim(self.action_space[-1])
        return action_size

    def render(self, mode="human"):
        '''
        #TODO: August 16, 2022: make this compatible with the browser code that steven has finished.
        '''
        return 0

    def step(self, actions):
        '''
        TODO: November 21 2022; this needs to be implemented and needs to give the right values
        [bid price, bid quantity, solar ask price, solar ask quantity, bess ask price, bess ask quantity]
        '''

        # SEND ACTIONS
        #actions are provided, so this method needs to put the actions into the
        # actions are tensor (n_actions, n_agent)
        # Trex will have price, quantity,
        for i, agent in enumerate(self.mem_lists):
            agent_action = actions[i]
            # insert the agents actions into the memlist
            self.mem_lists[agent]['actions'] = actions[i]

        ## TODO: read the observation from all the agents and put it into the self._obs array:

        self.read_obs_values()

        # terminated:
        # this will need to be able to get set on the end of each generation
        terminated = []

        # Reward:
        # Rewards are going to have to be sent over from the gym trader, which will be able to
        # get information from the reward

        # TODO: calculate reward

        reward = []

        # info:
        # Imma keep it as a open dictionary for now:
        info = {}

        return float(reward), all(terminated), info

    def read_obs_values(self):
        """
        This method cycles through the mem lists of the agents until they all have all read the information.
        """

        self._obs = []



    def reset(self):
        '''
        This method resets the trex environment.
        The reset would have to be able to kill all the TREX processes,
        then reboot them all and have the gym traders reconnect to the shared memory objects.
        TODO: Nov 18 2022: this needs to return the expected values
        '''

        return None

    def get_state(self):
        '''
        This method gets called for the pretransition data; this is the state data from the envcontroller
        This method will return a flattened version of all the agents observations.


        '''

        # get all the remote agent values and then put them into a list


        state = []


        return state

    def get_avail_actions(self):
        """
        This method will return a list of list that gives the availiable actions


        """
        # For now, all actions are availiable at all time
        ACTIONS = [1]*self.action_space[-1].shape[0]
        avail_actions = [ACTIONS *self.n_agents]

        return avail_actions