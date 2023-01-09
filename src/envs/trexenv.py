import sys

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
        self._obs = []
        self.action_array = [] # this is the array for the actions in the order that they are processed
        self.obs_array = []
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

        for ident in self.config['participants']:
            if self.config['participants'][ident]['trader']['type'] == 'gym_agent':
                try:
                    self.obs_array = self.config['participants'][ident]['trader']['observations']
                    obs_len = len(self.obs_array)
                    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,))
                except:
                    print('There was a problem loading the config observations')
                try:
                    actions = self.config['participants'][ident]['trader']['actions']
                    for action in actions:
                        if actions[action]['heuristic'] == 'learned':
                            self.action_array.append(action)

                    action_len = len(self.action_array)
                    action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(action_len,))
                except:
                    print("there was a problem loading the actions")

        # FIXME: November 22 2022; this works but still needs to be further defined in the trex config

        self.action_space = spaces.Tuple(tuple([action_space] * self.n_agents))
        self.observation_space = spaces.Tuple(tuple([obs_space] * self.n_agents))

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
                reward_name = ident+'_reward'

                # Flattened gym spaces. Actions are like this:
                # [bid price, bid quantity, solar ask price, solar ask quantity, bess ask price, bess ask quantity]
                length_of_obs = len(self.obs_array) + 1
                length_of_actions = len(self.action_array) + 1

                observations = [0.0] * length_of_obs

                actions_list = shared_memory.ShareableList([0.0]*length_of_actions, name=actions_name)
                obs_list = shared_memory.ShareableList([0.0]*length_of_obs, name=obs_name)
                reward_list = shared_memory.ShareableList([0.0, 0.0], name=reward_name)


                agent_dict[ident] = {
                    'obs':  obs_list,
                    'actions': actions_list,
                    'rewards': reward_list
                }
                self.agent_names = list(agent_dict.keys())
        return agent_dict

    def run_subprocess(self, args: list, delay=0):
        """
        This method runs the venv interpreter
        """
        import subprocess

        try:
            print('Trying to find the venv python worked')
            subprocess.run([sys.executable, args[0], *args[1]]) #FIXME: August 29 2022,  this is coded for postix system
            #subproces.run([sys.executable, args[0], *args[1]]) This is probably the most
        except:
            print('Excepting: using the atpetepymarl python')
            subprocess.run(['C:/Users/molly/.virtualenvs/atpeterpymarl-qnIKOvrx/Scripts/python', args[0], *args[1]])

    def launch_TREX(self):
        """
        this method launches trex as a process
        :return:
        """

        simulations = [
            # {'simulation_type': 'baseline'},
            {'simulation_type': 'training'}
            # {'simulation_type': 'validation'}
        ]

        # need to modify the config that is here with the sim type:
        confog = self.runner.modify_config(simulation_type='training')

        launch_list = self.runner.make_launch_list(confog)
        #launch the TREX launchlist from inside EPYMARL
        from multiprocessing import Pool
        pool_size = len(launch_list)
        pool = Pool(pool_size)
        pool.map(self.run_subprocess, launch_list)
        pool.close()



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

        return self._obs

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
        [bid price, bid quantity, solar ask price, solar ask quantity, bess ask price, bess ask quantity]
        '''

        # SEND ACTIONS
        #actions are provided, so this method needs to put the actions into the right action bufferes
        # actions are tensor (n_actions x n_agent)
        # Trex will have price, quantity,
        print('In Trexenv Step')
        for i, agent in enumerate(self.mem_lists):
            agent_action = actions[i]
            # insert the agents actions into the memlist
            self.mem_lists[agent]['actions'][1] = agent_action.item()
            self.write_flag(self.mem_lists[agent]['actions'], True)
        # this is where we would need to set the flag


        # terminated:
        # this will need to be able to get set on the end of each generation
        terminated = [0.0]*self.n_agents

        # Reward:
        # Rewards are going to have to be sent over from the gym trader, which will be able to
        # get information from the reward

        # TODO: get the reward from wherever I put the reward
        self.read_reward_values()
        reward = [float(reward) for reward in self._reward]

        # info:
        # Imma keep it as a open dictionary for now:
        info = {}

        return reward, all(terminated), info

    def read_obs_values(self):
        """
        This method cycles through the mem lists of the agents until they all have all read the information.
        """
        agent_status = [0] * self.n_agents
        self._obs = []
        flag = False
        while not flag:
            # main loop; keep checking agent_status is
            if all(agent_status):
                flag = True
            # loop through all the agents reward buffers and check their flags
            else:
                # print('Memlist before', self.mem_lists)
                for i, ident in enumerate(self.mem_lists):
                    # FIXME: Value error: too many values to unpack (expected 2)
                    # agent is a dictionary 'obs', 'actions', 'rewards'
                    if self.mem_lists[ident]['obs'][0]:
                        # rewards are good to read
                        # self._obs.append( self.mem_lists[ident]['obs'][1:] ) # this is hardcoded because rewards
                        self._obs.append([self.mem_lists[ident]['obs'][i] for i in range(1,
                                                                                         len(self.mem_lists[ident]['obs']))])
                        agent_status[i] = 1

    def write_flag(self, shared_list, flag):
        """
        This method sets the flag
        Parameters:
            shared_list ->  shared list object to be modified
            flag -> boolean that indicates write 0 or 1. True sets 1
        """
        print(shared_list)

        if flag:
            shared_list[0] = 1
            print("Flag was set ")
            print(shared_list)
        else:
            shared_list[0] = 0
            print("Flag was not set")

    def read_reward_values(self):
        """
        This method cycles through the reward mem lists of the agents until they all have read the information.
        """
        # encode the agents as one hot vectors:
        agent_status = [0] * self.n_agents
        self._reward = []
        flag = False
        while not flag:
            # main loop; keep checking agent_status is
            if all(agent_status):
                flag = True
            #loop through all the agents reward buffers and check their flags
            else:
                for i, ident in enumerate(self.mem_lists):
                    # agent is a dictionary 'obs', 'actions', 'rewards'
                    if self.mem_lists[ident]['rewards'][0]:
                        # rewards are good to read
                        self._reward.append(self.mem_lists[ident]['rewards'][1])
                        agent_status[i] = 1


    def reset(self):
        '''
        This method resets the trex environment.
        The reset would have to be able to kill all the TREX processes,
        then reboot them all and have the gym traders reconnect to the shared memory objects.
        TODO Peter: November 30, 2022; This is going to need to reset the TREX instance
        '''

        return None

    def get_state(self):
        '''
        This method gets called for the pretransition data; this is the state data from the envcontroller
        This method will return a flattened version of all the agents observations.

        Return: State -> list ; size [individual agent obs] * n_agents
        '''

        # get all the remote agent values and then put them into a list
        # TODO: December 02, test this functionality
        state = []
        self.read_obs_values()
        state = np.concatenate(self._obs).tolist()
        return state

    def get_avail_actions(self):
        """
        This method will return a list of list that gives the available actions
        return: avail_actions -> list of [1]* n_agents

        """
        # For now, all actions are availiable at all time
        ACTIONS = [1]*self.action_space[-1].shape[0]
        avail_actions = [ACTIONS *self.n_agents]

        return avail_actions