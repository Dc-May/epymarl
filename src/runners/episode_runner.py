from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import math
from torch import nn

import wandb



class EpisodeRunner:

    def __init__(self, args, logger):

        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # This is where env.make() is called
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)

        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # add the agent returns here: 
        self.agent_returns = []

        # Log the first run
        self.log_train_stats_t = -1000000 #Fixme: this is a hack?

        wandb.init(project="GymRex",
                   entity="dc_may" #obviously this needs to be changed if ur sbdy else
                   )
        wandb.config = {
            "algorithms": "Alg_Name",
            "scenario": "Price_only_constant_profile",
        }

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0 #ToDo: unhack

    def run(self, test_mode=False):
        '''
        Items that have been added by Peter:
        * current_episode_agent_returns
        '''
        self.reset()
        # this clears the agent returns list every episode. 
        # This was added by me
        current_episode_agent_returns = [0]*self.args.n_agents
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            state = [self.env.get_state()]
            obs = [self.env.get_obs()]
            pre_transition_data = {
                "state": obs, #[self.env.get_state()], #ToDo: Peter, Jan6th 2023 check if this can be unhacked
                "avail_actions": [self.env.get_avail_actions()],
                "obs": obs
            }

            self.batch.update(pre_transition_data, ts=self.t)
            print(self.t, self.t_env)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            reward, terminated, env_info = self.env.step(actions[0])

            for agent_name, agent_reward in zip(self.env.agent_names, reward):
                wandb.log({agent_name + '_reward': agent_reward})

            # TODO: March 14 2022, this appeases the algorithm but still needs to be toggled for the scenario
            # TODO: May 24 2022: this toggle can be done through self.args.name .
            # Parse the string and then check if coop is in the name
            episode_return += math.fsum(reward)
            # episode_return += reward
            # TODO: works 
            # for i, value in enumerate(env_info['agent_rewards']):
            #         current_episode_agent_returns[i] += value
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
        state = [self.env.get_state()]
        obs = [self.env.get_obs()]
        last_data = {
            "state": obs,  # [self.env.get_state()], #ToDo: Peter, Jan6th 2023 check if this can be unhacked
            "avail_actions": [self.env.get_avail_actions()],
            "obs": obs
        }

        for agent_name, agent_obs in zip(self.env.agent_names, obs): #ToDo: DCM 040123 - check if this autoseparates into invididual obs and disentangle and add proper names by fetching names from trex-env
            for i, value in enumerate(agent_obs):
                wandb.log({agent_name + '_obs_' + str(i): value})

        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        # this is where I can append current_episode_agent_returns
        self.agent_returns.append(current_episode_agent_returns)
        # env_info.pop('agent_rewards')


        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        #ToDo (Daniel, Peter, Jan9th 2023): check if this change is appropriate
        # if not test_mode:
        self.t_env += self.t

        cur_returns.append(episode_return)
       

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
            self._log_info(log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            #TODO: DCM 040123 this is where I will log the histograms:
            # if self.args.use_rnn:
            #     self._log_hidden_states(self.mac.hidden_states, self.t_env)

            # print('modules', self.mac.agent)
            # for index, layer in enumerate(self.mac.agent._modules):
            #     print('layer', layer)
            # for layer in self.mac.agent.children():
            #     if isinstance(layer, nn.Linear):
            #         print('linear layer', layer)
            #         print(layer.state_dict()['weight'])
            #         print(layer.state_dict()['bias'])
            #     elif isinstance(layer,nn.GRUCell):
            #         print("GRU layer")
            #         print(layer)
            #         print(layer.state_dict()['weight_ih'])
            #         print(layer.state_dict()['bias_ih'])


            # time to log
            self._log_info(log_prefix)

            if hasattr(self.mac.action_selector, "epsilon"):
                # self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
                wandb.log({"epsilon": self.mac.action_selector.epsilon})
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        # self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        # self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        wandb.log({"return_mean": np.mean(returns),
                     "return_std": np.std(returns)})
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                # self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
                wandb.log({k + "_mean": v/stats["n_episodes"]})

        stats.clear()

    #info logger written by me:
    def _log_info(self,prefix):
        #this function will log each agent's mean reward value for each logging period.
        # TODO: this needs to be automated for any number of agents 
    
        n_agents = self.args.n_agents
    
        agent_returns = np.array(self.agent_returns)
        
        # self.logger.log_stat(prefix + "agent_0_mean_returns",np.mean(array[:,0]), self.t_env)
        # self.logger.log_stat(prefix + 'agent_0_return_std', np.std(array[:,0]), self.t_env)
               
        # self.logger.log_stat(prefix + "agent_1_mean_returns",np.mean(array[:,1]), self.t_env)
        # self.logger.log_stat(prefix + 'agent_1_return_std', np.std(array[:,1]), self.t_env)
        # TODO: sept 17, this needs to be tested
        for n in range(n_agents):
            # self.logger.log_stat(prefix + 'agent_' + str(n) + '_mean_returns', np.mean(array[:,n]), self.t_env)
            # self.logger.log_stat(prefix + 'agent_' + str(n) + '_return_std', np.std(array[:,n]), self.t_env)
            wandb.log({'agent_' + str(n) + '_mean_returns': np.mean(agent_returns[:,n]),
                        'agent_' + str(n) + '_return_std': np.std(agent_returns[:,n])})

        # TODO: works
        self.agent_returns = [] 

    #Might be total overkill, only use if you need to log hidden states for inspetction!
    def _log_hidden_states(self, weights, stats):
        '''
        This method logs the hidden states of the GRU if it is present.
        '''
        for i, weight in enumerate(weights):
            layer = 'Hidden State' + str(i)
            # print(weight)
            self.logger.log_hist(layer, weight, stats)

        #ToDo: this is where DCM will log the histograms for hidden states, needs to be cleaned up
        data = [[w] for w in weights]
        table = wandb.Table(data=data, columns=["scores"])
        wandb.log({'my_histogram': wandb.plot.histogram(table, "scores",
                                                        title="Prediction Score Distribution")})

    def _log_weights(self, weights, stats):
        print('In _log_weights')
        print(weights)
        print(stats)