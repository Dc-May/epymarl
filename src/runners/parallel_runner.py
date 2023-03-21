import charset_normalizer.utils
from epymarl.src.envs import REGISTRY as env_REGISTRY
from functools import partial
from epymarl.src.components.episode_buffer import EpisodeBatch
import multiprocessing as mp
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
import os
import sys
import subprocess
from main_launch import _prep_trex #ToDo: obv this needs to be somewhere else at some point
import TREX_Core

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger, external_launch=False):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        pool_size = mp.cpu_count() - 2
        pool = mp.Pool(processes=pool_size)

        # Create the environment arguments with unique env_id
        # self.args.env['use_tensorboard'] = self.args.use_tensorboard
        #self.args.env['unique_token'] = self.args.use_tensorboard #ToDO: find a betetr place for this line
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i
            env_args[i]['env_id'] = i #ToDO: adjust this with above env-id

        #create the trex-launch lists, force proper port allocation and add env_ids to the gym agent config
        # port separation is done by env_id, since both have to be unique

        # trex_results = pool.map_async(run_subprocess, trex_launch_list)
        trex_launch_lists = [_prep_trex() for i in range(self.batch_size)] #create independent copies, [x]*y would not do so
        for trex_launch_list_nbr in range(len(trex_launch_lists)): #for every launch list
            print('launch list number:', trex_launch_list_nbr)
            env_id =env_args[trex_launch_list_nbr]['env_id']
            print('env_number:', env_id)

            for client_nbr in range(len(trex_launch_lists[trex_launch_list_nbr])):


                #get the index of the client's port number
                client_args = trex_launch_lists[trex_launch_list_nbr]
                client_args = client_args[client_nbr]
                client_args = client_args[1]
                # FixMe: Steven tells me that this should not be necessary
                port_is_in_client = [True for i, s in enumerate(client_args) if s.startswith('--port=')]
                if port_is_in_client:
                    port_index = [i for i, s in enumerate(client_args) if s.startswith('--port=')]
                    assert len(port_index) == 1
                    port_index = port_index[0]
                # #make sure the port index of each launch list is unique, but also the same for each client in the launch list
                    trex_launch_lists[trex_launch_list_nbr][client_nbr][1][port_index] = '--port=' + str(42069 + env_id*10)

                #add the env_id to the gym agent config
                is_trader = [True for i, s in enumerate(client_args) if s.startswith('--trader=')]
                if is_trader:
                    #if we haev a gym agent, find the trader info
                    trader_index = [i for i, s in enumerate(client_args) if s.startswith('--trader=')]
                    assert len(trader_index) == 1
                    trader_index = trader_index[0]
                    trader_info = trex_launch_lists[trex_launch_list_nbr][client_nbr][1][trader_index]
                    # print('trader info:', trader_info)
                    #add the env_id to the trader info
                    added_env_argument = '"env_id": ' + str(env_id)
                    trader_info = trader_info[:-1] + ',' + added_env_argument + trader_info[-1]
                    # print('trader info with env added:', trader_info)
                    trex_launch_lists[trex_launch_list_nbr][client_nbr][1][trader_index] = trader_info

        launch_list = []
        for trex_launch_list in trex_launch_lists:
            launch_list.extend(trex_launch_list)
        trex_results = pool.map_async(run_subprocess, launch_list)

        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_worker_args = [(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))) for env_arg, worker_conn in zip(env_args, self.worker_conns)]
        env_results = pool.starmap_async(env_worker, env_worker_args)
        # pool.close()
        # pool.terminate()

        print('expecting', self.batch_size, 'envs')
        [parent_conn.send(('ping', None)) for parent_conn in self.parent_conns]
        env_nbrs = [parent_conn.recv() for parent_conn in self.parent_conns]
        # print('got', env_nbrs, 'envs')
        # these three lines get the episode limit from the environment 
        self.parent_conns[0].send(("get_env_info", None))

        self.env_info = self.parent_conns[0].recv()
        print('env_0 info:', self.env_info)
        print('self.env_info["episode_limit"]')
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0
        # this sets up the arrays and dictionaries necessary for saveing the metrics 
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        #TODO: this is what i have added sept 16
        self.agent_returns = []
        self.agent_returns_array = []

        self.log_train_stats_t = -100000
        print('init of parallel runner done')

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch,
                                 scheme,
                                 groups,
                                 self.batch_size,
                                 self.episode_limit + 1,
                                 preprocess=preprocess,
                                 device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()
        self.agent_returns = []
        
        # Reset the envs
        [parent_conn.send(("reset", None)) for parent_conn in self.parent_conns]


        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        data_envs = [parent_conn.recv() for parent_conn in self.parent_conns]
        for data in data_envs:
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        # send the pre_transition data to the batch for the 0th timestep
        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        agent_returns = [0 for _ in range(self.batch_size)]
        curr_agent_returns=[[0]*self.env_info['n_agents'] for _ in range(self.batch_size)] #TODO: Sept 20; I think that this is what is causing the extension is not working properly.

        
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()
        
            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx])) #SEND THE STEP command to all the parallel envs
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            # if envs are all done, break the loop 
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv() #GETS THE DATA DICTIONARY FROM EACH PARALLEL ENV 
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    # ToDo: Mar3, Daniel - Ask @Peter what 'info' should all contain and where to get
                    # This seems to be the information for individual agent rewards for summing sth whatever??
                    #collect the individual reward data from the info parameter.
                    agent_returns[idx] = data["info"]["agent_rewards"] # clean data info
                    data['info'].pop('agent_rewards')

                    #ToDO: later on we need to adjust this code so that we can also track individual agent returns!
                    episode_returns[idx] += np.sum(data["reward"]) # Add the current step return to the episode returns for each process
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False

                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            self.agent_returns_array.append(tuple(agent_returns))

            # print('after append',self.agent_returns)
            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)
        
        if not test_mode:
            self.t_env += self.env_steps_this_run


        #ToDo: figure out what get stats is supposed to provide, eh?
        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        for k in set.union(*[set(d) for d in infos]):
            buffer = 0
            for d in infos:
                num = d.get(k, 0)
                try:
                    buffer = np.sum([buffer, num])
                except:
                    print("error adding", num, "to", buffer)
                    print("k", k)
                    print("d", d)
                    raise ValueError
            cur_stats[k] = buffer
        # cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)
        # print('Current returns array',cur_returns)
        
        #FIXME: Sept 20, this extension does nothing 
        # TODO: sept 20, collapse the 50 samples into one return sum for each world here 
        # env_dummy = [0 for agent in range(self.args.n_agents)]

        for _, run in enumerate(self.agent_returns):
            env_dummy = [0 for agent in range(self.args.n_agents)]
            for env_index, env_result in enumerate(run):
                print('eun',env_index)
                # This is where we add each run result up for each agent so that we are left with one sum per env
                # env_result = [agent0, agent1]
                # need a dummy list here
                # print('run, env_result', run, env_result)
                print('curr_agent_returns', len(curr_agent_returns))
                for agent in range(self.args.n_agents):
                    print('run, env_result, agent', len(run),len(env_result),agent)
                    curr_agent_returns[env_index][agent] += env_result[agent]


                
            # this is where you would send the env_dummy to the curr_agent_returns 
            
        # FIXME: Sept 23, this does not extend the array         
        # agent_returns_array.append(curr_agent_returns)
        self.agent_returns_array.extend(curr_agent_returns)
        # print('Current agent returns array', len(self.agent_returns_array))
        
        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            
            # self._log_agent_rewards(self.agent_returns_array, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        # print('returns that they are logging logging', len(returns))
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                shape = v.shape if hasattr(v, "shape") else None
                if shape is not None:
                    if len(shape) == 0:
                        v = np.mean(v)
                    elif len(shape) >=1:
                        v = np.mean(v, axis=-1)
                    else:
                        raise ValueError("Unknown shape: {}".format(shape))

                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
    
    def _log_agent_rewards(self, agent_returns, prefix):
        '''

        '''
        # print('returns that i am logging' , len(agent_returns))
        np_agent_returns = np.array(agent_returns)
       
        test = np.mean(np_agent_returns, axis=0)
        # print(test)
        n_agents = self.args.n_agents
        # final_array = [list() for n in range(n_agents)]
        # for _, batch in enumerate(agent_returns):
        #     for _, step in enumerate(batch):
        #         for agent in range(n_agents):
        #             # print('step',agent ,step[agent])
        #             final_array[agent].append(step[agent])

        # print(final_array) 
        # np_final_array= np.array(final_array)
        # for n in range(n_agents):
        #     mean = np.mean(np_final_array[n])
            # print('n', mean)
        # FIXME:Sept 17 this records but i have no idea why what it records does not line up with what is logged. 
        for n in range(n_agents): 
            self.logger.log_stat(prefix + 'agent_' + str(n) + '_mean_returns',np.mean(np_agent_returns[:,n]) , self.t_env)
            self.logger.log_stat(prefix + 'agent_' + str(n) + '_return_std', np.std(np_agent_returns[:,n]), self.t_env)
        # TODO:Sept 20 agent_returns cannot be cleared like this. 
        agent_returns.clear()


def run_subprocess(args : list):

    #epymarl launch command is
    # ('D:\\TREX-DRL-Project\\ePyMarl\\src\\main.py',
    # ['--config=ippo_ns', '--env-config=TREX', 'with', 'env_args.n_agents=4', 'env_args.TREX_config=GymIntegration_1Agent_Price'],
    # )
    print('attempting to launch TREX', flush=True)
    # try:

    subprocess.run([sys.executable, args[0], *args[1]])  # FIXME: August 29 2022,  this is coded for postix system

        # subproces.run([sys.executable, args[0], *args[1]]) This is probably the most
    # except:
    # print('failed to launch subprocess, plz investigate')


def env_worker(remote, env_fn):

    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            # TODO: This is where my env change interacts with the code
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info  
            })
            
            # env_info.pop('agent_rewards')
        elif cmd == "reset":
            # print('resetting env', flush=True)
            reset = env.reset()
            # print('getting_state', flush=True)
            state = env.get_state()
            # print('getting available actions', flush=True)
            avail_actions = env.get_avail_actions()
            # print('getting obs', flush=True)
            obs = env.get_obs()
            # print('sending data', flush=True)
            package = {
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs
            }
            remote.send(package)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            try:
                info = env.get_env_info()
            except:
                print('failed to access env info of environment')
                info = None
            print('env info', info)
            remote.send(info)
        elif cmd == "get_stats":
            stats = env.get_stats()
            remote.send(stats)
        elif cmd == "ping":
            remote.send(env.ping())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        # print('pickle Default protocol', pickle.DEFAULT_PROTOCOL)
        self.x = pickle.loads(ob)

