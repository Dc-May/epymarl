import asyncio
import datetime
import sys
import json
import os
import time
import asyncio

import socketio
from sqlalchemy_utils import database_exists
import databases
import dataset
from TREX_Core._clients.sim_controller.training_controller import TrainingController
from TREX_Core._utils import utils, db_utils


class Controller:
    '''
    Sim controller takes over timing control from market
    in RT mode, the market will move onto the next round regardless of whether participants finish their actions or not
    in sim mode, market rounds will not start until all the participants have joined
    auction rounds (therefore time) will not advance until all participants have completed all of their actions

    in order to do this, the sim controller needs to know the following things about the simulation
    this information can be obtained from the config json file

    1. sim start time
    2. number of participants, as well as their IDs

    The sim controller has special permission to see when participants join the market
    '''
    # Intialize client related data
    def __init__(self, sio_client, market_id, **kwargs):
        self.__client = sio_client
        self.__market_id = market_id


        # self.__learning_agents = [participant for participant in self.__config['participants'] if
        #                          'learning' in self.__config['participants'][participant]['trader'] and
        #                          self.__config['participants'][participant]['trader']['learning']]
        #
        # self.__static_agents = [participant for participant in self.__config['participants'] if
        #                         'learning' not in self.__config['participants'][participant]['trader'] or
        #                          not self.__config['participants'][participant]['trader']['learning']]




    async def delay(self, s):
        '''This function delays the sim by s seconds using the client sleep method so as not to interrupt the thread control. 

        Params: 
            int or float : number of seconds to 
        '''
        await self.__client.sleep(s)

    # Register client in server
    async def register(self):
        client_data = {
            'type': ('env_controller', ''),
            'id': 'env_controller',
            'market_id': self.__market_id
        }
        await self.__client.emit('register', client_data, namespace='/simulation', callback=self.register_success)

    # If client has not connected, retry registration
    async def register_success(self, success):
        await self.delay(utils.secure_random.random() * 10)
        print("register success", success)
        if not success:
            await self.register()
        # self.status['registered_on_server'] = True



class NSSimulation(socketio.AsyncClientNamespace):

    def __init__(self, controller):
        super().__init__(namespace='/simulation')
        self.controller = controller

    async def on_connect(self):
        print("Connected or something")
        await self.controller.register()

    async def on_pre_transition_data(self, data):
        print(data)
    # async def on_disconnect(self):
    #   print('disconnected from simulation')




