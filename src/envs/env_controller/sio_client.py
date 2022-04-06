import sys
import os
import asyncio
import socket
import socketio
import tenacity
from TREX_Core._utils import jkson
from env_controller import Controller as EnvController
from TREX_Core._clients.sim_controller.ns_common import NSDefault
from env_controller import NSSimulation

if os.name == 'posix':
    import uvloop
    uvloop.install()

class Client:
    # Initialize client data for sim controller
    def __init__(self, server_address, market_id):
        self.server_address = server_address
        self.sio_client = socketio.AsyncClient(reconnection=True,
                                               reconnection_attempts=100,
                                               reconnection_delay=1,
                                               reconnection_delay_max=5,
                                               randomization_factor=0.5,
                                               json=jkson)

        # Set client to controller class
        self.controller = EnvController(self.sio_client, market_id)
        self.sio_client.register_namespace(NSDefault(controller=self.controller))
        self.sio_client.register_namespace(NSSimulation(controller=self.controller))

    @tenacity.retry(wait=tenacity.wait_fixed(1) + tenacity.wait_random(0, 2))
    async def start_client(self):
        await self.sio_client.connect(self.server_address)
        await self.sio_client.wait()

    async def run(self):

        tasks = [
            asyncio.create_task(self.start_client())
        ]

        try:
            await asyncio.gather(*tasks)
        except SystemExit:
            for t in tasks:
                t.cancel()
            raise SystemExit

def __main():
    import socket
    import argparse
    import json
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--host', default=socket.gethostbyname(socket.getfqdn()), help='')
    parser.add_argument('--port', default=42069, help='')
    parser.add_argument('--market_id', default='', help='')
    args = parser.parse_args()

    # configs = json.loads(args.config)
    client = Client(server_address=''.join(['http://', args.host, ':', str(args.port)]),
                    market_id=args.market_id)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.run())

if __name__ == '__main__':
    import sys
    sys.exit(__main())
