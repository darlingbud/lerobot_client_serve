"""LeRobot WebSocket Policy Server - runs on cloud (4090 server).

Receives observations from edge, runs policy inference, returns actions.
"""

import asyncio
import http
import logging
import socket
import time
import traceback
from typing import Optional

import websockets.asyncio.server as ws_server
import websockets.frames

from lerobot_remote.msgpack_numpy import Packer, NumpyExt

logger = logging.getLogger(__name__)


class LeRobotWebsocketPolicyServer:
    """LeRobot Policy Server with WebSocket interface.

    Similar to OpenPI's WebsocketPolicyServer but for LeRobot policies.

    Protocol:
        1. Client connects -> Server sends metadata (policy info)
        2. Client sends observation -> Server returns action
        3. Repeat...
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: Optional[dict] = None,
    ):
        """Initialize server.

        Args:
            policy: LeRobot policy with select_action(obs) method
            host: Host to bind to
            port: Port to listen on
            metadata: Optional metadata to send to client on connect
        """
        self.policy = policy
        self.host = host
        self.port = port
        self.metadata = metadata or {
            "framework": "lerobot",
            "model": "act",
        }
        self._running = False

        # Set websocket log level
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self):
        """Start the server synchronously."""
        asyncio.run(self.run())

    async def run(self):
        """Run the async server."""
        logger.info(f"Starting LeRobot Policy Server on {self.host}:{self.port}")

        async with ws_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: ws_server.ServerConnection):
        """Handle a client connection."""
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")

        packer = Packer()

        try:
            # Send metadata on connect
            await websocket.send(packer.pack(self.metadata))
            logger.info(f"Sent metadata to {client_addr}")

            prev_total_time = None

            while True:
                try:
                    start_time = time.monotonic()

                    # Receive observation
                    raw_data = await websocket.recv()
                    obs = self._unpackb(raw_data)

                    # Run inference
                    infer_start = time.monotonic()
                    result = self.policy.infer(obs)
                    infer_time = time.monotonic() - infer_start

                    # Add timing info
                    result["server_timing"] = {
                        "infer_ms": infer_time * 1000,
                    }
                    if prev_total_time is not None:
                        result["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                    # Send action
                    await websocket.send(packer.pack(result))
                    prev_total_time = time.monotonic() - start_time

                    logger.debug(
                        f"Inference: {infer_time*1000:.1f}ms, "
                        f"Total: {prev_total_time*1000:.1f}ms"
                    )

                except websockets.ConnectionClosed:
                    logger.info(f"Client disconnected: {client_addr}")
                    break

        except Exception as e:
            logger.error(f"Error handling client {client_addr}: {e}")
            logger.error(traceback.format_exc())
            try:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error",
                )
            except Exception:
                pass
            raise

    @staticmethod
    def _unpackb(data):
        """Unpack msgpack data with numpy support."""
        import msgpack

        def decode(obj):
            if isinstance(obj, dict) and "__ndarray__" in obj:
                import numpy as np

                arr = np.frombuffer(obj["data"], dtype=obj["dtype"])
                return arr.reshape(obj["shape"])
            return obj

        return msgpack.unpackb(data, raw=False, object_hook=decode)

    @staticmethod
    async def _health_check(connection, request):
        """Handle health check requests."""
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None


def get_local_ip() -> str:
    """Get the local IP address of this machine."""
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)
