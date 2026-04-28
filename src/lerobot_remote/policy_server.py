"""Generic Policy Server for LeRobot Remote.

This module defines the interface for policy servers. Any algorithm
(ACT, YOLO+IK, hardcoded, etc.) can be used as long as it implements
the PolicyServer interface.
"""

import asyncio
import http
import logging
import socket
import time
import traceback
from abc import ABC, abstractmethod
from typing import Optional

import websockets.asyncio.server as ws_server
import websockets.frames

from lerobot_remote.protocol import (
    Action,
    ErrorCode,
    HealthCheckPath,
    Observation,
    ProtocolVersion,
    ServerMetadata,
    Serialization,
)

logger = logging.getLogger(__name__)


class PolicyServer(ABC):
    """Abstract base class for policy servers.

    Implement this interface to create any policy server for LeRobot Remote.

    Example:
        class MyACTPolicy(PolicyServer):
            def __init__(self, checkpoint_path):
                self.model = load_model(checkpoint_path)

            def infer(self, obs: Observation) -> Action:
                action = self.model.predict(obs.image, obs.state)
                return Action(action=action)

            def get_metadata(self) -> ServerMetadata:
                return ServerMetadata(
                    name="my_act",
                    algorithm="act",
                    description="ACT policy from checkpoint"
                )
    """

    @abstractmethod
    def infer(self, obs: Observation) -> Action:
        """Run inference on observation.

        This is the main method that edge clients call via WebSocket.

        Args:
            obs: Observation from the robot (image + state)

        Returns:
            Action to be executed by the robot
        """
        pass

    @abstractmethod
    def get_metadata(self) -> ServerMetadata:
        """Get server metadata.

        Returns:
            ServerMetadata containing server info
        """
        pass

    def start(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the server.

        Args:
            host: Host to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.serialization = Serialization()

        logger.info(f"Starting {self.get_metadata().name} on {host}:{port}")

        asyncio.run(self._run_server())

    async def _run_server(self) -> None:
        """Run the async WebSocket server."""
        async with ws_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: ws_server.ServerConnection) -> None:
        """Handle a client connection."""
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")

        try:
            # Send metadata on connect
            metadata = self.get_metadata().to_dict()
            await websocket.send(self.serialization.pack(metadata))
            logger.info(f"Sent metadata to {client_addr}: {metadata}")

            prev_total_time = None

            while True:
                try:
                    start_time = time.monotonic()

                    # Receive observation
                    raw_data = await websocket.recv()
                    obs_dict = self.serialization.unpack(raw_data)
                    obs = Observation.from_dict(obs_dict)

                    # Run inference
                    infer_start = time.monotonic()
                    action = self.infer(obs)
                    infer_time = time.monotonic() - infer_start

                    # Prepare response
                    response = action.to_dict()
                    if action.server_timing is None:
                        response["server_timing"] = {}

                    response["server_timing"]["infer_ms"] = infer_time * 1000
                    if prev_total_time is not None:
                        response["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                    # Send action
                    await websocket.send(self.serialization.pack(response))
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
    async def _health_check(connection, request) -> Optional[ws_server.Response]:
        """Handle health check requests."""
        if request.path == HealthCheckPath:
            return connection.respond(http.HTTPStatus.OK, b"OK\n")
        return None


def get_local_ip() -> str:
    """Get the local IP address of this machine."""
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)
