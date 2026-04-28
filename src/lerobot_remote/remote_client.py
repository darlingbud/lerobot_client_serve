"""WebSocket Client for LeRobot Remote.

Connects to a PolicyServer and sends observations.
"""

import logging
import time
from typing import Optional

import websockets.sync.client as ws_client

from lerobot_remote.protocol import Action, Observation, ServerMetadata, Serialization

logger = logging.getLogger(__name__)


class RemoteRobotClient:
    """Client that connects to a PolicyServer and sends observations.

    This is the edge-side component that:
    1. Collects observations from the robot (via RobotClient)
    2. Sends them to the cloud PolicyServer
    3. Receives actions and forwards them to the robot

    Usage:
        # Create robot client (implements RobotClient interface)
        robot = SO101RobotClient(host="192.168.1.100", port=8765)

        # Create remote client
        client = RemoteRobotClient(
            robot=robot,
            server_url="ws://100.89.143.11:8000",
        )

        # Connect
        client.connect()

        # Control loop
        while True:
            obs = robot.get_observation()
            action = client.infer(obs)
            robot.execute(action.action)
    """

    def __init__(
        self,
        robot,
        server_url: str,
        api_key: Optional[str] = None,
        reconnect_interval: float = 5.0,
    ):
        """Initialize remote robot client.

        Args:
            robot: RobotClient instance (e.g., SO101RobotClient)
            server_url: WebSocket URL of the PolicyServer
            api_key: Optional API key for authentication
            reconnect_interval: Seconds between reconnection attempts
        """
        self.robot = robot
        self.server_url = server_url
        self.api_key = api_key
        self.reconnect_interval = reconnect_interval

        self._ws: Optional[ws_client.ClientConnection] = None
        self._server_metadata: Optional[ServerMetadata] = None
        self._serialization = Serialization()

    def connect(self) -> bool:
        """Connect to the PolicyServer.

        Returns:
            True if connected successfully, False otherwise.
        """
        logger.info(f"Connecting to {self.server_url}...")

        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None

        while True:
            try:
                self._ws = ws_client.connect(
                    self.server_url,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )

                # Receive metadata from server
                metadata_raw = self._ws.recv()
                metadata_dict = self._serialization.unpack(metadata_raw)
                self._server_metadata = ServerMetadata(**metadata_dict)

                logger.info(f"Connected! Server: {self._server_metadata}")
                return True

            except ConnectionRefusedError:
                logger.warning(f"Connection refused. Retrying in {self.reconnect_interval}s...")
                time.sleep(self.reconnect_interval)
            except Exception as e:
                logger.warning(f"Connection failed: {e}. Retrying in {self.reconnect_interval}s...")
                time.sleep(self.reconnect_interval)

    def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._ws:
            self._ws.close()
            self._ws = None

    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._ws is not None

    def infer(self, obs: Observation) -> Action:
        """Send observation to server and receive action.

        Args:
            obs: Observation from robot

        Returns:
            Action from server
        """
        if not self._ws:
            raise ConnectionError("Not connected to server")

        # Send observation
        data = self._serialization.pack(obs.to_dict())
        self._ws.send(data)

        # Receive response
        response = self._ws.recv()

        if isinstance(response, str):
            # String response means error
            raise RuntimeError(f"Inference error: {response}")

        response_dict = self._serialization.unpack(response)

        return Action(
            action=response_dict.get("action"),
            server_timing=response_dict.get("server_timing"),
        )

    def get_server_metadata(self) -> Optional[ServerMetadata]:
        """Get metadata from the server."""
        return self._server_metadata

    def infer_and_execute(self, obs: Optional[Observation] = None) -> Action:
        """Convenience method: get obs, infer, execute.

        Args:
            obs: Optional observation. If None, will get from robot.

        Returns:
            Action that was executed.
        """
        if obs is None:
            obs = self.robot.get_observation()

        action = self.infer(obs)
        self.robot.execute(action.action)
        return action
