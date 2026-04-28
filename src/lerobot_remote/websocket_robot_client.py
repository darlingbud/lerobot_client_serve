"""LeRobot WebSocket Robot Client - runs on edge (local robot machine).

Connects to cloud policy server, sends observations, receives actions,
and forwards them to the robot.
"""

import logging
import time
from typing import Dict, Optional, Tuple

import websockets.sync.client as ws_client

from lerobot_remote.msgpack_numpy import Packer

logger = logging.getLogger(__name__)


class LeRobotWebsocketRobotClient:
    """Robot client that connects to cloud LeRobot Policy Server.

    Sends observations (image + robot state) and receives actions.

    Protocol (matching OpenPI's WebsocketClientPolicy):
        1. Connect to server -> receive metadata
        2. Send observation -> receive action
        3. Repeat...

    Usage:
        client = LeRobotWebsocketRobotClient(
            robot=robot_agent,
            server_url="ws://100.89.143.11:8000",
        )

        # In control loop:
        obs = robot.get_observation()
        action = client.infer(obs)
        robot.execute(action)
    """

    def __init__(
        self,
        robot,
        server_url: str,
        api_key: Optional[str] = None,
        reconnect_interval: float = 5.0,
    ):
        """Initialize client.

        Args:
            robot: Robot client object with get_observation() and execute(action) methods
            server_url: WebSocket URL of the policy server (e.g., "ws://100.89.143.11:8000")
            api_key: Optional API key for authentication
            reconnect_interval: Seconds between reconnection attempts
        """
        self.robot = robot
        self.server_url = server_url
        self.api_key = api_key
        self.reconnect_interval = reconnect_interval

        self._packer = Packer()
        self._ws: Optional[ws_client.ClientConnection] = None
        self._server_metadata: Optional[Dict] = None

        # Connect to server
        self._connect()

    def _connect(self):
        """Connect to the policy server with retry logic."""
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
                self._server_metadata = self._unpackb(metadata_raw)

                logger.info(f"Connected! Server metadata: {self._server_metadata}")
                return

            except ConnectionRefusedError:
                logger.warning(f"Connection refused. Retrying in {self.reconnect_interval}s...")
                time.sleep(self.reconnect_interval)
            except Exception as e:
                logger.warning(f"Connection failed: {e}. Retrying in {self.reconnect_interval}s...")
                time.sleep(self.reconnect_interval)

    def get_server_metadata(self) -> Dict:
        """Get the metadata sent by the server on connect."""
        return self._server_metadata

    def infer(self, obs: Dict) -> Dict:
        """Send observation to server and receive action.

        Args:
            obs: Dictionary with:
                - image: Numpy array (H, W, C) or dict with 'data'
                - state: List or array of joint positions

        Returns:
            Dictionary with:
                - action: Numpy array of action values
                - server_timing: (optional) Timing info from server
        """
        if self._ws is None:
            raise ConnectionError("Not connected to server")

        # Pack and send observation
        data = self._packer.pack(obs)
        self._ws.send(data)

        # Receive response
        response = self._ws.recv()

        if isinstance(response, str):
            # String response means error
            raise RuntimeError(f"Inference error: {response}")

        return self._unpackb(response)

    def reset(self):
        """Reset the policy (placeholder for compatibility)."""
        pass

    def _unpackb(self, data):
        """Unpack msgpack data with numpy support."""
        import msgpack

        def decode(obj):
            if isinstance(obj, dict) and "__ndarray__" in obj:
                import numpy as np

                arr = np.frombuffer(obj["data"], dtype=obj["dtype"])
                return arr.reshape(obj["shape"])
            return obj

        return msgpack.unpackb(data, raw=False, object_hook=decode)

    def close(self):
        """Close the WebSocket connection."""
        if self._ws:
            self._ws.close()
            self._ws = None


class LeRobotRemoteRobot:
    """Robot wrapper that sends actions to cloud for inference.

    This provides a drop-in replacement for local robot control.
    Just use this instead of the local policy, and inference happens on cloud.
    """

    def __init__(
        self,
        robot,
        server_url: str,
        api_key: Optional[str] = None,
    ):
        """Initialize.

        Args:
            robot: Local robot client (e.g., RobotAgent) with get_observation() and execute() methods
            server_url: WebSocket URL of the policy server
            api_key: Optional API key
        """
        self.robot = robot
        self.ws_client = LeRobotWebsocketRobotClient(
            robot=robot,
            server_url=server_url,
            api_key=api_key,
        )

    def get_observation(self) -> Dict:
        """Get observation from robot."""
        return self.robot.get_observation()

    def execute(self, action) -> Dict:
        """Execute action on robot.

        Args:
            action: Action dict from ws_client.infer(), or directly from self.ws_client.infer(obs)
        """
        if isinstance(action, dict) and "action" in action:
            action = action["action"]
        return self.robot.execute(action)

    def infer_and_execute(self, obs: Dict) -> Dict:
        """Get observation, run inference on cloud, execute action.

        Convenience method combining get_observation() + infer() + execute().
        """
        action = self.ws_client.infer(obs)
        return self.execute(action)
