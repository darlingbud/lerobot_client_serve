"""Generic LeRobot Robot wrapper for lerobot_remote.

This module provides a generic robot interface that uses LeRobot's robot implementations.
Any LeRobot-compatible robot can be used by providing the appropriate config.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np

from lerobot_remote.protocol import Observation
from lerobot_remote.robot_client import RobotClient

logger = logging.getLogger(__name__)


@dataclass
class LeRobotRobotConfig:
    """Configuration for a LeRobot robot.

    Attributes:
        robot_type: Type of robot (e.g., "so101_follower", "so100_follower")
        port: Serial port for robot connection (e.g., "/dev/ttyUSB0")
        cameras: Dictionary of camera configs
        use_degrees: Whether to use degrees for motor positions
    """

    robot_type: str = "so101_follower"
    port: str = "/dev/ttyUSB0"
    cameras: Dict[str, Any] = field(default_factory=dict)
    use_degrees: bool = False

    @classmethod
    def from_dict(cls, d: Dict) -> "LeRobotRobotConfig":
        """Create config from dictionary."""
        return cls(**d)


class LeRobotRobot(RobotClient):
    """Generic robot client using LeRobot's robot implementations.

    This wraps any LeRobot-compatible robot to provide the RobotClient interface
    for lerobot_remote.

    Supported robots:
        - so101_follower: SO-101 robot arm
        - so100_follower: SO-100 robot arm
        - And any other LeRobot-compatible robot

    Usage:
        config = LeRobotRobotConfig(
            robot_type="so101_follower",
            port="/dev/ttyUSB0",
            cameras={"front": {"type": "opencv", "port": 0}},
        )
        robot = LeRobotRobot(config)
        robot.connect()

        obs = robot.get_observation()
        robot.execute(action)
    """

    def __init__(self, config: LeRobotRobotConfig):
        """Initialize the robot.

        Args:
            config: LeRobotRobotConfig configuration
        """
        self.config = config
        self._robot = None

    def connect(self) -> bool:
        """Connect to the robot.

        Returns:
            True if connected successfully, False otherwise.
        """
        try:
            from lerobot.robots.so_follower.so_follower import SOFollower
            from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
            from lerobot.cameras import CameraConfig

            # Build cameras config
            cameras = {}
            for name, cam_cfg in self.config.cameras.items():
                if isinstance(cam_cfg, dict):
                    cameras[name] = CameraConfig(
                        type=cam_cfg.get("type", "opencv"),
                        port=cam_cfg.get("port", 0),
                        fps=cam_cfg.get("fps", 30),
                        width=cam_cfg.get("width", 640),
                        height=cam_cfg.get("height", 480),
                    )
                else:
                    cameras[name] = cam_cfg

            # Create LeRobot config
            lr_config = SOFollowerRobotConfig(
                id="local",
                port=self.config.port,
                cameras=cameras,
                use_degrees=self.config.use_degrees,
            )

            # Create and connect robot
            self._robot = SOFollower(lr_config)
            self._robot.connect()

            logger.info(f"Connected to {self.config.robot_type} on {self.config.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            import traceback
            traceback.print_exc()
            return False

    def disconnect(self) -> None:
        """Disconnect from the robot."""
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None

    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._robot is not None and self._robot.is_connected

    def get_observation(self) -> Observation:
        """Get current observation from the robot.

        Returns:
            Observation containing image and joint state.
        """
        if not self._robot:
            raise ConnectionError("Robot not connected")

        # Get observation from LeRobot robot
        lr_obs = self._robot.get_observation()

        # Convert LeRobot observation to our format
        return self._to_observation(lr_obs)

    def execute(self, action: np.ndarray) -> None:
        """Execute action on the robot.

        Args:
            action: Action values (joint positions or deltas)
        """
        if not self._robot:
            raise ConnectionError("Robot not connected")

        # Convert action to LeRobot format
        lr_action = self._to_lerobot_action(action)

        # Execute
        self._robot.send_action(lr_action)

    def _to_lerobot_action(self, action: np.ndarray) -> Dict[str, float]:
        """Convert numpy action array to LeRobot action dict."""
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

        lr_action = {}
        for i, name in enumerate(joint_names):
            if i < len(action):
                lr_action[f"{name}.pos"] = float(action[i])

        return lr_action

    def _to_observation(self, lr_obs) -> Observation:
        """Convert LeRobot observation to our Observation format."""
        # Extract image
        image = None
        for key in ["observation.images.front", "observation.images.cam", "camera"]:
            if hasattr(lr_obs, key) or (isinstance(lr_obs, dict) and key in lr_obs):
                image = lr_obs[key] if isinstance(lr_obs, dict) else getattr(lr_obs, key)
                break

        if image is None:
            # No image found, return blank
            image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Convert to numpy if needed
        if hasattr(image, "numpy"):
            image = image.numpy()
        if isinstance(image, np.ndarray) and image.ndim == 4:
            # Remove batch dim
            image = image[0]
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[0] == 3:
            # CHW -> HWC
            image = np.transpose(image, (1, 2, 0))

        # Extract state
        state = []
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        for name in joint_names:
            key = f"{name}.pos"
            if isinstance(lr_obs, dict) and key in lr_obs:
                state.append(float(lr_obs[key]))
            elif hasattr(lr_obs, key):
                state.append(float(getattr(lr_obs, key)))
            else:
                state.append(0.0)

        return Observation(image=image, state=state)


class SO101Robot(LeRobotRobot):
    """Convenience class for SO-101 robot.

    预设 SO-101 的默认配置。
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        camera_port: int = 0,
    ):
        """Initialize SO-101 robot.

        Args:
            port: Serial port for robot arm
            camera_port: Camera device port
        """
        config = LeRobotRobotConfig(
            robot_type="so101_follower",
            port=port,
            cameras={"front": {"type": "opencv", "port": camera_port}},
        )
        super().__init__(config)
