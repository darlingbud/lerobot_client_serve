"""LeRobot Robot wrapper for lerobot_remote.

Uses LeRobot's SOFollower (SO-100/101) to interface with physical robots.
Supports any LeRobot-compatible robot through configuration.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from lerobot_remote.protocol import Observation
from lerobot_remote.robot_client import RobotClient

logger = logging.getLogger(__name__)


@dataclass
class LeRobotRobotConfig:
    """Configuration for a LeRobot robot."""

    robot_type: str = "so_follower"
    port: str = "/dev/robot_follower"
    cameras: Dict[str, Any] = field(default_factory=dict)
    use_degrees: bool = False
    calibration: Optional[Dict[str, Any]] = None  # Optional pre-loaded calibration

    @classmethod
    def from_dict(cls, d: Dict) -> "LeRobotRobotConfig":
        return cls(**d)


class LeRobotRobot(RobotClient):
    """Robot client using LeRobot's robot implementations.

    Supports SO-100, SO-101, and other LeRobot-compatible robots.
    """

    def __init__(self, config: LeRobotRobotConfig):
        self.config = config
        self._robot = None

    def connect(self) -> bool:
        """Connect to the robot.

        Returns:
            True if connected successfully, False otherwise.
        """
        try:
            from lerobot.robots.so_follower import SOFollower
            from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
            from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

            # Build cameras config
            cameras = {}
            for name, cam_cfg in self.config.cameras.items():
                if isinstance(cam_cfg, dict):
                    cameras[name] = OpenCVCameraConfig(
                        cam_cfg.get("port", 0),
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

            # Create robot and connect (skip interactive calibration)
            self._robot = SOFollower(lr_config)
            self._robot.connect(calibrate=False)

            # Read and set calibration from motors (non-interactive)
            cal = self._robot.bus.read_calibration()
            self._robot.bus.calibration = cal
            logger.info(f"Robot calibration loaded from motors: {list(cal.keys())}")

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

        # Get raw observation from LeRobot robot
        obs_dict = self._robot.get_observation()

        # Extract image (if cameras are configured)
        image = None
        for key in ["observation.images.front", "observation.images.cam"]:
            if key in obs_dict:
                image = obs_dict[key]
                break

        if image is None:
            image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Convert to numpy if needed
        if hasattr(image, "numpy"):
            image = image.numpy()
        if isinstance(image, np.ndarray) and image.ndim == 4:
            image = image[0]
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

        # Extract state (joint positions)
        state = []
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        for name in joint_names:
            key = f"{name}.pos"
            if key in obs_dict:
                state.append(float(obs_dict[key]))
            else:
                state.append(0.0)

        return Observation(image=image, state=state)

    def execute(self, action: np.ndarray) -> None:
        """Execute action on the robot.

        Args:
            action: Action values for joints in this order:
                    [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        """
        if not self._robot:
            raise ConnectionError("Robot not connected")

        # Convert numpy action to LeRobot format
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        lr_action = {}
        for i, name in enumerate(joint_names):
            if i < len(action):
                lr_action[f"{name}.pos"] = float(action[i])

        self._robot.send_action(lr_action)


class SO101Robot(LeRobotRobot):
    """Convenience class for SO-101 robot.

    预设 SO-101 的默认配置。
    """

    def __init__(
        self,
        port: str = "/dev/robot_follower",
        camera_port: int = 0,
    ):
        """Initialize SO-101 robot.

        Args:
            port: Serial port for robot arm
            camera_port: Camera device port
        """
        config = LeRobotRobotConfig(
            robot_type="so_follower",
            port=port,
            cameras={"front": {"port": camera_port}},
        )
        super().__init__(config)
