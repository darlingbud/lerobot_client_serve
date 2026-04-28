"""Generic Robot Client for LeRobot Remote.

This module defines the interface for robot clients. It is agnostic to the
specific robot hardware used - any robot that implements the interface can be used.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from lerobot_remote.protocol import Observation, Serialization

logger = logging.getLogger(__name__)


class RobotClient(ABC):
    """Abstract base class for robot clients.

    Implement this interface to use any robot with LeRobot Remote.

    Example:
        class MyRobot(RobotClient):
            def get_observation(self) -> Observation:
                # Capture image and get joint state
                image = self.camera.capture()
                state = self.robot.get_joint_positions()
                return Observation(image=image, state=state)

            def execute(self, action: np.ndarray) -> None:
                self.robot.send_joint_targets(action)
    """

    @abstractmethod
    def get_observation(self) -> Observation:
        """Get current observation from the robot.

        Returns:
            Observation containing image and joint state.
        """
        pass

    @abstractmethod
    def execute(self, action: np.ndarray) -> None:
        """Execute action on the robot.

        Args:
            action: Action values (e.g., joint deltas or absolute positions)
        """
        pass

    def connect(self) -> bool:
        """Connect to the robot.

        Returns:
            True if connection successful, False otherwise.
        """
        return True

    def disconnect(self) -> None:
        """Disconnect from the robot."""
        pass

    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return True


class SO101RobotClient(RobotClient):
    """SO-101 robot client using the RobotAgent interface.

    This is a concrete implementation for the SO-101/RO-101 robot arm.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        camera_id: int = 0,
    ):
        """Initialize SO-101 robot client.

        Args:
            host: Robot server host
            port: Robot server port
            camera_id: Camera device ID
        """
        self.host = host
        self.port = port
        self.camera_id = camera_id
        self._robot = None
        self._camera = None

    def connect(self) -> bool:
        """Connect to the SO-101 robot."""
        try:
            from robot_agent import RobotAgent

            self._robot = RobotAgent(host=self.host, port=self.port)
            if not self._robot.connect():
                logger.error("Failed to connect to robot")
                return False

            logger.info(f"Connected to SO-101 robot at {self.host}:{self.port}")
            return True

        except ImportError:
            logger.error("robot_agent not found. Make sure SO-101 skill is installed.")
            return False

    def disconnect(self) -> None:
        """Disconnect from the robot."""
        if self._robot:
            self._robot.disconnect()
            self._robot = None

    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._robot is not None

    def get_observation(self) -> Observation:
        """Get current observation from the robot."""
        if not self._robot:
            raise ConnectionError("Robot not connected")

        # Get state from robot
        state_resp = self._robot.get_observation()
        state = self._extract_joint_state(state_resp)

        # Get image from camera
        image = self._capture_image()

        return Observation(image=image, state=state)

    def execute(self, action: np.ndarray) -> None:
        """Execute action on the robot."""
        if not self._robot:
            raise ConnectionError("Robot not connected")

        # Convert action array to joint positions
        # Assuming action is [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

        if len(action) != len(joint_names):
            raise ValueError(f"Action size {len(action)} doesn't match joint count {len(joint_names)}")

        # Create position dict
        positions = dict(zip(joint_names, action))
        self._robot.set_positions(**positions)

    def _extract_joint_state(self, state_resp: Dict) -> List[float]:
        """Extract joint positions from robot state response."""
        # Expected format: {"joints": {joint_name: position, ...}}
        joints = state_resp.get("joints", {})

        joint_order = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

        state = []
        for name in joint_order:
            if name in joints:
                state.append(float(joints[name]))
            else:
                state.append(0.0)

        return state

    def _capture_image(self) -> np.ndarray:
        """Capture image from camera."""
        import cv2

        if self._camera is None:
            self._camera = cv2.VideoCapture(self.camera_id)

        ret, frame = self._camera.read()
        if not ret:
            # Return blank image if camera fails
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Convert BGR to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class SimulatedRobotClient(RobotClient):
    """Simulated robot for testing without hardware."""

    def __init__(self):
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        logger.info("Simulated robot connected")
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_observation(self) -> Observation:
        """Return random observation for testing."""
        import time

        # Random image and state
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        state = [0.0] * 6  # 6 joints

        return Observation(
            image=image,
            state=state,
            timestamp=time.time(),
        )

    def execute(self, action: np.ndarray) -> None:
        """Log action for testing."""
        logger.debug(f"Simulated execute: {action}")
