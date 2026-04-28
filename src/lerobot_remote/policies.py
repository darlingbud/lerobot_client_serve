"""LeRobot ACT Policy Server.

Concrete implementation of PolicyServer for LeRobot ACT policies.
Uses official LeRobot preprocessor/postprocessor pipelines for proper normalization.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from lerobot_remote.policy_server import PolicyServer
from lerobot_remote.protocol import Action, Observation, ServerMetadata

logger = logging.getLogger(__name__)


class LeRobotACTPolicy(PolicyServer):
    """LeRobot ACT policy server.

    Loads a LeRobot ACT checkpoint and runs inference.
    Uses LeRobot's preprocessor/postprocessor pipelines for proper normalization.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        """Initialize LeRobot ACT policy.

        Args:
            checkpoint_path: Path to the checkpoint directory
            device: Device to run inference on
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._load_policy()
        self._load_processors()

    def _load_policy(self) -> None:
        """Load the LeRobot policy from checkpoint."""
        logger.info(f"Loading LeRobot ACT policy from {self.checkpoint_path}")

        try:
            from lerobot.policies.act.modeling_act import ACTPolicy
        except ImportError as e:
            logger.error(f"LeRobot not found: {e}")
            logger.error("Install with: pip install -e /path/to/lerobot")
            raise

        # Load policy
        self._policy = ACTPolicy.from_pretrained(self.checkpoint_path)
        self._policy.to(self.device)
        self._policy.eval()

        logger.info("LeRobot ACT policy loaded successfully")

    def _load_processors(self) -> None:
        """Load preprocessor and postprocessor from LeRobot factory."""
        try:
            from lerobot.policies.factory import make_pre_post_processors
        except ImportError as e:
            logger.error(f"LeRobot factory not found: {e}")
            raise

        # Create pre/post processors using LeRobot factory
        # Override device to 'cpu' if CUDA is not available
        preprocessor_overrides = {"device_processor": {"device": self.device}}
        postprocessor_overrides = {"device_processor": {"device": self.device}}

        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=self._policy.config,
            pretrained_path=str(self.checkpoint_path),
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides=postprocessor_overrides,
        )

        logger.info("Pre/post processors loaded successfully")

    def infer(self, obs: Observation) -> Action:
        """Run ACT inference on observation."""
        # Convert observation to LeRobot format
        le_obs = self._to_lerobot_obs(obs)

        # Run preprocessor (normalization, batching, device placement)
        le_obs = self._preprocessor(le_obs)

        with torch.inference_mode():
            action = self._policy.select_action(le_obs)

        # Run postprocessor (unnormalization) - PolicyAction is torch.Tensor
        action = self._postprocessor(action)

        # Convert to numpy array
        action = action.cpu().numpy().flatten()  # shape: (6,)

        return Action(action=action)

    def get_metadata(self) -> ServerMetadata:
        """Get server metadata."""
        return ServerMetadata(
            name="LeRobot ACT Policy",
            algorithm="act",
            version="1.0",
            description=f"ACT policy from {self.checkpoint_path}",
        )

    def _to_lerobot_obs(self, obs: Observation) -> Dict[str, Any]:
        """Convert generic observation to LeRobot format.

        Mirrors the lerobot_record flow:
        1. build_dataset_frame() - structure as dataset features
        2. prepare_observation_for_inference() - convert to tensor, normalize images to [0,1], CHW format
        """
        le_obs = {}

        # Handle image - same as prepare_observation_for_inference
        # Image format from robot: (H, W, C) uint8 RGB
        if obs.image is not None:
            image = obs.image

            # Handle dict format
            if isinstance(image, dict) and "data" in image:
                image = image["data"]

            # Convert to tensor, normalize to [0, 1], then permute to (C, H, W)
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float() / 255.0  # normalize to [0,1]
                if image.ndim == 3 and image.shape[2] in [1, 3, 4]:
                    image = image.permute(2, 0, 1)  # HWC -> CHW

            le_obs["observation.images.front"] = image

        # Handle state - same format as build_dataset_frame output
        if obs.state is not None:
            state = obs.state
            if isinstance(state, list):
                state = np.array(state, dtype=np.float32)
            elif isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            le_obs["observation.state"] = torch.from_numpy(state).float()

        # Add task and robot_type (required by ACT policy)
        le_obs["task"] = ""
        le_obs["robot_type"] = ""

        return le_obs


class YOLOIKPolicy(PolicyServer):
    """YOLO + Inverse Kinematics policy server.

    Detects objects with YOLO, then computes inverse kinematics for grasping.
    """

    def __init__(
        self,
        yolo_model_path: str,
        robot_config: Dict[str, Any],
    ):
        """Initialize YOLO + IK policy.

        Args:
            yolo_model_path: Path to YOLO model
            robot_config: Robot configuration for IK
        """
        self.yolo_model_path = yolo_model_path
        self.robot_config = robot_config
        self._yolo = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model."""
        logger.info(f"Loading YOLO model from {self.yolo_model_path}")

        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not found. Install with: pip install ultralytics")
            raise

        self._yolo = YOLO(self.yolo_model_path)
        logger.info("YOLO model loaded")

    def infer(self, obs: Observation) -> Action:
        """Run YOLO detection + IK to get grasp action."""
        # Detect object
        results = self._yolo.predict(obs.image, verbose=False)
        detections = results[0]

        if len(detections.boxes) == 0:
            # No detection, return home position
            action = np.array([0, 0, 0, 0, 0, 0])
        else:
            # Get center of first detection
            box = detections.boxes[0]
            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
            y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2

            # Simple IK: map pixel to joint angles
            action = self._pixel_to_joints(x_center.item(), y_center.item())

        return Action(action=action)

    def get_metadata(self) -> ServerMetadata:
        """Get server metadata."""
        return ServerMetadata(
            name="YOLO + IK Policy",
            algorithm="yolo_ik",
            version="1.0",
            description="YOLO detection with inverse kinematics",
        )

    def _pixel_to_joints(self, x: float, y: float) -> np.ndarray:
        """Map pixel coordinates to joint angles."""
        # Normalize to [-1, 1]
        x_norm = (x / 640) * 2 - 1
        y_norm = (y / 480) * 2 - 1

        # Simple mapping (tune for your robot)
        shoulder_pan = x_norm * 30  # -30 to +30 degrees
        shoulder_lift = -y_norm * 20 - 20  # adjust based on camera angle
        elbow_flex = -20
        wrist_flex = 50
        wrist_roll = 0
        gripper = 100 if y_norm > 0 else 0  # close gripper if object is low

        return np.array([shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper])


class HardcodedTrajectoryPolicy(PolicyServer):
    """Hardcoded trajectory policy for testing.

    Always returns the same predefined trajectory.
    """

    def __init__(self, trajectory: list):
        """Initialize with a hardcoded trajectory.

        Args:
            trajectory: List of actions to cycle through
        """
        self.trajectory = [np.array(t) for t in trajectory]
        self._step = 0

    def infer(self, obs: Observation) -> Action:
        """Return next action in trajectory."""
        action = self.trajectory[self._step % len(self.trajectory)]
        self._step += 1
        return Action(action=action)

    def get_metadata(self) -> ServerMetadata:
        """Get server metadata."""
        return ServerMetadata(
            name="Hardcoded Trajectory",
            algorithm="hardcoded",
            version="1.0",
            description=f"Hardcoded trajectory with {len(self.trajectory)} steps",
        )
