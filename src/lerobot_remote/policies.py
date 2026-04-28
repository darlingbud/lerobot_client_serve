"""LeRobot ACT Policy Server.

Concrete implementation of PolicyServer for LeRobot ACT policies.
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
        self._action_mean = None
        self._action_std = None
        self._load_policy()
        self._load_normalization_stats()

    def _load_policy(self) -> None:
        """Load the LeRobot policy from checkpoint."""
        logger.info(f"Loading LeRobot ACT policy from {self.checkpoint_path}")

        try:
            from lerobot.policies.act.modeling_act import ACTPolicy
        except ImportError as e:
            logger.error(f"LeRobot not found: {e}")
            logger.error("Install with: pip install -e /path/to/lerobot")
            raise

        # Load config
        config_path = self.checkpoint_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        # Load policy
        self._policy = ACTPolicy.from_pretrained(self.checkpoint_path)
        self._policy.to(self.device)
        self._policy.eval()

        logger.info("LeRobot ACT policy loaded successfully")

    def _load_normalization_stats(self) -> None:
        """Load action normalization stats for denormalization."""
        try:
            from safetensors.numpy import load_file
        except ImportError:
            logger.warning("safetensors not available, skipping normalization stats")
            return

        stats_path = self.checkpoint_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        if not stats_path.exists():
            logger.warning(f"Normalization stats not found: {stats_path}")
            return

        try:
            stats = load_file(stats_path)
            self._action_mean = stats["action.mean"]  # shape: (6,)
            self._action_std = stats["action.std"]    # shape: (6,)
            logger.info(f"Loaded action stats - mean: {self._action_mean}, std: {self._action_std}")
        except Exception as e:
            logger.warning(f"Failed to load normalization stats: {e}")

    def infer(self, obs: Observation) -> Action:
        """Run ACT inference on observation."""
        import time

        # Convert observation to LeRobot format
        le_obs = self._to_lerobot_obs(obs)

        with torch.no_grad():
            action = self._policy.select_action(le_obs)

        # Convert to numpy
        if hasattr(action, "cpu"):
            action = action.cpu().numpy()
        elif isinstance(action, torch.Tensor):
            action = action.numpy()

        # Ensure it's 1D
        action = action.flatten()  # shape: (6,)

        # Denormalize action (MEAN_STD normalization)
        if self._action_mean is not None and self._action_std is not None:
            action = action * self._action_std + self._action_mean
            logger.debug(f"Denormalized action: {action}")

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
        """Convert generic observation to LeRobot format."""
        import torch

        le_obs = {}

        # Handle image
        if obs.image is not None:
            image = obs.image

            # Handle dict format
            if isinstance(image, dict) and "data" in image:
                image = image["data"]

            # Convert to tensor (HWC -> CHW)
            if isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[2] in [1, 3, 4]:
                    image = np.transpose(image, (2, 0, 1))
                image = torch.from_numpy(image).float()

            le_obs["observation.images.front"] = image.unsqueeze(0).to(self.device)

        # Handle state
        if obs.state is not None:
            state = obs.state
            if isinstance(state, list):
                state = torch.tensor(state, dtype=torch.float32)
            elif isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            le_obs["observation.state"] = state.unsqueeze(0).to(self.device)

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
