"""Load LeRobot policy from checkpoint."""

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def load_lerobot_policy(
    checkpoint_path: str,
    device: str = "cuda",
) -> "LeRobotPolicyWrapper":
    """Load a LeRobot policy from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory (e.g., .../checkpoints/100000/pretrained_model/)
        device: Device to load the model on ('cuda' or 'cpu')

    Returns:
        A LeRobotPolicyWrapper that implements the infer(obs) interface.
    """
    from lerobot.configs import RobotConfig
    from lerobot.policies import policy_factory
    from lerobot.scripts.train_model import load_config

    checkpoint_path = Path(checkpoint_path)

    # Load config
    config_path = checkpoint_path / "config.json"
    train_config_path = checkpoint_path / "train_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Load policy using lerobot's factory
    policy, _ = policy_factory(
        config_path=config_path,
        pretrained_model_name_or_path=str(checkpoint_path),
        device=device,
    )

    policy.eval()
    logger.info(f"Loaded LeRobot policy from {checkpoint_path}")

    return LeRobotPolicyWrapper(policy, device)


class LeRobotPolicyWrapper:
    """Wrapper around LeRobot policy to provide a simple infer(obs) interface.

    LeRobot policies expect observation dict with:
        - observation.images.{camera_name}: Numpy array or dict with 'data' key
        - observation.state: Robot joint positions

    Returns action dict with:
        - action: Numpy array of action values
    """

    def __init__(self, policy, device: str = "cuda"):
        self.policy = policy
        self.device = device

    def infer(self, obs: dict) -> dict:
        """Run inference on observation.

        Args:
            obs: Dictionary with:
                - image: Numpy array (H, W, C) or dict with 'data'
                - state: List or array of joint positions

        Returns:
            Dictionary with:
                - action: Numpy array of action values
        """
        # Convert observation to LeRobot format
        leRobot_obs = self._convert_obs(obs)

        with torch.no_grad():
            # Run policy inference
            action = self.policy.select_action(leRobot_obs)

        # Convert to numpy and return
        if hasattr(action, "cpu"):
            action = action.cpu().numpy()

        return {"action": action}

    def _convert_obs(self, obs: dict) -> dict:
        """Convert generic obs format to LeRobot format.

        LeRobot expects:
        {
            "observation.images.<camera>": NCHW tensor or dict,
            "observation.state": tensor,
        }
        """
        import torch

        le_obs = {}

        # Handle image
        if "image" in obs:
            image = obs["image"]
            if isinstance(image, dict) and "data" in image:
                image = image["data"]

            if isinstance(image, np.ndarray):
                # Convert HWC -> CHW
                if image.ndim == 3 and image.shape[2] in [1, 3, 4]:
                    image = np.transpose(image, (2, 0, 1))
                image = torch.from_numpy(image).float()

            le_obs["observation.images.cam"] = image.unsqueeze(0).to(self.device)

        # Handle state
        if "state" in obs:
            state = obs["state"]
            if isinstance(state, (list, np.ndarray)):
                state = torch.tensor(state, dtype=torch.float32)
            le_obs["observation.state"] = state.unsqueeze(0).to(self.device)

        return le_obs
