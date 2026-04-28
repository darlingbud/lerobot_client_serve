#!/usr/bin/env python3
"""Run Policy Server on cloud.

Usage:
    python scripts/run_server.py --config configs/config.yaml

Or with arguments:
    python scripts/run_server.py --checkpoint /path/to/checkpoint --port 8000
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot_remote.policy_server import get_local_ip


def main():
    parser = argparse.ArgumentParser(description="Run Policy Server")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="act",
        choices=["act", "yolo_ik", "hardcoded"],
        help="Policy type to use",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (for act/yolo_ik policies)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: cuda or cpu",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)

        # Override with config values
        args.host = config.get("server", {}).get("host", args.host)
        args.port = config.get("server", {}).get("port", args.port)
        args.checkpoint = config.get("server", {}).get("checkpoint", args.checkpoint)
        args.device = config.get("server", {}).get("device", args.device)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create policy based on type
    if args.policy == "act":
        if not args.checkpoint:
            logging.error("--checkpoint required for act policy")
            sys.exit(1)

        from lerobot_remote.policies import LeRobotACTPolicy

        policy = LeRobotACTPolicy(
            checkpoint_path=args.checkpoint,
            device=args.device,
        )

    elif args.policy == "yolo_ik":
        from lerobot_remote.policies import YOLOIKPolicy

        policy = YOLOIKPolicy(
            yolo_model_path=args.checkpoint or "yolov8n.pt",
            robot_config={},
        )

    elif args.policy == "hardcoded":
        from lerobot_remote.policies import HardcodedTrajectoryPolicy

        # Example: simple waving trajectory
        trajectory = [
            [0, 0, 0, 0, 0, 0],      # home
            [30, -20, 0, 50, 0, 0],  # wave up
            [30, -20, 0, 50, 0, 100],  # grip
            [-30, -20, 0, 50, 0, 0],  # wave left
            [0, 0, 0, 0, 0, 0],      # home
        ]

        policy = HardcodedTrajectoryPolicy(trajectory=trajectory)

    # Get server IP
    local_ip = get_local_ip()

    logging.info(f"Starting {args.policy} server on {args.host}:{args.port}")
    logging.info(f"Edge clients should connect to: ws://{local_ip}:{args.port}")
    logging.info("Press Ctrl+C to stop")

    try:
        policy.start(host=args.host, port=args.port)
    except KeyboardInterrupt:
        logging.info("Server stopped")


if __name__ == "__main__":
    main()
