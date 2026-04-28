#!/usr/bin/env python3
"""Run LeRobot Policy Server on cloud (4090 server).

Usage:
    python scripts/run_server.py --checkpoint /path/to/checkpoint --port 8000
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot_remote.policy_loader import load_lerobot_policy
from lerobot_remote.websocket_policy_server import (
    LeRobotWebsocketPolicyServer,
    get_local_ip,
)


def main():
    parser = argparse.ArgumentParser(description="Run LeRobot Policy Server")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., .../checkpoints/100000/pretrained_model/)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load policy
    logging.info(f"Loading policy from {args.checkpoint}...")
    policy = load_lerobot_policy(args.checkpoint, device=args.device)

    # Get server IP
    local_ip = get_local_ip()
    logging.info(f"Local IP: {local_ip}")

    # Create and start server
    server = LeRobotWebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={
            "framework": "lerobot",
            "model": "act",
            "checkpoint": str(args.checkpoint),
        },
    )

    logging.info(f"Starting server on {args.host}:{args.port}")
    logging.info(f"Edge clients should connect to: ws://{local_ip}:{args.port}")
    logging.info("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Server stopped")


if __name__ == "__main__":
    main()
