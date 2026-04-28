#!/usr/bin/env python3
"""Run LeRobot Remote Robot Client on edge (local robot machine).

This connects to the cloud policy server and controls the robot.

Usage:
    # From this machine (edge):
    python scripts/run_client.py --server-url ws://100.89.143.11:8000 --robot-host 127.0.0.1 --robot-port 8765

    # Then control the robot:
    python scripts/control_robot.py --server-url ws://100.89.143.11:8000
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from lerobot_remote.websocket_robot_client import LeRobotRemoteRobot


def main():
    parser = argparse.ArgumentParser(description="Run LeRobot Remote Robot Client")
    parser.add_argument(
        "--server-url",
        type=str,
        default="ws://100.89.143.11:8000",
        help="WebSocket URL of the policy server",
    )
    parser.add_argument(
        "--robot-host",
        type=str,
        default="127.0.0.1",
        help="Host of the local robot server",
    )
    parser.add_argument(
        "--robot-port",
        type=int,
        default=8765,
        help="Port of the local robot server",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key for server authentication",
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

    # Import robot agent
    try:
        from robot_agent import RobotAgent
    except ImportError:
        logging.error("robot_agent not found. Make sure SO-101 skill is installed.")
        sys.exit(1)

    # Connect to robot
    logging.info(f"Connecting to robot at {args.robot_host}:{args.robot_port}...")
    robot = RobotAgent(host=args.robot_host, port=args.robot_port)
    if not robot.connect():
        logging.error("Failed to connect to robot")
        sys.exit(1)

    logging.info("Robot connected!")

    # Create remote robot client
    logging.info(f"Connecting to policy server at {args.server_url}...")
    remote_robot = LeRobotRemoteRobot(
        robot=robot,
        server_url=args.server_url,
        api_key=args.api_key,
    )

    logging.info("Connected to cloud policy server!")
    logging.info("Ready to receive commands via infer_and_execute()")

    return remote_robot, robot


if __name__ == "__main__":
    remote_robot, robot = main()

    # Keep running
    print("\nRobot ready! Use remote_robot.infer_and_execute(obs) to control.")
    print("Press Ctrl+C to stop")

    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        robot.disconnect()
