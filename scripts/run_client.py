#!/usr/bin/env python3
"""Run Robot Client on edge.

Usage:
    python scripts/run_client.py --config configs/config.yaml

Or with arguments:
    python scripts/run_client.py --server-url ws://100.89.143.11:8000 --robot-type so101 --robot-port /dev/ttyUSB0
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Add lerobot to path
lerobot_path = Path.home() / "lerobot" / "src"
if lerobot_path.exists():
    sys.path.insert(0, str(lerobot_path))


def main():
    parser = argparse.ArgumentParser(description="Run Robot Client")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="WebSocket URL of the policy server",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="so101",
        choices=["so101", "simulate"],
        help="Type of robot to use",
    )
    parser.add_argument(
        "--robot-port",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port for robot arm (Linux) or COM port (Windows)",
    )
    parser.add_argument(
        "--camera-port",
        type=int,
        default=0,
        help="Camera device port",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display camera feed using OpenCV window",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use simulated robot (for testing without hardware)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Get observation and inference only, do NOT execute actions",
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

    # Load config if provided
    config = {}
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)

        # Override with config values
        args.server_url = args.server_url or config.get("client", {}).get("server_url")
        args.robot_type = args.robot_type or config.get("robot", {}).get("type", "so101")
        args.robot_port = args.robot_port or config.get("robot", {}).get("port", "/dev/ttyUSB0")
        args.camera_port = args.camera_port or config.get("robot", {}).get("camera_port", 0)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create robot client
    if args.simulate:
        from lerobot_remote.robot_client import SimulatedRobotClient

        robot = SimulatedRobotClient()
        logging.info("Using simulated robot")
    elif args.robot_type == "so101":
        from lerobot_remote.lerobot_robot import SO101Robot

        robot = SO101Robot(
            port=args.robot_port,
            camera_port=args.camera_port,
        )
        logging.info(f"Connecting to SO-101 robot on {args.robot_port} (camera={args.camera_port})")
    else:
        logging.error(f"Unknown robot type: {args.robot_type}")
        sys.exit(1)

    # Connect to robot
    if not robot.connect():
        logging.error("Failed to connect to robot")
        sys.exit(1)

    logging.info("Robot connected!")

    # Move to safe starting position (for grasping task)
    import numpy as np
    SAFE_POSITIONS = np.array([
        1.126972201352359,   # shoulder_pan
        -97.32999582811848,  # shoulder_lift (in degrees, LeRobot uses degrees internally)
        100.0,               # elbow_flex
        71.68443496801706,   # wrist_flex
        0.024420024420024333, # wrist_roll
        0.9946949602122015,   # gripper
    ])
    logging.info(f"Moving to safe starting position: {SAFE_POSITIONS}")
    robot.execute(SAFE_POSITIONS)
    import time
    time.sleep(3)  # Wait for robot to reach position
    logging.info("Ready at safe position")

    # Create remote client
    if not args.server_url:
        logging.error("--server-url required (or set in config)")
        sys.exit(1)

    from lerobot_remote.remote_client import RemoteRobotClient

    client = RemoteRobotClient(
        robot=robot,
        server_url=args.server_url,
        api_key=args.api_key,
    )

    # Connect to server
    try:
        client.connect()
    except KeyboardInterrupt:
        logging.info("Disconnecting...")
        robot.disconnect()
        sys.exit(0)

    logging.info("Connected to cloud policy server!")

    if args.dry_run:
        logging.info("DRY RUN MODE: Will NOT execute actions, only print them")
        print("\n=== DRY RUN: Getting observation, running inference, printing action ===\n")

    # Run control loop
    print("\nControl loop started. Press Ctrl+C to stop")
    try:
        while True:
            try:
                # Get observation
                obs = robot.get_observation()
                logging.debug(f"Got observation: state={obs.state}")

                # Send to cloud and get action
                action = client.infer(obs)

                if args.dry_run:
                    # Only print, don't execute
                    print(f"[DRY RUN] Action received: {action.action}")
                    print(f"[DRY RUN] Server timing: {action.server_timing}")
                else:
                    # Execute action
                    robot.execute(action.action)
                    logging.debug(f"Executed action: {action.action}")

                # Display camera feed if enabled
                if args.display:
                    import cv2
                    if obs.image is not None:
                        cv2.imshow("Robot Camera", obs.image)
                        cv2.waitKey(1)

            except Exception as e:
                logging.error(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        client.disconnect()
        robot.disconnect()
        logging.info("Disconnected")


if __name__ == "__main__":
    main()
