#!/usr/bin/env python3
"""Run Robot Client on edge.

Usage:
    python scripts/run_client.py --config configs/config.yaml

Or with arguments:
    python scripts/run_client.py --server-url ws://100.89.143.11:8000 --robot-host 192.168.3.164
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Add robot_agent to path (from SO-101 skill)
skill_robot_agent = Path.home() / ".nvm/versions/node/v22.22.2/lib/node_modules/openclaw/skills/so101robot/scripts"
sys.path.insert(0, str(skill_robot_agent))

# Also add workspace robot_agent if exists
workspace_robot_agent = Path.home() / ".openclaw/workspace/so_follower_robot/robot_agent"
if workspace_robot_agent.exists():
    sys.path.insert(0, str(workspace_robot_agent))


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
        "--robot-host",
        type=str,
        default=None,
        help="Host of the robot server",
    )
    parser.add_argument(
        "--robot-port",
        type=int,
        default=None,
        help="Port of the robot server",
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
        args.robot_host = args.robot_host or config.get("robot", {}).get("host", "127.0.0.1")
        args.robot_port = args.robot_port or config.get("robot", {}).get("port", 8765)

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
    else:
        from lerobot_remote.robot_client import SO101RobotClient

        robot = SO101RobotClient(
            host=args.robot_host or "127.0.0.1",
            port=args.robot_port or 8765,
        )
        logging.info(f"Connecting to SO-101 robot at {args.robot_host}:{args.robot_port}")

    # Connect to robot
    if not robot.connect():
        logging.error("Failed to connect to robot")
        sys.exit(1)

    logging.info("Robot connected!")

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
