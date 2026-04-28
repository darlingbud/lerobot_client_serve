"""LeRobot Remote - Generic cloud-edge robotics framework.

This module provides a generic interface for cloud-edge robotics control.
Any robot (implementing RobotClient) can connect to any policy server
(implementing PolicyServer) via WebSocket.
"""

from lerobot_remote.protocol import (
    Action,
    ErrorCode,
    HealthCheckPath,
    Observation,
    ProtocolVersion,
    ServerMetadata,
    Serialization,
)

from lerobot_remote.robot_client import RobotClient, SimulatedRobotClient
from lerobot_remote.lerobot_robot import LeRobotRobot, LeRobotRobotConfig, SO101Robot
from lerobot_remote.policy_server import PolicyServer, get_local_ip
from lerobot_remote.remote_client import RemoteRobotClient

__version__ = "0.2.0"
__all__ = [
    # Protocol
    "Action",
    "ErrorCode",
    "HealthCheckPath",
    "Observation",
    "ProtocolVersion",
    "ServerMetadata",
    "Serialization",
    # Robot clients
    "RobotClient",
    "SimulatedRobotClient",
    "LeRobotRobot",
    "LeRobotRobotConfig",
    "SO101Robot",
    # Policy servers
    "PolicyServer",
    "get_local_ip",
    # Remote client
    "RemoteRobotClient",
]
