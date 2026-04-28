"""Common protocol definitions for LeRobot Remote.

This module defines the communication protocol between edge (robot) and cloud (policy).
It is agnostic to the specific algorithm used on the cloud side.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import msgpack
import numpy as np


# =============================================================================
# Protocol Messages
# =============================================================================

@dataclass
class Observation:
    """Observation sent from edge to cloud.

    Attributes:
        image: Robot camera image (H, W, C) as numpy array
        state: Robot joint positions as list/numpy array
        timestamp: Unix timestamp when observation was captured
    """

    image: np.ndarray
    state: List[float]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "image": self.image,
            "state": self.state,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Observation":
        """Create from dictionary."""
        return cls(
            image=d["image"],
            state=d["state"],
            timestamp=d.get("timestamp", time.time()),
        )


@dataclass
class Action:
    """Action returned from cloud to edge.

    Attributes:
        action: Action values (e.g., joint deltas or absolute positions)
        server_timing: Optional timing information from server
    """

    action: np.ndarray
    server_timing: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action": self.action,
            "server_timing": self.server_timing,
        }


@dataclass
class ServerMetadata:
    """Metadata sent from server to client on connection.

    Attributes:
        name: Server implementation name
        algorithm: Algorithm type (e.g., "act", "yolo_ik", "hardcoded")
        version: Protocol version
        description: Human-readable description
    """

    name: str
    algorithm: str
    version: str = "1.0"
    description: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "algorithm": self.algorithm,
            "version": self.version,
            "description": self.description,
        }


# =============================================================================
# Serialization
# =============================================================================

class Serialization:
    """Handles serialization/deserialization with numpy support."""

    @staticmethod
    def pack(obj: Any) -> bytes:
        """Serialize object to bytes using msgpack with numpy support."""
        return msgpack.packb(obj, use_bin_type=True, default=Serialization._encode_ndarray)

    @staticmethod
    def unpack(data: bytes) -> Any:
        """Deserialize bytes using msgpack with numpy support."""
        return msgpack.unpackb(
            data,
            raw=False,
            object_hook=Serialization._decode_ndarray,
        )

    @staticmethod
    def _encode_ndarray(obj):
        """Encode numpy array to msgpack-compatible format."""
        if isinstance(obj, np.ndarray):
            if obj.size == 0:
                return {"__ndarray__": True, "dtype": str(obj.dtype), "shape": (), "data": b""}
            return {
                "__ndarray__": True,
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": obj.tobytes(),
            }
        return None

    @staticmethod
    def _decode_ndarray(obj):
        """Decode msgpack-compatible format back to numpy array."""
        if isinstance(obj, dict) and "__ndarray__" in obj:
            dtype = np.dtype(obj["dtype"])
            shape = tuple(obj["shape"]) if obj["shape"] else ()
            data = obj["data"]
            if isinstance(data, bytes) and len(data) > 0:
                return np.frombuffer(data, dtype=dtype).reshape(shape)
            elif isinstance(data, (list, np.ndarray)):
                return np.array(data, dtype=dtype).reshape(shape)
            else:
                return np.array([], dtype=dtype)
        return obj


# =============================================================================
# Protocol Constants
# =============================================================================

DEFAULT_SERVER_PORT = 8000
HealthCheckPath = "/healthz"
ProtocolVersion = "1.0"

# Error codes
class ErrorCode:
    SUCCESS = 0
    CONNECTION_ERROR = 1
    INFERENCE_ERROR = 2
    TIMEOUT = 3
    INVALID_OBSERVATION = 4
    UNKNOWN_ERROR = 99
