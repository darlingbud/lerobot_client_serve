"""Msgpack serialization with numpy support. Adapted from OpenPI."""

import msgpack
import numpy as np


class Packer:
    """Msgpack packer with numpy array support."""

    def pack(self, obj):
        """Pack an object, converting numpy arrays to bytes."""
        return msgpack.packb(obj, use_bin_type=True)

    def unpackb(self, data):
        """Unpack data, converting numpy array bytes back to arrays."""
        return msgpack.unpackb(data, raw=False, object_hook=self._n_decode)


class NumpyExt:
    """Numpy extension for msgpack."""

    @staticmethod
    def encode(obj):
        if isinstance(obj, np.ndarray):
            if obj.size == 0:
                return {"__ndarray__": True, "dtype": str(obj.dtype), "shape": obj.shape, "data": obj.tobytes()}
            return {
                "__ndarray__": True,
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": obj.tobytes(),
            }
        return None

    @staticmethod
    def decode(obj):
        if isinstance(obj, dict) and "__ndarray__" in obj:
            data = obj["data"]
            if isinstance(data, bytes):
                arr = np.frombuffer(data, dtype=obj["dtype"])
            else:
                arr = np.array(data, dtype=obj["dtype"])
            return arr.reshape(obj["shape"])
        return None


def numpy_packer():
    """Create a msgpack Packer with numpy support."""
    return msgpack.Packer(default=NumpyExt.encode, use_bin_type=True)


def numpy_unpacker():
    """Create a msgpack Unpacker with numpy support."""
    return msgpack.Unpacker(raw=False, object_hook=NumpyExt.decode)
