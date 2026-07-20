from __future__ import annotations

import pickle
from typing import Any

import zmq


def ipc_path(prefix: str, name: str) -> str:
    # ZMQ ipc:// requires an absolute filesystem path after the scheme.
    base = prefix if prefix.startswith("/") else f"/tmp/{prefix}"
    return f"ipc://{base}_{name}"


class ZmqCodec:
    @staticmethod
    def dumps(obj: Any) -> bytes:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loads(data: bytes) -> Any:
        return pickle.loads(data)


def bind_pull(ctx: zmq.Context, path: str) -> zmq.Socket:
    sock = ctx.socket(zmq.PULL)
    sock.bind(path)
    return sock


def connect_push(ctx: zmq.Context, path: str) -> zmq.Socket:
    sock = ctx.socket(zmq.PUSH)
    sock.connect(path)
    return sock


def bind_push(ctx: zmq.Context, path: str) -> zmq.Socket:
    sock = ctx.socket(zmq.PUSH)
    sock.bind(path)
    return sock


def connect_pull(ctx: zmq.Context, path: str) -> zmq.Socket:
    sock = ctx.socket(zmq.PULL)
    sock.connect(path)
    return sock
