"""MPClient：DP 多副本客户端（对齐 vLLM V1 的 ``v1/engine/core_client.py``）。

每个副本是一个独立 ``EngineCoreProc``（跑在各自进程 / 各卡上，自主前向），
门面通过 ZMQ PUSH/PULL 与它们通信，从而让 dp_size 个副本并发推理。

- 请求线格式：``RequestWire``（请求序列化）/ ``CoreRequest``（入站）/ ``CoreOutput``（出站）。
- ``run_engine_core_proc``：EngineCoreProc 进程入口 + 忙循环（排空输入 → 有活 step → 空闲让出）。
- ``MPClient``：spawn 副本 + round-robin 路由 + 输出汇总 + 销毁。
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import tempfile
import time

import msgspec
import torch
import zmq

from ..config import EngineConfig
from ..data import EngineCoreOutput, Sequence
from .client import EngineCoreClient
from .core import EngineCore


class RequestWire(msgspec.Struct):
    """一条请求的最小可序列化表示（对齐 vLLM 的 EngineCoreRequest 思路）。"""

    seq_id: int
    token_ids: list[int]
    block_size: int
    temperature: float
    top_p: float | None
    top_k: int | None
    min_p: float | None
    max_tokens: int
    ignore_eos: bool

    @classmethod
    def from_sequence(cls, seq: Sequence) -> "RequestWire":
        return cls(
            seq_id=seq.seq_id,
            token_ids=list(seq.token_ids),
            block_size=seq.block_size,
            temperature=seq.temperature,
            top_p=seq.top_p,
            top_k=seq.top_k,
            min_p=seq.min_p,
            max_tokens=seq.max_tokens,
            ignore_eos=seq.ignore_eos,
        )

    def to_sequence(self) -> Sequence:
        """在 EngineCore 进程中重建一条等价的 Sequence（seq_id 保持一致）。"""
        return Sequence(
            token_ids=list(self.token_ids),
            block_size=self.block_size,
            seq_id=self.seq_id,
            num_prompt_tokens=len(self.token_ids),
            last_token=self.token_ids[-1],
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            max_tokens=self.max_tokens,
            ignore_eos=self.ignore_eos,
        )


class CoreRequest(msgspec.Struct):
    """门面 → EngineCoreProc：加请求或关停。"""

    op: str  # "add" | "shutdown"
    req: RequestWire | None = None


class CoreOutput(msgspec.Struct):
    """EngineCoreProc → 门面：一步的增量输出 + token 指标。"""

    outputs: list[EngineCoreOutput]
    metric: int = 0


def run_engine_core_proc(
    config: EngineConfig,
    *,
    eos_token_id: int,
    device: str,
    dtype: torch.dtype,
    in_addr: str,
    out_addr: str,
    ready,  # multiprocessing.Event：绑定完成后置位，避免门面提前 connect
) -> None:
    """EngineCoreProc 进程入口：构建 EngineCore 并进入服务忙循环。"""
    core = EngineCore(
        config, eos_token_id=eos_token_id, device=torch.device(device), dtype=dtype
    )
    ctx = zmq.Context()
    in_sock = ctx.socket(zmq.PULL)
    in_sock.bind(in_addr)
    out_sock = ctx.socket(zmq.PUSH)
    out_sock.bind(out_addr)

    poller = zmq.Poller()
    poller.register(in_sock, zmq.POLLIN)
    enc = msgspec.msgpack.Encoder()
    dec = msgspec.msgpack.Decoder(CoreRequest)

    ready.set()
    try:
        while True:
            # 1) 排空所有待处理的输入（非阻塞）。
            shutdown = False
            while True:
                socks = dict(poller.poll(0))
                if in_sock not in socks:
                    break
                msg = dec.decode(in_sock.recv())
                if msg.op == "shutdown":
                    shutdown = True
                    break
                if msg.op == "add" and msg.req is not None:
                    core.add_request(msg.req.to_sequence())
            if shutdown:
                break

            # 2) 有活就前进一步；空闲让出 CPU，等新请求或关停。
            if core.has_unfinished():
                outputs, metric = core.step()
                out_sock.send(enc.encode(CoreOutput(outputs=outputs, metric=metric)))
            else:
                time.sleep(0.0002)
    finally:
        core.destroy()
        in_sock.close(0)
        out_sock.close(0)
        ctx.term()


class MPClient(EngineCoreClient):
    """spawn dp_size 个 EngineCoreProc 进程，ZMQ 路由 + 汇总输出。"""

    def __init__(
        self,
        config: EngineConfig,
        *,
        eos_token_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.config = config
        self.dp_size = config.dp_size
        assert self.dp_size > 1

        self._ctx = zmq.Context()
        self._enc = msgspec.msgpack.Encoder()
        self._dec = msgspec.msgpack.Decoder(CoreOutput)
        self._in: list[zmq.Socket] = []
        self._out: list[zmq.Socket] = []
        self._poller = zmq.Poller()
        self._procs: list[mp.Process] = []
        self._pending: set[int] = set()  # 未观察到 finished 的 seq_id
        self._next = 0

        ctx = mp.get_context("spawn")
        tmp = tempfile.mkdtemp(prefix="v4_dp_")
        for r in range(self.dp_size):
            replica_cfg = self._replica_config(r)
            in_addr = f"ipc://{tmp}/in_{r}.ipc"
            out_addr = f"ipc://{tmp}/out_{r}.ipc"
            ready = ctx.Event()
            p = ctx.Process(
                target=run_engine_core_proc,
                args=(),
                kwargs=dict(
                    config=replica_cfg,
                    eos_token_id=eos_token_id,
                    device=f"cuda:{replica_cfg.device_ids[0]}",
                    dtype=dtype,
                    in_addr=in_addr,
                    out_addr=out_addr,
                    ready=ready,
                ),
                daemon=True,
            )
            p.start()
            self._procs.append(p)
            ready.wait(timeout=60)
            if not ready.is_set():
                raise RuntimeError(f"DP replica {r} failed to initialize")

            in_sock = self._ctx.socket(zmq.PUSH)
            in_sock.connect(in_addr)
            out_sock = self._ctx.socket(zmq.PULL)
            out_sock.connect(out_addr)
            self._in.append(in_sock)
            self._out.append(out_sock)
            self._poller.register(out_sock, zmq.POLLIN)

    # ------------------------------------------------------------------ #
    # 副本配置
    # ------------------------------------------------------------------ #

    def _replica_config(self, r: int) -> EngineConfig:
        """每个副本是一份独立引擎：dp_size 归 1，分配自己的卡与 NCCL 端口。

        用 ``copy.copy`` 复用已加载的 hf_config，避免每副本重读磁盘。
        """
        per = self.config.tp_size * self.config.pp_size
        if self.config.device_ids is not None:
            devices = self.config.device_ids[r * per : (r + 1) * per]
        else:
            start = r * per
            devices = list(range(start, start + per))
        replica = copy.copy(self.config)
        replica.dp_size = 1
        replica.device_ids = devices
        replica.nccl_port = self.config.nccl_port + r * 16
        return replica

    # ------------------------------------------------------------------ #
    # EngineCoreClient 接口
    # ------------------------------------------------------------------ #

    def add_request(self, seq: Sequence) -> None:
        self._pending.add(seq.seq_id)
        r = self._next
        self._next = (self._next + 1) % self.dp_size
        req = CoreRequest(op="add", req=RequestWire.from_sequence(seq))
        self._in[r].send(self._enc.encode(req))

    def step(self) -> tuple[list[EngineCoreOutput], int]:
        return self._drain(block=True)

    def has_unfinished(self) -> bool:
        return bool(self._pending)

    # ------------------------------------------------------------------ #
    # 输出汇总
    # ------------------------------------------------------------------ #

    def _drain(self, *, block: bool) -> tuple[list[EngineCoreOutput], int]:
        """聚合各副本已就绪的输出；``block`` 时至少等到一条（避免门面空转）。"""
        outputs: list[EngineCoreOutput] = []
        metric = 0
        first = True
        while True:
            timeout = None if (block and first) else 0
            socks = dict(self._poller.poll(timeout))
            if not socks:
                break
            for sock in socks:
                msg = self._dec.decode(sock.recv())
                outputs.extend(msg.outputs)
                metric += msg.metric
                for o in msg.outputs:
                    if o.is_finished:
                        self._pending.discard(o.seq_id)
            first = False
        return outputs, metric

    # ------------------------------------------------------------------ #
    # 销毁
    # ------------------------------------------------------------------ #

    def destroy(self) -> None:
        for sock in self._in:
            try:
                sock.send(self._enc.encode(CoreRequest(op="shutdown")))
            except zmq.ZMQError:
                pass
        for p in self._procs:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()
        for s in self._in:
            s.close(0)
        for s in self._out:
            s.close(0)
        self._ctx.term()
        self._procs.clear()
        self._pending.clear()
