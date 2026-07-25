#!/usr/bin/env python3
"""Export vllm-v7 rank-local sharded checkpoints from a HF model directory.

Example:
  source ../.venv/bin/activate
  source ../.env.cuda.sh
  PYTHONPATH=. python scripts/save_sharded_state.py \
    --model /data/huggingface-local/Qwen3-8B \
    --tp 2 --device-ids 2,4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def _worker(
    rank: int,
    *,
    model: str,
    tp_size: int,
    pp_size: int,
    device_ids: list[int],
    master_port: int,
) -> None:
    import torch
    from transformers import AutoConfig

    from src.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from src.models.loader import load_model_hf
    from src.models.sharded_loader import save_sharded_checkpoint, write_manifest

    world_size = tp_size * pp_size
    tp_rank = rank % tp_size
    pp_rank = rank // tp_size
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{master_port}",
        local_device=device_ids[rank] if device_ids else 0,
    )
    initialize_model_parallel(tp_size, pp_size, tp_rank, pp_rank)

    hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    is_moe = int(getattr(hf_config, "num_experts", 0) or 0) > 0 or any(
        "Moe" in a for a in (getattr(hf_config, "architectures", None) or [])
    )
    is_vl = any("VL" in a for a in (getattr(hf_config, "architectures", None) or []))
    text_cfg = getattr(hf_config, "text_config", hf_config)
    dtype = getattr(text_cfg, "dtype", None) or getattr(text_cfg, "torch_dtype", None)
    if not isinstance(dtype, torch.dtype):
        dtype = torch.bfloat16

    from src.models.weight_utils import default_dtype_context

    if is_vl:
        from src.models.qwen3_vl import Qwen3VLForCausalLM

        with default_dtype_context(dtype):
            model_obj = Qwen3VLForCausalLM(hf_config).eval()
        device = torch.device("cpu")
        model_obj.load_weights(
            model,
            device,
            dtype,
            rank=rank,
            tp_size=tp_size,
            pp_size=pp_size,
        )
    elif is_moe:
        from src.models.qwen3_moe import Qwen3MoeForCausalLM

        with default_dtype_context(dtype):
            model_obj = Qwen3MoeForCausalLM(hf_config)
        load_model_hf(model_obj, model)
        model_obj = model_obj.eval()
    else:
        from src.models.qwen3 import Qwen3ForCausalLM

        with default_dtype_context(dtype):
            model_obj = Qwen3ForCausalLM(hf_config)
        load_model_hf(model_obj, model)
        model_obj = model_obj.eval()

    files = save_sharded_checkpoint(
        model_obj,
        model,
        rank=rank,
        tp_size=tp_size,
        pp_size=pp_size,
        dtype=dtype,
    )
    if rank == 0:
        write_manifest(
            model,
            tp_size=tp_size,
            pp_size=pp_size,
            world_size=world_size,
            dtype=str(dtype).replace("torch.", ""),
        )
        print(
            f"Wrote manifest + {len(files)} shard file(s) for rank 0; "
            f"world_size={world_size}"
        )
    else:
        print(f"rank={rank}: wrote {len(files)} shard file(s)")

    destroy_model_parallel()
    destroy_distributed_environment()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model directory")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--device-ids", default="", help="comma-separated PCI ids")
    ap.add_argument("--master-port", type=int, default=29611)
    args = ap.parse_args()

    model = str(Path(args.model).expanduser())
    if not Path(model).is_dir():
        raise SystemExit(f"missing model dir: {model}")
    if args.tp < 1 or args.pp < 1:
        raise SystemExit("tp/pp must be >= 1")

    world_size = args.tp * args.pp
    if args.device_ids.strip():
        device_ids = [int(x) for x in args.device_ids.split(",") if x.strip()]
    else:
        device_ids = list(range(world_size))
    if len(device_ids) != world_size:
        raise SystemExit(f"need {world_size} device ids, got {device_ids}")

    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker,
            kwargs=dict(
                rank=rank,
                model=model,
                tp_size=args.tp,
                pp_size=args.pp,
                device_ids=device_ids,
                master_port=args.master_port,
            ),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise SystemExit(p.exitcode or 1)
    print("OK")


if __name__ == "__main__":
    main()
