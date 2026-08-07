from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn

from .vit import Qwen3VisionTransformer, _normalize_grids

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VisionEncodeOutput:
    embeds: torch.Tensor  # [total_tokens, out_dim] concatenated over images
    deepstack: list[torch.Tensor] | None
    sizes: list[int]

    def split(self) -> list[tuple[torch.Tensor, list[torch.Tensor] | None]]:
        outs = []
        cur = 0
        for sz in self.sizes:
            emb = self.embeds[cur : cur + sz]
            ds = (
                [t[cur : cur + sz].contiguous() for t in self.deepstack]
                if self.deepstack is not None
                else None
            )
            outs.append((emb, ds))
            cur += sz
        return outs


@dataclass
class _CacheEntry:
    embeds: torch.Tensor
    deepstack: list[torch.Tensor] | None
    tokens: int


class EncoderCache:
    """LRU cache of vision encoder outputs (official-style: no ref counting).

    Eviction order = insertion order (oldest first), so ``get`` must
    ``move_to_end`` and ``put`` evicts from the front with ``popitem``.
    """

    def __init__(self, max_tokens: int = 16384):
        self.max_tokens = max(1, int(max_tokens))
        self.used_tokens = 0
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()

    @staticmethod
    def _tokens(embeds: torch.Tensor, deepstack: list[torch.Tensor] | None) -> int:
        n = int(embeds.shape[0])
        if deepstack:
            n += sum(int(t.shape[0]) for t in deepstack)
        return n

    def get(self, key: str) -> tuple[torch.Tensor, list[torch.Tensor] | None] | None:
        e = self._cache.get(key)
        if e is None:
            return None
        self._cache.move_to_end(key)
        return e.embeds, e.deepstack

    def put(
        self,
        key: str,
        embeds: torch.Tensor,
        deepstack: list[torch.Tensor] | None = None,
    ) -> None:
        # Store detached GPU tensors; callers must treat as read-only.
        embeds = embeds.detach()
        ds = [t.detach() for t in deepstack] if deepstack is not None else None
        tokens = self._tokens(embeds, ds)
        if key in self._cache:
            old = self._cache.pop(key)
            self.used_tokens -= old.tokens
        while self.used_tokens + tokens > self.max_tokens and self._cache:
            _, e = self._cache.popitem(last=False)
            self.used_tokens -= e.tokens
        self._cache[key] = _CacheEntry(embeds=embeds, deepstack=ds, tokens=tokens)
        self.used_tokens += tokens

    def clear(self) -> None:
        self._cache.clear()
        self.used_tokens = 0

    def __contains__(self, k: str) -> bool:
        return k in self._cache

    def __len__(self) -> int:
        return len(self._cache)


@dataclass
class _BudgetGraph:
    token_budget: int
    max_batch_size: int
    graph: torch.cuda.CUDAGraph
    input_buffers: dict[str, torch.Tensor]
    output_buffer: torch.Tensor
    max_patches: int
    max_seqlen: int
    out_dim_cat: int


class VisionEncoderCudaGraph:
    def __init__(
        self,
        vit: Qwen3VisionTransformer,
        *,
        token_budgets: list[int] | None = None,
        max_batch_size: int = 4,
    ):
        self.vit = vit
        self.max_batch_size = max(1, int(max_batch_size))
        self.token_budgets = sorted(token_budgets or [64, 128, 256, 512, 1024, 2048])
        self.graphs: dict[int, _BudgetGraph] = {}
        self.hits = 0
        self.misses = 0
        self._enabled = False
        self._graph_pool = None

    @property
    def enabled(self) -> bool:
        return self._enabled and bool(self.graphs)

    def capture(self) -> None:
        device = self.vit.pos_embed.weight.device
        dtype = self.vit.pos_embed.weight.dtype
        m = self.vit.spatial_merge_size
        pe = self.vit.patch_embed
        flat_ps = pe.in_channels * pe.temporal_patch_size * pe.patch_size * pe.patch_size
        out_dim = self.vit.out_hidden_size
        n_ds = len(self.vit.deepstack_indexes)
        out_dim_cat = out_dim * (1 + n_ds)

        self.graphs.clear()
        self._graph_pool = torch.cuda.graph_pool_handle()
        # Warmup backend once outside capture.
        _ = self.vit._backend(dtype)

        for budget in sorted(self.token_budgets, reverse=True):
            if budget < self.max_batch_size:
                continue
            per_item = (budget + self.max_batch_size - 1) // self.max_batch_size
            grid_config = [
                [1, m, per_item * m] for _ in range(self.max_batch_size)
            ]
            max_patches = sum(t * h * w for t, h, w in grid_config)
            max_seqlen = budget * (m * m)

            md = self.vit.prepare_metadata(
                grid_config,
                max_num_seqs=self.max_batch_size,
                max_seqlen_override=max_seqlen,
                max_patches=max_patches,
            )
            buffers = {
                "pixel_values": torch.randn(
                    max_patches, flat_ps, device=device, dtype=dtype
                ),
                "pos_embeds": md["pos_embeds"].contiguous(),
                "rotary_cos": md["rotary_cos"].contiguous(),
                "rotary_sin": md["rotary_sin"].contiguous(),
                "cu_seqlens": md["cu_seqlens"].contiguous(),
            }
            meta = {
                "pos_embeds": buffers["pos_embeds"],
                "rotary_cos": buffers["rotary_cos"],
                "rotary_sin": buffers["rotary_sin"],
                "cu_seqlens": buffers["cu_seqlens"],
                "cu_seqlens_cpu": md["cu_seqlens_cpu"],
                "seqlens": md["seqlens"],
                "offsets": md["offsets"],
                "sizes": md["sizes"],
                "max_seqlen": max_seqlen,
            }

            attn = self.vit._backend(dtype)
            attn.bind(buffers["cu_seqlens"], max_seqlen)

            with torch.inference_mode():
                out = self.vit(
                    buffers["pixel_values"],
                    metadata=meta,
                    concat_deepstack=True,
                    plan_attn=False,
                    split_images=False,
                )
                tmp = torch.empty_like(out)

            g = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(g, pool=self._graph_pool):
                out = self.vit(
                    buffers["pixel_values"],
                    metadata=meta,
                    concat_deepstack=True,
                    plan_attn=False,
                    split_images=False,
                )
                tmp.copy_(out)

            self.graphs[budget] = _BudgetGraph(
                token_budget=budget,
                max_batch_size=self.max_batch_size,
                graph=g,
                input_buffers=buffers,
                output_buffer=tmp,
                max_patches=max_patches,
                max_seqlen=max_seqlen,
                out_dim_cat=out_dim_cat,
            )
            logger.info(
                "Vision CG captured budget=%d max_batch=%d patches=%d",
                budget,
                self.max_batch_size,
                max_patches,
            )
            # Free capture intermediates before next budget.
            torch.cuda.empty_cache()

        self._enabled = bool(self.graphs)

    def _find_budget(self, total_out_tokens: int, num_items: int) -> int | None:
        if num_items > self.max_batch_size:
            return None
        for b in self.token_budgets:
            if b >= total_out_tokens and b in self.graphs:
                return b
        return None

    @staticmethod
    def _copy_pad(dst: torch.Tensor, src: torch.Tensor) -> None:
        dst.zero_()
        n = min(dst.shape[0], src.shape[0])
        dst[:n].copy_(src[:n])

    def try_encode(
        self, pixel_values: torch.Tensor, grid_thw
    ) -> VisionEncodeOutput | None:
        if not self.enabled:
            return None
        grids = _normalize_grids(grid_thw)
        m = self.vit.spatial_merge_size
        sizes = [int(t * h * w) // (m * m) for t, h, w in grids]
        total = sum(sizes)
        budget = self._find_budget(total, len(grids))
        if budget is None:
            self.misses += 1
            return None

        bg = self.graphs[budget]
        # Build real metadata (padded seq count to max_batch).
        md = self.vit.prepare_metadata(
            grids,
            max_num_seqs=bg.max_batch_size,
            max_seqlen_override=bg.max_seqlen,
            max_patches=bg.max_patches,
        )
        bufs = bg.input_buffers
        self._copy_pad(bufs["pixel_values"], pixel_values)
        self._copy_pad(bufs["pos_embeds"], md["pos_embeds"])
        self._copy_pad(bufs["rotary_cos"], md["rotary_cos"])
        self._copy_pad(bufs["rotary_sin"], md["rotary_sin"])
        # cu_seqlens: copy then ensure length matches static buffer.
        cu_dst = bufs["cu_seqlens"]
        cu_src = md["cu_seqlens"]
        cu_dst.zero_()
        cu_dst[: cu_src.numel()].copy_(cu_src)
        if cu_src.numel() < cu_dst.numel():
            # Pad remaining slots with last offset (zero-length seqs).
            cu_dst[cu_src.numel() :].fill_(int(cu_src[-1].item()))

        bg.graph.replay()
        self.hits += 1

        out = bg.output_buffer[:total]
        out_dim = self.vit.out_hidden_size
        n_ds = len(self.vit.deepstack_indexes)
        if n_ds:
            chunks = out.split(out_dim, dim=1)
            embeds = chunks[0].contiguous()
            deepstack = [c.contiguous() for c in chunks[1:]]
        else:
            embeds = out.contiguous()
            deepstack = None
        return VisionEncodeOutput(embeds=embeds, deepstack=deepstack, sizes=sizes)


class Qwen3VisionTower(nn.Module):
    def __init__(self, vision_config) -> None:
        super().__init__()
        self._cfg = vision_config
        self.vit: Qwen3VisionTransformer | None = None
        self.cudagraph: VisionEncoderCudaGraph | None = None
        self._use_cudagraph = True

    @property
    def merge_size(self) -> int:
        return getattr(self._cfg, "spatial_merge_size", 2)

    @property
    def out_dim(self) -> int:
        return getattr(self._cfg, "out_hidden_size", self._cfg.hidden_size)

    @property
    def deepstack_features(self):
        return self.vit.deepstack_features if self.vit else None

    def load_weights(self, state_dict: dict[str, torch.Tensor], vision_config) -> None:
        device = next(iter(state_dict.values())).device
        self.vit = Qwen3VisionTransformer(vision_config).to(device)
        n = self.vit.load_weights(state_dict)
        logger.info("Native ViT loaded: %d params", n)

    def warm_cudagraph(
        self,
        token_budgets: list[int] | None = None,
        max_batch_size: int = 4,
    ) -> None:
        assert self.vit is not None
        self.cudagraph = VisionEncoderCudaGraph(
            self.vit, token_budgets=token_budgets, max_batch_size=max_batch_size
        )
        self.cudagraph.capture()
        self._use_cudagraph = True

    def prepare_metadata(self, grid_thw: torch.Tensor) -> dict:
        assert self.vit is not None
        return self.vit.prepare_metadata(grid_thw)

    def forward(self, pixel_values, grid_thw=None, metadata=None):
        """Backward-compatible eager forward → per-image tuple."""
        assert self.vit is not None
        return self.vit(pixel_values, grid_thw=grid_thw, metadata=metadata)

    @torch.inference_mode()
    def encode(
        self,
        pixel_values: torch.Tensor,
        grid_thw,
        *,
        use_cudagraph: bool | None = None,
    ) -> VisionEncodeOutput:
        """Batch-encode images; prefer CUDA Graph when warmed and shapes fit."""
        assert self.vit is not None
        grids = _normalize_grids(grid_thw)
        m = self.merge_size
        sizes = [int(t * h * w) // (m * m) for t, h, w in grids]
        use_cg = self._use_cudagraph if use_cudagraph is None else use_cudagraph
        patch_counts = [int(t * h * w) for t, h, w in grids]

        if use_cg and self.cudagraph is not None and self.cudagraph.enabled:
            hit = self.cudagraph.try_encode(pixel_values, grids)
            if hit is not None:
                return hit
            # Greedy pack: run CG per item (or small groups) when batch is too wide.
            if len(grids) > 1:
                embeds_l, ds_l = [], []
                off = 0
                for g, pc, sz in zip(grids, patch_counts, sizes):
                    pv_i = pixel_values[off : off + pc]
                    off += pc
                    sub = self.cudagraph.try_encode(pv_i, [g])
                    if sub is None:
                        sub = self._encode_eager(pv_i, [g], [sz])
                    embeds_l.append(sub.embeds)
                    if sub.deepstack is not None:
                        ds_l.append(sub.deepstack)
                embeds = torch.cat(embeds_l, dim=0)
                deepstack = (
                    [torch.cat([d[i] for d in ds_l], dim=0) for i in range(len(ds_l[0]))]
                    if ds_l
                    else None
                )
                return VisionEncodeOutput(embeds=embeds, deepstack=deepstack, sizes=sizes)

        return self._encode_eager(pixel_values, grids, sizes)

    def _encode_eager(self, pixel_values, grids, sizes) -> VisionEncodeOutput:
        assert self.vit is not None
        parts = self.vit(pixel_values, grid_thw=grids)
        embeds = torch.cat(list(parts), dim=0) if isinstance(parts, tuple) else parts
        ds = self.vit.deepstack_features
        if ds is not None:
            ds = [t.contiguous() for t in ds]
        return VisionEncodeOutput(embeds=embeds, deepstack=ds, sizes=sizes)
