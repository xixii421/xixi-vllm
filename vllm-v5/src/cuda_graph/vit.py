"""ViT encoder CUDA Graph runner (fixed base grid + token budgets)."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import torch

from ..config.vit_cg import ViTCGConfig

logger = logging.getLogger(__name__)

# Re-export for callers that import from cuda_graph.vit
__all__ = ["ViTCGConfig", "ViTCUDAGraphRunner"]


@dataclass
class _BudgetGraph:
    budget: int
    n_slots: int
    graph: torch.cuda.CUDAGraph
    pv_buf: torch.Tensor
    grid_buf: torch.Tensor
    out_buf: torch.Tensor
    dps_bufs: list[torch.Tensor]
    meta: dict


class ViTCUDAGraphRunner:
    """Capture ViT encode for a discrete set of token budgets on a fixed base grid."""

    def __init__(
        self,
        vit_forward,
        metadata_fn,
        config: ViTCGConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self._forward = vit_forward
        self._metadata_fn = metadata_fn
        self._cfg = config
        self._device = device
        self._dtype = dtype
        self._m = config.merge_size
        self._base_grid = tuple(config.base_grid)
        self._base_tokens = int(
            self._base_grid[0]
            * (self._base_grid[1] // self._m)
            * (self._base_grid[2] // self._m)
        )
        self._base_patches = int(
            self._base_grid[0] * self._base_grid[1] * self._base_grid[2]
        )
        if self._base_tokens <= 0:
            raise ValueError(f"invalid base_tokens from grid={self._base_grid}")
        self._budgets = self._make_budgets()
        self._graphs: dict[int, _BudgetGraph] = {}
        self._captured = False

    def _make_budgets(self) -> list[int]:
        b, out = (self._cfg.min_budget, [])
        if b % self._base_tokens != 0:
            b = (b + self._base_tokens - 1) // self._base_tokens * self._base_tokens
        while b <= self._cfg.max_budget:
            out.append(b)
            b *= 2
        if not out or out[-1] < self._cfg.max_budget:
            mx = self._cfg.max_budget
            if mx % self._base_tokens != 0:
                mx = mx // self._base_tokens * self._base_tokens
            if mx >= self._base_tokens and (not out or out[-1] != mx):
                out.append(mx)
        return out

    def _find_budget(self, n_tokens: int) -> int:
        for b in self._budgets:
            if b >= n_tokens:
                return b
        return -1

    def can_use(self, grid_thw: torch.Tensor) -> bool:
        if not self._captured:
            return False
        g = grid_thw.view(-1, 3)
        base = torch.tensor(self._base_grid, device=g.device, dtype=g.dtype)
        if not torch.equal(g, base.unsqueeze(0).expand_as(g)):
            return False
        n_tok = int((g.prod(-1) // (self._m * self._m)).sum().item())
        return self._find_budget(n_tok) > 0

    def capture(self, graph_pool=None) -> None:
        if self._captured:
            return
        torch.cuda.synchronize(self._device)
        torch.cuda.empty_cache()
        logger.info(
            "ViT CUDA Graph capture: budgets=%s, base=%s",
            self._budgets,
            self._base_grid,
        )
        for budget in sorted(self._budgets, reverse=True):
            torch.cuda.empty_cache()
            bg = self._capture_one(budget, graph_pool)
            self._graphs[budget] = bg
            if graph_pool is None:
                graph_pool = bg.graph.pool()
            torch.cuda.synchronize(self._device)
        self._captured = True
        logger.info("ViT CUDA Graph captured (%d budgets)", len(self._graphs))

    def _capture_one(self, budget: int, graph_pool) -> _BudgetGraph:
        n_slots = budget // self._base_tokens
        assert n_slots * self._base_tokens == budget
        grid = torch.tensor(
            [list(self._base_grid) for _ in range(n_slots)],
            dtype=torch.int32,
            device=self._device,
        )
        total_patches = n_slots * self._base_patches
        pv = torch.randn(
            total_patches, self._cfg.patch_dim, dtype=self._dtype, device=self._device
        )
        meta = self._metadata_fn(grid)
        for k, v in list(meta.items()):
            if isinstance(v, torch.Tensor):
                meta[k] = v.to(
                    device=self._device,
                    dtype=self._dtype
                    if v.dtype in (torch.float16, torch.bfloat16, torch.float32)
                    else v.dtype,
                )
        out_buf = torch.empty(
            budget, self._cfg.out_hidden_size, dtype=self._dtype, device=self._device
        )
        dps_bufs = [
            torch.empty(
                budget, self._cfg.out_hidden_size, dtype=self._dtype, device=self._device
            )
            for _ in range(self._cfg.num_deepstack)
        ]
        # Warmup outside the graph, then capture.
        self._forward(pv, grid, meta, out_buf, dps_bufs)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=graph_pool):
            self._forward(pv, grid, meta, out_buf, dps_bufs)
        return _BudgetGraph(
            budget=budget,
            n_slots=n_slots,
            graph=g,
            pv_buf=pv,
            grid_buf=grid,
            out_buf=out_buf,
            dps_bufs=dps_bufs,
            meta=meta,
        )

    def replay(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        g = grid_thw.view(-1, 3)
        actual_tokens = int((g.prod(-1) // (self._m * self._m)).sum().item())
        budget = self._find_budget(actual_tokens)
        if budget <= 0 or not self.can_use(g):
            raise RuntimeError("ViT CG replay called when can_use is False")
        bg = self._graphs[budget]
        n_patches = int(pixel_values.shape[0])
        if n_patches > bg.pv_buf.shape[0]:
            raise RuntimeError("pixel_values exceed captured buffer")
        # Only clear the unused pad region; real patches are overwritten below.
        if n_patches < bg.pv_buf.shape[0]:
            bg.pv_buf[n_patches:].zero_()
        bg.pv_buf[:n_patches].copy_(pixel_values.to(dtype=self._dtype))
        bg.graph.replay()
        merged = bg.out_buf[:actual_tokens]
        dps = None
        if bg.dps_bufs:
            dps = [buf[:actual_tokens] for buf in bg.dps_bufs]
        return (merged, dps)

    def destroy(self) -> None:
        self._graphs.clear()
        self._captured = False
        gc.collect()
        torch.cuda.empty_cache()
