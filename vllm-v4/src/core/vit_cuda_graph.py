from __future__ import annotations

import gc
from dataclasses import dataclass

import torch


@dataclass
class ViTCGConfig:
    min_budget: int = 64
    max_budget: int = 256
    out_hidden_size: int = 2048
    merge_size: int = 2
    patch_dim: int = 1536
    base_grid: tuple[int, int, int] = (1, 16, 16)
    num_deepstack: int = 0


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
    def __init__(
        self,
        vit_forward,
        metadata_fn,
        config: ViTCGConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self._forward = vit_forward  # (pv, grid, meta, out, dps) → None
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
        assert self._base_tokens > 0
        self._budgets = self._make_budgets()
        self._graphs: dict[int, _BudgetGraph] = {}
        self._captured = False

    def _make_budgets(self) -> list[int]:
        b, out = self._cfg.min_budget, []

        if b % self._base_tokens != 0:
            b = ((b + self._base_tokens - 1) // self._base_tokens) * self._base_tokens
        while b <= self._cfg.max_budget:
            out.append(b)
            b *= 2
        if not out or out[-1] < self._cfg.max_budget:
            mx = self._cfg.max_budget
            if mx % self._base_tokens != 0:
                mx = (mx // self._base_tokens) * self._base_tokens
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
        print(f"ViT CUDA Graph capture: budgets={self._budgets}, base={self._base_grid}")

        for budget in sorted(self._budgets, reverse=True):
            torch.cuda.empty_cache()
            bg = self._capture_one(budget, graph_pool)
            self._graphs[budget] = bg
            if graph_pool is None:
                graph_pool = bg.graph.pool()
            torch.cuda.synchronize(self._device)

        self._captured = True
        print(f"ViT CUDA Graph captured ({len(self._graphs)} budgets)")

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

        # Warmup outside graph.
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

        bg.pv_buf.zero_()
        bg.pv_buf[:n_patches].copy_(pixel_values.to(dtype=self._dtype))
        # Keep capture-time grid / metadata structure (pad slots stay empty).
        bg.graph.replay()

        merged = bg.out_buf[:actual_tokens]
        dps = None
        if bg.dps_bufs:
            dps = [buf[:actual_tokens] for buf in bg.dps_bufs]
        return merged, dps

    def destroy(self) -> None:
        self._graphs.clear()
        self._captured = False
        gc.collect()
        torch.cuda.empty_cache()
