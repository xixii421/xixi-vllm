"""Sampler: greedy, mixed temp, top-k / top-p / min-p."""

from __future__ import annotations

import torch

from src.runtime.sampler import Sampler


class TestSampler:
    def test_greedy(self):
        s = Sampler()
        logits = torch.tensor([[0.1, 0.9, 0.0], [0.5, 0.2, 0.3]])
        out = s(logits, torch.zeros(2))
        assert out.tolist() == [1, 0]

    def test_mixed_temperature_no_div0(self):
        s = Sampler()
        logits = torch.randn(4, 32)
        logits[0] = -100
        logits[0, 3] = 50
        logits[2] = -100
        logits[2, 7] = 50
        temps = torch.tensor([0.0, 0.8, 0.0, 1.2])
        out = s(logits, temps)
        assert int(out[0]) == 3 and int(out[2]) == 7

    def test_top_k_restricts(self):
        s = Sampler()
        # top-2 of these logits are indices 0 and 1 (both beat the rest).
        logits = torch.tensor([[5.0, 4.0, 0.0, -1.0, -2.0, -3.0]])
        torch.manual_seed(0)
        outs = [
            int(
                s(
                    logits.clone(),
                    torch.tensor([1.0]),
                    top_ks=torch.tensor([2]),
                    top_ps=torch.tensor([1.0]),
                ).item()
            )
            for _ in range(20)
        ]
        assert set(outs) <= {0, 1}
        assert 0 in outs  # typically sees both under random Gumbel

    def test_top_p_nucleus(self):
        s = Sampler()
        # After softmax roughly [0.7, 0.2, 0.1] → with top_p=0.75 keep only first
        logits = torch.tensor([[2.0, 0.5, -0.5]])
        # Force high temp→softmax; tiny temperature keeps greedy on filtered set.
        # Use deterministic path: temperature near 0 after filter still argmax of
        # filtered logits → should be 0 when top_p keeps nucleus including argmax.
        out = s(
            logits.clone(),
            torch.tensor([0.5]),
            top_ps=torch.tensor([0.75]),
            top_ks=torch.tensor([0]),
        )
        assert int(out[0]) == 0

    def test_min_p_masks_tail(self):
        s = Sampler()
        logits = torch.tensor([[10.0, 0.0, -5.0]])
        # max_p ~1, min_p=0.5 masks tokens with p < 0.5
        out = s(
            logits,
            torch.tensor([1.0]),
            min_ps=torch.tensor([0.5]),
            top_ps=torch.tensor([1.0]),
            top_ks=torch.tensor([0]),
        )
        assert int(out[0]) == 0

    def test_greedy_ignores_filters(self):
        s = Sampler()
        logits = torch.tensor([[1.0, 0.0, 5.0]])
        out = s(
            logits,
            torch.tensor([0.0]),
            top_ks=torch.tensor([1]),  # would keep only index 0 if applied
            top_ps=torch.tensor([0.1]),
        )
        assert int(out[0]) == 2
