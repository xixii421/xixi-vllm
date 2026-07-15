# vllm-v5 DESIGN

## Goal

Ship a **single multimodal teaching path** on top of v4: Qwen3-VL image → text.

## Why not many VL families?

| Family | Projector / ViT | Position | Cost to support |
| --- | --- | --- | --- |
| Qwen3-VL | Native ViT + DeepStack | MRoPE (3D) | Full path in v5 |
| Qwen2-VL | Similar | MRoPE | Small delta later |
| LLaVA | CLIP + MLP | 1D RoPE | Separate merge / pack |
| InternVL | Dynamic tiles | Custom | Separate encoder API |

Different families do **not** share one `encode_and_merge`. v5 keeps a registry stub and implements Qwen3-VL only.

## Components

1. **MultiModalProcessor** — HF chat template + `pixel_values` + image token spans  
2. **Qwen3VisionTower** — patch tokens (+ optional deepstack features)  
3. **encode_and_merge** — splice visual embeds into text embeds; build MRoPE positions  
4. **ModelRunner** — on multimodal prefill pass `inputs_embeds` / deepstack; KV dims from `text_config`  
5. **Decode** — unchanged from v4 (CG when positions are 1D)
6. **ViT CUDA Graph (optional)** — `enable_vit_cuda_graph=True` captures budget graphs
   (`src/core/vit_cuda_graph.py`); replay when every image matches base grid `[1,16,16]`.

## Correctness notes

- `mf.start` is absolute in the prompt; merge/MRoPE map into the *scheduled* window via `start - num_cached_tokens`.
- After an image span, text MRoPE positions continue from `max(grid) + 1` (HF ``rope_deltas``); decode uses `len-1 + rope_delta`.
- Interleaved MRoPE follows HF ``apply_interleaved_mrope`` (not naive T/H/W gather).
- HF ViT weights use `linear_fc{1,2}` → remapped to our `fc{1,2}`; patch embed layout is ``[L,C,T,H,W]``.
- Prefers fitting the full multimodal prompt in one prefill (`max_num_batched_tokens` large enough).
- Pixel tensors are **per-image** slices (not the whole batch repeated).
- ViT CG is orthogonal to decode CG; variable-resolution images fall back to eager.
