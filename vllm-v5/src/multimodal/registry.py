from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MultimodalModelSpec:
    name: str
    families: tuple[str, ...]
    status: str  # "implemented" | "planned"
    notes: str


REGISTRY: dict[str, MultimodalModelSpec] = {
    "qwen3_vl": MultimodalModelSpec(
        name="Qwen3-VL",
        families=("Qwen3VLForConditionalGeneration",),
        status="implemented",
        notes="Native ViT + MRoPE + DeepStack; primary v5 teaching path.",
    ),
    "qwen2_vl": MultimodalModelSpec(
        name="Qwen2-VL",
        families=("Qwen2VLForConditionalGeneration",),
        status="planned",
        notes="Same overall recipe as Qwen3-VL; needs own weight mapping / processor IDs.",
    ),
    "llava": MultimodalModelSpec(
        name="LLaVA",
        families=("LlavaForConditionalGeneration",),
        status="planned",
        notes="CLIP/SigLIP + MLP projector; 1D RoPE (no MRoPE).",
    ),
    "internvl": MultimodalModelSpec(
        name="InternVL",
        families=("InternVLChatModel",),
        status="planned",
        notes="Dynamic resolution + pixel shuffle; different merge rules.",
    ),
}


def supported_implemented() -> list[str]:
    return [s.name for s in REGISTRY.values() if s.status == "implemented"]
