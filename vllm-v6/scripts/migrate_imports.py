import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPL = [
    (r"from \.\.memory\.", "from ..scheduling."),
    (r"from \.memory\.", "from .scheduling."),
    (r"from src\.memory\.", "from src.scheduling."),
    (r"from src\.memory import", "from src.scheduling import"),
    (r"from \.\.engine\.", "from ..runtime."),
    (r"from \.engine\.", "from .runtime."),
    (r"from src\.engine\.", "from src.runtime."),
    (r"from src\.engine import", "from src.runtime import"),
    (r"from \.\.core\.cuda_graph", "from ..cuda_graph.decode"),
    (r"from \.\.core\.vit_cuda_graph", "from ..cuda_graph.vit"),
    (r"from src\.core\.cuda_graph", "from src.cuda_graph.decode"),
    (r"from src\.core\.vit_cuda_graph", "from src.cuda_graph.vit"),
    (r"from \.\.utils\.context", "from ..context"),
    (r"from src\.utils\.context", "from src.context"),
    (r"from src\.utils import", "from src.context import"),
    (r"from \.\.layers\.utils_parallel import divide, ensure_divisibility", "from ..distributed.math import divide, ensure_divisibility"),
    (r"from src\.layers\.utils_parallel import divide, ensure_divisibility", "from src.distributed.math import divide, ensure_divisibility"),
    (r"from \.\.layers\.utils_parallel", "from ..distributed.pp"),
    (r"from src\.layers\.utils_parallel", "from src.distributed.pp"),
    (r"from \.\.distributed\.utils", "from ..distributed.pp"),
    (r"from src\.distributed\.utils", "from src.distributed.pp"),
    (r"import src\.distributed\.utils as", "import src.distributed.pp as"),
    (r"from \.\.memory\.kv_ops", "from ..kernels.kv_ops"),
    (r"from src\.memory\.kv_ops", "from src.kernels.kv_ops"),
    (r"from \.\._mrope", "from ..positions"),
    (r"from \._mrope", "from .positions"),
    (r"from src\.multimodal\._mrope", "from src.multimodal.positions"),
    (r"from \.\.multimodal\.vision import EncoderCache", "from ..multimodal.encoder_cache import EncoderCache"),
    (r"from \.\.multimodal\.vision import Qwen3VisionTower", "from ..multimodal.tower import Qwen3VisionTower"),
    (r"from src\.models\.hub", "from src.tools.hub"),
    (r"from \.engine import Engine", "from .runtime import Engine"),
]

for path in ROOT.rglob("*.py"):
    if ".venv" in path.parts or "__pycache__" in path.parts:
        continue
    text = path.read_text()
    new = text
    for a, b in REPL:
        new = re.sub(a, b, new)
    if new != text:
        path.write_text(new)
        print(path.relative_to(ROOT))
