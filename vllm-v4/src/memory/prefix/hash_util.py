import struct
import numpy as np
import xxhash

def compute_block_hash(token_ids: list[int] | np.ndarray, prefix: int = -1) -> int:
    if xxhash is not None:
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little", signed=False))
        arr = np.asarray(token_ids, dtype=np.int32)
        h.update(arr.tobytes())
        return int(h.intdigest())
    # Fallback: same layout as v2 blake2b path.
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little", signed=False))
    h.update(struct.pack(f"<{len(token_ids)}i", *token_ids))
    return int.from_bytes(h.digest(), "little", signed=False)
