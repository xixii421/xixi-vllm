import hashlib
import numpy as np


def hash_pil_image(image) -> str:
    h = hashlib.sha1()
    try:
        exif = getattr(image, "getexif", lambda: None)()
        if exif:
            uid = exif.get(0xA420)
            if uid:
                h.update(f"exif:{uid}".encode())
    except Exception:
        pass
    arr = np.asarray(image)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    h.update(arr.tobytes())
    h.update(f"|{tuple(getattr(image, 'size', ()))}|{getattr(image, 'mode', '')}".encode())
    return h.hexdigest()
