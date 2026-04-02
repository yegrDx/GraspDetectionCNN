from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class PCDHeader:
    width: int
    height: int
    fields: Tuple[str, ...]
    size: Tuple[int, ...]
    type: Tuple[str, ...]
    count: Tuple[int, ...]
    data: str
    points: int


def _parse_header(lines) -> PCDHeader:
    d: Dict[str, str] = {}
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        key = parts[0].upper()
        if key == "DATA":
            d["DATA"] = parts[1].lower()
            break
        d[key] = " ".join(parts[1:])
    width = int(d.get("WIDTH", "0"))
    height = int(d.get("HEIGHT", "1"))
    fields = tuple(d.get("FIELDS", "").split())
    size = tuple(int(x) for x in d.get("SIZE", "").split())
    typ = tuple(d.get("TYPE", "").split())
    cnt = tuple(int(x) for x in d.get("COUNT", "").split()) if "COUNT" in d else tuple([1] * len(fields))
    pts = int(d.get("POINTS", str(width * height)))
    data = d.get("DATA", "ascii")
    return PCDHeader(width=width, height=height, fields=fields, size=size, type=typ, count=cnt, data=data, points=pts)


def read_pcd_xyz(path: str):
    with open(path, "rb") as f:
        content = f.read()

    lines = content.splitlines()
    header_lines = []
    data_start = None
    for i, ln in enumerate(lines):
        header_lines.append(ln.decode("utf-8", errors="ignore"))
        if ln.strip().upper().startswith(b"DATA"):
            header_bytes = b"\n".join(lines[: i + 1]) + b"\n"
            data_start = len(header_bytes)
            break
    if data_start is None:
        raise ValueError("Invalid PCD: no DATA line")

    hdr = _parse_header(header_lines)
    fields_lower = [f.lower() for f in hdr.fields]
    if not all(k in fields_lower for k in ["x", "y", "z"]):
        raise ValueError(f"PCD does not contain xyz fields: {hdr.fields}")

    raw = content[data_start:]
    if hdr.data == "ascii":
        txt = raw.decode("utf-8", errors="ignore").strip().split()
        arr = np.array(txt, dtype=np.float32)
        n_fields = len(hdr.fields)
        if arr.size % n_fields != 0:
            raise ValueError("ASCII PCD parse size mismatch")
        arr = arr.reshape((-1, n_fields))
        name_to_i = {n.lower(): i for i, n in enumerate(hdr.fields)}
        xyz = arr[:, [name_to_i["x"], name_to_i["y"], name_to_i["z"]]]
    elif hdr.data == "binary":
        # build dtype
        np_types = []
        for t, s in zip(hdr.type, hdr.size):
            t = t.upper()
            if t == "F" and s == 4:
                np_types.append(np.float32)
            elif t == "F" and s == 8:
                np_types.append(np.float64)
            elif t == "I" and s == 4:
                np_types.append(np.int32)
            elif t == "U" and s == 4:
                np_types.append(np.uint32)
            elif t == "U" and s == 2:
                np_types.append(np.uint16)
            else:
                raise ValueError(f"Unsupported PCD field type: {t}{s}")
        dtype = np.dtype([(name, tp) for name, tp in zip(hdr.fields, np_types)])
        point_step = int(sum(hdr.size))
        npts = hdr.points
        expected = npts * point_step
        if len(raw) < expected:
            npts = len(raw) // point_step
            expected = npts * point_step
        buf = raw[:expected]
        arr = np.frombuffer(buf, dtype=dtype, count=npts)
        xyz = np.stack([arr["x"].astype(np.float32), arr["y"].astype(np.float32), arr["z"].astype(np.float32)], axis=1)
    else:
        raise ValueError(f"Unsupported PCD DATA mode: {hdr.data}")

    if hdr.width > 0 and hdr.height > 0 and hdr.width * hdr.height == xyz.shape[0]:
        xyz = xyz.reshape((hdr.height, hdr.width, 3))
    return xyz, hdr


def depth_from_pcd(path: str, invalid_value: float = 0.0) -> np.ndarray:
    xyz, hdr = read_pcd_xyz(path)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError("PCD is not organized (H,W,3). Cornell should be organized.")
    z = xyz[..., 2].astype(np.float32)
    z[~np.isfinite(z)] = invalid_value
    z[z < 0] = invalid_value
    return z
