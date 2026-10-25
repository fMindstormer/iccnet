"""Read profiles back: header, tag table, and multiProcessElements tags as stages."""
import struct

import numpy as np

from .icc import Curve


def u16(b, o): return struct.unpack_from('>H', b, o)[0]
def u32(b, o): return struct.unpack_from('>I', b, o)[0]
def f32(b, o, n): return np.frombuffer(b, '>f4', n, o).astype(np.float64)


def header(data):
    v = u32(data, 8)
    return {
        'size': u32(data, 0), 'version': f'{v >> 24}.{(v >> 20) & 15}.{(v >> 16) & 15}',
        'class': data[12:16].decode(), 'space': data[16:20].decode(), 'pcs': data[20:24].decode(),
        'intent': u32(data, 64), 'id': data[84:100].hex(),
    }


def tags(data):
    """[(signature, offset, size, type signature)]"""
    n = u32(data, 128)
    out = []
    for i in range(n):
        o = 132 + 12 * i
        sig, off, size = data[o:o + 4].decode(), u32(data, o + 4), u32(data, o + 8)
        out.append((sig, off, size, data[off:off + 4].decode(errors='replace')))
    return out


def tag_data(data, sig):
    for s, off, size, _ in tags(data):
        if s == sig:
            return data[off:off + size]
    raise KeyError(sig)


def _curve(b, o):
    assert b[o:o + 4] == b'curf', b[o:o + 4]
    n = u16(b, o + 8)
    breaks = list(f32(b, o + 12, n - 1))
    p = o + 12 + 4 * (n - 1)
    segs = []
    for _ in range(n):
        kind = b[p:p + 4]
        if kind == b'parf':
            ftype = u16(b, p + 8)
            assert ftype == 0, 'only formula type 0 is read'
            segs.append(('f',) + tuple(f32(b, p + 12, 4)))
            p += 12 + 16
        else:
            assert kind == b'samf', kind
            count = u32(b, p + 8)
            segs.append(('s', f32(b, p + 12, count)))
            p += 12 + 4 * count
    return Curve(breaks, segs)


def mpet_stages(t):
    """The elements of one 'mpet' tag (its bytes) as stages."""
    assert t[:4] == b'mpet'
    n = u32(t, 12)
    stages = []
    for i in range(n):
        off = u32(t, 16 + 8 * i)
        kind, p, q = t[off:off + 4], u16(t, off + 8), u16(t, off + 10)
        if kind == b'matf':
            W = f32(t, off + 12, p * q).reshape(q, p)
            stages.append(('mat', W, f32(t, off + 12 + 4 * p * q, q)))
        elif kind == b'cvst':
            stages.append(('cur', [_curve(t, off + u32(t, off + 12 + 8 * j)) for j in range(p)]))
        else:
            raise ValueError(f'element {kind!r} is not read')
    return stages
