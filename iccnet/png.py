"""PNG with an embedded ICC profile (iCCP chunk), 8 or 16 bits per channel, no dependencies."""
import struct
import zlib

import numpy as np


def _chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)


def write(path, img, icc=None, name=b'iccnet'):
    """img: uint8 or uint16 array (h, w, 3 or 4). Rows use the Sub filter, which turns the
    coordinate ramps of a carrier into runs of equal bytes."""
    img = np.asarray(img)
    depth = 16 if img.dtype == np.uint16 else 8
    h, w, c = img.shape
    bpp = c * depth // 8
    rows = np.ascontiguousarray(img.astype('>u2' if depth == 16 else 'u1')).view(np.uint8).reshape(h, w * bpp)
    sub = rows.copy()
    sub[:, bpp:] = rows[:, bpp:] - rows[:, :-bpp]
    raw = np.concatenate([np.ones((h, 1), np.uint8), sub], 1).tobytes()
    out = b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, depth, {3: 2, 4: 6}[c], 0, 0, 0))
    if icc is not None:
        out += _chunk(b'iCCP', name + b'\0\0' + zlib.compress(icc, 9))
    out += _chunk(b'IDAT', zlib.compress(raw, 9)) + _chunk(b'IEND', b'')
    with open(path, 'wb') as f:
        f.write(out)


def read(path):
    """(pixels as uint8 or uint16 array (h, w, channels), embedded ICC profile or None)."""
    data = open(path, 'rb').read()
    assert data[:8] == b'\x89PNG\r\n\x1a\n', 'not a PNG'
    pos, idat, icc = 8, [], None
    while pos < len(data):
        n = struct.unpack('>I', data[pos:pos + 4])[0]
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        if kind == b'IHDR':
            w, h, depth, ctype, _, _, interlace = struct.unpack('>IIBBBBB', body)
            assert interlace == 0, 'interlaced PNGs are not supported'
        elif kind == b'iCCP':
            icc = zlib.decompress(body[body.index(b'\0') + 2:])
        elif kind == b'IDAT':
            idat.append(body)
        pos += 12 + n
    assert depth in (8, 16), 'only 8 and 16 bits per channel'
    c = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
    bpp = c * depth // 8
    stride = w * bpp
    raw = np.frombuffer(zlib.decompress(b''.join(idat)), np.uint8).reshape(h, stride + 1)
    out = np.zeros((h, stride), np.uint8)
    prev = np.zeros(stride, np.int64)
    for y in range(h):
        f, line = raw[y, 0], raw[y, 1:].astype(np.int64)
        if f == 0:
            cur = line
        elif f == 1:                         # each byte of a pixel is a running sum of its column
            cur = np.cumsum(line.reshape(w, bpp), axis=0).reshape(-1) & 255
        elif f == 2:
            cur = (line + prev) & 255
        else:
            cur = _sequential(line, prev, f, bpp)
        out[y] = cur
        prev = cur.astype(np.int64)
    if depth == 16:
        return out.view('>u2').reshape(h, w, c).astype(np.uint16), icc
    return out.reshape(h, w, c), icc


def _sequential(line, prev, f, bpp):
    cur = np.zeros_like(line)
    for i in range(len(line)):
        a = cur[i - bpp] if i >= bpp else 0
        b = prev[i]
        cc = prev[i - bpp] if i >= bpp else 0
        if f == 1:
            p = a
        elif f == 3:
            p = (a + b) // 2
        else:
            pa, pb, pc = abs(b - cc), abs(a - cc), abs(a + b - 2 * cc)
            p = a if pa <= pb and pa <= pc else (b if pb <= pc else cc)
        cur[i] = (line[i] + p) & 255
    return cur
