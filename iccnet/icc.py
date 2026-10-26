"""Write ICC v4 profiles whose DToB tags evaluate a network (ICC.1:2022 §10.16).

A network is a list of stages, each one multiProcessElement:
    ('mat', W, b)        y = W x + b, W of shape (out, in)       -> 'matf'
    ('cur', [Curve...])  y_i = curve_i(x_i), one curve a channel -> 'cvst'
`forward()` evaluates stages exactly as the profile encodes them (float32 parameters and
float32 storage between elements, as LittleCMS does), so a written profile can be checked
against it. `profile()` writes the published layout: see its docstring.
"""
import hashlib
import struct

import numpy as np

D50 = np.array([0.9642, 1.0, 0.8249])
# sRGB colorants adapted to D50 with Bradford, as in the sRGB v4 profiles
SRGB_TO_XYZ = np.array([
    [0.4360747, 0.3850649, 0.1430804],
    [0.2225045, 0.7168786, 0.0606169],
    [0.0139322, 0.0971045, 0.7141733],
])
BRADFORD_D65_TO_D50 = np.array([
    [1.0478112, 0.0228866, -0.0501270],
    [0.0295424, 0.9904844, -0.0170491],
    [-0.0092345, 0.0150436, 0.7521316],
])
SRGB_PARAMS = (2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045)
U1F15 = 32768 / 65535        # integer LUT types carry XYZ as u1Fixed15: 1.0 means 65535/32768
MAX_CHANNELS = 15            # LittleCMS reads 'matf' elements with at most 15 inputs and outputs


def u16(v): return struct.pack('>H', v)
def u32(v): return struct.pack('>I', v)
def f32s(a): return np.asarray(a, dtype='>f4').tobytes()
def s15f16(a): return b''.join(struct.pack('>i', int(round(v * 65536))) for v in np.ravel(a))
def pad4(b): return b + b'\0' * (-len(b) % 4)
def as_f32(a): return np.asarray(a, dtype=np.float32).astype(np.float64)


def srgb_encode(lin):
    lin = np.asarray(lin, dtype=np.float64)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.abs(lin) ** (1 / 2.4) * np.sign(lin) - 0.055)


def srgb_decode(v):
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


# ---- segmented curves --------------------------------------------------------------------------

class Curve:
    """Segmented curve. Segment i covers (break[i-1], break[i]]: the first starts at -inf, the last
    ends at +inf. A segment is ('f', g, a, b, c), meaning y = (a x + b)^g + c (formula type 0), or
    ('s', values): samples evenly spaced over the segment, the first point being the previous
    segment's value at the breakpoint (so the first and last segments are formulas)."""

    def __init__(self, breaks, segs):
        assert len(segs) == len(breaks) + 1 and segs[0][0] == 'f' and segs[-1][0] == 'f'
        self.breaks = [float(np.float32(x)) for x in breaks]
        self.segs = [('s', as_f32(s[1])) if s[0] == 's' else ('f',) + tuple(as_f32(s[1:])) for s in segs]

    def icc(self):
        out = b'curf' + u32(0) + u16(len(self.segs)) + u16(0) + f32s(self.breaks)
        for s in self.segs:
            if s[0] == 'f':
                out += b'parf' + u32(0) + u16(0) + u16(0) + f32s(s[1:])
            else:
                out += b'samf' + u32(0) + u32(len(s[1])) + f32s(s[1])
        return out

    @staticmethod
    def formula(s, x):
        g, a, b, c = s[1:]
        e = a * x + b
        if g == 1.0:
            return e + c
        # LittleCMS returns c for a negative base; this module only writes g != 1 where e > 0
        return np.where(e < 0, c, np.power(np.maximum(e, 0), g) + c)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        y = np.empty_like(x)
        edges = [-np.inf] + self.breaks + [np.inf]
        for i, s in enumerate(self.segs):
            m = (x > edges[i]) & (x <= edges[i + 1])
            if s[0] == 'f':
                y[m] = self.formula(s, x[m])
            elif m.any():
                lo, hi = edges[i], edges[i + 1]
                prev = self.segs[i - 1]
                first = self.formula(prev, np.array([lo]))[0] if prev[0] == 'f' else prev[1][-1]
                pts = np.concatenate([[first], s[1]])
                y[m] = np.interp(x[m], np.linspace(lo, hi, len(pts)), pts)
        return y


def identity():
    """y = x for x >= 0. LittleCMS before 2.16 returns c where a x + b < 0, even at gamma 1, so a
    formula segment that may see a negative base is written differently (piecewise_linear); this
    one is only used on channels that are never negative."""
    return Curve([], [('f', 1, 1, 0, 0)])


def relu():
    return Curve([0.0], [('f', 1, 0, 0, 0), ('f', 1, 1, 0, 0)])


def sampled(fn, lo, hi, n):
    """fn sampled at n + 1 points over [lo, hi], constant outside."""
    xs = lo + (hi - lo) * np.arange(1, n + 1) / n
    return Curve([lo, hi], [('f', 1, 0, 0, float(fn(np.float64(lo)))), ('s', fn(xs)),
                            ('f', 1, 0, 0, float(fn(np.float64(hi))))])


def piecewise_linear(lo, hi, values, reach=None):
    """Linear interpolation of values at evenly spaced knots on [lo, hi], continued with the end
    slopes out to reach = (left, right) and constant beyond (default: 100 knot ranges either side).
    One sampled segment, so it is exact. Each linear end is measured from the end of its segment
    where it is 0, so its base a x + b is never negative there: LittleCMS before 2.16 returns c
    for a negative base even at gamma 1."""
    v = np.asarray(values, dtype=np.float64)
    h = (hi - lo) / (len(v) - 1)
    sl, sr = (v[1] - v[0]) / h, (v[-1] - v[-2]) / h
    left, right = reach if reach is not None else (lo - 100 * (hi - lo), hi + 100 * (hi - lo))
    assert left < lo and right > hi
    yl, yr = v[0] + sl * (left - lo), v[-1] + sr * (right - hi)
    lin_l = ('f', 1, sl, -sl * left, yl) if sl >= 0 else ('f', 1, sl, -sl * lo, v[0])
    lin_r = ('f', 1, sr, -sr * hi, v[-1]) if sr >= 0 else ('f', 1, sr, -sr * right, yr)
    return Curve([left, lo, hi, right], [('f', 1, 0, 0, yl), lin_l, ('s', v[1:]), lin_r, ('f', 1, 0, 0, yr)])


def srgb_eotf():
    g, a, b, c, d = SRGB_PARAMS
    return Curve([d], [('f', 1, c, 0, 0), ('f', g, a, b, 0)])


# ---- multiProcessElements ----------------------------------------------------------------------

def _position_table(header, items, share):
    """Header, then (offset, size) for each item from the start of the element, then the items."""
    pos = len(header) + 8 * len(items)
    table, blobs, seen = [], [], {}
    for it in items:
        if not (share and it in seen):
            seen[it] = (pos, len(it))
            blobs.append(pad4(it))
            pos += len(blobs[-1])
        table.append(seen[it])
    return header + b''.join(u32(o) + u32(n) for o, n in table) + b''.join(blobs)


def matf(W, b):
    W = np.asarray(W, dtype=np.float64)
    q, p = W.shape
    assert p <= MAX_CHANNELS and q <= MAX_CHANNELS, f'{p} -> {q} channels: LittleCMS reads at most 15'
    return b'matf' + u32(0) + u16(p) + u16(q) + f32s(W) + f32s(b)


def cvst(curves, share=True):
    n = len(curves)
    return _position_table(b'cvst' + u32(0) + u16(n) + u16(n), [c.icc() for c in curves], share)


def gate(stages):
    """Add a gate channel that paints the plane z <= 0 black, leaving z > 0 unchanged.

    Black point compensation makes LittleCMS run the profile on RGB (0, 0, 0) and take the result
    as the profile's black point, and what it does with a color that is not black depends on the
    version: 2.14 clips L* to 50, 2.16 and 2.17 return 0 above 50, 2.18 and 2.19 return 0 above 95
    and clip to 50 between. A black point of L* 50 maps mid-gray to black. The gate is 1 where
    z <= 0 and 0 elsewhere, rides through every layer in a channel of its own (so hidden layers
    may be at most 14 wide), and the last matrix subtracts 1000 times it from every output."""
    step = Curve([0.0], [('f', 1, 0, 0, 1), ('f', 1, 0, 0, 0)])
    out = [('mat', np.vstack([np.eye(3), [[0, 0, 1]]]), np.zeros(4)),     # R, G, B, B
           ('cur', [identity()] * 3 + [step])]
    last = max(i for i, st in enumerate(stages) if st[0] == 'mat')
    for i, st in enumerate(stages):
        if st[0] == 'mat' and i < last:
            q, p = st[1].shape
            M = np.zeros((q + 1, p + 1))
            M[:q, :p], M[q, p] = st[1], 1
            out.append(('mat', M, np.append(st[2], 0)))
        elif st[0] == 'mat':
            out.append(('mat', np.hstack([st[1], np.full((st[1].shape[0], 1), -1000.0)]), st[2]))
        elif i < last:
            out.append(('cur', list(st[1]) + [identity()]))
        else:
            out.append(st)
    return out


def channels(stages):
    first = stages[0]
    nin = first[1].shape[1] if first[0] == 'mat' else len(first[1])
    n = nin
    for st in stages:
        width = st[1].shape[1] if st[0] == 'mat' else len(st[1])
        assert width == n, 'channel counts of consecutive stages differ'
        n = st[1].shape[0] if st[0] == 'mat' else n
    return nin, n


def mpet(stages, share=True):
    nin, nout = channels(stages)
    elems = [matf(st[1], st[2]) if st[0] == 'mat' else cvst(st[1], share) for st in stages]
    return _position_table(b'mpet' + u32(0) + u16(nin) + u16(nout) + u32(len(elems)), elems, share)


def forward(stages, x, f32=True):
    """Evaluate stages on x of shape (..., inputs), the way the profile encodes them."""
    x = np.asarray(x, dtype=np.float64)
    for st in stages:
        if st[0] == 'mat':
            x = x @ as_f32(st[1]).T + as_f32(st[2])
        else:
            x = np.stack([c(x[..., i]) for i, c in enumerate(st[1])], axis=-1)
        if f32:
            x = as_f32(x)
    return x


def srgb_tail():
    """sRGB-encoded values to the XYZ PCS."""
    return [('cur', [srgb_eotf()] * 3), ('mat', SRGB_TO_XYZ, np.zeros(3))]


# ---- tags and profile --------------------------------------------------------------------------

def mluc(text):
    s = text.encode('utf-16-be')
    return b'mluc' + u32(0) + u32(1) + u32(12) + b'enUS' + u32(len(s)) + u32(28) + s


def xyz_tag(v): return b'XYZ ' + u32(0) + s15f16(v)
def sf32_tag(m): return b'sf32' + u32(0) + s15f16(m)
def para(fn_type, params): return b'para' + u32(0) + u16(fn_type) + u16(0) + s15f16(params)


def _lut(sig, b_curves, matrix, m_curves):
    """lutAToBType ('mAB ', M curves -> matrix -> B curves) or lutBToAType ('mBA ', B curves ->
    matrix -> M curves), without CLUT and A curves. Data: B curves, matrix, M curves."""
    b = b''.join(pad4(c) for c in b_curves)
    mat = s15f16(matrix) + s15f16(np.zeros(3))
    m = b''.join(pad4(c) for c in m_curves)
    off_b, off_mat, off_m = 32, 32 + len(b), 32 + len(b) + len(mat)
    return (sig + u32(0) + bytes([3, 3, 0, 0]) + u32(off_b) + u32(off_mat) + u32(off_m) + u32(0) + u32(0)
            + b + mat + m)


def atob_srgb():
    """A2B as plain sRGB: what an engine that skips DToB shows (the pixel values as sRGB)."""
    ident = [para(0, (1.0,))] * 3
    return _lut(b'mAB ', ident, SRGB_TO_XYZ * U1F15, [para(3, SRGB_PARAMS)] * 3)


def btoa_srgb():
    ident = [para(0, (1.0,))] * 3
    # sRGB OETF as parametric type 4: (a x + b)^g + e for x >= d, c x + f below
    oetf = para(4, (1 / 2.4, 1.055 ** 2.4, 0, 12.92, 0.0031308, -0.055, 0))
    return _lut(b'mBA ', ident, np.linalg.inv(SRGB_TO_XYZ) / U1F15, [oetf] * 3)


def srgb_matrix_tags():
    """Plain sRGB as colorants and tone curves: rXYZ/gXYZ/bXYZ and rTRC/gTRC/bTRC."""
    return ([(s, xyz_tag(SRGB_TO_XYZ[:, i])) for i, s in enumerate(('rXYZ', 'gXYZ', 'bXYZ'))]
            + [(s, para(3, SRGB_PARAMS)) for s in ('rTRC', 'gTRC', 'bTRC')])


def srgb_profile():
    """A plain sRGB display profile (v4, matrix and tone curves, like LittleCMS's built-in one): the
    destination in `magick gallery/trio.png -profile sRGB.icc out.png`, so the command needs no
    profile from the operating system."""
    return assemble([('desc', mluc('sRGB (iccnet)')), ('cprt', mluc('No copyright, use freely')),
                     ('wtpt', xyz_tag(D50)), ('chad', sf32_tag(BRADFORD_D65_TO_D50))] + srgb_matrix_tags())


def profile(net, desc, fallback='full', cprt='No copyright, use freely'):
    """The published layout, for a network whose outputs are sRGB-encoded values.

    D2B0..D2B3 share one 'mpet': the network, then sRGB -> XYZ. LittleCMS reads the DToB tag of
    the intent it is asked for and never falls back from one to another, so all four are present.
    The other tags are plain sRGB, what engines that skip DToB show (the carrier's pixels):
    'matrix': rXYZ/gXYZ/bXYZ and rTRC/gTRC/bTRC. libvips, sharp and Krita drop profiles that
        LittleCMS's intent checks reject, and those checks look at these or at A2B tags. With them,
        LittleCMS finds the black point of the profile by running D2B1 on RGB (0, 0, 0), which the
        gate (see gate()) paints black, so black point compensation (forced for v4 destinations in
        the perceptual intent) leaves the picture alone. Without them it assumes a fixed dark gray
        and moves every color by up to 8 code values (probe/bpc.py).
    'lut': A2B0-A2B2 and B2A0-B2A2. Without them GIMP converts through babl's matrix path and
        darktable through its own, and neither reads DToB.
    'full' (default): both. None: DToB only, which most applications reject.
    """
    net_tag = mpet(net + srgb_tail())
    tags = [('desc', mluc(desc)), ('cprt', mluc(cprt)), ('wtpt', xyz_tag(D50)),
            ('chad', sf32_tag(BRADFORD_D65_TO_D50))] + [(s, net_tag) for s in ('D2B0', 'D2B1', 'D2B2', 'D2B3')]
    if fallback in ('matrix', 'full'):
        tags += srgb_matrix_tags()
    if fallback in ('lut', 'full'):
        tags += [(s, atob_srgb()) for s in ('A2B0', 'A2B1', 'A2B2')]
        tags += [(s, btoa_srgb()) for s in ('B2A0', 'B2A1', 'B2A2')]
    return assemble(tags)


def assemble(tags, cls=b'mntr', version=0x04300000, space=b'RGB ', pcs=b'XYZ ', intent=0):
    """Header, tag table and tag data; identical tag data is stored once and shared."""
    pos = 128 + 4 + 12 * len(tags)
    entries, blobs, seen = [], [], {}
    for sig, data in tags:
        if data not in seen:
            seen[data] = (pos, len(data))
            blobs.append(pad4(data))
            pos += len(blobs[-1])
        entries.append((sig, seen[data]))
    body = u32(len(tags)) + b''.join(s.encode() + u32(o) + u32(n) for s, (o, n) in entries) + b''.join(blobs)
    h = bytearray(128)
    h[0:4] = u32(128 + len(body))
    h[8:12] = u32(version)
    h[12:16], h[16:20], h[20:24] = cls, space, pcs
    h[24:36] = struct.pack('>6H', 2026, 10, 25, 0, 0, 0)
    h[36:40] = b'acsp'
    h[64:68] = u32(intent)
    h[68:80] = s15f16(D50)
    prof = bytearray(bytes(h) + body)
    tmp = bytearray(prof)                    # profile ID: MD5 with flags, intent and ID zeroed
    tmp[44:48], tmp[64:68], tmp[84:100] = b'\0' * 4, b'\0' * 4, b'\0' * 16
    prof[84:100] = hashlib.md5(tmp).digest()
    return bytes(prof)
