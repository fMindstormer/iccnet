"""Written profiles, read back and run by an unmodified LittleCMS, match iccnet.icc.forward().

The engine facts the project depends on are tested too, so a LittleCMS that behaves differently
(CI runs Ubuntu's 2.14, macOS has 2.17) fails here first.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iccnet import icc, lcms, png, read  # noqa: E402

# NumPy built against Apple's Accelerate raises spurious floating-point warnings in matmul
pytestmark = pytest.mark.filterwarnings('ignore:.*encountered in matmul:RuntimeWarning')
rng = np.random.default_rng(1)
CODES = rng.integers(1, 256, (20000, 3)).astype(np.uint8)


def random_net(width=8, act='relu'):
    curves = {'relu': [icc.relu()] * width,
              'pwl': [icc.piecewise_linear(-3, 3, rng.normal(0, 1, 17)) for _ in range(width)],
              'sin': [icc.sampled(np.sin, -20, 20, 4096)] * width}[act]
    return [('mat', rng.normal(0, 2, (width, 3)), rng.uniform(-1, 1, width)), ('cur', curves),
            ('mat', rng.normal(0, 0.5, (3, width)), rng.uniform(-0.2, 0.2, 3)),
            ('cur', [icc.sampled(lambda t: 1 / (1 + np.exp(-t)), -8, 8, 1024)] * 3)]


@pytest.mark.parametrize('act', ['relu', 'pwl', 'sin'])
def test_littlecms_evaluates_the_stages(act):
    net = icc.gate(random_net(act=act))
    prof = icc.profile(net, f'test {act}')
    pcs = lcms.transform(lcms.Profile(prof), lcms.Profile(builtin='xyz'), CODES / 255.0, lcms.RGB_DBL, lcms.XYZ_DBL)
    want = icc.forward(net + icc.srgb_tail(), CODES / 255.0)
    assert np.abs(pcs - want).max() < 2e-6
    srgb = lcms.render(prof, CODES)
    assert np.abs(np.clip(srgb, 0, 1) - np.clip(icc.forward(net, CODES / 255.0), 0, 1)).max() * 255 < 0.3


def clamped_formula(s, x):
    """A formula segment as LittleCMS evaluated it before 2.16: c wherever a x + b < 0, even at gamma 1."""
    g, a, b, c = s[1:]
    e = a * x + b
    return np.where(e < 0, c, np.power(np.maximum(e, 0), g) + c)


@pytest.mark.parametrize('source', ['random', 'gallery'])
def test_older_littlecms_paints_the_same(monkeypatch, source):
    """The writer keeps the base of every formula segment non-negative where it is evaluated, so
    LittleCMS 2.14 and 2.15 (Ubuntu 24.04, Debian 12) paint what 2.16 and later paint."""
    if source == 'random':
        nets = [(icc.gate(random_net(act='pwl')), CODES / 255.0)]
    else:
        nets = []
        for name in ('mona', 'pearl', 'wave', 'duet', 'trio', 'lowpoly'):
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gallery', f'{name}.png')
            if not os.path.exists(path):
                pytest.skip('gallery not built')
            px, prof = png.read(path)
            nets.append((read.mpet_stages(read.tag_data(prof, 'D2B1')), px.reshape(-1, 3) / 255.0))
    want = [icc.forward(n, x) for n, x in nets]
    monkeypatch.setattr(icc.Curve, 'formula', staticmethod(clamped_formula))
    for (n, x), w in zip(nets, want):
        assert np.abs(icc.forward(n, x) - w).max() < 1e-5


def test_profile_reads_back():
    net = icc.gate(random_net(act='pwl'))
    data = icc.profile(net, 'round trip')
    sigs = [t[0] for t in read.tags(data)]
    for s in ('D2B0', 'D2B1', 'D2B2', 'D2B3', 'A2B0', 'A2B1', 'A2B2', 'B2A0', 'rXYZ', 'rTRC'):
        assert s in sigs
    back = read.mpet_stages(read.tag_data(data, 'D2B1'))
    x = CODES / 255.0
    assert np.array_equal(icc.forward(back, x), icc.forward(net + icc.srgb_tail(), x))
    assert read.header(data)['version'] == '4.3.0'


def test_every_intent_is_supported():
    p = lcms.Profile(icc.profile(icc.gate(random_net()), 'intents'))
    for intent in range(4):
        assert p.supports(intent, lcms.AS_INPUT) and p.supports(intent, lcms.AS_OUTPUT)


def test_intent_checks_ignore_dtob():
    """cmsIsIntentSupported looks at A2B tags and matrix/TRC tags only: a DToB-only profile
    supports nothing, and applications that check first (libvips, sharp, Krita) drop it."""
    net = icc.gate(random_net())
    assert not lcms.Profile(icc.profile(net, 'dtob only', fallback=None)).supports(lcms.PERCEPTUAL)
    for fb in ('matrix', 'lut', 'full'):
        assert lcms.Profile(icc.profile(net, fb, fallback=fb)).supports(lcms.PERCEPTUAL)


def test_each_intent_reads_its_own_dtob():
    """With D2B0 alone, the perceptual intent paints and the relative one falls back to A2B1, not to D2B0."""
    net = icc.gate(random_net())
    tags = [('desc', icc.mluc('D2B0 only')), ('cprt', icc.mluc('none')), ('wtpt', icc.xyz_tag(icc.D50)),
            ('D2B0', icc.mpet(net + icc.srgb_tail()))]
    tags += [(s, icc.xyz_tag(icc.SRGB_TO_XYZ[:, i])) for i, s in enumerate(('rXYZ', 'gXYZ', 'bXYZ'))]
    tags += [(s, icc.para(3, icc.SRGB_PARAMS)) for s in ('rTRC', 'gTRC', 'bTRC')]
    tags += [(s, icc.atob_srgb()) for s in ('A2B0', 'A2B1', 'A2B2')] + [(s, icc.btoa_srgb()) for s in ('B2A0', 'B2A1', 'B2A2')]
    prof = icc.assemble(tags)
    painted = np.clip(icc.forward(net, CODES / 255.0), 0, 1)
    assert np.abs(lcms.render(prof, CODES, 'double', lcms.PERCEPTUAL) - painted).max() * 255 < 0.3
    # A2B1 is plain sRGB, its matrix rounded to s15Fixed16
    assert np.abs(lcms.render(prof, CODES, 'double', lcms.RELATIVE) - CODES / 255.0).max() * 255 < 1


def test_gate_paints_zero_black_and_leaves_the_rest():
    net = random_net()
    gated = icc.gate(net)
    x = CODES / 255.0
    # equal up to float32 rounding: the extra column changes the order of the sums
    assert np.abs(icc.forward(gated, x) - icc.forward(net, x)).max() < 1e-6
    zero = icc.forward(gated, np.zeros((1, 3)))          # the sigmoid's lower tail: black in 8 bits
    assert np.all(zero * 255 < 0.5)


@pytest.mark.parametrize('intent,flags', [(lcms.PERCEPTUAL, 0), (lcms.PERCEPTUAL, lcms.BPC), (lcms.RELATIVE, lcms.BPC)])
def test_black_point_compensation_leaves_the_picture(intent, flags):
    prof = icc.profile(icc.gate(random_net()), 'bpc')
    plain = lcms.render(prof, CODES, 'double', lcms.RELATIVE)
    assert np.abs(lcms.render(prof, CODES, 'double', intent, flags) - plain).max() * 255 < 0.01


def test_sixteen_channels_are_rejected():
    with pytest.raises(AssertionError):
        icc.matf(np.zeros((16, 3)), np.zeros(16))
    # written by hand past the writer's check: LittleCMS refuses the tag
    p, q = 3, 16
    m = b'matf' + icc.u32(0) + icc.u16(p) + icc.u16(q) + icc.f32s(np.zeros(p * q)) + icc.f32s(np.zeros(q))
    wide = icc._position_table(b'mpet' + icc.u32(0) + icc.u16(3) + icc.u16(16) + icc.u32(1), [m], True)
    prof = icc.assemble([('desc', icc.mluc('wide')), ('cprt', icc.mluc('none')), ('wtpt', icc.xyz_tag(icc.D50)),
                         ('D2B1', wide), ('A2B1', icc.atob_srgb())])
    with pytest.raises(ValueError):
        lcms.render(prof, CODES[:10])


def test_integer_formats_use_a_table():
    """8- and 16-bit formats with default flags sample the profile into a table: a 20-cycle sine is lost."""
    net = [('mat', np.diag([2 * np.pi * 20, 1, 1]), np.zeros(3)),
           ('cur', [icc.sampled(lambda t: 0.5 - 0.5 * np.cos(t), 0, 2 * np.pi * 20, 8192), icc.identity(), icc.identity()])]
    prof = icc.profile(net, 'sine')
    want = np.clip(icc.forward(net, CODES / 255.0), 0, 1)
    for path in ('16', '8'):
        assert np.abs(lcms.render(prof, CODES, path) - want).mean() * 255 > 20
        assert np.abs(lcms.render(prof, CODES, path, flags=lcms.NOOPTIMIZE) - want).max() * 255 < 1
    assert np.abs(lcms.render(prof, CODES, 'double') - want).max() * 255 < 0.6


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_png_round_trip(tmp_path, dtype):
    img = rng.integers(0, np.iinfo(dtype).max, (37, 53, 3)).astype(dtype)
    prof = icc.profile(random_net(), 'png')
    png.write(tmp_path / 'a.png', img, prof)
    back, embedded = png.read(tmp_path / 'a.png')
    assert np.array_equal(back, img) and embedded == prof
    from PIL import Image
    if dtype == np.uint8:                    # another decoder agrees
        im = Image.open(tmp_path / 'a.png')
        assert np.array_equal(np.asarray(im), img) and im.info['icc_profile'] == prof


def test_model_export_matches_pytorch():
    torch = pytest.importorskip('torch')
    from iccnet.model import Net
    from iccnet.train import tile
    x = tile(100).reshape(-1, 3) / 255.0
    for act in ('relu', 'sin', 'pwl'):
        torch.manual_seed(0)
        net = Net(act, 14, 3, 16)
        prof = icc.profile(net.stages(x), act)
        with torch.no_grad():
            want = net(torch.tensor(x, dtype=torch.float32)).double().numpy()
        got = lcms.render(prof, np.round(x * 255).astype(np.uint8))
        assert np.abs(got - want).max() * 255 < 0.1, act


@pytest.mark.skipif(shutil.which('node') is None, reason='needs Node 18+')
def test_page_evaluator_matches_littlecms(tmp_path):
    """pages/iccnet.js (the interactive page's evaluator) against LittleCMS, on a gallery carrier."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    carrier = os.path.join(root, 'gallery', 'mona.png')
    if not os.path.exists(carrier):
        pytest.skip('gallery not built')
    px, prof = png.read(carrier)
    flat = px.reshape(-1, 3)
    for path, name in (('double', 'double'), ('16', 'table16')):
        out = np.clip(lcms.render(prof, flat, path), 0, 1).reshape(px.shape)
        png.write(tmp_path / f'{name}.png', (out * 255).round().astype(np.uint8))
    r = subprocess.run(['node', os.path.join(root, 'tests', 'eval.mjs'), carrier, str(tmp_path / 'double.png'),
                        str(tmp_path / 'table16.png')], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
