"""The reproduction kit (release.py): sRGB.icc, the expected paintings, and the README's command
run by a native ImageMagick when one is on PATH. CI installs Ubuntu's ImageMagick 6."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from iccnet import icc, lcms, png  # noqa: E402

pytestmark = pytest.mark.filterwarnings('ignore:.*encountered in matmul:RuntimeWarning')
GALLERY = ['mona', 'pearl', 'wave', 'duet', 'trio', 'lowpoly']


def load(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float64)


def imagemagick():
    """magick (7) or convert (6), if it is ImageMagick's: Windows has a convert.exe of its own."""
    for name in ('magick', 'convert'):
        path = shutil.which(name)
        if path and 'ImageMagick' in subprocess.run([path, '-version'], capture_output=True, text=True).stdout:
            return path
    return None


def test_srgb_icc_is_plain_srgb():
    data = open(os.path.join(ROOT, 'sRGB.icc'), 'rb').read()
    assert data == icc.srgb_profile()
    rgb = np.random.default_rng(2).random((20000, 3))
    xyz = lcms.Profile(builtin='xyz')
    mine = lcms.transform(lcms.Profile(data), xyz, rgb, lcms.RGB_DBL, lcms.XYZ_DBL)
    builtin = lcms.transform(lcms.Profile(builtin='srgb'), xyz, rgb, lcms.RGB_DBL, lcms.XYZ_DBL)
    # the colorants are the ones the profiles' own sRGB tail uses, whose white has Z = 0.82521,
    # not LittleCMS's 0.8249
    assert np.abs(mine - builtin).max() < 5e-4


@pytest.mark.parametrize('name', GALLERY)
def test_expected_paintings_are_what_littlecms_paints(name):
    px, prof = png.read(os.path.join(ROOT, 'gallery', f'{name}.png'))
    srgb = icc.srgb_profile()
    # the rounding to 8 bits, plus what versions differ by in double precision: 2.17 and 2.19 agree
    # exactly, and 2.14 puts one value of the wave 0.511 away from its rounding
    for path, suffix, tol in (('double', '', 0.6), ('8', '-table', 1)):
        got = np.clip(lcms.render(prof, px[..., :3], path, lcms.PERCEPTUAL, dst=srgb), 0, 1) * 255
        want = load(os.path.join(ROOT, 'results', 'expected', f'{name}{suffix}.png'))
        assert np.abs(got - want).max() <= tol, f'{name}{suffix}'


@pytest.mark.skipif(imagemagick() is None, reason='needs ImageMagick 6 or 7 on PATH')
def test_imagemagick_paints_the_expected_picture(tmp_path):
    """The README's command, then the comparison REPRODUCE.md gives."""
    magick = imagemagick()
    out, want = str(tmp_path / 'painted.png'), os.path.join(ROOT, 'results', 'expected', 'trio.png')
    print(subprocess.run([magick, '-version'], capture_output=True, text=True).stdout.splitlines()[0])
    subprocess.run([magick, os.path.join(ROOT, 'gallery', 'trio.png'), '-profile', os.path.join(ROOT, 'sRGB.icc'), out],
                   check=True)
    assert np.abs(load(out) - load(want)).max() <= 1
    compare = [magick, 'compare'] if os.path.basename(magick).startswith('magick') else [shutil.which('compare')]
    r = subprocess.run(compare + ['-metric', 'AE', '-fuzz', '1%', out, want, 'null:'], capture_output=True, text=True)
    print('compare:', r.stderr.strip())
    assert r.returncode == 0 and r.stderr.split()[0] == '0'
