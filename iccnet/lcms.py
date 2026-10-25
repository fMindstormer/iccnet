"""An unmodified LittleCMS 2 through ctypes: $LCMS2, else the system's liblcms2."""
import ctypes
import ctypes.util
import os

import numpy as np


def _load():
    candidates = [os.environ.get('LCMS2'), ctypes.util.find_library('lcms2'),
                  '/opt/homebrew/lib/liblcms2.2.dylib', '/usr/local/lib/liblcms2.2.dylib',
                  'liblcms2.so.2']
    for path in filter(None, candidates):
        try:
            return ctypes.CDLL(path)
        except OSError:
            pass
    raise OSError('LittleCMS 2 not found: install it (brew install little-cms2, apt install liblcms2-2) '
                  'or set LCMS2 to the library path')


lib = _load()
vp, u32 = ctypes.c_void_p, ctypes.c_uint32
for name, res, args in [
    ('cmsOpenProfileFromMem', vp, [ctypes.c_char_p, u32]),
    ('cmsCreate_sRGBProfile', vp, []),
    ('cmsCreateXYZProfile', vp, []),
    ('cmsCreateTransform', vp, [vp, u32, vp, u32, u32, u32]),
    ('cmsDoTransform', None, [vp, vp, vp, u32]),
    ('cmsDeleteTransform', None, [vp]),
    ('cmsCloseProfile', ctypes.c_int, [vp]),
    ('cmsGetEncodedCMMversion', ctypes.c_int, []),
    ('cmsIsTag', ctypes.c_int, [vp, u32]),
    ('cmsReadTag', vp, [vp, u32]),
    ('cmsPipelineStageCount', u32, [vp]),
    ('cmsIsIntentSupported', ctypes.c_int, [vp, u32, u32]),
    ('cmsIsMatrixShaper', ctypes.c_int, [vp]),
]:
    fn = getattr(lib, name)
    fn.restype, fn.argtypes = res, args

ERRORS = []
_ERRFN = ctypes.CFUNCTYPE(None, vp, u32, ctypes.c_char_p)
_handler = _ERRFN(lambda ctx, code, text: ERRORS.append(text.decode()))
lib.cmsSetLogErrorHandler.argtypes = [_ERRFN]
lib.cmsSetLogErrorHandler(_handler)


def fmt(space, chans, nbytes, flt=False): return (flt << 22) | (space << 16) | (chans << 3) | nbytes


PT_RGB, PT_XYZ = 4, 9
RGB_8, RGB_16 = fmt(PT_RGB, 3, 1), fmt(PT_RGB, 3, 2)
RGB_DBL, XYZ_DBL = fmt(PT_RGB, 3, 0, True), fmt(PT_XYZ, 3, 0, True)
NOWHITEFIX, FORCE_CLUT, NOOPTIMIZE, HIGHRES, BPC = 0x0004, 0x0002, 0x0100, 0x0400, 0x2000
PERCEPTUAL, RELATIVE, SATURATION, ABSOLUTE = 0, 1, 2, 3
AS_INPUT, AS_OUTPUT = 0, 1
DTYPES = {RGB_8: np.uint8, RGB_16: np.uint16, RGB_DBL: np.float64, XYZ_DBL: np.float64}


def sig(s): return int.from_bytes(s.encode(), 'big')
def version(): return lib.cmsGetEncodedCMMversion()


class Profile:
    def __init__(self, data=None, builtin=None):
        if builtin == 'srgb':
            self.h = lib.cmsCreate_sRGBProfile()
        elif builtin == 'xyz':
            self.h = lib.cmsCreateXYZProfile()
        else:
            self._data = bytes(data)
            self.h = lib.cmsOpenProfileFromMem(self._data, len(self._data))
        if not self.h:
            raise ValueError('profile rejected: ' + '; '.join(ERRORS[-3:]))

    def stages(self, tag):
        p = lib.cmsReadTag(self.h, sig(tag))
        return lib.cmsPipelineStageCount(p) if p else None

    def supports(self, intent, direction=AS_INPUT):
        return bool(lib.cmsIsIntentSupported(self.h, intent, direction))

    def __del__(self):
        if getattr(self, 'h', None):
            lib.cmsCloseProfile(self.h)


def transform(src, dst, pixels, fin=RGB_DBL, fout=RGB_DBL, intent=RELATIVE, flags=0):
    """pixels: array (..., 3) in the dtype of fin; returns (..., 3) in the dtype of fout."""
    del ERRORS[:]
    x = lib.cmsCreateTransform(src.h, fin, dst.h, fout, intent, flags)
    if not x:
        raise ValueError('transform failed: ' + '; '.join(ERRORS))
    a = np.ascontiguousarray(pixels, dtype=DTYPES[fin])
    out = np.zeros(a.shape, dtype=DTYPES[fout])
    lib.cmsDoTransform(x, a.ctypes.data, out.ctypes.data, a.size // 3)
    lib.cmsDeleteTransform(x)
    return out


def render(icc, pixels, path='double', intent=RELATIVE, flags=0):
    """Carrier pixels (uint8, (..., 3)) through the profile to LittleCMS's sRGB, as 0..1 floats.
    path: 'double' (what ImageMagick does), '16' or '8' (integer formats, as Pillow, libvips and
    sharp use; LittleCMS resamples the profile into a table unless flags has NOOPTIMIZE)."""
    src, dst = Profile(icc), Profile(builtin='srgb')
    px = np.asarray(pixels)
    if path == 'double':
        return transform(src, dst, px / 255.0, RGB_DBL, RGB_DBL, intent, flags)
    if path == '16':
        return transform(src, dst, px.astype(np.uint16) * 257, RGB_16, RGB_16, intent, flags) / 65535.0
    return transform(src, dst, px.astype(np.uint8), RGB_8, RGB_8, intent, flags) / 255.0
