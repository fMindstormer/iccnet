"""One carrier PNG through every engine on this machine, each compared with LittleCMS in double.

usage: python probe/engines.py <carrier.png> [--figure out.png [--figure-engines a,b,..]] [--json out.json]
       [--save-dir dir] [--engines a,b,..]
Tools are found through the environment and skipped when missing:
    COLORSYNC_PROBE    the binary built from probe/colorsync.swift (macOS)
    NODE_MODULES_ROOT  a directory where `npm install sharp @imagemagick/magick-wasm` was run
    VIPS_PYTHON        a Python with pyvips (pip install "pyvips[binary]")
    tificc             LittleCMS's own TIFF tool, if it is on PATH (brew install little-cms2)
Each output is classed as exact (within 1.5 code values of the reference everywhere), table (the
picture, approximated), or ignored (the carrier's own pixels, shown as sRGB).
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iccnet import lcms, png  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {
    'colorsync': 'macOS ColorSync (Safari, Preview)',
    'imagemagick': 'ImageMagick 7, -profile (high-res)',
    'imagemagick_quantum': 'ImageMagick 7, quantum mode',
    'sharp': 'sharp (default pipeline)',
    'libvips': 'libvips icc_transform',
    'pillow': 'Pillow ImageCms (defaults)',
    'pillow_noopt': 'Pillow ImageCms, NOOPTIMIZE',
    'tificc': 'tificc (LittleCMS tool), defaults',
    'tificc_c0': 'tificc -c0 (no precalculation)',
    'lcms_16': 'LittleCMS 16-bit (default flags)',
    'lcms_8': 'LittleCMS 8-bit (default flags)',
    'lcms_double': 'LittleCMS double (reference)',
}


def load_rgb(path):
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255.0
    return a


def run_engines(carrier, wanted):
    px, prof = png.read(carrier)
    px = px[..., :3]
    h, w, _ = px.shape
    flat = px.reshape(-1, 3)
    out, versions = {}, {'littlecms': lcms.version()}
    out['lcms_double'] = lcms.render(prof, flat, 'double').reshape(h, w, 3)
    out['lcms_16'] = lcms.render(prof, flat, '16').reshape(h, w, 3)
    out['lcms_8'] = lcms.render(prof, flat, '8').reshape(h, w, 3)
    im = Image.open(carrier).convert('RGB')
    src = ImageCms.ImageCmsProfile(io.BytesIO(prof))
    dst = ImageCms.createProfile('sRGB')
    out['pillow'] = np.asarray(ImageCms.profileToProfile(im, src, dst), dtype=np.float64) / 255.0
    out['pillow_noopt'] = np.asarray(ImageCms.profileToProfile(im, src, dst, flags=ImageCms.Flags.NOOPTIMIZE),
                                     dtype=np.float64) / 255.0
    versions['pillow'] = f'{Image.__version__} (LittleCMS {ImageCms.core.littlecms_version})'
    with tempfile.TemporaryDirectory() as tmp:
        if os.environ.get('NODE_MODULES_ROOT'):
            r = subprocess.run(['node', os.path.join(HERE, 'node.mjs'), os.environ['NODE_MODULES_ROOT'], carrier, tmp],
                               check=True, capture_output=True, text=True)
            versions.update(json.loads(r.stdout.strip().splitlines()[-1]))
            out['imagemagick'] = load_rgb(os.path.join(tmp, 'im_highres.png'))
            out['imagemagick_quantum'] = load_rgb(os.path.join(tmp, 'im_quantum.png'))
            out['sharp'] = load_rgb(os.path.join(tmp, 'sharp.png'))
        if os.environ.get('VIPS_PYTHON'):
            r = subprocess.run([os.environ['VIPS_PYTHON'], os.path.join(HERE, 'vips.py'), carrier,
                                os.path.join(tmp, 'vips.png')], check=True, capture_output=True, text=True)
            versions.update(json.loads(r.stdout.strip().splitlines()[-1]))
            out['libvips'] = load_rgb(os.path.join(tmp, 'vips.png'))
        if shutil.which('tificc'):
            tif = os.path.join(tmp, 'carrier.tif')
            im.save(tif, icc_profile=prof)
            banner = subprocess.run(['tificc'], capture_output=True, text=True)
            versions['tificc'] = (banner.stdout + banner.stderr).splitlines()[0].split('[')[-1].rstrip(']')
            for key, args in (('tificc', []), ('tificc_c0', ['-c0'])):
                subprocess.run(['tificc', *args, tif, os.path.join(tmp, f'{key}.tif')], check=True, capture_output=True)
                out[key] = load_rgb(os.path.join(tmp, f'{key}.tif'))
        if os.environ.get('COLORSYNC_PROBE'):
            icc_path = os.path.join(tmp, 'p.icc')
            open(icc_path, 'wb').write(prof)
            r = subprocess.run([os.environ['COLORSYNC_PROBE'], icc_path, carrier, os.path.join(tmp, 'cs')],
                               check=True, capture_output=True, text=True)
            versions['colorsync'] = r.stdout.strip().splitlines()[0]
            cg = np.fromfile(os.path.join(tmp, 'cs.cg.f32'), '<f4').reshape(h, w, 4)[..., :3]
            out['colorsync'] = cg.astype(np.float64)
    out = {k: np.clip(v, 0, 1) for k, v in out.items() if not wanted or k in wanted or k == 'lcms_double'}
    return px / 255.0, out, versions


def classify(pixels, out):
    ref = out['lcms_double']
    rows = {}
    for k, a in out.items():
        e = np.abs(a - ref) * 255
        raw = np.abs(a - pixels).mean() * 255
        kind = 'exact' if e.max() <= 1.5 else 'ignored' if raw < 1.5 else 'table'
        rows[k] = dict(label=LABELS[k], kind=kind, mean=round(float(e.mean()), 2), max=round(float(e.max()), 1),
                       mean_vs_carrier=round(float(raw), 2))
    return rows


def font(size):
    for path in ('/System/Library/Fonts/Helvetica.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def figure(out, rows, order, path):
    h, w, _ = out['lcms_double'].shape
    head, gap = 40, 8
    canvas = Image.new('RGB', (w, len(order) * (h + head + gap) - gap), 'white')
    d = ImageDraw.Draw(canvas)
    big, small = font(15), font(12)
    for i, k in enumerate(order):
        y = i * (h + head + gap)
        canvas.paste(Image.fromarray((out[k] * 255).round().astype(np.uint8)), (0, y + head))
        r = rows[k]
        d.text((2, y + 3), r['label'], fill='black', font=big)
        note = {'exact': f'exact: within {r["max"]:.1f} code values of LittleCMS in double',
                'table': f'approximated: mean {r["mean"]:.1f}, max {r["max"]:.0f} code values off',
                'ignored': 'profile not evaluated: the carrier pixels, shown as sRGB'}[r['kind']]
        d.text((2, y + 21), note, fill=(90, 90, 90), font=small)
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('carrier')
    ap.add_argument('--figure')
    ap.add_argument('--json')
    ap.add_argument('--save-dir', help='write each engine\'s output there as <engine>.png')
    ap.add_argument('--engines', default='', help='subset, in figure order (default: all found)')
    ap.add_argument('--figure-engines', default='', help='the figure\'s rows, if not every engine run')
    a = ap.parse_args()
    wanted = [e for e in a.engines.split(',') if e]
    shown = [e for e in a.figure_engines.split(',') if e]
    pixels, out, versions = run_engines(a.carrier, wanted)
    rows = classify(pixels, out)
    for k, r in rows.items():
        print(f'{r["label"]:40s} {r["kind"]:8s} mean {r["mean"]:6.2f}  max {r["max"]:6.1f}')
    if a.figure:
        figure(out, rows, shown or wanted or [k for k in LABELS if k in out], a.figure)
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
        for k, img in out.items():
            Image.fromarray((img * 255).round().astype(np.uint8)).save(os.path.join(a.save_dir, f'{k}.png'))
    if a.json:
        json.dump(dict(carrier=os.path.basename(a.carrier), versions=versions, engines=rows), open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
