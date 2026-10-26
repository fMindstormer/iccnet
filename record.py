"""The README's animation, docs/images/run.png, rendered from a real run of its command.

    python record.py [--out docs/images/run.png]
Runs `magick -version` and `magick trio.png -profile sRGB.icc painted.png` (or ImageMagick 6's
convert) in a scratch directory, times the conversion, and draws the frames from what it got: the
carrier's own pixels, which are what a browser shows, the command and the version line as typed and
printed, and ImageMagick's painted.png, shown once the command returns. It is not a screen
capture; everything in it comes from the run. The file is an animated PNG, lossless, so the
gradient and the paintings keep the run's exact pixels (a GIF's 256 colors would dither the
gradient); viewers without APNG support show the last frame.
"""
import argparse
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from iccnet import png

W, PAD, CAP, GAP = 800, 16, 28, 14
TERM_LINE, TERM_PAD = 22, 14
BG, RULE, INK, MUTED = (246, 247, 249), (208, 213, 221), (31, 35, 40), (101, 109, 118)
TERM_BG, TERM_INK, TERM_DIM, PROMPT = (20, 22, 27), (230, 232, 236), (154, 161, 173), (126, 231, 135)


def font(names, size):
    for path in names:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


MONO = font(['/System/Library/Fonts/Menlo.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'], 14)
SANS = font(['/System/Library/Fonts/Helvetica.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'], 14)


def imagemagick():
    for name in ('magick', 'convert'):
        path = shutil.which(name)
        if path and 'ImageMagick' in subprocess.run([path, '-version'], capture_output=True, text=True).stdout:
            return path
    raise SystemExit('needs ImageMagick 6 or 7 on PATH (brew install imagemagick, apt install imagemagick)')


def run():
    """The real run: version line, command, elapsed time, and the two pictures."""
    magick = imagemagick()
    cmd = os.path.basename(magick)
    version = subprocess.run([magick, '-version'], capture_output=True, text=True).stdout.splitlines()[0]
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy('gallery/trio.png', tmp)
        shutil.copy('sRGB.icc', tmp)
        t = time.perf_counter()
        subprocess.run([magick, 'trio.png', '-profile', 'sRGB.icc', 'painted.png'], cwd=tmp, check=True)
        seconds = time.perf_counter() - t
        painted = np.asarray(Image.open(os.path.join(tmp, 'painted.png')).convert('RGB'))
    carrier = png.read('gallery/trio.png')[0][..., :3]
    return dict(cmd=cmd, version=version, seconds=seconds, carrier=carrier, painted=painted,
                line=f'{cmd} trio.png -profile sRGB.icc painted.png', kb=os.path.getsize('gallery/trio.png') / 1024)


def layout(h_img):
    y_cap1 = PAD
    y_img1 = y_cap1 + CAP
    y_term = y_img1 + h_img + GAP
    h_term = 2 * TERM_PAD + 4 * TERM_LINE
    y_cap2 = y_term + h_term + GAP
    y_img2 = y_cap2 + CAP
    return dict(y_img1=y_img1, y_term=y_term, h_term=h_term, y_cap2=y_cap2, y_img2=y_img2, height=y_img2 + h_img + PAD)


def frame(r, L, typed, state):
    """state: 'typing' (cursor after the typed text), 'running' (cursor on the next line), 'done'."""
    h_img, w_img = r['carrier'].shape[:2]
    x0 = (W - w_img) // 2
    im = Image.new('RGB', (W, L['height']), BG)
    d = ImageDraw.Draw(im)
    d.text((x0, PAD + 4), 'trio.png, as a browser shows it', fill=INK, font=SANS)
    d.text((x0 + w_img, PAD + 4), f'{r["kb"]:.0f} KB, its profile embedded', fill=MUTED, font=SANS, anchor='ra')
    d.rectangle([x0 - 1, L['y_img1'] - 1, x0 + w_img, L['y_img1'] + h_img], outline=RULE)
    im.paste(Image.fromarray(r['carrier']), (x0, L['y_img1']))
    d.rounded_rectangle([x0 - 1, L['y_term'], x0 + w_img, L['y_term'] + L['h_term']], 8, fill=TERM_BG)
    tx, ty = x0 + TERM_PAD, L['y_term'] + TERM_PAD
    cw = MONO.getlength('m')
    d.text((tx, ty), '$', fill=PROMPT, font=MONO)
    d.text((tx + 2 * cw, ty), f'{r["cmd"]} -version | head -1', fill=TERM_INK, font=MONO)
    d.text((tx, ty + TERM_LINE), r['version'], fill=TERM_DIM, font=MONO)
    y = ty + 2 * TERM_LINE
    d.text((tx, y), '$', fill=PROMPT, font=MONO)
    d.text((tx + 2 * cw, y), typed, fill=TERM_INK, font=MONO)
    if state == 'typing':
        cx = tx + (2 + len(typed)) * cw
        d.rectangle([cx, y + 2, cx + cw - 1, y + 17], fill=TERM_INK)
    elif state == 'running':
        d.rectangle([tx, y + TERM_LINE + 2, tx + cw - 1, y + TERM_LINE + 17], fill=TERM_INK)
    else:
        d.text((tx, y + TERM_LINE), '$', fill=PROMPT, font=MONO)
        d.rectangle([tx + 2 * cw, y + TERM_LINE + 2, tx + 3 * cw - 1, y + TERM_LINE + 17], fill=TERM_INK)
    done = state == 'done'
    d.text((x0, L['y_cap2'] + 4), 'painted.png, written by ImageMagick' if done else 'painted.png',
           fill=INK if done else MUTED, font=SANS)
    if done:
        d.text((x0 + w_img, L['y_cap2'] + 4), f'{r["seconds"]:.2f} s', fill=MUTED, font=SANS, anchor='ra')
    d.rectangle([x0 - 1, L['y_img2'] - 1, x0 + w_img, L['y_img2'] + h_img], outline=RULE)
    if done:
        im.paste(Image.fromarray(r['painted']), (x0, L['y_img2']))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='docs/images/run.png')
    a = ap.parse_args()
    r = run()
    print(r['version'])
    print(f'{r["line"]}: {r["seconds"]:.2f} s')
    L = layout(r['carrier'].shape[0])
    frames, ms = [frame(r, L, '', 'typing')], [1800]
    for n in range(2, len(r['line']) + 2, 2):
        frames.append(frame(r, L, r['line'][:n], 'typing'))
        ms.append(60)
    ms[-1] = 500
    frames.append(frame(r, L, r['line'], 'running'))
    ms.append(max(100, round(r['seconds'] * 1000)))
    frames.append(frame(r, L, r['line'], 'done'))
    ms.append(4500)
    # the first image is the default one, outside the animation: the finished frame
    frames[-1].save(a.out, save_all=True, append_images=frames, default_image=True, duration=ms, loop=0,
                    disposal=0, blend=0, optimize=True)
    print(f'{a.out}: {len(frames)} frames, {sum(ms) / 1000:.1f} s, {os.path.getsize(a.out) / 1024:.0f} KB')


if __name__ == '__main__':
    main()
