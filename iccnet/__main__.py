"""python -m iccnet train | carrier | render | inspect  (-h on each for options)"""
import argparse
import json
import sys

import numpy as np


def cmd_train(a):
    from .train import run
    names = a.paintings.split(',')
    codes = [int(c) for c in a.codes.split(',')] if a.codes else ([128] if len(names) == 1 else
                                                                   list(np.linspace(16, 239, len(names)).round().astype(int)))
    assert len(codes) == len(names), 'one latent code per painting'
    r = run(names, codes, a.name, a.out, act=a.act, width=a.width, depth=a.depth, knots=a.knots,
            steps=a.steps, batch=a.batch, lr=a.lr, seed=a.seed, threads=a.threads)
    print(json.dumps(r, indent=1))


def cmd_carrier(a):
    from . import png
    from .train import carrier
    icc = open(a.profile, 'rb').read()
    png.write(a.out, carrier([int(c) for c in a.codes.split(',')], a.side), icc)


def cmd_render(a):
    from PIL import Image
    from . import lcms, png
    px, icc = png.read(a.carrier)
    if icc is None:
        sys.exit(f'{a.carrier} has no embedded profile')
    px = px[..., :3] if px.dtype == np.uint8 else (px[..., :3] // 257).astype(np.uint8)
    intent = {'perceptual': lcms.PERCEPTUAL, 'relative': lcms.RELATIVE}[a.intent]
    out = lcms.render(icc, px.reshape(-1, 3), a.path, intent, lcms.NOOPTIMIZE if a.noopt else 0)
    Image.fromarray((np.clip(out, 0, 1).reshape(px.shape) * 255).round().astype(np.uint8)).save(a.out)


def cmd_inspect(a):
    from . import lcms, png, read
    data = open(a.file, 'rb').read()
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        data = png.read(a.file)[1]
        if data is None:
            sys.exit('no embedded profile')
    print(json.dumps(read.header(data)))
    prof = lcms.Profile(data)
    for sig, off, size, kind in read.tags(data):
        line = f'  {sig} {kind} {size:7d} bytes at {off}'
        if kind == 'mpet':
            st = read.mpet_stages(data[off:off + size])
            widths = [s[1].shape[0] if s[0] == 'mat' else len(s[1]) for s in st]
            line += f': {len(st)} elements, channels {st[0][1].shape[1]} -> ' + ' -> '.join(map(str, widths))
        print(line)
    names = ['perceptual', 'relative', 'saturation', 'absolute']
    print('LittleCMS', lcms.version(), 'supports, as input:',
          ', '.join(n for i, n in enumerate(names) if prof.supports(i)) or 'nothing')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='iccnet')
    sub = ap.add_subparsers(required=True)
    t = sub.add_parser('train', help='fit paintings, write <name>.icc/.png/.npz/.json')
    t.add_argument('--paintings', default='mona', help='comma-separated: mona, pearl, wave')
    t.add_argument('--codes', default=None, help='latent code (blue, 0-255) per painting')
    t.add_argument('--act', default='pwl', choices=('relu', 'sin', 'pwl'))
    t.add_argument('--width', type=int, default=14, help='hidden channels; the black gate takes one more (LittleCMS caps at 15)')
    t.add_argument('--depth', type=int, default=8)
    t.add_argument('--knots', type=int, default=64)
    t.add_argument('--steps', type=int, default=8000)
    t.add_argument('--batch', type=int, default=8192)
    t.add_argument('--lr', type=float, default=None)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--threads', type=int, default=None)
    t.add_argument('--name', required=True)
    t.add_argument('--out', default='runs')
    t.set_defaults(fn=cmd_train)
    c = sub.add_parser('carrier', help='a carrier PNG for a profile')
    c.add_argument('profile')
    c.add_argument('out')
    c.add_argument('--codes', default='128')
    c.add_argument('--side', type=int, default=256)
    c.set_defaults(fn=cmd_carrier)
    r = sub.add_parser('render', help='paint a carrier with LittleCMS')
    r.add_argument('carrier')
    r.add_argument('out')
    r.add_argument('--path', default='double', choices=('double', '16', '8'))
    r.add_argument('--intent', default='perceptual', choices=('perceptual', 'relative'))
    r.add_argument('--noopt', action='store_true', help='cmsFLAGS_NOOPTIMIZE: no table on 8/16-bit paths')
    r.set_defaults(fn=cmd_render)
    i = sub.add_parser('inspect', help='header, tags and DToB pipelines of a profile or PNG')
    i.add_argument('file')
    i.set_defaults(fn=cmd_inspect)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
