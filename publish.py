"""Export every trained run with the current writer, measure it, and build the gallery and figures.

    python publish.py
reads   runs/<name>.npz (from train_all.py)
writes  gallery/<name>.png   carriers, each with its profile embedded
        gallery/<name>.npz   the weights
        gallery/README.md    the list
        results/capacity.json  the figures quoted in README.md
        docs/images/capacity.png, docs/images/curves.png, docs/images/capacity/<run>.png
"""
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from iccnet import icc, lcms, png
from iccnet.model import load
from iccnet.train import SIDE, carrier, painting, psnr, tile
from train_all import RUNS

GALLERY = {'mona': 'mona_pwl_d8', 'pearl': 'pearl_pwl_d8', 'wave': 'wave_pwl_d8', 'duet': 'duet_pwl_d8',
           'trio': 'trio_pwl_d8', 'lowpoly': 'mona_relu_d8'}
TITLES = {'mona': 'Leonardo da Vinci, Mona Lisa', 'pearl': 'Johannes Vermeer, Girl with a Pearl Earring',
          'wave': 'Katsushika Hokusai, The Great Wave off Kanagawa'}
ACT = {'relu': 'ReLU', 'sin': 'sine', 'pwl': 'learned curves'}


def gallery_readme(rows, path):
    runs = {r['name']: r for r in rows}
    out = ['# Gallery', '',
           'Carriers: R = column, G = row, B = the code of a painting, one 256-pixel tile per code, with the',
           'profile embedded. `<name>.npz` holds the weights (`iccnet.model.load`). Written by `publish.py`;',
           'PSNR against the 256 × 256 painting, for LittleCMS\'s exact render and for its 8-bit table.', '',
           '| File | Network | Profile | Code: painting, PSNR exact / table |', '|---|---|---:|---|']
    for short, name in GALLERY.items():
        if name not in runs:
            continue
        r = runs[name]
        net = f'{ACT[r["act"]]}, {r["depth"]} layers of {r["width"]}, {r["parameters"]:,} parameters'
        cells = '; '.join(f'{p["code"]}: {TITLES[p["painting"]].split(", ")[1]}, {p["psnr"]:.2f} / {p["psnr_table_8bit"]:.2f} dB'
                          for p in r['paintings'])
        out.append(f'| `{short}.png` | {net} | {r["profile_bytes"] / 1024:.0f} KB | {cells} |')
    open(path, 'w').write('\n'.join(out) + '\n')


def export(name):
    net, cfg = load(f'runs/{name}.npz')
    names, codes = cfg['paintings'], cfg['codes']
    X = np.concatenate([tile(c).reshape(-1, 3) / 255.0 for c in codes])
    Y = np.concatenate([painting(n).reshape(-1, 3) for n in names])
    prof = icc.profile(net.stages(X), f'iccnet: {"; ".join(TITLES[n] for n in names)}')
    with torch.no_grad():
        P = net(torch.tensor(X, dtype=torch.float32)).double().numpy()
    px = np.round(X * 255).astype(np.uint8)
    rel = np.clip(lcms.render(prof, px, 'double', lcms.RELATIVE), 0, 1)
    per = np.clip(lcms.render(prof, px, 'double', lcms.PERCEPTUAL, lcms.BPC), 0, 1)
    t8 = np.clip(lcms.render(prof, px, '8', lcms.PERCEPTUAL), 0, 1)
    t16 = np.clip(lcms.render(prof, px, '16', lcms.PERCEPTUAL), 0, 1)
    n = SIDE * SIDE
    per_painting = []
    for i, (p, code) in enumerate(zip(names, codes)):
        s = slice(i * n, (i + 1) * n)
        per_painting.append(dict(
            painting=p, code=int(code), psnr=round(psnr(rel[s], Y[s]), 2), psnr_pytorch=round(psnr(P[s], Y[s]), 2),
            psnr_table_8bit=round(psnr(t8[s], Y[s]), 2), psnr_table_16bit=round(psnr(t16[s], Y[s]), 2),
            max_littlecms_vs_pytorch=round(float(np.abs(rel[s] - np.clip(P[s], 0, 1)).max() * 255), 3),
            max_perceptual_bpc_vs_relative=round(float(np.abs(per[s] - rel[s]).max() * 255), 3)))
    report = json.load(open(f'runs/{name}.json'))
    params = sum(v.numel() for v in net.parameters())
    row = dict(name=name, act=cfg['act'], width=cfg['width'], depth=cfg['depth'], knots=cfg['knots'] if cfg['act'] == 'pwl' else None,
               steps=report['steps'], parameters=params, profile_bytes=len(prof), train_seconds=report['train_seconds'],
               paintings=per_painting)
    return row, prof, codes, rel.reshape(len(names), SIDE, SIDE, 3), net


def font(size):
    for path in ('/System/Library/Fonts/Helvetica.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def capacity_figure(rows, renders, path):
    cols = [('target', None)] + [(r['name'], r) for r in rows if r['name'].startswith('mona_')]
    head, gap = 46, 6
    canvas = Image.new('RGB', (len(cols) * (SIDE + gap) - gap, SIDE + head), 'white')
    d = ImageDraw.Draw(canvas)
    for i, (name, r) in enumerate(cols):
        x = i * (SIDE + gap)
        img = painting('mona') if r is None else renders[name][0]
        canvas.paste(Image.fromarray((img * 255).round().astype(np.uint8)), (x, head))
        if r is None:
            d.text((x + 2, 4), 'the painting, 256 × 256', fill='black', font=font(14))
            continue
        act = {'relu': 'ReLU', 'sin': 'sine', 'pwl': f'learned curves, {r["knots"]} knots'}[r['act']]
        d.text((x + 2, 4), f'{act}, depth {r["depth"]}', fill='black', font=font(14))
        d.text((x + 2, 24), f'{r["paintings"][0]["psnr"]:.1f} dB · {r["parameters"]:,} parameters · '
               f'{r["profile_bytes"] / 1024:.0f} KB', fill=(90, 90, 90), font=font(11))
    canvas.save(path)


def curves_figure(net, path, layer=3):
    """The learned activation of every neuron in one layer, as the profile stores them."""
    curves = net.pwl[layer].curves()
    cw, ch, gap = 120, 90, 8
    cols = 7
    rows = (len(curves) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * (cw + gap) - gap, rows * (ch + gap) - gap), 'white')
    d = ImageDraw.Draw(canvas)
    xs = np.linspace(-5, 5, 400)
    for k, c in enumerate(curves):
        ys = c(xs)
        x0, y0 = (k % cols) * (cw + gap), (k // cols) * (ch + gap)
        d.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], outline=(220, 220, 220))
        lo, hi = ys.min(), ys.max()
        pts = [(x0 + i * (cw - 1) / (len(xs) - 1), y0 + ch - 4 - (y - lo) / (hi - lo + 1e-9) * (ch - 8)) for i, y in enumerate(ys)]
        d.line(pts, fill=(40, 60, 150), width=2)
    canvas.save(path)


def main():
    os.makedirs('gallery', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('docs/images/capacity', exist_ok=True)
    Image.fromarray((painting('mona') * 255).round().astype(np.uint8)).save('docs/images/capacity/target.png')
    rows, renders, nets = [], {}, {}
    for name, *_ in RUNS:
        if not os.path.exists(f'runs/{name}.npz'):
            print(f'{name}: not trained, skipped')
            continue
        row, prof, codes, rend, net = export(name)
        rows.append(row)
        renders[name], nets[name] = rend, net
        if name.startswith('mona_'):
            Image.fromarray((rend[0] * 255).round().astype(np.uint8)).save(f'docs/images/capacity/{name}.png')
        print(name, ' '.join(f'{p["painting"]} {p["psnr"]} dB (table {p["psnr_table_8bit"]})' for p in row['paintings']),
              f'{row["profile_bytes"] / 1024:.0f} KB')
        for short, run in GALLERY.items():
            if run == name:
                png.write(f'gallery/{short}.png', carrier(codes), prof)
                np.savez(f'gallery/{short}.npz', **dict(np.load(f'runs/{name}.npz')))
    json.dump(dict(littlecms=lcms.version(), runs=rows), open('results/capacity.json', 'w'), indent=1)
    gallery_readme(rows, 'gallery/README.md')
    capacity_figure(rows, renders, 'docs/images/capacity.png')
    if 'mona_pwl_d8' in nets:
        curves_figure(nets['mona_pwl_d8'], 'docs/images/curves.png')


if __name__ == '__main__':
    main()
