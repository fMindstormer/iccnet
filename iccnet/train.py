"""Fit a network to one or more paintings, then write its profile and a carrier PNG.

Painting i is drawn at blue = codes[i]: the carrier tile for it has R = column, G = row, B = that
code. Codes start at 1: the exported profile paints z = 0 black (see model.Net.stages).
"""
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from . import icc, lcms, png
from .model import Net, save

PAINTINGS = {                    # file in paintings/, vertical position of the square crop (0 = top)
    'mona': ('mona.jpg', 0.18),
    'pearl': ('pearl.jpg', 0.5),
    'wave': ('wave.jpg', 0.5),
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDE = 256


def painting(name, side=SIDE):
    path, pos = PAINTINGS[name]
    im = Image.open(os.path.join(ROOT, 'paintings', path)).convert('RGB')
    w, h = im.size
    s = min(w, h)
    left, top = (w - s) // 2, int((h - s) * pos)
    im = im.crop((left, top, left + s, top + s)).resize((side, side), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0


def tile(code, side=SIDE):
    """Carrier pixels for one latent code: (side, side, 3) uint8, R = column, G = row, B = code."""
    r, g = np.meshgrid(np.arange(side) * 255 // (side - 1), np.arange(side) * 255 // (side - 1), indexing='xy')
    return np.stack([r, g, np.full_like(r, code)], -1).astype(np.uint8)


def carrier(codes, side=SIDE):
    return np.concatenate([tile(c, side) for c in codes], 1)


def psnr(a, b):
    return float(10 * np.log10(1 / np.mean((np.clip(a, 0, 1) - b) ** 2)))


def fit(names, codes, act='pwl', width=14, depth=8, knots=64, steps=8000, batch=8192, lr=None,
        seed=0, threads=None, log=print):
    assert min(codes) >= 1, 'z = 0 is reserved for the gate'
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    X = np.concatenate([tile(c).reshape(-1, 3) / 255.0 for c in codes])
    Y = np.concatenate([painting(n).reshape(-1, 3) for n in names])
    Xt, Yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    net = Net(act, width, depth, knots)
    lr = lr or {'relu': 3e-3, 'sin': 1e-3, 'pwl': 3e-3}[act]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, len(Xt), (batch,))
        loss = torch.mean((net(Xt[idx]) - Yt[idx]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and (step + 1) % 1000 == 0:
            log(f'  step {step + 1}: loss {loss.item():.5f} ({time.time() - t0:.0f} s)')
    return net, X, Y, time.time() - t0


def evaluate(net, X, Y, prof, names, codes):
    """PSNR of PyTorch and of LittleCMS's renders, per painting, and how far LittleCMS's
    perceptual and black-point-compensated renders are from its plain relative one."""
    with torch.no_grad():
        P = net(torch.tensor(X, dtype=torch.float32)).double().numpy()
    corner = lcms.render(prof, np.zeros((1, 3), np.uint8), 'double')[0]
    px = np.round(X * 255).astype(np.uint8)
    rel = np.clip(lcms.render(prof, px, 'double', lcms.RELATIVE), 0, 1)
    per = np.clip(lcms.render(prof, px, 'double', lcms.PERCEPTUAL, lcms.BPC), 0, 1)
    tab = np.clip(lcms.render(prof, px, '8', lcms.PERCEPTUAL), 0, 1)
    out = []
    n = SIDE * SIDE
    for i, (name, code) in enumerate(zip(names, codes)):
        s = slice(i * n, (i + 1) * n)
        out.append(dict(painting=name, code=int(code), psnr_pytorch=psnr(P[s], Y[s]),
                        psnr_littlecms=psnr(rel[s], Y[s]), psnr_littlecms_8bit_table=psnr(tab[s], Y[s]),
                        max_diff_relative=float(np.abs(rel[s] - np.clip(P[s], 0, 1)).max() * 255),
                        max_diff_perceptual_bpc=float(np.abs(per[s] - rel[s]).max() * 255)))
    return out, corner, rel


def run(names, codes, name, out='runs', **kw):
    os.makedirs(out, exist_ok=True)
    net, X, Y, secs = fit(names, codes, **kw)
    stages = net.stages(X)
    prof = icc.profile(stages, desc=f'iccnet {name}: {", ".join(names)}')
    with open(os.path.join(out, f'{name}.icc'), 'wb') as f:
        f.write(prof)
    png.write(os.path.join(out, f'{name}.png'), carrier(codes), prof)
    save(net, os.path.join(out, f'{name}.npz'), paintings=names, codes=list(codes))
    metrics, corner, rel = evaluate(net, X, Y, prof, names, codes)
    Image.fromarray((np.concatenate([rel[i * SIDE * SIDE:(i + 1) * SIDE * SIDE].reshape(SIDE, SIDE, 3)
                                     for i in range(len(names))], 1) * 255).round().astype(np.uint8)
                    ).save(os.path.join(out, f'{name}_render.png'))
    params = sum(p.numel() for p in net.parameters())
    report = dict(name=name, **net.config, steps=kw.get('steps', 8000), batch=kw.get('batch', 8192),
                  parameters=params, profile_bytes=len(prof), elements=len(stages) + 2,
                  train_seconds=round(secs), corner=[round(float(v), 4) for v in corner],
                  littlecms=lcms.version(), paintings=metrics)
    with open(os.path.join(out, f'{name}.json'), 'w') as f:
        json.dump(report, f, indent=1)
    return report
