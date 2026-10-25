"""Train every network the README reports on, three at a time. Runs that are done are skipped.

    python train_all.py            # runs/<name>.{icc,png,npz,json,_render.png}
Then `python publish.py` copies the gallery and collects results/capacity.json.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

RUNS = [
    # capacity sweep on one painting: activation, depth, knots (width 14: with the gate, LittleCMS's limit of 15)
    ('mona_relu_d8', 'mona', None, dict(act='relu', depth=8)),
    ('mona_sin_d8', 'mona', None, dict(act='sin', depth=8)),
    ('mona_pwl_d4', 'mona', None, dict(act='pwl', depth=4)),
    ('mona_pwl_d8', 'mona', None, dict(act='pwl', depth=8)),
    ('mona_pwl_d12', 'mona', None, dict(act='pwl', depth=12)),
    ('mona_pwl_d8_k16', 'mona', None, dict(act='pwl', depth=8, knots=16)),
    ('mona_pwl_d8_k256', 'mona', None, dict(act='pwl', depth=8, knots=256)),
    # gallery
    ('pearl_pwl_d8', 'pearl', None, dict(act='pwl', depth=8)),
    ('wave_pwl_d8', 'wave', None, dict(act='pwl', depth=8)),
    ('duet_pwl_d8', 'mona,pearl', '16,239', dict(act='pwl', depth=8, steps=12000)),
    ('trio_pwl_d8', 'mona,pearl,wave', '32,128,224', dict(act='pwl', depth=8, steps=12000)),
]


def one(spec):
    name, paintings, codes, kw = spec
    if os.path.exists(f'runs/{name}.json'):
        return name, 'done before'
    from iccnet.train import run
    names = paintings.split(',')
    codes = [int(c) for c in codes.split(',')] if codes else [128]
    r = run(names, codes, name, 'runs', threads=2, log=None, **{'steps': 8000, **kw})
    return name, ' '.join(f'{p["painting"]} {p["psnr_littlecms"]:.2f} dB' for p in r['paintings'])


if __name__ == '__main__':
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    with ProcessPoolExecutor(workers) as ex:
        for name, msg in ex.map(one, RUNS):
            print(f'{name}: {msg}', flush=True)
    print(json.dumps(sorted(os.listdir('runs'))))
