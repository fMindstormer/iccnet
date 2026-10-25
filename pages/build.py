"""Build the interactive page into pages/dist/ (published as a Claude artifact).

    python pages/build.py
Needs gallery/ and results/capacity.json (publish.py) and results/engines.json plus
results/engines/<engine>.png (probe/engines.py on gallery/trio.png, see README).
The page inlines pages/iccnet.js and loads the gallery PNGs, the engine outputs and the capacity
renders as files published next to it.
"""
import json
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DIST = os.path.join(HERE, 'dist')

GALLERY = [('mona', 'Mona Lisa'), ('pearl', 'Girl with a Pearl Earring'), ('wave', 'The Great Wave'),
           ('duet', 'Duet: two paintings'), ('trio', 'Trio: three paintings'), ('lowpoly', 'Mona Lisa, ReLU')]
RUN_OF = {'mona': 'mona_pwl_d8', 'pearl': 'pearl_pwl_d8', 'wave': 'wave_pwl_d8', 'duet': 'duet_pwl_d8',
          'trio': 'trio_pwl_d8', 'lowpoly': 'mona_relu_d8'}
PAINTINGS = {
    'mona': dict(title='Mona Lisa', artist='Leonardo da Vinci', date='c. 1503–1519'),
    'pearl': dict(title='Girl with a Pearl Earring', artist='Johannes Vermeer', date='c. 1665'),
    'wave': dict(title='The Great Wave off Kanagawa', artist='Katsushika Hokusai', date='c. 1831'),
}
ENGINES = ['colorsync', 'imagemagick', 'sharp', 'pillow', 'lcms_double']


def what(r):
    act = {'relu': 'ReLU', 'sin': 'Sine', 'pwl': f'Learned curves, {r["knots"]} knots'}[r['act']]
    return f'{act}, depth {r["depth"]}'


def main():
    cap = json.load(open(os.path.join(ROOT, 'results', 'capacity.json')))
    runs = {r['name']: r for r in cap['runs']}
    eng = json.load(open(os.path.join(ROOT, 'results', 'engines.json')))
    gallery = []
    for gid, name in GALLERY:
        r = runs[RUN_OF[gid]]
        gallery.append(dict(id=gid, name=name, codes=[p['code'] for p in r['paintings']], act=r['act'], depth=r['depth'],
                            parameters=r['parameters'], profile_bytes=r['profile_bytes'], paintings=r['paintings']))
    capacity = [dict(name=r['name'], what=what(r), parameters=r['parameters'], profile_bytes=r['profile_bytes'],
                     psnr=r['paintings'][0]['psnr'], psnr_table=r['paintings'][0]['psnr_table_16bit'])
                for r in cap['runs'] if r['name'].startswith('mona_')]
    versions = eng['versions']
    labels = {'colorsync': 'macOS ColorSync (Safari, Preview, Quick Look)', 'imagemagick': 'ImageMagick 7, -profile sRGB.icc',
              'sharp': 'sharp, default pipeline', 'pillow': 'Pillow ImageCms, default flags',
              'lcms_double': 'LittleCMS in double precision'}
    vtext = {'colorsync': 'macOS 27.0.1',
             'imagemagick': ' '.join(versions.get('imagemagick', '').split()[:2]) + ', WebAssembly build',
             'sharp': f'sharp {versions.get("sharp", "")}, libvips {versions.get("libvips", "")}',
             'pillow': versions.get('pillow', ''), 'lcms_double': f'LittleCMS {versions["littlecms"] / 1000:.2f}'}
    engines = [dict(id=k, label=labels[k], version=vtext[k], **{f: eng['engines'][k][f] for f in ('kind', 'mean', 'max')})
               for k in ENGINES if k in eng['engines']]
    data = dict(gallery=gallery, paintings=PAINTINGS, engines=engines, capacity=capacity,
                note=f'Measured on an Apple M4 with macOS 27.0.1 and LittleCMS {cap["littlecms"] / 1000:.2f}. The paintings above are '
                     f'painted by this page\'s own evaluator of the profile, which matches LittleCMS in double precision to within '
                     f'one code value and its 33-point table to within three.')

    js = open(os.path.join(HERE, 'iccnet.js')).read()
    js = re.sub(r'^export (async function|function|const)', r'\1', js, flags=re.M)
    html = open(os.path.join(HERE, 'index.template.html')).read()
    html = html.replace('/*__ICCNET_JS__*/', js).replace('/*__DATA__*/null', json.dumps(data, ensure_ascii=False))
    shutil.rmtree(DIST, ignore_errors=True)
    for sub in ('gallery', 'engines', 'capacity'):
        os.makedirs(os.path.join(DIST, sub))
    open(os.path.join(DIST, 'index.html'), 'w').write(html)
    for gid, _ in GALLERY:
        shutil.copy(os.path.join(ROOT, 'gallery', f'{gid}.png'), os.path.join(DIST, 'gallery'))
    for k in ENGINES:
        shutil.copy(os.path.join(ROOT, 'results', 'engines', f'{k}.png'), os.path.join(DIST, 'engines'))
    for name in ['target'] + [c['name'] for c in capacity]:
        shutil.copy(os.path.join(ROOT, 'docs', 'images', 'capacity', f'{name}.png'), os.path.join(DIST, 'capacity'))
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(DIST) for f in fs)
    print(f'pages/dist: {total / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
