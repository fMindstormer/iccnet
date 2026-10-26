"""Build the interactive page into pages/dist/: https://fmindstormer.github.io/iccnet/

    python pages/build.py              a complete document, for GitHub Pages (.github/workflows/pages.yml)
    python pages/build.py --artifact   the page without <!doctype>, <head> and <body>, for a Claude
                                       artifact, whose host adds them
Needs gallery/ and results/capacity.json (publish.py), results/engines.json plus
results/engines/<engine>.png (probe/engines.py on gallery/trio.png, see README), and sRGB.icc with
results/expected/ (release.py). The page inlines pages/iccnet.js and loads the gallery PNGs, the
engine outputs and the capacity renders as files next to it, where sRGB.icc and the expected
painting are also offered for download.
"""
import argparse
import json
import os
import re
import shutil

DOCUMENT = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<meta name="description" content="PNG gradients whose ICC color profiles are neural networks: '
            'a color engine converting the file paints the Mona Lisa.">\n'
            '<style>:root { color-scheme: light; padding-top: env(safe-area-inset-top, 0px); '
            'padding-bottom: env(safe-area-inset-bottom, 0px) } body { margin: 0 } img { max-width: 100% } '
            '[hidden] { display: none !important }</style>\n'
            '{head}</head>\n<body>\n{body}</body>\n</html>\n')

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
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', action='store_true', help='leave out the document skeleton')
    a = ap.parse_args()
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
    labels = {'colorsync': 'macOS ColorSync (Safari, Preview, Quick Look)', 'imagemagick': 'ImageMagick, -profile sRGB.icc',
              'sharp': 'sharp, default pipeline', 'pillow': 'Pillow ImageCms, default flags',
              'lcms_double': 'LittleCMS in double precision'}
    vtext = {'colorsync': 'macOS 27.0.1',
             'imagemagick': ' '.join(versions.get('imagemagick', '').split()[:3]) + ', native (Homebrew)',
             'sharp': f'sharp {versions.get("sharp", "")}, libvips {versions.get("libvips", "")}',
             'pillow': 'Pillow ' + versions.get('pillow', ''), 'lcms_double': f'LittleCMS {versions["littlecms"] / 1000:.2f}'}
    engines = [dict(id=k, label=labels[k], version=vtext[k], **{f: eng['engines'][k][f] for f in ('kind', 'mean', 'max')})
               for k in ENGINES if k in eng['engines']]
    data = dict(gallery=gallery, paintings=PAINTINGS, engines=engines, capacity=capacity,
                note=f'Measured on an Apple M4 with macOS 27.0.1: the engines with LittleCMS {versions["littlecms"] / 1000:.2f}, '
                     f'the networks with LittleCMS {cap["littlecms"] / 1000:.2f}. The paintings above are painted by this page\'s '
                     f'own evaluator of the profile, which matches LittleCMS in double precision to within one code value and its '
                     f'33-point table to within three.')

    js = open(os.path.join(HERE, 'iccnet.js')).read()
    js = re.sub(r'^export (async function|function|const)', r'\1', js, flags=re.M)
    html = open(os.path.join(HERE, 'index.template.html')).read()
    html = html.replace('/*__ICCNET_JS__*/', js).replace('/*__DATA__*/null', json.dumps(data, ensure_ascii=False))
    if not a.artifact:
        head, body = html.split('\n<div class="wrap">', 1)
        head = head.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', '')
        html = DOCUMENT.replace('{head}', head + '\n').replace('{body}', '<div class="wrap">' + body)
    shutil.rmtree(DIST, ignore_errors=True)
    for sub in ('gallery', 'engines', 'capacity', 'expected'):
        os.makedirs(os.path.join(DIST, sub))
    open(os.path.join(DIST, 'index.html'), 'w').write(html)
    shutil.copy(os.path.join(ROOT, 'sRGB.icc'), DIST)
    shutil.copy(os.path.join(ROOT, 'results', 'expected', 'trio.png'), os.path.join(DIST, 'expected'))
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
