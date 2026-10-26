# Reproducing iccnet

Each PNG in `gallery/` is a gradient whose ICC profile holds a trained network. Converting the
file to sRGB with a color engine that evaluates the profile's DToB tags runs the network and
paints the pictures. Reproducing this needs no training, no Python and no GPU: one ImageMagick
command, and a comparison with the expected output.

The same steps work in a clone of the repository and in the release bundle,
[iccnet-v1.0.0.zip](https://github.com/fMindstormer/iccnet/releases/tag/v1.0.0), which holds
the carriers, `sRGB.icc`, the expected outputs, every engine's result and `SHA256SUMS`.

## 1. Check the files (release bundle)

```sh
shasum -a 256 -c SHA256SUMS      # macOS
sha256sum -c SHA256SUMS          # Linux
```

## 2. Convert

```sh
magick gallery/trio.png -profile sRGB.icc painted.png      # ImageMagick 7
convert gallery/trio.png -profile sRGB.icc painted.png     # ImageMagick 6
```

ImageMagick reads the profile embedded in `trio.png` as the source and converts to `sRGB.icc`.
That file is a plain sRGB display profile (540 bytes, ICC v4, three colorants and three tone
curves, written by `iccnet.icc.srgb_profile()`), so the command needs nothing from the operating
system. Any sRGB profile works as the destination: with macOS's
`"/System/Library/ColorSync/Profiles/sRGB Profile.icc"` or LittleCMS's built-in sRGB, the
paintings move by at most 0.05 code values.

## 3. Compare

```sh
magick compare -metric AE -fuzz 1% painted.png results/expected/trio.png null:    # ImageMagick 7
compare -metric AE -fuzz 1% painted.png results/expected/trio.png null:           # ImageMagick 6
```

This prints `0` when the painting matches the expected one to within 1% (`-fuzz 1%`); ImageMagick
7 prints `0 (0)`. A painting through the 33-point table, or the gradient itself, prints a larger
number. The other carriers work the same way, against `results/expected/<name>.png`.

## What other software shows

| Software | Result | Compare with |
|---|---|---|
| ImageMagick 6 and 7, `-profile` | the paintings, exact | `results/expected/<name>.png` |
| Pillow with `flags=ImageCms.Flags.NOOPTIMIZE`; LittleCMS on float pixels; `tificc -c0` | the paintings, exact | `results/expected/<name>.png` |
| Pillow with default flags; `tificc`; sharp; libvips | the paintings through LittleCMS's 33-point table: blocky | `results/expected/<name>-table.png` |
| Browsers, macOS Preview and Quick Look, GitHub | the gradient: they don't evaluate DToB tags | the carrier itself |

`results/expected/<name>.png` is LittleCMS's double-precision render, rounded to 8 bits, and
`<name>-table.png` its 8-bit default path. Both go to `sRGB.icc` in the perceptual intent,
ImageMagick's default (the relative intent gives the same picture). `release.py` writes them.

With Python and Pillow, exact:

```python
import io
from PIL import Image, ImageCms

im = Image.open('gallery/trio.png')
src = ImageCms.ImageCmsProfile(io.BytesIO(im.info['icc_profile']))
dst = ImageCms.getOpenProfile('sRGB.icc')
ImageCms.profileToProfile(im, src, dst, flags=ImageCms.Flags.NOOPTIMIZE).save('painted.png')
```

Without `flags`, Pillow lets LittleCMS sample the transform into a table, and the result is
`results/expected/trio-table.png`.

## Tested

| ImageMagick | Where | LittleCMS | `trio.png` against the expected painting |
|---|---|---|---|
| 7.1.2-34 Q16-HDRI | Homebrew, macOS 27.0.1, Apple M4 | 2.19.1 | max 1 code value (3 rounding ties of 589,824 values) |
| 6.9.12-98 Q16 | Ubuntu 24.04 (apt), GitHub Actions | 2.14 | max 1 code value, `tests/test_reproduce.py` on every push |

`results/engines.json` and `results/engines/` hold every engine's output for `trio.png` on the
Mac (`probe/engines.py`), and `docs/images/engines.png` shows three of them.
