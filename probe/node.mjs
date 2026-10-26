// ImageMagick 7 (its official WebAssembly build) and sharp (libvips) on one carrier PNG.
// usage: node probe/node.mjs <dir with node_modules: sharp, @imagemagick/magick-wasm> <carrier.png> <out dir>
// Writes im_highres.png, im_quantum.png (ImageMagick's two transform modes, -profile to $SRGB_ICC)
// and sharp.png (sharp's default pipeline, which converts embedded profiles to sRGB).
import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join } from 'node:path';

const [root, carrier, out] = process.argv.slice(2);
const require = createRequire(join(root, 'noop.js'));
const sharp = require('sharp');
const im = require('@imagemagick/magick-wasm');
await im.initializeImageMagick(readFileSync(require.resolve('@imagemagick/magick-wasm/magick.wasm')));
const srgb = new im.ColorProfile(readFileSync(process.env.SRGB_ICC || '/System/Library/ColorSync/Profiles/sRGB Profile.icc'));
const png = readFileSync(carrier);
for (const [mode, label] of [[im.ColorTransformMode.HighRes, 'highres'], [im.ColorTransformMode.Quantum, 'quantum']]) {
  im.ImageMagick.read(png, (image) => {
    image.transformColorSpace(srgb, mode);
    image.write(im.MagickFormat.Png24, (data) => writeFileSync(join(out, `im_${label}.png`), data));
  });
}
await sharp(png).png().toFile(join(out, 'sharp.png'));
console.log(JSON.stringify({ imagemagick_wasm: im.Magick.imageMagickVersion, sharp: sharp.versions.sharp, libvips: sharp.versions.vips }));
