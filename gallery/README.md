# Gallery

Carriers: R = column, G = row, B = the code of a painting, one 256-pixel tile per code, with the
profile embedded. `<name>.npz` holds the weights (`iccnet.model.load`). Written by `publish.py`;
PSNR against the 256 × 256 painting, for LittleCMS's exact render and for its 8-bit table.

| File | Network | Profile | Code: painting, PSNR exact / table |
|---|---|---:|---|
| `mona.png` | learned curves, 8 layers of 14, 8,739 parameters | 64 KB | 128: Mona Lisa, 28.42 / 25.79 dB |
| `pearl.png` | learned curves, 8 layers of 14, 8,739 parameters | 64 KB | 128: Girl with a Pearl Earring, 28.24 / 21.33 dB |
| `wave.png` | learned curves, 8 layers of 14, 8,739 parameters | 64 KB | 128: The Great Wave off Kanagawa, 18.71 / 15.21 dB |
| `duet.png` | learned curves, 8 layers of 14, 8,739 parameters | 64 KB | 16: Mona Lisa, 27.05 / 25.58 dB; 239: Girl with a Pearl Earring, 26.65 / 21.41 dB |
| `trio.png` | learned curves, 8 layers of 14, 8,739 parameters | 64 KB | 32: Mona Lisa, 25.96 / 24.97 dB; 128: Girl with a Pearl Earring, 24.66 / 21.22 dB; 224: The Great Wave off Kanagawa, 16.97 / 15.45 dB |
| `lowpoly.png` | ReLU, 8 layers of 14, 1,571 parameters | 25 KB | 128: Mona Lisa, 23.98 / 23.68 dB |
