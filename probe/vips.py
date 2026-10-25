"""libvips on one carrier PNG: icc_transform to sRGB (its default, relative intent).

usage: <python with pyvips> probe/vips.py <carrier.png> <out.png>      (pip install "pyvips[binary]")
"""
import json
import sys

import pyvips

im = pyvips.Image.new_from_file(sys.argv[1])
im.icc_transform('srgb').write_to_file(sys.argv[2])
print(json.dumps({'libvips': '.'.join(str(pyvips.version(i)) for i in range(3))}))
