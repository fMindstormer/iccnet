// Read a PNG and its embedded ICC profile, and evaluate the profile's DToB tag as LittleCMS does
// in floating point. No dependencies: the page inlines this file and tests/eval.mjs imports it.

const ASCII = (b, o, n = 4) => String.fromCharCode(...b.subarray(o, o + n));
const view = (b) => new DataView(b.buffer, b.byteOffset, b.byteLength);
const f32 = Math.fround;

async function inflate(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// PNG: 8- or 16-bit gray, RGB or RGBA, not interlaced. Returns 8-bit RGB pixels and the profile.
export async function readPNG(bytes) {
  const v = view(bytes);
  if (v.getUint32(0) !== 0x89504e47) throw new Error('not a PNG file');
  let pos = 8, w = 0, h = 0, depth = 0, ctype = 0, icc = null;
  const idat = [];
  while (pos < bytes.length) {
    const n = v.getUint32(pos), kind = ASCII(bytes, pos + 4), body = bytes.subarray(pos + 8, pos + 8 + n);
    if (kind === 'IHDR') {
      w = view(body).getUint32(0); h = view(body).getUint32(4); depth = body[8]; ctype = body[9];
      if (body[12]) throw new Error('interlaced PNGs are not supported');
    } else if (kind === 'iCCP') {
      icc = await inflate(body.subarray(body.indexOf(0) + 2));
    } else if (kind === 'IDAT') idat.push(body);
    pos += 12 + n;
  }
  const chans = { 0: 1, 2: 3, 4: 2, 6: 4 }[ctype];
  if (!chans || (depth !== 8 && depth !== 16)) throw new Error('only 8- and 16-bit gray, RGB and RGBA PNGs');
  const bpp = chans * depth / 8, stride = w * bpp;
  const raw = await inflate(new Uint8Array(await new Blob(idat).arrayBuffer()));
  const cur = new Uint8Array(stride), prev = new Uint8Array(stride);
  const rgb = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)], line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let i = 0; i < stride; i++) {
      const a = i >= bpp ? cur[i - bpp] : 0, b = prev[i], c = i >= bpp ? prev[i - bpp] : 0;
      let p = 0;
      if (f === 1) p = a; else if (f === 2) p = b; else if (f === 3) p = (a + b) >> 1;
      else if (f === 4) { const pa = Math.abs(b - c), pb = Math.abs(a - c), pc = Math.abs(a + b - 2 * c); p = pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      cur[i] = (line[i] + p) & 255;
    }
    for (let x = 0; x < w; x++) for (let k = 0; k < 3; k++) {
      const ch = chans < 3 ? 0 : k;                        // gray: R = G = B
      rgb[(y * w + x) * 3 + k] = cur[x * bpp + ch * depth / 8];  // 16-bit: high byte
    }
    prev.set(cur);
  }
  return { width: w, height: h, depth, rgb, icc };
}

// ICC header and tag table
export function readProfile(icc) {
  const v = view(icc), n = v.getUint32(128), tags = [];
  for (let i = 0; i < n; i++) {
    const o = 132 + 12 * i, offset = v.getUint32(o + 4), size = v.getUint32(o + 8);
    tags.push({ sig: ASCII(icc, o), offset, size, type: ASCII(icc, offset) });
  }
  const ver = v.getUint32(8);
  return { size: v.getUint32(0), version: `${ver >>> 24}.${(ver >>> 20) & 15}`, cls: ASCII(icc, 12),
    space: ASCII(icc, 16), pcs: ASCII(icc, 20), tags };
}

function readCurve(icc, o) {
  const v = view(icc);
  if (ASCII(icc, o) !== 'curf') throw new Error('expected a segmented curve');
  const n = v.getUint16(o + 8), breaks = [];
  for (let i = 0; i < n - 1; i++) breaks.push(v.getFloat32(o + 12 + 4 * i));
  let p = o + 12 + 4 * (n - 1);
  const segs = [];
  for (let i = 0; i < n; i++) {
    const kind = ASCII(icc, p);
    if (kind === 'parf') {
      const type = v.getUint16(p + 8), np = [4, 5, 5][type], q = [];
      for (let j = 0; j < np; j++) q.push(v.getFloat32(p + 12 + 4 * j));
      segs.push({ type, q });
      p += 12 + 4 * np;
    } else if (kind === 'samf') {
      const count = v.getUint32(p + 8), s = new Float32Array(count);
      for (let j = 0; j < count; j++) s[j] = v.getFloat32(p + 12 + 4 * j);
      segs.push({ samples: s });
      p += 12 + 4 * count;
    } else throw new Error(`unknown curve segment ${kind}`);
  }
  const x0 = [-Infinity, ...breaks], x1 = [...breaks, Infinity];
  segs.forEach((s, i) => { s.x0 = x0[i]; s.x1 = x1[i]; });
  // a sampled segment's first point is the previous segment's value at the breakpoint
  segs.forEach((s, i) => {
    if (s.samples) { const full = new Float32Array(s.samples.length + 1); full[0] = evalSeg(segs[i - 1], s.x0); full.set(s.samples, 1); s.samples = full; }
  });
  return segs;
}

// ICC.1:2022 Table 60: type 0 (g, a, b, c) y = (a x + b)^g + c; type 1 (g, a, b, c, d)
// y = a log10(b x^g + c) + d; type 2 (a, b, c, d, e) y = a b^(c x + d) + e. Where the spec leaves
// a negative base or log argument open, this does what LittleCMS does.
function formula(s, x) {
  const q = s.q;
  if (s.type === 0) {
    const t = q[1] * x + q[2];
    return q[0] === 1 ? t + q[3] : t < 0 ? q[3] : Math.pow(t, q[0]) + q[3];
  }
  if (s.type === 1) { const t = q[2] * Math.pow(x, q[0]) + q[3]; return t <= 0 ? q[4] : q[1] * Math.log10(t) + q[4]; }
  return q[0] * Math.pow(q[1], q[2] * x + q[3]) + q[4];
}

function evalSeg(s, x) {
  if (!s.samples) return formula(s, x);
  const n = s.samples.length - 1;
  let t = f32(f32(x - s.x0) / (s.x1 - s.x0));
  t = t < 0 || t !== t ? 0 : t > 1 ? 1 : t;
  if (t === 1) return s.samples[n];
  const u = t * n, i = Math.floor(u), r = u - i;
  return s.samples[i] + (s.samples[Math.min(i + 1, n)] - s.samples[i]) * r;
}

export function evalCurve(segs, x) {
  for (let i = segs.length - 1; i >= 0; i--) if (x > segs[i].x0 && x <= segs[i].x1) return evalSeg(segs[i], x);
  return -Infinity;
}

// The elements of a multiProcessElements tag, as stages
export function readStages(icc, sig = 'D2B1') {
  const tag = readProfile(icc).tags.find((t) => t.sig === sig);
  if (!tag) throw new Error(`no ${sig} tag`);
  if (tag.type !== 'mpet') throw new Error(`${sig} is ${tag.type}, not mpet`);
  const v = view(icc), base = tag.offset, n = v.getUint32(base + 12), stages = [];
  for (let i = 0; i < n; i++) {
    const o = base + v.getUint32(base + 16 + 8 * i), kind = ASCII(icc, o), p = v.getUint16(o + 8), q = v.getUint16(o + 10);
    if (kind === 'matf') {
      const W = new Float64Array(p * q), b = new Float64Array(q);
      for (let j = 0; j < p * q; j++) W[j] = v.getFloat32(o + 12 + 4 * j);
      for (let j = 0; j < q; j++) b[j] = v.getFloat32(o + 12 + 4 * (p * q + j));
      stages.push({ kind, p, q, W, b });
    } else if (kind === 'cvst') {
      const curves = [];
      for (let j = 0; j < p; j++) curves.push(readCurve(icc, o + v.getUint32(o + 12 + 8 * j)));
      stages.push({ kind, p, q, curves });
    } else throw new Error(`element ${kind} is not supported here`);
  }
  return stages;
}

// sRGB colorants adapted to D50 (Bradford), inverted: the PCS back to linear sRGB
const XYZ_TO_SRGB = (() => {
  const m = [0.4360747, 0.3850649, 0.1430804, 0.2225045, 0.7168786, 0.0606169, 0.0139322, 0.0971045, 0.7141733];
  const [a, b, c, d, e, f, g, h, i] = m, det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  return [e * i - f * h, c * h - b * i, b * f - c * e, f * g - d * i, a * i - c * g, c * d - a * f,
    d * h - e * g, b * g - a * h, a * e - b * d].map((x) => x / det);
})();
const encode = (l) => (l <= 0.0031308 ? 12.92 * l : 1.055 * Math.pow(l, 1 / 2.4) - 0.055);

// Compile stages into a function (r, g, b in 0..1) -> [R, G, B] sRGB-encoded, unclipped
export function compile(stages) {
  const width = Math.max(4, ...stages.map((s) => Math.max(s.p, s.q)));
  const A = new Float32Array(width), B = new Float32Array(width);
  return (r, g, b) => {
    let x = A, y = B;
    x[0] = r; x[1] = g; x[2] = b;
    for (const s of stages) {
      if (s.kind === 'matf') {
        for (let i = 0; i < s.q; i++) {
          let t = 0;
          for (let j = 0; j < s.p; j++) t += x[j] * s.W[i * s.p + j];
          y[i] = t + s.b[i];
        }
      } else {
        for (let i = 0; i < s.p; i++) y[i] = evalCurve(s.curves[i], x[i]);
      }
      const t = x; x = y; y = t;
    }
    const [X, Y, Z] = [x[0], x[1], x[2]], m = XYZ_TO_SRGB;
    return [encode(m[0] * X + m[1] * Y + m[2] * Z), encode(m[3] * X + m[4] * Y + m[5] * Z),
      encode(m[6] * X + m[7] * Y + m[8] * Z)];
  };
}

const clip = (v) => (v < 0 || v !== v ? 0 : v > 1 ? 1 : v);

// Paint 8-bit RGB pixels into RGBA bytes. table: 0 for exact evaluation, else the number of grid
// points per axis of the table LittleCMS builds for 8- and 16-bit formats (33 by default).
export function paint(fn, rgb, out, table = 0) {
  const n = rgb.length / 3;
  if (!table) {
    for (let k = 0; k < n; k++) {
      const c = fn(rgb[3 * k] / 255, rgb[3 * k + 1] / 255, rgb[3 * k + 2] / 255);
      out[4 * k] = Math.round(clip(c[0]) * 255); out[4 * k + 1] = Math.round(clip(c[1]) * 255);
      out[4 * k + 2] = Math.round(clip(c[2]) * 255); out[4 * k + 3] = 255;
    }
    return out;
  }
  const g = sample(fn, table);
  for (let k = 0; k < n; k++) {
    const c = tetra(g, table, rgb[3 * k] / 255, rgb[3 * k + 1] / 255, rgb[3 * k + 2] / 255);
    out[4 * k] = Math.round(c[0] * 255); out[4 * k + 1] = Math.round(c[1] * 255);
    out[4 * k + 2] = Math.round(c[2] * 255); out[4 * k + 3] = 255;
  }
  return out;
}

const samples = new Map();
function sample(fn, n) {
  if (samples.has(fn) && samples.get(fn).n === n) return samples.get(fn).g;
  const g = new Float32Array(n * n * n * 3);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) {
    const c = fn(i / (n - 1), j / (n - 1), k / (n - 1)), o = ((i * n + j) * n + k) * 3;
    g[o] = clip(c[0]); g[o + 1] = clip(c[1]); g[o + 2] = clip(c[2]);
  }
  g.fill(1, ((n * n * n) - 1) * 3);              // white-on-white fix-up: input white -> white
  samples.set(fn, { n, g });
  return g;
}

function tetra(g, n, r, gg, b) {
  const px = r * (n - 1), py = gg * (n - 1), pz = b * (n - 1);
  const x0 = Math.min(Math.floor(px), n - 2), y0 = Math.min(Math.floor(py), n - 2), z0 = Math.min(Math.floor(pz), n - 2);
  const rx = px - x0, ry = py - y0, rz = pz - z0;
  const at = (i, j, k, c) => g[(((x0 + i) * n + (y0 + j)) * n + (z0 + k)) * 3 + c];
  const out = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const c0 = at(0, 0, 0, c), c111 = at(1, 1, 1, c);
    let v;
    if (rx >= ry && ry >= rz) v = c0 + (at(1, 0, 0, c) - c0) * rx + (at(1, 1, 0, c) - at(1, 0, 0, c)) * ry + (c111 - at(1, 1, 0, c)) * rz;
    else if (rx >= rz && rz >= ry) v = c0 + (at(1, 0, 0, c) - c0) * rx + (at(1, 0, 1, c) - at(1, 0, 0, c)) * rz + (c111 - at(1, 0, 1, c)) * ry;
    else if (rz >= rx && rx >= ry) v = c0 + (at(0, 0, 1, c) - c0) * rz + (at(1, 0, 1, c) - at(0, 0, 1, c)) * rx + (c111 - at(1, 0, 1, c)) * ry;
    else if (ry >= rx && rx >= rz) v = c0 + (at(0, 1, 0, c) - c0) * ry + (at(1, 1, 0, c) - at(0, 1, 0, c)) * rx + (c111 - at(1, 1, 0, c)) * rz;
    else if (ry >= rz && rz >= rx) v = c0 + (at(0, 1, 0, c) - c0) * ry + (at(0, 1, 1, c) - at(0, 1, 0, c)) * rz + (c111 - at(0, 1, 1, c)) * rx;
    else v = c0 + (at(0, 0, 1, c) - c0) * rz + (at(0, 1, 1, c) - at(0, 0, 1, c)) * ry + (c111 - at(0, 1, 1, c)) * rx;
    out[c] = clip(v);
  }
  return out;
}
