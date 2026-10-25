// The page's evaluator against LittleCMS: node tests/eval.mjs <carrier.png> <double.png> <table16.png>
// <double.png> is LittleCMS's render in double precision, <table16.png> its render on 16-bit formats
// with default flags (a 33-point table). Prints max and mean differences in 8-bit code values as
// JSON, and exits with 1 if the exact path is off by more than 1 anywhere.
import { readFileSync } from 'node:fs';
import { compile, paint, readPNG, readStages } from '../pages/iccnet.js';

const [carrierPath, doublePath, tablePath] = process.argv.slice(2);
const carrier = await readPNG(new Uint8Array(readFileSync(carrierPath)));
const fn = compile(readStages(carrier.icc, 'D2B1'));
const n = carrier.width * carrier.height;
const result = {};
for (const [label, path, table] of [['exact', doublePath, 0], ['table', tablePath, 33]]) {
  const want = (await readPNG(new Uint8Array(readFileSync(path)))).rgb;
  const got = paint(fn, carrier.rgb, new Uint8ClampedArray(n * 4), table);
  let max = 0, sum = 0;
  for (let k = 0; k < n; k++) for (let c = 0; c < 3; c++) {
    const e = Math.abs(got[4 * k + c] - want[3 * k + c]);
    max = Math.max(max, e); sum += e;
  }
  result[label] = { max, mean: +(sum / (3 * n)).toFixed(4) };
}
console.log(JSON.stringify(result));
process.exit(result.exact.max > 1 ? 1 : 0);
