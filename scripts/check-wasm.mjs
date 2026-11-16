// Verifies that the WebAssembly build reproduces the native library bit for bit.
//
//   ./build-wasm.sh nodejs && node scripts/check-wasm.mjs
//
// It runs every preset on every built-in chart through pkg/nodejs, prints the digest table and
// compares it with `cargo run --release --example digest` (skipped with --no-native, e.g. when
// cargo is not installed; the combined digest is then compared with EXPECTED below).
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
const sd = require(path.join(root, 'pkg/nodejs/smart_downscaler.js'));

// Combined digest of release 1.0.0 (all presets × all charts).
const EXPECTED = '9e38d13c08509990';

const presets = [['fast', 'fast'], ['balanced', 'balanced'], ['high', 'high'], ['maximum', 'maximum'], ['exact', 'exact_colors'], ['legacy', 'legacy']];
const charts = [['portrait', 256, 256, 64, 64], ['illustration', 320, 180, 96, 54], ['gradient', 200, 120, 37, 23], ['texture', 192, 192, 48, 48], ['pixel_art', 192, 192, 40, 40]];
const MASK = (1n << 64n) - 1n;
const PRIME = 0x100000001b3n;
const pad = (s, n) => (s + ' '.repeat(n)).slice(0, n);

let all = 0xcbf29ce484222325n;
const lines = [];
const add = (label, digest) => {
  lines.push(`${label} ${digest}`);
  all = ((all ^ BigInt('0x' + digest)) * PRIME) & MASK;
};
for (const [name, preset] of presets) {
  for (const [chart, w, h, tw, th] of charts) {
    const r = sd.downscale(sd.test_chart(chart, w, h), w, h, tw, th, sd.WasmDownscaleConfig[preset]());
    add(`${pad(name, 9)} ${pad(chart, 13)}`, r.digest);
    r.free();
  }
  for (const soft of [false, true]) {
    const r = sd.downscale(sd.test_chart(soft ? 'sprite_soft' : 'sprite', 160, 160), 160, 160, 40, 40, sd.WasmDownscaleConfig[preset]());
    add(`${pad(name, 9)} sprite(soft=${pad(String(soft), 5)})`, r.digest);
    r.free();
  }
}
const combined = all.toString(16).padStart(16, '0');
lines.push(`combined ${combined}`);
console.log(lines.join('\n'));

let ok = true;
if (process.argv.includes('--no-native')) {
  ok = combined === EXPECTED;
  console.log(ok ? `\nOK: combined digest matches release ${sd.version()}` : `\nMISMATCH: expected ${EXPECTED}`);
} else {
  const native = execFileSync('cargo', ['run', '--quiet', '--release', '--example', 'digest'], { cwd: root, encoding: 'utf8' }).trim().split('\n');
  const diff = lines.filter((l, i) => l !== native[i]);
  ok = diff.length === 0 && native.length === lines.length;
  console.log(ok ? `\nOK: WebAssembly == native (${lines.length - 1} results)` : `\nMISMATCH in ${diff.length} line(s):\n${diff.join('\n')}`);
}
process.exit(ok ? 0 : 1);
