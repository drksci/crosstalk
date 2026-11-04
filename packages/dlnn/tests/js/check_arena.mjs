// usage: node tests/js/check_arena.mjs [dataDir]   (default docs/artifacts/data)
// 1. every arena model's JS forward pass vs the numpy logits stored in examples_arena.json (tolerance 1e-5, top-1 agreement)
// 2. JS signal generator vs the Python generator: per-class means of the 25 cumulant-baseline features, two-sample z-test.
// Prints a JSON summary on the last line; exit code 1 on failure.
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
const dataDir = resolve(process.argv[2] || resolve(root, "docs/artifacts/data"));
const js = (f) => import(resolve(root, "docs/artifacts/assets/js", f));
const { buildArenaModels, decodeArenaExamples, cumulantFeatures, FEATURE_NAMES } = await js("models.js");
const { makeRng, genWindow, CLASSES, SNRS } = await js("signals.js");
const rd = (f) => JSON.parse(readFileSync(resolve(dataDir, f), "utf8"));

const LIMIT = 1e-5, ZMAX = 5.0;
const models = buildArenaModels(rd("arena_models.json"));
const ex = decodeArenaExamples(rd("examples_arena.json"));
const argmax = (z) => { let b = 0; for (let i = 1; i < z.length; i++) if (z[i] > z[b]) b = i; return b; };
const summary = { limit: LIMIT, models: {}, generator: {} };
let ok = true;
for (const m of models) {
  let maxErr = 0, agree = 0, correct = 0;
  const t0 = process.hrtime.bigint();
  for (let e = 0; e < ex.n; e++) {
    const out = m.forward(ex.x[e]), ref = ex.logits[m.id][e];
    for (let i = 0; i < 8; i++) maxErr = Math.max(maxErr, Math.abs(out[i] - ref[i]));
    if (argmax(out) === argmax(ref)) agree++;
    if (argmax(out) === ex.y[e]) correct++;
  }
  const ms = Number(process.hrtime.bigint() - t0) / 1e6 / ex.n;
  summary.models[m.id] = { max_abs_err: maxErr, top1_agree: `${agree}/${ex.n}`, js_correct: `${correct}/${ex.n}`, ms_per_classification: +ms.toFixed(3) };
  if (!(maxErr <= LIMIT) || agree !== ex.n) ok = false;
}

// generator statistics
const gs = rd("examples_gen_stats.json");
const NJ = 400, rng = makeRng(20261103);
let maxZ = 0, nBad = 0, nTests = 0, worst = null;
CLASSES.forEach((cls, c) => {
  const feats = [];
  for (let i = 0; i < NJ; i++) feats.push(cumulantFeatures(genWindow(cls, SNRS[i % SNRS.length], rng)));
  const ref = gs.per_class[cls];
  for (let f = 0; f < FEATURE_NAMES.length; f++) {
    let mu = 0; for (const v of feats) mu += v[f][0] ?? v[f]; mu /= NJ;
    let va = 0; for (const v of feats) va += ((v[f]) - mu) ** 2; va /= NJ;
    const se = Math.sqrt(ref.std[f] ** 2 / ref.n + va / NJ) + 1e-12;
    const z = Math.abs(mu - ref.mean[f]) / se;
    nTests++;
    if (z > ZMAX) nBad++;
    if (z > maxZ) { maxZ = z; worst = { cls, feature: FEATURE_NAMES[f], js_mean: mu, py_mean: ref.mean[f], z }; }
  }
});
summary.generator = { windows_per_class_js: NJ, windows_per_class_py: gs.per_class[CLASSES[0]].n, tests: nTests, zmax_allowed: ZMAX, max_z: +maxZ.toFixed(2), n_over: nBad, worst };
if (nBad > 0) ok = false;
summary.ok = ok;
console.log(JSON.stringify(summary, null, 1));
process.exit(ok ? 0 : 1);
