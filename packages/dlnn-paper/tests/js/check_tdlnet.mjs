// usage: node tests/js/check_tdlnet.mjs model.json examples.json  -> prints a JSON summary (max |JS - numpy| logits)
import { readFileSync } from "node:fs";
import { decodeModel, decodeExamples, forward, argmax } from "../../docs/artifacts/assets/js/tdlnet.js";
const [mp, ep] = process.argv.slice(2);
const model = decodeModel(JSON.parse(readFileSync(mp, "utf8")));
const exj = JSON.parse(readFileSync(ep, "utf8"));
const ex = decodeExamples(exj);
const T = ex.shape[1];
let maxErr = 0, agree = 0, correct = 0, maxLayerErr = null;
const t0 = process.hrtime.bigint();
for (let e = 0; e < ex.n; e++) {
  const r = forward(model, ex.x[e], T);
  const ref = ex.logits[e];
  for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(r.out[i] - ref[i]));
  if (argmax(r.out) === argmax(ref)) agree++;
  if (argmax(r.out) === ex.y[e]) correct++;
}
const ms = Number(process.hrtime.bigint() - t0) / 1e6;
console.log(JSON.stringify({ n: ex.n, max_abs_err: maxErr, top1_agree: agree, js_correct: correct, ms_total: ms, ms_per_example: ms / ex.n }));
