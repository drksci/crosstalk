// models.js: JS forward passes of the arena models, decoded from data/arena_models.json ("dlnn-arena-models/1").
// Input convention for every model: x = Float32Array [2*128], I row then Q row, raw (unnormalised) as in the dataset.
// Output: Float32Array(8) logits in the order of CLASSES. Python references: dlnn/viz/export_web.py reference_logits().
//   dlnn       tdlnet.js (dlnn-web-model/1): ternary taps, tanh, pooled digital readout
//   mlp        relu(W h + b) on raw I/Q; mlp-feat: the same on 319 phase-invariant features (powCompactFeatures)
//   mlp (raw)  relu(W h + b) on the flattened I/Q, linear last layer
//   cnn        3 x (left-pad 7, cross-correlation k=8, ReLU), time mean, linear
//   reservoir  random ternary delay layers (y = tanh((u + bias)/vsat)), pooled mean [and mean^2], standardise, ridge readout
//   cumulant   25 classical features (cumulants, spectral lines, ...), nan_to_num, standardise, softmax regression
// Works in node >= 18 and in browsers.

import { decodeModel, forward as dlnnForward, layerForward } from "./tdlnet.js";
import { fft, makeRng } from "./signals.js";

const T = 128;

function b64ToBytes(s) {
  if (typeof atob === "function") { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
  return new Uint8Array(Buffer.from(s, "base64"));
}
const f32 = (s) => { const b = b64ToBytes(s); return new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2).slice(); };
const i8 = (s) => { const b = b64ToBytes(s); return new Int8Array(b.buffer, b.byteOffset, b.byteLength).slice(); };
export { f32 as decodeF32 };

// ----------------------------------------------------------------------------- MLP
function buildMlp(d) {
  const layers = d.layers.map((l) => ({ W: f32(l.W), b: f32(l.b), out: l.shape[0], inn: l.shape[1] }));
  return (x) => {
    let h = Float64Array.from(x);
    for (let li = 0; li < layers.length; li++) {
      const { W, b, out, inn } = layers[li], y = new Float64Array(out);
      for (let o = 0; o < out; o++) {
        let s = b[o]; const base = o * inn;
        for (let i = 0; i < inn; i++) s += W[base + i] * h[i];
        y[o] = li < layers.length - 1 && s < 0 ? 0 : s;
      }
      h = y;
    }
    return Float32Array.from(h);
  };
}

// ----------------------------------------------------------------------------- feature MLP (phase-invariant 'pow_compact' features)
/** Port of dlnn.model.arena.mlp_features(x, "pow_compact"): unit-RMS z; streams |z|, angle(z[n] z*[n-1])/2pi, |FFT z|/sqrt(N),
 *  |FFT z^2|/sqrt(N), |FFT z^4|/sqrt(N); each stride-2 pooled (mean of pairs) -> 64+63+64+64+64 = 319 features. */
export function powCompactFeatures(x) {
  const N = T, re = new Float64Array(N), im = new Float64Array(N);
  let p = 0; for (let n = 0; n < N; n++) { re[n] = x[n]; im[n] = x[N + n]; p += re[n] * re[n] + im[n] * im[n]; }
  const g = 1 / Math.sqrt(p / N); for (let n = 0; n < N; n++) { re[n] *= g; im[n] *= g; }
  const sq = Math.sqrt(N), out = [];
  const pool = (a) => { const o = new Float64Array(a.length >> 1); for (let k = 0; k < o.length; k++) o[k] = 0.5 * (a[2 * k] + a[2 * k + 1]); return o; };
  const amp = new Float64Array(N); for (let n = 0; n < N; n++) amp[n] = Math.hypot(re[n], im[n]);
  const f1 = new Float64Array(N - 1);
  for (let n = 1; n < N; n++) f1[n - 1] = Math.atan2(im[n] * re[n - 1] - re[n] * im[n - 1], re[n] * re[n - 1] + im[n] * im[n - 1]) / (2 * Math.PI);
  const mag = (a, b) => { const fr = Float64Array.from(a), fi = Float64Array.from(b); fft(fr, fi); const o = new Float64Array(N); for (let k = 0; k < N; k++) o[k] = Math.hypot(fr[k], fi[k]) / sq; return o; };
  const z2r = new Float64Array(N), z2i = new Float64Array(N), z4r = new Float64Array(N), z4i = new Float64Array(N);
  for (let n = 0; n < N; n++) { z2r[n] = re[n] * re[n] - im[n] * im[n]; z2i[n] = 2 * re[n] * im[n]; z4r[n] = z2r[n] * z2r[n] - z2i[n] * z2i[n]; z4i[n] = 2 * z2r[n] * z2i[n]; }
  for (const s of [amp, f1, mag(re, im), mag(z2r, z2i), mag(z4r, z4i)]) out.push(...pool(s));
  return Float64Array.from(out);
}
function buildMlpFeat(d) {
  if (d.feature_set !== "pow_compact") throw new Error("unsupported feature set " + d.feature_set);
  const run = buildMlp(d);
  return (x) => run(powCompactFeatures(x));
}

// ----------------------------------------------------------------------------- 1-D CNN
function buildCnn(d) {
  const conv = d.conv.map((l) => ({ W: f32(l.W), b: f32(l.b), O: l.shape[0], I: l.shape[1], K: l.shape[2] }));
  const fcW = f32(d.fc_W), fcB = f32(d.fc_b), nOut = d.fc_shape[0], nIn = d.fc_shape[1];
  return (x) => {
    let h = Float64Array.from(x), cin = 2;
    for (const L of conv) {
      const y = new Float64Array(L.O * T);
      for (let o = 0; o < L.O; o++) {
        for (let n = 0; n < T; n++) {
          let s = L.b[o];
          for (let i = 0; i < L.I; i++) {
            const wb = (o * L.I + i) * L.K, xb = i * T;
            // out[n] = sum_j w[j] * xpad[n + j], xpad = K-1 zeros then x  =>  x[n + j - (K-1)]
            for (let j = 0; j < L.K; j++) { const m = n + j - (L.K - 1); if (m >= 0) s += L.W[wb + j] * h[xb + m]; }
          }
          y[o * T + n] = s > 0 ? s : 0;
        }
      }
      h = y; cin = L.O;
    }
    const f = new Float64Array(cin);
    for (let c = 0; c < cin; c++) { let s = 0; for (let n = 0; n < T; n++) s += h[c * T + n]; f[c] = s / T; }
    const out = new Float32Array(nOut);
    for (let r = 0; r < nOut; r++) { let s = fcB[r]; for (let k = 0; k < nIn; k++) s += fcW[r * nIn + k] * f[k]; out[r] = s; }
    return out;
  };
}

// ----------------------------------------------------------------------------- reservoir
function sparseTaps(W, O, I, K, dil) {
  const taps = [];
  for (let o = 0; o < O; o++) {
    const t = [];
    for (let i = 0; i < I; i++) for (let k = 0; k < K; k++) { const w = W[(o * I + i) * K + k]; if (w !== 0) t.push(i, k * dil, w); }
    taps.push(Int32Array.from(t));
  }
  return taps;
}
function buildReservoir(d) {
  const layers = d.layers.map((l) => ({
    O: l.O, I: l.I, K: l.K, dilation: l.dilation, vsat: Math.fround(l.vsat), gain: new Float32Array(l.O).fill(1),
    bias: f32(l.bias), taps: sparseTaps(i8(l.W), l.O, l.I, l.K, l.dilation),
  }));
  const mean = f32(d.feat_mean), std = f32(d.feat_std), RW = f32(d.readout_W), RB = f32(d.readout_b);
  const nOut = d.readout_shape[0], nF = d.readout_shape[1], sq = d.pool === "mean_sq";
  return (x) => {
    let h = x;
    const f = new Float64Array(nF); let c = 0;
    for (const ly of layers) {
      h = layerForward(h, T, ly);
      for (let o = 0; o < ly.O; o++) { let s = 0; for (let n = 0; n < T; n++) s += h[o * T + n]; f[c++] = s / T; }
      if (sq) for (let o = 0; o < ly.O; o++) { let s = 0; for (let n = 0; n < T; n++) s += h[o * T + n] ** 2; f[c++] = s / T; }
    }
    for (let k = 0; k < nF; k++) f[k] = (f[k] - mean[k]) / std[k];
    const out = new Float32Array(nOut);
    for (let r = 0; r < nOut; r++) { let s = RB[r]; for (let k = 0; k < nF; k++) s += RW[r * nF + k] * f[k]; out[r] = s; }
    return out;
  };
}

// ----------------------------------------------------------------------------- cumulant features (port of dlnn.model.arena.features)
export const FEATURE_NAMES = [
  "C20abs", "C40abs", "C41abs", "C42", "C60abs", "C63", "rawC40abs", "rawC41abs", "line2", "line4", "line8", "amp_cv", "amp_kurt",
  "amp_skew", "f1_std", "f1_kurt", "f1_absmean", "f4_std", "f4_kurt", "f8_std", "phd_std", "phd_kurt", "bw_std", "spec_flat", "spec_peak",
];

const kurt = (v) => {
  const N = v.length; let mu = 0; for (let i = 0; i < N; i++) mu += v[i]; mu /= N;
  let m2 = 0, m4 = 0; for (let i = 0; i < N; i++) { const d = v[i] - mu, d2 = d * d; m2 += d2; m4 += d2 * d2; }
  m2 /= N; m4 /= N; return m4 / (m2 * m2 + 1e-12);
};
const stdv = (v) => {
  const N = v.length; let mu = 0; for (let i = 0; i < N; i++) mu += v[i]; mu /= N;
  let s = 0; for (let i = 0; i < N; i++) s += (v[i] - mu) ** 2; return Math.sqrt(s / N);
};

/** Complex helpers on (re, im) arrays: mean over n of z^a * conj(z)^b. */
function momentAB(re, im, a, b) {
  const N = re.length; let sr = 0, si = 0;
  for (let n = 0; n < N; n++) {
    let pr = 1, pi = 0;
    for (let k = 0; k < a; k++) { const r = pr * re[n] - pi * im[n]; pi = pr * im[n] + pi * re[n]; pr = r; }
    for (let k = 0; k < b; k++) { const r = pr * re[n] + pi * im[n]; pi = pi * re[n] - pr * im[n]; pr = r; }
    sr += pr; si += pi;
  }
  return [sr / N, si / N];
}
const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const cabs = (a) => Math.hypot(a[0], a[1]);
const cadd = (...v) => v.reduce((s, a) => [s[0] + a[0], s[1] + a[1]], [0, 0]);
const cscale = (a, k) => [a[0] * k, a[1] * k];

function cumulants(re, im) {
  const m20 = momentAB(re, im, 2, 0), m40 = momentAB(re, im, 4, 0), m41 = momentAB(re, im, 3, 1), m42 = momentAB(re, im, 2, 2);
  const m21 = momentAB(re, im, 1, 1)[0], m60 = momentAB(re, im, 6, 0), m63 = momentAB(re, im, 3, 3);
  const a20 = cabs(m20);
  const c40 = cadd(m40, cscale(cmul(m20, m20), -3));
  const c41 = cadd(m41, cscale(m20, -3 * m21));
  const c42 = m42[0] - a20 * a20 - 2 * m21 * m21;
  const c60 = cadd(m60, cscale(cmul(m40, m20), -15), cscale(cmul(m20, cmul(m20, m20)), 30));
  const c63 = m63[0] - 9 * m42[0] * m21 - 6 * a20 * a20 * m21 + 12 * m21 ** 3;
  return [a20, cabs(c40), cabs(c41), c42, cabs(c60), c63];
}

function lineStrength(re, im, p) { // max |FFT(z^p)| / (mean |z|^p * N), plus argmax index
  const N = re.length, pr = new Float64Array(N), pi = new Float64Array(N);
  let pw = 0;
  for (let n = 0; n < N; n++) {
    let ar = 1, ai = 0;
    for (let k = 0; k < p; k++) { const r = ar * re[n] - ai * im[n]; ai = ar * im[n] + ai * re[n]; ar = r; }
    pr[n] = ar; pi[n] = ai; pw += Math.pow(Math.hypot(re[n], im[n]), p);
  }
  fft(pr, pi);
  let best = -1, bi = 0;
  for (let k = 0; k < N; k++) { const a = Math.hypot(pr[k], pi[k]); if (a > best) { best = a; bi = k; } }
  return { strength: best / ((pw / N) * N), idx: bi };
}

/** x: Float32Array [2*128] -> Float64Array(25), the same 25 features, same order, as dlnn.model.arena.features. */
export function cumulantFeatures(x) {
  const N = T, re = new Float64Array(N), im = new Float64Array(N);
  let p = 0;
  for (let n = 0; n < N; n++) { re[n] = x[n]; im[n] = x[N + n]; p += re[n] * re[n] + im[n] * im[n]; }
  const g = 1 / Math.sqrt(p / N);
  for (let n = 0; n < N; n++) { re[n] *= g; im[n] *= g; }
  const l4 = lineStrength(re, im, 4), k = l4.idx > N / 2 ? l4.idx - N : l4.idx, fhat = k / (4 * N);
  const cr = new Float64Array(N), ci = new Float64Array(N);
  for (let n = 0; n < N; n++) { const a = -2 * Math.PI * fhat * n, c = Math.cos(a), s = Math.sin(a); cr[n] = re[n] * c - im[n] * s; ci[n] = re[n] * s + im[n] * c; }
  const C = cumulants(cr, ci), Craw = cumulants(re, im);
  const lines = [lineStrength(re, im, 2).strength, l4.strength, lineStrength(re, im, 8).strength];
  const amp = new Float64Array(N); let am = 0;
  for (let n = 0; n < N; n++) { amp[n] = Math.hypot(re[n], im[n]); am += amp[n]; }
  am /= N;
  const asd = stdv(amp); let sk = 0; for (let n = 0; n < N; n++) sk += (amp[n] - am) ** 3;
  const cv = asd / am, ak = kurt(amp), ask = sk / N / (asd ** 3 + 1e-12);
  const instF = (lag) => { const o = new Float64Array(N - lag); for (let n = lag; n < N; n++) { const pr = re[n] * re[n - lag] + im[n] * im[n - lag], pi = im[n] * re[n - lag] - re[n] * im[n - lag]; o[n - lag] = Math.atan2(pi, pr) / (2 * Math.PI * lag); } return o; };
  const f1 = instF(1), f4 = instF(4), f8 = instF(8);
  let f1abs = 0; for (const v of f1) f1abs += Math.abs(v); f1abs /= f1.length;
  const ph8 = new Float64Array(N - 8);
  for (let n = 8; n < N; n++) { const pr = re[n] * re[n - 8] + im[n] * im[n - 8], pi = im[n] * re[n - 8] - re[n] * im[n - 8]; ph8[n - 8] = pr / (Math.hypot(pr, pi) + 1e-9); }
  const sr = Float64Array.from(re), si = Float64Array.from(im);
  fft(sr, si);
  const S = new Float64Array(N); let ss = 0;
  for (let n = 0; n < N; n++) { S[n] = sr[n] * sr[n] + si[n] * si[n]; ss += S[n]; }
  let mf = 0; const fr = (n) => (n < N / 2 ? n : n - N) / N;
  for (let n = 0; n < N; n++) { S[n] /= ss; mf += S[n] * fr(n); }
  let bw = 0, lg = 0, sm = 0; for (let n = 0; n < N; n++) { bw += S[n] * (fr(n) - mf) ** 2; lg += Math.log(S[n] + 1e-12); sm += S[n]; }
  bw = Math.sqrt(bw); const flat = Math.exp(lg / N) / (sm / N);
  const sorted = Float64Array.from(S).sort(); let peak = 0; for (let n = N - 4; n < N; n++) peak += sorted[n];
  return Float64Array.from([C[0], C[1], C[2], C[3], C[4], C[5], Craw[1], Craw[2], ...lines, cv, ak, ask, stdv(f1), kurt(f1), f1abs, stdv(f4), kurt(f4), stdv(f8), stdv(ph8), kurt(ph8), bw, flat, peak]);
}

function buildCumulant(d) {
  const mean = f32(d.feat_mean), std = f32(d.feat_std), W = f32(d.W), b = f32(d.b), nF = d.W_shape[0], nC = d.W_shape[1];
  return (x) => {
    const f = cumulantFeatures(x), out = new Float32Array(nC);
    for (let k = 0; k < nF; k++) { let v = f[k]; if (Number.isNaN(v)) v = 0; else if (v === Infinity) v = 1e3; else if (v === -Infinity) v = -1e3; f[k] = (v - mean[k]) / std[k]; }
    for (let c = 0; c < nC; c++) { let s = b[c]; for (let k = 0; k < nF; k++) s += f[k] * W[k * nC + c]; out[c] = s; }
    return out;
  };
}

// ----------------------------------------------------------------------------- DLNN with tap-amplitude tolerance
/** Same forward pass as tdlnet.js, but every non-zero tap gain is multiplied by 10^(sigmaDb * N(0,1) / 20) (fixed by `seed`):
 *  the "tap amplitude error" of the tolerance sweeps. sigmaDb = 0 is exactly the ideal ternary net. */
export function tolerantForward(model, sigmaDb, seed = 1) {
  const rng = makeRng(seed);
  const layers = model.layers.map((ly) => ({
    ...ly,
    w: ly.taps.map((t) => { const w = new Float32Array(t.length / 3); for (let p = 0; p < t.length; p += 3) w[p / 3] = t[p + 2] * Math.pow(10, (sigmaDb * rng.n()) / 20); return w; }),
  }));
  return (x) => {
    let h = Float32Array.from(x, (v) => v * model.inputScale);
    const outs = [];
    for (const ly of layers) {
      const y = new Float32Array(ly.O * T);
      for (let o = 0; o < ly.O; o++) {
        const t = ly.taps[o], w = ly.w[o], g = ly.gain[o], b = ly.bias[o];
        for (let n = 0; n < T; n++) {
          let u = 0;
          for (let p = 0, q = 0; p < t.length; p += 3, q++) { const m = n - t[p + 1]; if (m >= 0) u += w[q] * h[t[p] * T + m]; }
          y[o * T + n] = g * Math.tanh((Math.fround(u) + b) / ly.vsat);
        }
      }
      outs.push(y); h = y;
    }
    const nOut = model.readoutShape[0], nF = model.readoutShape[1], last = outs[model.readoutFrom[0]], O = model.layers[model.readoutFrom[0]].O;
    const f = new Float32Array(nF);
    for (let o = 0; o < O; o++) { let s = 0; for (let n = 0; n < T; n++) s += last[o * T + n]; f[o] = s / T; }
    const out = new Float32Array(nOut);
    for (let r = 0; r < nOut; r++) { let s = model.readoutB[r]; for (let k = 0; k < nF; k++) s += model.readoutW[r * nF + k] * f[k]; out[r] = s; }
    return { layers: outs, features: f, out };
  };
}

// ----------------------------------------------------------------------------- public API
/** Decode "dlnn-arena-models/1" into runnable adapters: {id, label, desc, params, ops, acc, acc_ge10dB, forward(x), ...}. */
export function buildArenaModels(json) {
  if (json.schema !== "dlnn-arena-models/1") throw new Error("unsupported arena schema " + json.schema);
  return json.models.map((m) => {
    const out = { id: m.id, label: m.label, desc: m.desc, params: m.params, ops: m.ops, acc: m.acc, accGe10: m.acc_ge10dB, accPerClass: m.acc_per_class, standIn: !!m.stand_in, provisional: m.provisional || null, source: m.source };
    const sch = m.data.schema;
    if (sch === "dlnn-web-model/1") {
      out.isDlnn = true;
      out.model = decodeModel(m.data);
      out.forward = (x) => dlnnForward(out.model, x, T).out;
      out.trace = (x) => dlnnForward(out.model, x, T);
      out.setTolerance = (sigmaDb, seed = 1) => {          // pooled single-layer readout only (true for the exported DLNNs)
        if (!(sigmaDb > 0)) { out.forward = (x) => dlnnForward(out.model, x, T).out; out.trace = (x) => dlnnForward(out.model, x, T); return; }
        const f = tolerantForward(out.model, sigmaDb, seed);
        out.forward = (x) => f(x).out; out.trace = f;
      };
    } else if (sch === "dlnn-arena-mlp/1") out.forward = buildMlp(m.data);
    else if (sch === "dlnn-arena-mlp-feat/1") out.forward = buildMlpFeat(m.data);
    else if (sch === "dlnn-arena-cnn/1") out.forward = buildCnn(m.data);
    else if (sch === "dlnn-arena-reservoir/1") out.forward = buildReservoir(m.data);
    else if (sch === "dlnn-arena-cumulant/1") out.forward = buildCumulant(m.data);
    else throw new Error("unknown model schema " + sch);
    return out;
  });
}

export function decodeArenaExamples(j) {
  if (j.schema !== "dlnn-arena-examples/1") throw new Error("unsupported arena examples schema " + j.schema);
  const X = f32(j.x), sx = 2 * T, x = [], logits = {};
  for (let e = 0; e < j.n; e++) x.push(X.subarray(e * sx, (e + 1) * sx));
  for (const [id, s] of Object.entries(j.logits)) { const L = f32(s), a = []; for (let e = 0; e < j.n; e++) a.push(L.subarray(e * 8, (e + 1) * 8)); logits[id] = a; }
  return { n: j.n, x, y: j.y, snr: j.snr, classes: j.classes, logits };
}
