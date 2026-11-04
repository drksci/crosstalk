// tdlnet.js: hand-written ternary tapped-delay-line (TDL) forward pass, no dependencies.
// Mirrors dlnn/tiers/virtual.py (tier V) and the "dlnn-model/1" schema (dlnn/model/npz.py), as exported by
// dlnn/viz/export_web.py ("dlnn-web-model/1"):
//   u[o,n] = sum_{i,k} W[o,i,k] * x[i, n - k*dilation]      (zeros before n = 0)
//   y[o,n] = gain[o] * tanh((u[o,n] + bias[o]) / vsat)
//   features = concat over readout_from of y ("per_step") or of time-mean(y) ("pooled")
//   out = readout_W @ features + readout_b
// Arrays are stored as Float32Array (float32 like numpy); sums run in double and round on store.
// Works as an ES module in the browser and in node (>= 18).

function b64ToBytes(s) {
  if (typeof atob === "function") { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
  return new Uint8Array(Buffer.from(s, "base64"));
}
const f32 = (s) => { const b = b64ToBytes(s); return new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2).slice(); };
const i8 = (s) => { const b = b64ToBytes(s); return new Int8Array(b.buffer, b.byteOffset, b.byteLength).slice(); };

/** Decode a "dlnn-web-model/1" JSON object into typed arrays plus a sparse tap list per output. */
export function decodeModel(j) {
  if (j.schema !== "dlnn-web-model/1") throw new Error("unsupported model schema " + j.schema);
  const layers = j.layers.map((L) => {
    const W = i8(L.W), gain = f32(L.gain), bias = f32(L.bias);
    const taps = []; // per output o: Int32Array [i, shift, sign, i, shift, sign, ...] for nonzero W
    for (let o = 0; o < L.O; o++) {
      const t = [];
      for (let i = 0; i < L.I; i++) for (let k = 0; k < L.K; k++) {
        const w = W[(o * L.I + i) * L.K + k];
        if (w !== 0) t.push(i, k * L.dilation, w);
      }
      taps.push(Int32Array.from(t));
    }
    const ly = { O: L.O, I: L.I, K: L.K, dilation: L.dilation, vsat: Math.fround(L.vsat), gain, bias, W, taps };
    if (j.line_fir) { // board line: tap k sees the input through its measured FIR h_k[m] (d = 1): Wl[o,i,m] = sum_k W[o,i,k] h_k[m]
      const fir = j.line_fir, M = fir[0].length, Wl = new Float32Array(L.O * L.I * M);
      for (let o = 0; o < L.O; o++) for (let i = 0; i < L.I; i++) for (let m = 0; m < M; m++) {
        let s = 0; for (let k = 0; k < L.K; k++) s += W[(o * L.I + i) * L.K + k] * Math.fround(fir[k][m]);
        Wl[(o * L.I + i) * M + m] = s;
      }
      ly.wl = Wl; ly.M = M;
    }
    return ly;
  });
  return {
    kind: j.kind, layers, readoutW: f32(j.readout_W), readoutShape: j.readout_W_shape, readoutB: f32(j.readout_b),
    readoutFrom: j.readout_from, readoutKind: j.readout_kind, inputScale: Math.fround(j.input_scale),
    inputFullscale: j.input_fullscale, meta: j.meta, sha256: j.sha256,
  };
}

/** One layer: x is a Float32Array [I*T] (row-major [I,T]); returns Float32Array [O*T]. */
export function layerForward(x, T, ly) {
  const y = new Float32Array(ly.O * T);
  const vs = ly.vsat;
  if (ly.wl) { // line-FIR layer: u[o,n] = sum_{i,m} Wl[o,i,m] x[i, n-m]
    const { wl, M, I } = ly;
    for (let o = 0; o < ly.O; o++) {
      const g = ly.gain[o], b = ly.bias[o];
      for (let n = 0; n < T; n++) {
        let u = 0;
        for (let i = 0; i < I; i++) { const wb = (o * I + i) * M, xb = i * T; for (let m = 0; m < M && m <= n; m++) u += wl[wb + m] * x[xb + n - m]; }
        y[o * T + n] = g * Math.tanh((Math.fround(u) + b) / vs);
      }
    }
    return y;
  }
  for (let o = 0; o < ly.O; o++) {
    const t = ly.taps[o], g = ly.gain[o], b = ly.bias[o];
    for (let n = 0; n < T; n++) {
      let u = 0;
      for (let p = 0; p < t.length; p += 3) {
        const m = n - t[p + 1];
        if (m >= 0) u += t[p + 2] * x[t[p] * T + m];
      }
      y[o * T + n] = g * Math.tanh((Math.fround(u) + b) / vs);
    }
  }
  return y;
}

/** Forward pass for one example x: Float32Array [C*T]. Returns {layers: [Float32Array [O*T]], features, out}.
 *  out is [n_out] (pooled) or [n_out*T] (per_step). */
export function forward(model, x, T) {
  let h = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) h[i] = x[i] * model.inputScale;
  const layers = [];
  for (const ly of model.layers) { h = layerForward(h, T, ly); layers.push(h); }
  const nOut = model.readoutShape[0], nFeat = model.readoutShape[1];
  const parts = model.readoutFrom.map((l) => ({ y: layers[l], O: model.layers[l].O }));
  if (model.readoutKind === "pooled") {
    const f = new Float32Array(nFeat); let c = 0;
    for (const p of parts) for (let o = 0; o < p.O; o++) { let s = 0; for (let n = 0; n < T; n++) s += p.y[o * T + n]; f[c++] = s / T; }
    const out = new Float32Array(nOut);
    for (let r = 0; r < nOut; r++) { let s = 0; for (let k = 0; k < nFeat; k++) s += model.readoutW[r * nFeat + k] * f[k]; out[r] = s + model.readoutB[r]; }
    return { layers, features: f, out };
  }
  const out = new Float32Array(nOut * T);
  for (let n = 0; n < T; n++) for (let r = 0; r < nOut; r++) {
    let s = 0, c = 0;
    for (const p of parts) for (let o = 0; o < p.O; o++) s += model.readoutW[r * nFeat + c++] * p.y[o * T + n];
    out[r * T + n] = s + model.readoutB[r];
  }
  return { layers, features: null, out };
}

/** Decode a "dlnn-web-examples/1" object; returns {n, shape, x: Float32Array[], logits: Float32Array[], y, snr}. */
export function decodeExamples(j) {
  if (j.schema !== "dlnn-web-examples/1") throw new Error("unsupported examples schema " + j.schema);
  const X = f32(j.x), L = f32(j.logits), sx = j.shape[0] * j.shape[1], sl = j.logits_shape.reduce((a, b) => a * b, 1);
  const x = [], logits = [];
  for (let e = 0; e < j.n; e++) { x.push(X.subarray(e * sx, (e + 1) * sx)); logits.push(L.subarray(e * sl, (e + 1) * sl)); }
  return { n: j.n, shape: j.shape, x, logits, y: j.y, snr: j.snr, classes: j.classes, layers: j.layers };
}

export function softmax(z) {
  let m = -Infinity; for (const v of z) m = Math.max(m, v);
  const e = Array.from(z, (v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}
export const argmax = (z) => { let b = 0; for (let i = 1; i < z.length; i++) if (z[i] > z[b]) b = i; return b; };
