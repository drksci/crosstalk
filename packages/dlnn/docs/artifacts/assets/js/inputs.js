// inputs.js: the arena's signal sources. Each yields CLEAN complex samples one at a time (the engine adds CFO and noise):
//   CyclingSource  random modulations from the dataset generator (signals.js), labelled, cycling through all 8 classes
//   KeySource      hold a key / touch to key OOK (carrier on while held) or 2FSK (two tones)
//   MicSource      Web Audio: loudness gate -> OOK, whistle pitch -> 2FSK/4FSK; needs permission, fails gracefully
//   DrawSource     a hand-drawn I/Q trajectory, looped (one lap = 128 / laps samples)
// Frequencies are in cycles/sample of the 8 samples/symbol dataset: 2FSK tones are +/-1/16, 4FSK +/-1/16, +/-3/16.

import { makeRng, cleanSegment, CLASSES, NSAMP } from "./signals.js";

const SQRT2 = Math.SQRT2;       // OOK "on" amplitude: unit mean power at 50 % duty, as in the dataset
const TWO_PI = 2 * Math.PI;

export class CyclingSource {
  constructor(seed = 11) {
    this.name = "cycle"; this.labelled = true; this.rng = makeRng(seed);
    this.lock = -1;          // -1 = cycle through all classes, else a fixed class index
    this.seg = null; this.pos = 0; this.segId = 0; this.cls = -1; this.last = -1;
  }
  setLock(c) { this.lock = c; this.seg = null; }
  skip() { this.seg = null; }
  fresh() {
    let c = this.lock;
    if (c < 0) { do { c = this.rng.int(CLASSES.length); } while (c === this.last); }
    this.last = c; this.cls = c; this.segId++;
    // 64 symbols, crop 448 samples after a 24-sample margin: the RRC / fractional-delay edge transients are cropped away
    this.seg = cleanSegment(CLASSES[c], this.rng, { nsym: 64, margin: 24, len: 448 });
    this.pos = 0;
  }
  next() {
    if (!this.seg || this.pos >= this.seg.i.length) this.fresh();
    const k = this.pos++;
    return { i: this.seg.i[k], q: this.seg.q[k], label: this.cls, seg: this.segId };
  }
  get info() { return { cls: this.cls, name: this.cls >= 0 ? CLASSES[this.cls] : "" }; }
}

export class KeySource {
  constructor() {
    this.name = "key"; this.labelled = false; this.down = false; this.mode = "ook"; // "ook" | "fsk"
    this.env = 0; this.phase = 0; this.segId = 1; this.ones = 0; this.count = 0;
  }
  setDown(d) { this.down = !!d; }
  setMode(m) { this.mode = m; this.env = 0; this.segId++; }
  next() {
    this.count++;
    if (this.mode === "ook") {
      this.env += ((this.down ? SQRT2 : 0) - this.env) * 0.4; // one-pole edge smoothing, about the RRC rise time
      this.phase = 0.3;
      return { i: this.env * Math.cos(this.phase), q: this.env * Math.sin(this.phase), label: -1, seg: this.segId };
    }
    this.phase += TWO_PI * (this.down ? 1 : -1) / 16;
    return { i: Math.cos(this.phase), q: Math.sin(this.phase), label: -1, seg: this.segId };
  }
  get hint() { return this.mode === "ook" ? "OOK" : "2FSK"; }
}

/** Autocorrelation pitch estimate of a mono buffer; returns Hz or 0. Searches 180-2500 Hz (a whistle is 500-2500 Hz). */
export function detectPitch(buf, sampleRate, fMin = 180, fMax = 2500) {
  const N = buf.length;
  let e = 0; for (let i = 0; i < N; i++) e += buf[i] * buf[i];
  if (e / N < 1e-6) return 0;
  const lagMin = Math.floor(sampleRate / fMax), lagMax = Math.min(N >> 1, Math.ceil(sampleRate / fMin));
  let best = 0, bestLag = 0;
  const corr = new Float32Array(lagMax + 2);
  for (let lag = lagMin; lag <= lagMax + 1; lag++) {
    let s = 0, n1 = 0, n2 = 0;
    for (let i = 0; i + lag < N; i += 2) { s += buf[i] * buf[i + lag]; n1 += buf[i] * buf[i]; n2 += buf[i + lag] * buf[i + lag]; }
    corr[lag] = s / (Math.sqrt(n1 * n2) + 1e-12);
  }
  for (let lag = lagMin + 1; lag <= lagMax; lag++) {
    const c = corr[lag];
    if (c > best && c >= corr[lag - 1] && c >= corr[lag + 1]) { best = c; bestLag = lag; if (c > 0.92) break; }
  }
  if (best < 0.6 || !bestLag) return 0;
  const a = corr[bestLag - 1], b = corr[bestLag], c = corr[bestLag + 1], den = a - 2 * b + c;     // parabolic refinement
  const off = Math.abs(den) > 1e-9 ? (0.5 * (a - c)) / den : 0;
  return sampleRate / (bestLag + off);
}

/** Map a pitch in Hz to an FSK level index (0..nLevels-1) on a log scale between fLo and fHi. */
export function pitchToLevel(hz, nLevels, fLo = 500, fHi = 2200) {
  const t = Math.min(1, Math.max(0, Math.log(hz / fLo) / Math.log(fHi / fLo)));
  return Math.min(nLevels - 1, Math.floor(t * nLevels));
}

export class MicSource {
  constructor() {
    this.name = "mic"; this.labelled = false; this.mode = "ook"; this.nLevels = 4; // mode: "ook" (loudness) | "fsk" (pitch)
    this.state = "idle"; this.error = ""; this.level = 0; this.pitch = 0; this.gate = 0.02; this.sens = 1;
    this.env = 0; this.phase = 0; this.freq = 1 / 16; this.segId = 1; this.fsk = 0;
    this.ctx = null; this.analyser = null; this.stream = null; this.buf = null; this.frame = 0;
  }
  async start() {
    if (this.state === "running") return true;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { this.state = "unavailable"; this.error = "This browser has no microphone API (needs https or localhost)."; return false; }
    try {
      this.state = "asking";
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
      const AC = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AC();
      if (this.ctx.state === "suspended") await this.ctx.resume().catch(() => {});
      const src = this.ctx.createMediaStreamSource(this.stream);
      this.analyser = this.ctx.createAnalyser(); this.analyser.fftSize = 2048; this.analyser.smoothingTimeConstant = 0;
      src.connect(this.analyser);
      this.buf = new Float32Array(this.analyser.fftSize);
      this.state = "running"; this.error = "";
      return true;
    } catch (e) {
      this.state = e && e.name === "NotAllowedError" ? "denied" : "unavailable";
      this.error = this.state === "denied" ? "Microphone permission was denied. Tap the lock icon in the address bar to allow it, or use another input." : "No microphone could be opened (" + (e && e.name ? e.name : "error") + ").";
      return false;
    }
  }
  stop() {
    try { if (this.stream) this.stream.getTracks().forEach((t) => t.stop()); if (this.ctx) this.ctx.close(); } catch (e) { /* ignore */ }
    this.stream = null; this.ctx = null; this.analyser = null; if (this.state === "running") this.state = "idle";
  }
  setMode(m) { this.mode = m; this.segId++; this.env = 0; }
  /** Call once per animation frame: updates loudness (RMS) and, every other frame, the pitch. */
  poll() {
    if (this.state !== "running" || !this.analyser) return;
    this.analyser.getFloatTimeDomainData(this.buf);
    let e = 0; for (let i = 0; i < this.buf.length; i++) e += this.buf[i] * this.buf[i];
    this.level = Math.sqrt(e / this.buf.length);
    if ((this.frame++ & 1) === 0) this.pitch = this.level > this.gate / this.sens ? detectPitch(this.buf.subarray(0, 1024), this.ctx.sampleRate) : 0;
    if (this.pitch > 0) {
      const lv = pitchToLevel(this.pitch, this.nLevels);
      this.fsk = lv;
      const order = this.nLevels === 2 ? [-1, 1] : [-3, -1, 1, 3];
      this.freq = order[lv] / 16;
    }
  }
  get on() { return this.level * this.sens > this.gate; }
  next() {
    if (this.mode === "ook") {
      this.env += ((this.on ? SQRT2 : 0) - this.env) * 0.4;
      return { i: this.env * Math.cos(0.3), q: this.env * Math.sin(0.3), label: -1, seg: this.segId };
    }
    this.phase += TWO_PI * this.freq;       // holds the last tone while you are silent
    return { i: Math.cos(this.phase), q: Math.sin(this.phase), label: -1, seg: this.segId };
  }
}

export class DrawSource {
  constructor() {
    this.name = "draw"; this.labelled = false; this.laps = 1; this.segId = 1; this.pos = 0;
    this.pts = []; this.path = null; this.setPreset("circle");
  }
  /** Points are in the unit square [-1, 1]^2 (I right, Q up). */
  setPoints(pts) {
    this.pts = pts.slice();
    this.path = pts.length >= 2 ? resample(pts, NSAMP) : null;
    this.segId++; this.pos = 0;
  }
  setPreset(name) {
    const p = [];
    if (name === "circle") for (let k = 0; k <= 64; k++) p.push([0.8 * Math.cos((2 * Math.PI * k) / 64), 0.8 * Math.sin((2 * Math.PI * k) / 64)]);
    else if (name === "square") p.push([-0.7, -0.7], [0.7, -0.7], [0.7, 0.7], [-0.7, 0.7], [-0.7, -0.7]);
    else if (name === "eight") for (let k = 0; k <= 96; k++) { const t = (2 * Math.PI * k) / 96; p.push([0.8 * Math.sin(t), 0.8 * Math.sin(2 * t) * 0.8]); }
    else if (name === "star") for (let k = 0; k <= 16; k++) { const a = (2 * Math.PI * (k * 5 % 16)) / 16 + 0.4; p.push([0.85 * Math.cos(a), 0.85 * Math.sin(a)]); }
    else if (name === "clear") { this.pts = []; this.path = null; this.segId++; return; }
    this.setPoints(p);
  }
  setLaps(n) { this.laps = n; this.segId++; }
  next() {
    if (!this.path) return { i: 0, q: 0, label: -1, seg: this.segId };
    const lap = Math.max(1, Math.round(NSAMP / this.laps)), k = this.pos++ % lap, u = (k / lap) * NSAMP;
    const a = Math.floor(u) % NSAMP, b = (a + 1) % NSAMP, f = u - Math.floor(u);
    return { i: this.path.i[a] * (1 - f) + this.path.i[b] * f, q: this.path.q[a] * (1 - f) + this.path.q[b] * f, label: -1, seg: this.segId };
  }
}

/** Resample a polyline to n points uniformly in arc length; return unit-mean-power I and Q (about the origin). */
function resample(pts, n) {
  const d = [0];
  for (let k = 1; k < pts.length; k++) d.push(d[k - 1] + Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]));
  const L = d[d.length - 1] || 1e-9, I = new Float32Array(n), Q = new Float32Array(n);
  let j = 0;
  for (let k = 0; k < n; k++) {
    const s = (k / (n - 1)) * L;
    while (j < pts.length - 2 && d[j + 1] < s) j++;
    const seg = d[j + 1] - d[j] || 1e-9, f = Math.min(1, Math.max(0, (s - d[j]) / seg));
    I[k] = pts[j][0] + f * (pts[j + 1][0] - pts[j][0]); Q[k] = pts[j][1] + f * (pts[j + 1][1] - pts[j][1]);
  }
  let p = 0; for (let k = 0; k < n; k++) p += I[k] * I[k] + Q[k] * Q[k];
  const g = 1 / Math.sqrt(p / n + 1e-12);
  for (let k = 0; k < n; k++) { I[k] *= g; Q[k] *= g; }
  return { i: I, q: Q };
}
