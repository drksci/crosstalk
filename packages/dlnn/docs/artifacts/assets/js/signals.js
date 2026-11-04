// signals.js: JS port of the synthetic modulation generator in dlnn/data/modrec.py, plus a small FFT and PRNG.
// Same definitions as Python: 8 samples/symbol, RRC beta 0.35 (span 6), CPFSK h = 1, GMSK BT 0.3, random carrier phase,
// CFO U(-0.01, 0.01) cycles/sample, random timing offset U(0, 8) samples (fractional delay in the frequency domain),
// unit signal power, AWGN at the requested SNR. The random numbers differ from numpy's (different PRNG), so the
// check in tests/js/check_arena.mjs compares per-class feature statistics, not individual windows.
// Works as an ES module in the browser and in node >= 18. No dependencies.

export const CLASSES = ["OOK", "BPSK", "QPSK", "8PSK", "16QAM", "2FSK", "4FSK", "GMSK"];
export const SNRS = [0, 5, 10, 15, 20];
export const SPS = 8;
export const NSAMP = 128;
const TWO_PI = 2 * Math.PI;

// ----------------------------------------------------------------------------- PRNG (mulberry32 + Box-Muller)
export function makeRng(seed = 1) {
  let s = seed >>> 0, spare = null;
  const u = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const n = () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let a = 0;
    while (a < 1e-12) a = u();
    const r = Math.sqrt(-2 * Math.log(a)), th = TWO_PI * u();
    spare = r * Math.sin(th);
    return r * Math.cos(th);
  };
  return { u, n, int: (k) => Math.floor(u() * k), range: (a, b) => a + (b - a) * u(), choice: (arr) => arr[Math.floor(u() * arr.length)] };
}

// ----------------------------------------------------------------------------- FFT (any length: radix-2 or direct DFT)
const twCache = new Map();
function twiddles(N) {
  let t = twCache.get(N);
  if (!t) {
    const c = new Float64Array(N), s = new Float64Array(N);
    for (let k = 0; k < N; k++) { c[k] = Math.cos((TWO_PI * k) / N); s[k] = Math.sin((TWO_PI * k) / N); }
    t = { c, s };
    twCache.set(N, t);
  }
  return t;
}

/** In-place complex DFT of (re, im); sign -1 forward (numpy fft), +1 inverse (unnormalised). */
export function fft(re, im, inverse = false) {
  const N = re.length, sg = inverse ? 1 : -1;
  if ((N & (N - 1)) === 0) {
    for (let i = 1, j = 0; i < N; i++) {
      let bit = N >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    const { c, s } = twiddles(N);
    for (let len = 2; len <= N; len <<= 1) {
      const step = N / len, half = len >> 1;
      for (let i = 0; i < N; i += len) {
        for (let k = 0; k < half; k++) {
          const wr = c[k * step], wi = sg * s[k * step];
          const a = i + k, b = a + half;
          const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
        }
      }
    }
    return;
  }
  const { c, s } = twiddles(N), or = new Float64Array(N), oi = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    let sr = 0, si = 0;
    for (let n = 0; n < N; n++) {
      const idx = (k * n) % N, wr = c[idx], wi = sg * s[idx];
      sr += re[n] * wr - im[n] * wi; si += re[n] * wi + im[n] * wr;
    }
    or[k] = sr; oi[k] = si;
  }
  re.set(or); im.set(oi);
}

// ----------------------------------------------------------------------------- waveform definitions (modrec.py)
function rrc(beta = 0.35, span = 6, sps = SPS) {
  const half = (span * sps) / 2, h = new Float64Array(2 * half + 1);
  for (let i = 0; i < h.length; i++) {
    const t = (i - half) / sps;
    if (Math.abs(t) < 1e-9) h[i] = 1 - beta + (4 * beta) / Math.PI;
    else if (Math.abs(Math.abs(t) - 1 / (4 * beta)) < 1e-9)
      h[i] = (beta / Math.SQRT2) * ((1 + 2 / Math.PI) * Math.sin(Math.PI / (4 * beta)) + (1 - 2 / Math.PI) * Math.cos(Math.PI / (4 * beta)));
    else h[i] = (Math.sin(Math.PI * t * (1 - beta)) + 4 * beta * t * Math.cos(Math.PI * t * (1 + beta))) / (Math.PI * t * (1 - (4 * beta * t) ** 2));
  }
  let e = 0; for (const v of h) e += v * v;
  const nrm = Math.sqrt(e);
  return h.map((v) => v / nrm);
}
const RRC = rrc();

function gaussianTaps(bt = 0.3, span = 4) {
  const half = (span * SPS) / 2, g = new Float64Array(2 * half + 1), sigma = Math.sqrt(Math.log(2)) / (TWO_PI * bt);
  let sum = 0;
  for (let i = 0; i < g.length; i++) { const t = (i - half) / SPS; g[i] = Math.exp((-t * t) / (2 * sigma * sigma)); sum += g[i]; }
  return g.map((v) => v / sum);
}
const GAUSS = gaussianTaps();

/** numpy.convolve(a, h, mode="same") for a real series a (len N) and odd-length kernel h. */
function convSame(a, h) {
  const N = a.length, M = h.length, off = (M - 1) >> 1, out = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let j = 0; j < M; j++) { const m = i + off - j; if (m >= 0 && m < N) s += a[m] * h[j]; }
    out[i] = s;
  }
  return out;
}

const LEVELS16 = [-3, -1, 1, 3];
/** Baseband waveform of `nsym` symbols of class `cls`: {re, im} Float64Array(nsym*SPS). */
export function waveform(cls, nsym, rng) {
  const L = nsym * SPS;
  if (cls === "2FSK" || cls === "4FSK") {
    const lv = cls === "2FSK" ? [-1, 1] : LEVELS16, re = new Float64Array(L), im = new Float64Array(L);
    let ph = 0;
    for (let s = 0; s < nsym; s++) {
      const f = rng.choice(lv) / (2 * SPS);
      for (let k = 0; k < SPS; k++) { ph += f; re[s * SPS + k] = Math.cos(TWO_PI * ph); im[s * SPS + k] = Math.sin(TWO_PI * ph); }
    }
    return { re, im };
  }
  if (cls === "GMSK") {
    const a = new Float64Array(L);
    for (let s = 0; s < nsym; s++) { const b = 2 * rng.int(2) - 1; for (let k = 0; k < SPS; k++) a[s * SPS + k] = b; }
    const sm = convSame(a, GAUSS), re = new Float64Array(L), im = new Float64Array(L);
    let acc = 0;
    for (let n = 0; n < L; n++) { acc += sm[n]; const ph = (Math.PI * 0.5 * acc) / SPS; re[n] = Math.cos(ph); im[n] = Math.sin(ph); }
    return { re, im };
  }
  const sr = new Float64Array(L), si = new Float64Array(L); // upsampled symbols
  for (let s = 0; s < nsym; s++) {
    let r = 0, i = 0;
    if (cls === "OOK") r = rng.int(2);
    else if (cls === "BPSK") r = 2 * rng.int(2) - 1;
    else if (cls === "QPSK") { const a = Math.PI / 4 + (Math.PI / 2) * rng.int(4); r = Math.cos(a); i = Math.sin(a); }
    else if (cls === "8PSK") { const a = (Math.PI / 4) * rng.int(8); r = Math.cos(a); i = Math.sin(a); }
    else if (cls === "16QAM") { r = LEVELS16[rng.int(4)]; i = LEVELS16[rng.int(4)]; }
    else throw new Error("unknown class " + cls);
    sr[s * SPS] = r; si[s * SPS] = i;
  }
  return { re: convSame(sr, RRC), im: convSame(si, RRC) };
}

const freqCache = new Map();
function fftFreqs(L) { // numpy.fft.fftfreq(L)
  let f = freqCache.get(L);
  if (!f) { f = new Float64Array(L); for (let k = 0; k < L; k++) f[k] = (k < Math.ceil(L / 2) ? k : k - L) / L; freqCache.set(L, f); }
  return f;
}

/** Fractional-sample circular delay (frequency domain), exactly as modrec.generate. Modifies re/im in place. */
function fracDelay(re, im, frac) {
  const L = re.length, f = fftFreqs(L);
  fft(re, im, false);
  for (let k = 0; k < L; k++) {
    const a = -TWO_PI * f[k] * frac, wr = Math.cos(a), wi = Math.sin(a), r = re[k] * wr - im[k] * wi;
    im[k] = re[k] * wi + im[k] * wr; re[k] = r;
  }
  fft(re, im, true);
  for (let k = 0; k < L; k++) { re[k] /= L; im[k] /= L; }
}

/** A clean (noise-free) signal segment: unit power, random carrier phase and CFO as in the dataset.
 *  Python crops 128 samples from 24 symbols after an 8-sample margin; `len`/`margin`/`nsym` let the live stream use a
 *  longer segment with a bigger margin (so the RRC edge transient is cropped away). */
export function cleanSegment(cls, rng, { nsym = 24, margin = 8, len = NSAMP, cfoMax = 0.01 } = {}) {
  const w = waveform(cls, nsym, rng), delay = rng.range(0, SPS), frac = delay - Math.floor(delay);
  fracDelay(w.re, w.im, frac);
  const s0 = margin + Math.floor(delay), I = new Float32Array(len), Q = new Float32Array(len);
  let p = 0;
  for (let n = 0; n < len; n++) p += w.re[s0 + n] ** 2 + w.im[s0 + n] ** 2;
  const g = 1 / Math.sqrt(p / len), cfo = rng.range(-cfoMax, cfoMax), ph0 = rng.range(0, TWO_PI);
  for (let n = 0; n < len; n++) {
    const a = ph0 + TWO_PI * cfo * n, c = Math.cos(a), s = Math.sin(a), r = w.re[s0 + n] * g, i = w.im[s0 + n] * g;
    I[n] = r * c - i * s; Q[n] = r * s + i * c;
  }
  return { i: I, q: Q, cfo };
}

/** One dataset-style window x [2*128] (I row then Q row, Float32Array) with AWGN at snrDb, as modrec.generate. */
export function genWindow(cls, snrDb, rng) {
  const seg = cleanSegment(cls, rng), x = new Float32Array(2 * NSAMP), sigma = Math.sqrt(Math.pow(10, -snrDb / 10) / 2);
  for (let n = 0; n < NSAMP; n++) { x[n] = seg.i[n] + sigma * rng.n(); x[NSAMP + n] = seg.q[n] + sigma * rng.n(); }
  return x;
}

/** Noise standard deviation per quadrature for a unit-power signal at snrDb. */
export const sigmaFor = (snrDb) => Math.sqrt(Math.pow(10, -snrDb / 10) / 2);
