// arena.js: model-agnostic arena engine (Amendments A1/A2). DOM-free, so it runs in node for tests.
//
// Loads N model adapters (models.js, built from data/arena_models.json) and runs them in sync on ONE continuously
// cycling input stream. A "source" yields clean complex samples one at a time; the engine adds the user's carrier
// offset and AWGN at the chosen SNR, keeps the last 128 samples (the window every model was trained on), and every
// `hop` samples classifies that window with every enabled model. Work is time-sliced across animation frames so
// a slow model never blocks rendering. The labelled cycling source supplies ground truth for the live accuracy counter.
//
//   const arena = new Arena({ models, source, energy, now });
//   arena.tick(dtMs)  // from requestAnimationFrame; emits samples at `rate`/s while running, advances classification
//   arena.step()      // paused: emit exactly one sample and classify immediately
//   arena.on("classify", (res) => ...)
//
// Source interface: { name, labelled, next() -> { i, q, label (class index or -1), seg (segment id) } }.

import { makeRng, sigmaFor, NSAMP } from "./signals.js";

export const NCLASS = 8;
export const BASE_RATE = 120; // samples per second at 1x playback (cosmetic: 1 sample = 0.96 us on the board)

export function softmax(z) {
  let m = -Infinity; for (const v of z) if (v > m) m = v;
  const e = new Float64Array(z.length); let s = 0;
  for (let i = 0; i < z.length; i++) { e[i] = Math.exp(z[i] - m); s += e[i]; }
  for (let i = 0; i < z.length; i++) e[i] /= s;
  return e;
}
export const argmax = (z) => { let b = 0; for (let i = 1; i < z.length; i++) if (z[i] > z[b]) b = i; return b; };

/** Energy estimate from data/energy_refs.json: arithmetic only, 45 nm, Horowitz ISSCC 2014 (labelled estimate). */
export function energyModel(refs) {
  const h = refs && refs.verified && refs.verified.find((v) => v.pJ_per_op);
  if (!h) return null;
  const p = h.pJ_per_op, nonlinPJ = 10 * p.fp_mult_32bit; // tanh/exp/atan2: ASSUMPTION, 10 fp32 multiplies, not from the source
  // Verified MCU figure: STM32G474 DS12288 Table 21, IDD(Run) 29.5 mA at 170 MHz (typ, 25 C). VDD = 3.3 V is an ASSUMPTION (Nucleo supply),
  // 1 cycle per arithmetic op and NL_CYCLES per nonlinear op are ASSUMPTIONS, no loads/stores/loop overhead: a floor, not a measurement.
  const mc = refs.verified.find((v) => v.id === "stm32g474_run_mode_current");
  const VDD = 3.3, NL_CYCLES = 40;
  const mcu = mc && mc.values ? {
    mA: mc.values["170MHz_range1_boost_typ_25C_mA"], MHz: 170, vdd: VDD, nlCycles: NL_CYCLES, source: mc.source, caveat: mc.caveat,
    nJPerCycle: (mc.values["170MHz_range1_boost_typ_25C_mA"] * VDD) / 170,      // mA * V / MHz = nJ per cycle
    cycles: (ops) => ops.mults + ops.adds + ops.nonlin * NL_CYCLES,
    estimate(ops) { return (this.cycles(ops) * this.nJPerCycle) / 1e3; },        // microjoules
    timeUs(ops) { return this.cycles(ops) / this.MHz; },
  } : null;
  return {
    mcu,
    source: h.source, conditions: h.conditions, caveat: h.caveat, nonlinAssumption: "nonlinear op (tanh, exp, atan2) assumed = 10 fp32 multiplies",
    estimate(ops, ternary) {
      const add = ternary ? p.int_add_32bit : p.fp_add_32bit;
      return (ops.mults * p.fp_mult_32bit + ops.adds * add + ops.nonlin * nonlinPJ) / 1e6; // pJ -> microjoules
    },
  };
}

export class Arena {
  constructor({ models, source, energy = null, now = () => (typeof performance !== "undefined" ? performance.now() : Date.now()), seed = 7, budgetMs = 5 }) {
    // per-model state objects inherit the adapter (forward/trace/ops/...), so adapter-side changes such as setTolerance() apply live
    this.models = models.map((m) => Object.assign(Object.create(m), {
      enabled: m.id !== "mlp_l" && m.id !== "dlnn_fro", probs: new Float64Array(NCLASS).fill(1 / NCLASS), logits: new Float32Array(NCLASS), pred: -1, emaMs: 0, lastMs: 0,
      recent: [], correctAll: 0, totalAll: 0, ready: false,
    }));
    this.energy = energy ? energyModel(energy) : null;
    this.now = now;
    this.rng = makeRng(seed);
    this.budgetMs = budgetMs;
    this.source = source;
    this.snr = 12; this.cfo = 0; this.rate = BASE_RATE; this.hop = 16; this.paused = false;
    this.ringI = new Float32Array(NSAMP); this.ringQ = new Float32Array(NSAMP); this.ringSeg = new Int32Array(NSAMP).fill(-2); this.ringLab = new Int32Array(NSAMP).fill(-1);
    this.head = 0; this.filled = 0; this.sinceClassify = 0; this.cfoPhase = 0; this.acc = 0; this.nSamples = 0;
    this.pending = null; this.skipped = 0; this.classifications = 0;
    this.listeners = { classify: [], sample: [], source: [] };
    this.truth = -1; this.cleanI = 0; this.cleanQ = 0; this.lastIQ = [0, 0];
    this.tolerance = { sigmaDb: 0, seed: 1 };
    this.challenge = { active: false, duration: 20000, left: 0, score: 0, best: 0, log: [], rounds: 0 };
    this.lastResult = null;
  }

  on(ev, fn) { this.listeners[ev].push(fn); return () => { this.listeners[ev] = this.listeners[ev].filter((f) => f !== fn); }; }
  emit(ev, a) { for (const f of this.listeners[ev]) f(a); }

  setSource(src) {
    this.source = src; this.resetWindow(); this.emit("source", src);
  }
  resetWindow() {
    this.ringI.fill(0); this.ringQ.fill(0); this.ringSeg.fill(-2); this.ringLab.fill(-1); this.filled = 0; this.sinceClassify = 0; this.pending = null;
    for (const m of this.models) { m.recent = []; m.ready = false; }
  }
  resetScores() { for (const m of this.models) { m.recent = []; m.correctAll = 0; m.totalAll = 0; } }
  setEnabled(id, on) { const m = this.models.find((q) => q.id === id); if (m) m.enabled = on; }

  /** Emit one channel sample: clean source sample -> CFO rotation -> AWGN -> ring buffer. */
  pushSample() {
    const s = this.source.next();
    this.cleanI = s.i; this.cleanQ = s.q;
    const a = this.cfoPhase, c = Math.cos(a), sn = Math.sin(a);
    this.cfoPhase = (this.cfoPhase + 2 * Math.PI * this.cfo) % (2 * Math.PI);
    const sg = sigmaFor(this.snr);
    const i = s.i * c - s.q * sn + sg * this.rng.n(), q = s.i * sn + s.q * c + sg * this.rng.n();
    const h = this.head;
    this.ringI[h] = i; this.ringQ[h] = q; this.ringSeg[h] = s.seg ?? -1; this.ringLab[h] = s.label ?? -1;
    this.head = (h + 1) % NSAMP; this.filled = Math.min(NSAMP, this.filled + 1);
    this.truth = s.label ?? -1; this.lastIQ = [i, q]; this.nSamples++;
    this.emit("sample", { i, q, label: this.truth });
    return ++this.sinceClassify >= this.hop;
  }

  /** The current window in time order as Float32Array [I(128), Q(128)] (oldest sample first). */
  window() {
    const x = new Float32Array(2 * NSAMP);
    for (let n = 0; n < NSAMP; n++) { const k = (this.head + n) % NSAMP; x[n] = this.ringI[k]; x[NSAMP + n] = this.ringQ[k]; }
    return x;
  }
  /** Ground-truth class of the window if every sample belongs to one labelled segment, else -1. */
  windowLabel() {
    if (this.filled < NSAMP) return -1;
    const s0 = this.ringSeg[0], l0 = this.ringLab[0];
    if (l0 < 0) return -1;
    for (let k = 1; k < NSAMP; k++) if (this.ringSeg[k] !== s0 || this.ringLab[k] !== l0) return -1;
    return l0;
  }

  startClassify() {
    this.sinceClassify = 0;
    if (this.filled < NSAMP) return;
    if (this.pending) { this.skipped++; return; }
    this.pending = { x: this.window(), label: this.windowLabel(), todo: this.models.filter((m) => m.enabled), done: [], t0: this.now() };
  }

  /** Run pending model evaluations until the per-frame budget is spent. */
  work(force = false) {
    const p = this.pending;
    if (!p) return;
    const t0 = this.now();
    while (p.todo.length) {
      const m = p.todo.shift();
      const a = this.now();
      let lg;
      if (m.trace) { const r = m.trace(p.x); lg = r.out; m.lastTrace = r.layers; m.traceN = (m.traceN || 0) + 1; } else lg = m.forward(p.x);
      const ms = this.now() - a;
      m.lastMs = ms; m.emaMs = m.emaMs ? 0.85 * m.emaMs + 0.15 * ms : ms;
      m.logits = lg; m.probs = softmax(lg); m.pred = argmax(lg); m.ready = true;
      p.done.push(m);
      if (p.label >= 0) {
        const ok = m.pred === p.label ? 1 : 0;
        m.recent.push(ok); if (m.recent.length > 60) m.recent.shift();
        m.correctAll += ok; m.totalAll++;
      }
      if (!force && this.now() - t0 > this.budgetMs) break;
    }
    if (!p.todo.length) { this.finishWindow(p); this.pending = null; }
  }

  finishWindow(p) {
    this.classifications++;
    const preds = p.done.map((m) => ({ id: m.id, pred: m.pred, conf: m.probs[m.pred] }));
    const distinct = new Set(preds.map((q) => q.pred)).size;
    const dl = preds.find((q) => q.id === "dlnn");
    const others = preds.filter((q) => q.id !== "dlnn" && q.id !== "dlnn_fro" && q.id !== "dlnn_board");
    const dlnnOdd = !!dl && others.length > 0 && others.every((q) => q.pred !== dl.pred) && new Set(others.map((q) => q.pred)).size === 1;
    const res = { label: p.label, preds, disagree: distinct > 1, distinct, dlnnOdd, x: p.x, n: this.classifications };
    const ch = this.challenge;
    if (ch.active && res.disagree) {
      const pts = 1 + (dlnnOdd ? 2 : 0);
      ch.score += pts;
      ch.log.unshift({ n: res.n, pts, preds, label: p.label });
      if (ch.log.length > 6) ch.log.pop();
    }
    this.lastResult = res;
    this.emit("classify", res);
  }

  startChallenge() { const c = this.challenge; c.active = true; c.left = c.duration; c.score = 0; c.log = []; c.rounds++; }
  endChallenge() { const c = this.challenge; c.active = false; c.best = Math.max(c.best, c.score); this.emit("classify", this.lastResult); }

  /** Advance by dtMs of wall time: emit samples at `rate` while running, then spend the frame's compute budget. */
  tick(dtMs) {
    if (!this.paused) {
      this.acc += (this.rate * Math.min(dtMs, 100)) / 1000;
      let n = Math.min(Math.floor(this.acc), 12);
      this.acc -= Math.floor(this.acc);
      while (n-- > 0) if (this.pushSample()) this.startClassify();
      if (this.challenge.active) {
        this.challenge.left -= dtMs;
        if (this.challenge.left <= 0) { this.challenge.left = 0; this.endChallenge(); }
      }
    }
    this.work();
  }

  /** Paused single step: one channel sample, then classify the window right away (so each tap contribution is visible). */
  step() {
    this.pushSample();
    this.startClassify();
    this.work(true);
  }

  accuracy(m) { return m.recent.length ? m.recent.reduce((a, b) => a + b, 0) / m.recent.length : null; }
  /** Energy per classification in microjoules (digital-equivalent arithmetic estimate) or null when refs are missing. */
  energyUJ(m) { return this.energy ? this.energy.estimate(m.ops, m.id === "dlnn" || m.id === "dlnn_fro" || m.id === "dlnn_board" || m.id === "reservoir") : null; }
  /** Energy per classification in microjoules if run on an STM32G4 at 170 MHz (verified current, assumed cycles), or null. */
  mcuUJ(m) { return this.energy && this.energy.mcu ? this.energy.mcu.estimate(m.ops) : null; }
}
