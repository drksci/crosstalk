// arena-ui.js: the W0 arena widget. Builds the DOM, wires the four input sources (cycle / key / mic / draw), the SNR / CFO /
// tolerance knobs, pause + step, per-model probability bars with live accuracy, ops, energy-estimate and latency readouts,
// the D3 view of the DLNN's tap lines and the "fool the network" challenge. Engine: arena.js. Models: models.js.

import { Arena, BASE_RATE, NCLASS } from "./arena.js";
import { buildArenaModels } from "./models.js";
import { CyclingSource, KeySource, MicSource, DrawSource } from "./inputs.js";
import { CLASSES } from "./signals.js";
import { loadJSON, loadOptional, loadD3, el, cssVar, reducedMotion, isNarrow, store } from "./common.js";

const fmtOps = (n) => (n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e3 ? (n / 1e3).toFixed(n >= 1e5 ? 0 : 1) + "k" : String(n));
const fmtUJ = (u) => (u == null ? null : u < 0.01 ? u.toFixed(4) : u < 1 ? u.toFixed(3) : u < 100 ? u.toFixed(2) : u.toFixed(0));
const pct = (v) => (v == null ? "--" : Math.round(v * 100) + "%");
const hex2rgb = (h) => { h = h.replace("#", ""); if (h.length === 3) h = [...h].map((c) => c + c).join(""); return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]; };
const SPEEDS = [["0.1x", 0.1], ["0.25x", 0.25], ["0.5x", 0.5], ["1x", 1], ["2x", 2]];

export async function mountArena(host) {
  const [modelsJson, energyRefs] = await Promise.all([loadJSON("arena_models.json"), loadOptional("energy_refs.json"), loadD3()]);
  const d3 = window.d3;
  const models = buildArenaModels(modelsJson);
  const cycle = new CyclingSource(), key = new KeySource(), mic = new MicSource(), draw = new DrawSource();
  const sources = { cycle, key, mic, draw };
  const arena = new Arena({ models, source: cycle, energy: energyRefs });
  const dl = arena.models.find((m) => m.id === "dlnn");
  const dlAdapters = models.filter((m) => m.isDlnn && m.id !== "dlnn_board");      // the board net keeps its fixed measured line; tolerance applies to the ideal-line nets
  const stats = { fps: 0, frames: 0, longFrames: 0, source: "cycle" };
  window.__arena = { arena, stats, sources, setSource: (n) => setSource(n) };

  // ------------------------------------------------------------------------------------------------ static DOM
  const narrow = () => isNarrow();
  const tabs = el("div", { class: "seg", role: "tablist", "aria-label": "Input source" });
  const TAB = [["cycle", "Cycle"], ["key", "Key"], ["mic", "Mic"], ["draw", "Draw"]];
  for (const [id, label] of TAB) tabs.append(el("button", { role: "tab", id: "ar-tab-" + id, "data-src": id, "aria-selected": id === "cycle", type: "button", text: label }));
  const nowEl = el("div", { class: "ar-now", "aria-live": "off" });
  const btnPlay = el("button", { class: "btn", id: "ar-play", type: "button", "aria-pressed": "false", text: "Pause" });
  const btnStep = el("button", { class: "btn", id: "ar-step", type: "button", text: "Step", title: "Advance one sample (one tap time) and classify" });
  const speed = el("select", { id: "ar-speed", "aria-label": "Playback speed" }, SPEEDS.map(([l, v]) => el("option", { value: v, selected: v === 1, text: "Speed " + l })));
  const panelInput = el("div", { class: "ar-panel", id: "ar-input" });
  const scope = el("canvas", { class: "scope", id: "ar-scope", role: "img", "aria-label": "Last 128 samples: I and Q traces and the I/Q constellation" });
  const knobs = el("div", { class: "knobs" });
  const mkKnob = (id, label, min, max, step, val, fmt, onInput, hint) => {
    const out = el("output", { for: id, text: fmt(val) });
    const inp = el("input", { type: "range", id, min, max, step, value: val, "aria-label": label, title: hint || "" });
    inp.addEventListener("input", () => { out.textContent = fmt(+inp.value); onInput(+inp.value); });
    knobs.append(el("div", { class: "knob" }, el("label", { for: id, text: label, title: hint || "" }), inp, out));
    return inp;
  };
  const snrKnob = mkKnob("ar-snr", "SNR", 0, 30, 1, arena.snr, (v) => v + " dB", (v) => { arena.snr = v; }, "Signal to noise ratio of the AWGN added to the stream. The models were trained on 0 to 20 dB.");
  const cfoKnob = mkKnob("ar-cfo", "Carrier offset", -0.03, 0.03, 0.001, 0, (v) => (v >= 0 ? "+" : "") + v.toFixed(3), (v) => { arena.cfo = v; }, "Carrier frequency offset in cycles per sample. Training covered +/-0.010.");
  const tolKnob = mkKnob("ar-tol", "Tap tolerance", 0, 3, 0.1, 0, (v) => v.toFixed(1) + " dB", (v) => { dlAdapters.forEach((a) => a.setTolerance(v, 5)); }, "Ideal-line DLNNs only (the board-trained net has its own line model): every non-zero tap gain gets a random error with this standard deviation in dB (fixed pattern). The digital models have no such error.");
  const board = el("div", { class: "board", role: "list", "aria-label": "Model outputs" });
  const challenge = el("div", { class: "challenge" });
  const dlPanel = el("div", { class: "ar-panel ar-dlnn" });
  const foot = el("p", { class: "w-foot" });

  host.replaceChildren(
    el("div", { class: "w-head" }, el("h3", { class: "w-title", text: "Arena: the delay-line net against conventional classifiers" }), el("span", { class: "w-tag", text: "W0 · simulated · all models run in your browser" })),
    el("p", { class: "w-sub", text: "Accuracy is not where the DLNN wins: look at the operation counts and what each model needs in hardware. Every model sees the same 128 noisy I/Q samples and answers every 16 samples. Dots under the bars mark the true class of the last window when it came from a single labelled signal." }),
    el("div", { class: "ar-top" }, tabs, nowEl, el("div", { class: "ar-transport" }, btnPlay, btnStep, speed)),
    el("div", { class: "ar-grid" },
      el("div", { class: "ar-col" }, panelInput,
        el("div", { class: "ar-panel" }, el("h4", {}, el("span", { text: "What the models see" }), el("span", { class: "mono", id: "ar-fps", text: "" })), scope),
        el("div", { class: "ar-panel" }, el("h4", { text: "Channel" }), knobs)),
      el("div", { class: "ar-col" }, el("div", { class: "ar-panel" }, board), el("div", { class: "ar-panel" }, challenge))),
    dlPanel,
    foot,
  );

  // ------------------------------------------------------------------------------------------------ model board
  const mrows = {};
  const head = el("div", { class: "mhead", "aria-hidden": "true" }, el("span", { text: "Model" }), el("div", { class: "mbars" }, CLASSES.map((c) => el("span", { class: "lab", text: c }))),
    el("div", { class: "mm" }, el("span", { text: "Live acc" }), el("span", { text: "Cost" })));
  void head;
  const energyTip = arena.energy ? `Digital-equivalent arithmetic energy at ${arena.energy.conditions}, from per-operation costs (${arena.energy.source}). ${arena.energy.nonlinAssumption}. Ternary adds counted as 32-bit integer adds. No memory or instruction overhead. A labelled estimate, not a measurement.` : "";
  const mc = arena.energy && arena.energy.mcu;
  const mcuTip = mc ? `If run on an STM32G474 at ${mc.MHz} MHz: verified supply current ${mc.mA} mA (datasheet DS12288 Table 21, typical, 25 C) at an assumed ${mc.vdd} V is ${(mc.mA * mc.vdd).toFixed(0)} mW, ${mc.nJPerCycle.toFixed(2)} nJ per cycle. Cycles are ASSUMED: 1 per add or multiply, ${mc.nlCycles} per tanh/exp/atan2, no memory or loop overhead, so this is a floor, not a measurement. Core plus flash only.` : "";
  for (const m of arena.models) {
    const row = el("div", { class: "mrow", role: "listitem", "data-id": m.id, "data-off": !m.enabled, style: `--c: var(--m-${m.id})` });
    const cb = el("input", { type: "checkbox", id: "ar-en-" + m.id, checked: m.enabled, "aria-label": "Show " + m.label });
    cb.addEventListener("change", () => { arena.setEnabled(m.id, cb.checked); row.dataset.off = !cb.checked; });
    const bars = CLASSES.map((c, i) => el("div", { class: "bcol", "data-c": i }, el("span", { class: "pv" }), el("div", { class: "bar" }), el("span", { class: "lab", text: c })));
    const accEl = el("span", { class: "acc" }, "acc ", el("b", { text: "--" })), opsEl = el("span", {}, "ops ", el("b", { text: fmtOps(m.ops.adds + m.ops.mults + m.ops.nonlin) })),
      enEl = el("span", { class: arena.energy ? "est" : "pend", title: energyTip }, "energy ", el("b", { text: "pending" })), msEl = el("span", { title: "Measured JavaScript compute time per classification in this browser" }, "time ", el("b", { text: "--" })),
      mcuEl = el("span", { class: "est", title: mcuTip }, "on MCU ", el("b", { text: arena.energy && arena.energy.mcu ? "..." : "n/a" }));
    row.append(
      el("div", { class: "mname" }, el("label", { for: "ar-en-" + m.id }, cb, el("i"), m.label), el("small", { text: `${m.params.toLocaleString()} params` + (m.isDlnn ? ", ternary taps" : "") + (m.provisional ? "; provisional" : "") , title: m.provisional ? "Provisional: " + m.provisional : m.desc }), el("div", { class: "mmet" }, accEl, opsEl, enEl, msEl, mcuEl)),
      el("div", { class: "mbars" }, bars));
    cb.style.cssText = "accent-color: var(--c); margin: 0;";
    board.append(row);
    mrows[m.id] = { row, bars, accEl: accEl.lastChild, enEl, msEl: msEl.lastChild, last: {} };
    const eu = arena.mcuUJ(m);
    if (eu != null) mcuEl.lastChild.textContent = fmtUJ(eu) + " µJ";
    const e = arena.energyUJ(m);
    if (e != null) mrows[m.id].enEl.lastChild.textContent = fmtUJ(e) + " µJ";
  }
  for (const m of arena.models) {
    const r = mrows[m.id];
    if (m.id === "dlnn" && arena.energy) r.enEl.title = energyTip + " If built as the designed 3-tile board: about 150 µs per window at about 1.5 W is roughly 0.2 mJ (design estimate, not measured; docs/phase2_plan.md).";
  }

  // ------------------------------------------------------------------------------------------------ challenge
  const chScore = el("span", { class: "ch-score", id: "ch-score", text: "0" });
  const chBar = el("i");
  const chBtn = el("button", { class: "btn primary", id: "ch-start", type: "button", text: "Start a 20 s round" });
  const chBest = el("span", { class: "mono", id: "ch-best", text: "best 0" });
  const chLog = el("ul", { class: "ch-log", "aria-live": "polite" });
  const chHelp = el("p", { class: "ar-hint", html: "<b>Fool the network.</b> Score +1 for every answer round where the models disagree, +3 when the DLNN is the lone dissenter. Lower the SNR, add a carrier offset, key odd patterns, whistle, or draw something that is not a modulation." });
  challenge.append(el("h4", { class: "ar-h", text: "Fool the network" }), chHelp, el("div", { class: "ch-row" }, chBtn, chScore, el("div", { class: "ch-bar" }, chBar), chBest), chLog);
  chBest.textContent = "best " + (arena.challenge.best = +store.get("dlnn-ch-best", 0) || 0);
  chBtn.addEventListener("click", () => { if (arena.challenge.active) arena.endChallenge(); else { arena.paused = false; syncPlay(); arena.startChallenge(); } renderChallenge(true); });
  function renderChallenge(force) {
    const c = arena.challenge;
    chBtn.textContent = c.active ? "Stop round" : "Start a 20 s round";
    chScore.textContent = c.score;
    chBar.style.width = c.active ? (100 * c.left) / c.duration + "%" : "0%";
    if (c.best > (+store.get("dlnn-ch-best", 0) || 0)) store.set("dlnn-ch-best", c.best);
    chBest.textContent = "best " + c.best;
    if (force || c.log.length !== chLog.childElementCount || (c.log[0] && chLog.firstChild?.dataset.n !== String(c.log[0].n))) {
      chLog.replaceChildren(...c.log.map((e) => el("li", { "data-n": e.n }, el("b", { text: "+" + e.pts }), el("span", { class: "dis" }, e.preds.map((p) => el("span", { style: `--c: var(--m-${p.id})`, text: `${arena.models.find((m) => m.id === p.id).label}: ${CLASSES[p.pred]}` }))))));
      if (!c.log.length) chLog.replaceChildren(el("li", { text: c.active ? "Waiting for a disagreement..." : "No rounds yet." }));
    }
  }

  // ------------------------------------------------------------------------------------------------ input panels
  function panel(srcName) {
    panelInput.replaceChildren();
    panelInput.dataset.src = srcName;
    const h = (t) => el("h4", { class: "ar-h", text: t });
    if (srcName === "cycle") {
      const sel = el("select", { id: "ar-lock", "aria-label": "Modulation" }, el("option", { value: -1, text: "Cycle through all 8" }), CLASSES.map((c, i) => el("option", { value: i, text: c })));
      sel.addEventListener("change", () => { cycle.setLock(+sel.value); arena.resetWindow(); });
      const nxt = el("button", { class: "btn", id: "ar-next", type: "button", text: "Next signal" });
      nxt.addEventListener("click", () => cycle.skip());
      panelInput.append(h("Source: labelled stream"), el("div", { class: "ar-opts" }, sel, nxt), el("p", { class: "ar-hint", text: "Random modulations from the same generator as the training set, one after another. This is the stream the live accuracy counters use." }));
    } else if (srcName === "key") {
      const kb = el("button", { class: "keybtn", id: "ar-key", type: "button", "data-down": "false", "aria-label": "Hold to key the carrier" }, el("span", { text: "Hold SPACE or press here" }), el("small", { text: "carrier on while held" }));
      const down = (v) => { key.setDown(v); kb.dataset.down = v; };
      kb.addEventListener("pointerdown", (e) => { kb.setPointerCapture(e.pointerId); down(true); e.preventDefault(); });
      for (const t of ["pointerup", "pointercancel", "lostpointercapture"]) kb.addEventListener(t, () => down(false));
      kb.addEventListener("keydown", (e) => { if (e.code === "Space" || e.key === "Enter") e.preventDefault(); });
      const mode = el("div", { class: "seg", role: "group", "aria-label": "Keying mode" }, ["ook", "OOK (on/off)"], ["fsk", "2FSK (two tones)"]);
      mode.replaceChildren(...[["ook", "OOK on/off"], ["fsk", "FSK two tones"]].map(([m, l]) => el("button", { type: "button", "aria-pressed": m === key.mode, "data-m": m, text: l })));
      mode.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; key.setMode(b.dataset.m); mode.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b)); arena.resetWindow(); });
      panelInput.append(h("Source: you, with a key"), kb, el("div", { class: "ar-opts", style: "margin-top:8px" }, mode), el("p", { class: "ar-hint", text: "Key it like Morse. OOK: the carrier is on while you hold. FSK: holding switches to the upper tone, releasing to the lower. Slow the speed to watch it propagate." }));
      panelInput._down = down;
    } else if (srcName === "mic") {
      const status = el("p", { class: "ar-hint", id: "ar-mic-status", role: "status" });
      const en = el("button", { class: "btn primary", id: "ar-mic-en", type: "button", text: "Enable microphone" });
      const lvl = el("div", { class: "meter" }, el("i", { id: "ar-mic-lvl" }));
      const mode = el("div", { class: "seg", role: "group", "aria-label": "Microphone mapping" });
      mode.replaceChildren(...[["ook", "Loudness to OOK"], ["fsk", "Pitch to FSK"]].map(([m, l]) => el("button", { type: "button", "aria-pressed": m === mic.mode, "data-m": m, text: l })));
      mode.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; mic.setMode(b.dataset.m); mode.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b)); arena.resetWindow(); });
      const lv = el("select", { id: "ar-mic-levels", "aria-label": "FSK levels" }, el("option", { value: 2, text: "2 tones" }), el("option", { value: 4, selected: true, text: "4 tones" }));
      lv.addEventListener("change", () => { mic.nLevels = +lv.value; });
      const sens = el("input", { type: "range", id: "ar-mic-sens", min: 0.3, max: 6, step: 0.1, value: mic.sens, "aria-label": "Microphone sensitivity" });
      sens.addEventListener("input", () => { mic.sens = +sens.value; });
      const other = el("button", { class: "btn", type: "button", hidden: true, id: "ar-mic-other", text: "Use the cycling stream instead" });
      other.addEventListener("click", () => setSource("cycle"));
      const setStatus = () => {
        if (mic.state === "running") { status.textContent = "Listening. Hum or whistle: a loud sound turns the carrier on (OOK); a steady pitch picks a tone (FSK)."; en.hidden = true; other.hidden = true; }
        else if (mic.state === "asking") status.textContent = "Waiting for the browser permission prompt...";
        else if (mic.error) { status.textContent = mic.error + " The cycling stream is still available."; other.hidden = false; en.textContent = "Try again"; }
        else status.textContent = "The microphone is only used locally to set the carrier amplitude and pitch. Nothing is recorded or sent anywhere.";
      };
      en.addEventListener("click", async () => { mic.state = "asking"; setStatus(); await mic.start(); setStatus(); });
      panelInput.append(h("Source: your voice or a whistle"), el("div", { class: "ar-opts" }, en, mode), el("div", { class: "ar-opts", style: "margin-top:8px" }, lv, el("label", {}, "Sensitivity", sens)), el("div", { style: "margin-top:8px" }, lvl, el("div", { class: "ar-hint mono", id: "ar-mic-read", text: "" })), status, other);
      setStatus();
      panelInput._setStatus = setStatus;
    } else if (srcName === "draw") {
      const pad = el("canvas", { class: "pad", id: "ar-pad", role: "img", "aria-label": "Draw an I/Q trajectory with a mouse or finger. I is horizontal, Q vertical." });
      const presets = el("div", { class: "ar-opts" }, [["circle", "Circle"], ["square", "Square"], ["eight", "Figure 8"], ["star", "Star"], ["clear", "Clear"]].map(([p, l]) => { const b = el("button", { class: "btn", type: "button", text: l, "data-p": p }); b.addEventListener("click", () => { draw.setPreset(p); arena.resetWindow(); }); return b; }));
      const laps = el("select", { id: "ar-laps", "aria-label": "Laps per window" }, [1, 2, 4, 8].map((n) => el("option", { value: n, text: n + (n === 1 ? " lap per window" : " laps per window") })));
      laps.addEventListener("change", () => { draw.setLaps(+laps.value); arena.resetWindow(); });
      presets.append(laps);
      panelInput.append(h("Source: your I/Q drawing"), el("div", { class: "padwrap" }, pad, presets), el("p", { class: "ar-hint", text: "Draw the path of the carrier in the I/Q plane. It loops continuously: a circle is a pure tone, jumps between points look like phase shifts, a ring of dots is a PSK constellation. Shapes that are not modulations are the best way to fool the nets." }));
      setupPad(pad);
    }
  }

  function setupPad(pad) {
    const ctx = pad.getContext("2d");
    let cur = null, drawing = false;
    const resize = () => { const r = pad.getBoundingClientRect(), dpr = window.devicePixelRatio || 1; pad.width = Math.round(r.width * dpr); pad.height = Math.round(r.height * dpr); pad._dpr = dpr; };
    resize();
    new ResizeObserver(resize).observe(pad);
    const toUnit = (e) => { const r = pad.getBoundingClientRect(), s = Math.min(r.width, r.height) / 2; return [((e.clientX - r.left) - r.width / 2) / s, -((e.clientY - r.top) - r.height / 2) / s]; };
    pad.addEventListener("pointerdown", (e) => { pad.setPointerCapture(e.pointerId); drawing = true; cur = [toUnit(e)]; e.preventDefault(); });
    pad.addEventListener("pointermove", (e) => { if (!drawing) return; const p = toUnit(e), l = cur[cur.length - 1]; if (Math.hypot(p[0] - l[0], p[1] - l[1]) > 0.012) cur.push(p); });
    const end = () => { if (!drawing) return; drawing = false; if (cur.length >= 6) { draw.setPoints(cur); arena.resetWindow(); } cur = null; };
    pad.addEventListener("pointerup", end); pad.addEventListener("pointercancel", end);
    pad._render = () => {
      const dpr = pad._dpr || 1, W = pad.width, H = pad.height, s = Math.min(W, H) / 2;
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = cssVar("--scope-grid"); ctx.lineWidth = dpr;
      ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.stroke();
      ctx.beginPath(); ctx.arc(W / 2, H / 2, s * 0.8, 0, 7); ctx.stroke();
      const X = (p) => W / 2 + p[0] * s, Y = (p) => H / 2 - p[1] * s;
      const pts = drawing && cur ? cur : draw.pts;
      ctx.lineWidth = 2.5 * dpr; ctx.lineJoin = "round"; ctx.strokeStyle = cssVar("--scope-i");
      if (pts.length) { ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(X(p), Y(p)) : ctx.moveTo(X(p), Y(p)))); ctx.stroke(); }
      if (!drawing && draw.path) {
        const lap = Math.max(1, Math.round(128 / draw.laps)), k = ((draw.pos - 1 + 128 * 64) % lap) * (128 / lap) | 0;
        const gx = W / 2 + draw.path.i[k % 128] * s * 0.0 , _ = gx; // marker is drawn in normalised power units below
        ctx.fillStyle = cssVar("--accent");
        const pw = draw.pts.length ? Math.max(...draw.pts.map((p) => Math.hypot(p[0], p[1]))) : 1;
        // the path array is power-normalised; map back to the drawn scale for the dot
        const idx = draw.pts.length ? Math.min(draw.pts.length - 1, Math.floor((k / 127) * (draw.pts.length - 1))) : 0;
        const pp = draw.pts[idx] || [0, 0]; ctx.beginPath(); ctx.arc(X(pp), Y(pp), 5 * dpr, 0, 7); ctx.fill(); void pw;
      }
      ctx.fillStyle = cssVar("--muted"); ctx.font = `${10 * dpr}px ${cssVar("--font-mono")}`; ctx.fillText("I", W - 12 * dpr, H / 2 - 4 * dpr); ctx.fillText("Q", W / 2 + 5 * dpr, 12 * dpr);
    };
    panelInput._pad = pad;
  }

  function setSource(name) {
    if (!sources[name]) return;
    arena.setSource(sources[name]);
    stats.source = name;
    tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", b.dataset.src === name));
    panel(name);
    if (name !== "key") sources.key.setDown(false);
    if (name === "key" && document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
    arena.resetScores();
    renderNow();
  }
  tabs.addEventListener("click", (e) => { const b = e.target.closest("button[data-src]"); if (b) setSource(b.dataset.src); });
  tabs.addEventListener("keydown", (e) => {
    const bs = [...tabs.querySelectorAll("button")], i = bs.indexOf(document.activeElement);
    if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && i >= 0) { const n = bs[(i + (e.key === "ArrowRight" ? 1 : bs.length - 1)) % bs.length]; n.focus(); n.click(); e.preventDefault(); }
  });

  // space bar keying (only while the arena is on screen and the key source is selected)
  let inView = true;
  new IntersectionObserver((es) => { inView = es[0].isIntersecting; }, { threshold: 0.05 }).observe(host);
  const typing = (t) => t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName) && t.type !== "range" && t.type !== "checkbox";
  addEventListener("keydown", (e) => {
    if (e.code !== "Space" || stats.source !== "key" || !inView || typing(e.target)) return;
    e.preventDefault(); if (!e.repeat) panelInput._down && panelInput._down(true);
  });
  addEventListener("keyup", (e) => { if (e.code === "Space" && stats.source === "key") { e.preventDefault(); panelInput._down && panelInput._down(false); } });
  addEventListener("blur", () => panelInput._down && panelInput._down(false));

  // ------------------------------------------------------------------------------------------------ transport
  function syncPlay() { btnPlay.textContent = arena.paused ? "Play" : "Pause"; btnPlay.setAttribute("aria-pressed", arena.paused); btnStep.disabled = false; }
  btnPlay.addEventListener("click", () => { arena.paused = !arena.paused; syncPlay(); });
  btnStep.addEventListener("click", () => { if (!arena.paused) { arena.paused = true; syncPlay(); } arena.step(); });
  speed.addEventListener("change", () => { arena.rate = BASE_RATE * +speed.value; });
  if (reducedMotion()) { arena.paused = true; syncPlay(); }

  // ------------------------------------------------------------------------------------------------ DLNN inspector (D3)
  const L0 = dl.model.layers[0];
  const tapW = (o, i, k) => L0.W[(o * L0.I + i) * L0.K + k];
  let geo = null, svg = null, sel = 0, strips = null, dlDirty = true;
  const stripCanvases = [];
  function buildInspector() {
    dlPanel.replaceChildren();
    const W = Math.max(300, Math.floor(dlPanel.getBoundingClientRect().width || host.getBoundingClientRect().width - 24));
    const two = W >= 1000, svgW = two ? Math.floor(W * 0.58) : W - 24, small = svgW < 520;
    const left = 44, right = 12, K = L0.K, gap = small ? 4 : 8, cw = Math.max(18, (svgW - left - right - gap * (K - 1)) / K), ch = small ? 22 : 28;
    const rowY = [small ? 30 : 36, small ? 30 + ch + 14 : 36 + ch + 18];
    const sumY = rowY[1] + ch + (small ? 62 : 74), outY = sumY + (small ? 56 : 64);
    const H = outY + 30;
    geo = { svgW, H, left, cw, ch, gap, rowY, sumY, outY, small };
    const wrap = el("div", { class: "dl-wrap" });
    const sw = el("div", {});
    const strWrap = el("div", { class: "strips", "aria-label": "Layer activity over the last window" });
    wrap.style.gridTemplateColumns = two ? "minmax(0,7fr) minmax(0,5fr)" : "minmax(0,1fr)";
    wrap.append(sw, strWrap);
    dlPanel.append(el("h4", { class: "ar-h" }, el("span", { text: "Inside the DLNN: layer 1, tap by tap" }), el("span", { class: "mono", id: "dl-sel", text: "" })),
      el("p", { class: "ar-hint", style: "margin:0 0 8px", text: "Samples shift left to right along each delay line. Each tap is wired + (warm), - (cool) or off for the selected layer-1 output; wire weight shows how much it adds right now. Click an output node to follow another line. Use Step to advance one tap time." }), wrap);
    svg = d3.select(sw).append("svg").attr("class", "dl").attr("viewBox", `0 0 ${svgW} ${H}`).attr("role", "img").attr("aria-label", "Layer 1 of the DLNN: two input delay lines with eight taps each, summed into eight tanh outputs");
    const cellX = (k) => left + k * (cw + gap);
    const g = svg.append("g");
    // labels
    ["I", "Q"].forEach((t, i) => g.append("text").attr("x", 14).attr("y", rowY[i] + ch / 2 + 4).attr("font-weight", 700).attr("font-size", 14).text(t));
    for (let k = 0; k < K; k++) g.append("text").attr("x", cellX(k) + cw / 2).attr("y", rowY[0] - 8).attr("text-anchor", "middle").attr("font-family", "var(--font-mono)").attr("font-size", 10).text(k === 0 ? "now" : "z-" + k);
    g.append("g").attr("class", "wires");
    const cells = g.append("g");
    const cellSel = cells.selectAll("g.cell").data(d3.cross([0, 1], d3.range(K))).join("g").attr("class", "cell").attr("transform", (d) => `translate(${cellX(d[1])},${rowY[d[0]]})`);
    cellSel.append("rect").attr("width", cw).attr("height", ch).attr("rx", 4);
    cellSel.append("text").attr("class", "tapmark").attr("x", cw / 2).attr("y", ch + 13);
    const cv = cellSel.append("text").attr("class", "val simple-hide").attr("x", cw / 2).attr("y", ch / 2 + 3.5).attr("text-anchor", "middle").attr("font-family", "var(--font-mono)").attr("font-size", 9);
    void cv;
    g.append("circle").attr("class", "sumnode").attr("cx", svgW / 2 - 30).attr("cy", sumY).attr("r", 15).attr("fill", "var(--surface)").attr("stroke", "var(--ink-2)");
    g.append("text").attr("x", svgW / 2 - 30).attr("y", sumY + 5).attr("text-anchor", "middle").attr("font-size", 16).attr("font-weight", 700).text("Σ");
    g.append("path").attr("d", `M${svgW / 2 - 14},${sumY} H${svgW / 2 + 14}`).attr("stroke", "var(--ink-2)").attr("marker-end", "none");
    const tx = svgW / 2 + 18;
    g.append("rect").attr("x", tx).attr("y", sumY - 15).attr("width", 46).attr("height", 30).attr("rx", 6).attr("fill", "var(--surface)").attr("stroke", "var(--ink-2)");
    g.append("path").attr("d", `M${tx + 6},${sumY + 9} C${tx + 20},${sumY + 9} ${tx + 24},${sumY - 9} ${tx + 40},${sumY - 9}`).attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 2).style("filter", "var(--glow)");
    g.append("text").attr("class", "readout").attr("x", tx + 56).attr("y", sumY + 4).attr("font-family", "var(--font-mono)").attr("font-size", 11);
    g.append("text").attr("x", left).attr("y", outY - 16).attr("font-size", 10).attr("letter-spacing", "0.08em").text("LAYER 1 OUTPUTS (CLICK ONE)");
    const nodes = g.append("g").selectAll("g.node").data(d3.range(L0.O)).join("g").attr("class", "node").attr("tabindex", 0).attr("role", "button").attr("aria-label", (o) => "Follow layer 1 output " + (o + 1))
      .attr("transform", (o) => `translate(${left + 12 + o * ((svgW - left - right - 24) / (L0.O - 1))},${outY + 6})`)
      .on("click", (e, o) => { sel = o; dlDirty = true; }).on("keydown", (e, o) => { if (e.key === "Enter" || e.key === " ") { sel = o; dlDirty = true; e.preventDefault(); } });
    nodes.append("circle").attr("r", small ? 9 : 11);
    nodes.append("text").attr("y", small ? 22 : 25).attr("text-anchor", "middle").attr("font-family", "var(--font-mono)").attr("font-size", 9).text((o) => "o" + o);
    geo.cellX = cellX;
    // activity strips for layers 1-3 from the latest classified window
    stripCanvases.length = 0;
    dl.model.layers.forEach((ly, l) => {
      const c = el("canvas", { width: 128, height: ly.O, role: "img", "aria-label": `Layer ${l + 1} activity: ${ly.O} lines over the last 128 samples` });
      stripCanvases.push(c);
      strWrap.append(el("div", { class: "strip" }, el("span", { text: "L" + (l + 1) }), c));
    });
    strWrap.append(el("p", { class: "ar-hint", style: "margin:2px 0 0", text: "Rows are the lines of each layer, time runs left to right over the last classified window. Warm is positive, cool is negative." }));
    strips = strWrap; dlDirty = true;
  }

  const pc = { pos: [0, 0, 0], neg: [0, 0, 0], bg: [0, 0, 0] };
  function themeColors() { pc.pos = hex2rgb(cssVar("--pos")); pc.neg = hex2rgb(cssVar("--neg")); pc.bg = hex2rgb(cssVar("--scope-bg")); }
  function drawStrips() {
    if (!dl.lastTrace) return;
    dl.model.layers.forEach((ly, l) => {
      const c = stripCanvases[l]; if (!c) return;
      const ctx = c.getContext("2d"), img = ctx.createImageData(128, ly.O), y = dl.lastTrace[l];
      for (let o = 0; o < ly.O; o++) for (let n = 0; n < 128; n++) {
        const v = Math.max(-1, Math.min(1, y[o * 128 + n])), a = Math.abs(v), col = v >= 0 ? pc.pos : pc.neg, p = (o * 128 + n) * 4;
        img.data[p] = pc.bg[0] + (col[0] - pc.bg[0]) * a; img.data[p + 1] = pc.bg[1] + (col[1] - pc.bg[1]) * a; img.data[p + 2] = pc.bg[2] + (col[2] - pc.bg[2]) * a; img.data[p + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
    });
  }

  /** value of channel i (0 = I, 1 = Q) `age` samples ago (0 = newest); 0 before the stream has filled that far */
  const sampleAt = (i, age) => (age >= arena.filled ? 0 : (i ? arena.ringQ : arena.ringI)[(arena.head - 1 - age + 256) % 128]);
  const polColor = (w) => (w > 0 ? "var(--pos)" : w < 0 ? "var(--neg)" : "var(--zero)");
  const glyph = (w) => (w > 0 ? "+" : w < 0 ? "−" : "·");
  let lastTraceN = -1;
  function updateInspector() {
    if (!svg) return;
    const { cellX, cw, ch, rowY, sumY, svgW } = geo, K = L0.K, o = sel;
    const sx = svgW / 2 - 30, contrib = [];
    let u = 0;
    for (let i = 0; i < L0.I; i++) for (let k = 0; k < K; k++) {
      const w = tapW(o, i, k), x = sampleAt(i, k * L0.dilation), c = w * x; u += c; contrib.push({ i, k, w, x, c });
    }
    const y = L0.gain[o] * Math.tanh((u + L0.bias[o]) / L0.vsat);
    const peak = Math.max(0.5, ...contrib.map((q) => Math.abs(q.c))), xs = 1 / Math.max(1, ...contrib.map((q) => Math.abs(q.x)));
    svg.selectAll("g.cell").each(function (d) {
      const [i, k] = d, w = tapW(o, i, k), x = sampleAt(i, k * L0.dilation), a = Math.min(1, Math.abs(x) * xs * 0.9);
      const gsel = d3.select(this);
      gsel.select("rect").attr("fill", x >= 0 ? `color-mix(in srgb, var(--scope-i) ${Math.round(a * 85)}%, var(--surface))` : `color-mix(in srgb, var(--scope-q) ${Math.round(a * 85)}%, var(--surface))`);
      gsel.select(".tapmark").attr("fill", polColor(w)).text(glyph(w));
      gsel.select(".val").attr("fill", "var(--ink)").text((x >= 0 ? "+" : "") + x.toFixed(1));
    });
    const wires = svg.select("g.wires").selectAll("path").data(contrib, (d) => d.i * 8 + d.k);
    wires.join("path").attr("class", "wire")
      .attr("d", (d) => { const x0 = cellX(d.k) + cw / 2, y0 = rowY[d.i] + ch + 16, yy = sumY - 15; return `M${x0},${y0} C${x0},${y0 + 22} ${sx},${yy - 26} ${sx},${yy}`; })
      .attr("stroke", (d) => polColor(d.w)).attr("stroke-dasharray", (d) => (d.w === 0 ? "2 4" : null))
      .attr("stroke-width", (d) => (d.w === 0 ? 0.8 : 0.8 + 4.2 * Math.min(1, Math.abs(d.c) / peak)))
      .attr("opacity", (d) => (d.w === 0 ? 0.35 : 0.35 + 0.6 * Math.min(1, Math.abs(d.c) / peak)));
    svg.select(".sumnode").attr("fill", u >= 0 ? `color-mix(in srgb, var(--pos) ${Math.min(80, Math.abs(u) * 25)}%, var(--surface))` : `color-mix(in srgb, var(--neg) ${Math.min(80, Math.abs(u) * 25)}%, var(--surface))`);
    svg.select(".readout").text(`u=${u >= 0 ? "+" : ""}${u.toFixed(2)}  y=${y >= 0 ? "+" : ""}${y.toFixed(2)}`);
    // output nodes: all 8 layer-1 outputs at the newest sample
    svg.selectAll("g.node").each(function (oo) {
      let uu = 0;
      for (let i = 0; i < L0.I; i++) for (let k = 0; k < K; k++) { const w = tapW(oo, i, k); if (w) uu += w * sampleAt(i, k * L0.dilation); }
      const yy = L0.gain[oo] * Math.tanh((uu + L0.bias[oo]) / L0.vsat), a = Math.round(Math.min(1, Math.abs(yy)) * 90);
      const n = d3.select(this).classed("sel", oo === o);
      n.select("circle").attr("fill", `color-mix(in srgb, var(${yy >= 0 ? "--pos" : "--neg"}) ${a}%, var(--surface))`);
    });
    const ds = document.getElementById("dl-sel"); if (ds) ds.textContent = `following o${o}: ${contrib.filter((q) => q.w > 0).length} taps +, ${contrib.filter((q) => q.w < 0).length} taps −, ${contrib.filter((q) => q.w === 0).length} off`;
    if (dl.traceN !== lastTraceN) { lastTraceN = dl.traceN; drawStrips(); }
  }

  // ------------------------------------------------------------------------------------------------ scope canvas
  const sctx = scope.getContext("2d");
  let sdpr = 1;
  function resizeScope() { const r = scope.getBoundingClientRect(); sdpr = window.devicePixelRatio || 1; scope.width = Math.round(r.width * sdpr); scope.height = Math.round(r.height * sdpr); }
  resizeScope(); new ResizeObserver(resizeScope).observe(scope);
  function renderScope() {
    const W = scope.width, H = scope.height, d = sdpr, cw = Math.min(H, W * 0.38), tw = W - cw - 10 * d;
    sctx.setTransform(1, 0, 0, 1, 0, 0); sctx.clearRect(0, 0, W, H);
    sctx.strokeStyle = cssVar("--scope-grid"); sctx.lineWidth = d;
    sctx.beginPath(); sctx.moveTo(0, H / 2); sctx.lineTo(tw, H / 2); sctx.moveTo(tw + 10 * d + cw / 2, 0); sctx.lineTo(tw + 10 * d + cw / 2, H); sctx.moveTo(tw + 10 * d, H / 2); sctx.lineTo(W, H / 2); sctx.stroke();
    const fil = arena.filled, amp = 3.2;
    const trace = (arr, col) => {
      sctx.strokeStyle = col; sctx.lineWidth = 1.6 * d; sctx.beginPath();
      for (let n = 0; n < 128; n++) {
        const v = n < 128 - fil ? 0 : arr[(arena.head + n) % 128], x = (n / 127) * tw, y = H / 2 - (v / amp) * (H / 2);
        n ? sctx.lineTo(x, Math.max(2, Math.min(H - 2, y))) : sctx.moveTo(x, y);
      }
      sctx.stroke();
    };
    sctx.globalAlpha = 0.95; trace(arena.ringI, cssVar("--scope-i")); trace(arena.ringQ, cssVar("--scope-q")); sctx.globalAlpha = 1;
    const cx = tw + 10 * d + cw / 2, cy = H / 2, sc = (cw / 2) / amp;
    for (let n = Math.max(0, 128 - fil); n < 128; n++) {
      const k = (arena.head + n) % 128, a = 0.12 + 0.88 * (n / 127);
      sctx.fillStyle = cssVar("--accent"); sctx.globalAlpha = a * 0.9;
      sctx.beginPath(); sctx.arc(cx + arena.ringI[k] * sc, cy - arena.ringQ[k] * sc, (n > 120 ? 3 : 1.8) * d, 0, 7); sctx.fill();
    }
    sctx.globalAlpha = 1; sctx.fillStyle = cssVar("--muted"); sctx.font = `${10 * d}px ${cssVar("--font-mono")}`;
    sctx.fillText("I", 4 * d, 12 * d); sctx.fillStyle = cssVar("--scope-q"); sctx.fillText("Q", 18 * d, 12 * d); sctx.fillStyle = cssVar("--muted"); sctx.fillText("I/Q", tw + 14 * d, 12 * d);
  }

  // ------------------------------------------------------------------------------------------------ per-frame UI
  function renderNow() {
    const s = arena.source, t = arena.truth;
    let html;
    if (s.name === "cycle") html = `<span>Sending</span><span class="big">${t >= 0 ? CLASSES[t] : "..."}</span><span class="mono">at ${arena.snr} dB, CFO ${arena.cfo >= 0 ? "+" : ""}${arena.cfo.toFixed(3)}</span>`;
    else if (s.name === "key") html = `<span>You are keying</span><span class="big">${key.hint}</span><span class="mono">${key.down ? "key down" : "key up"}</span>`;
    else if (s.name === "mic") html = `<span>Microphone</span><span class="big">${mic.mode === "ook" ? (mic.on ? "on" : "quiet") : mic.pitch ? Math.round(mic.pitch) + " Hz" : "no pitch"}</span><span class="mono">${mic.state}</span>`;
    else html = `<span>Your drawing</span><span class="big">${draw.path ? "looping" : "empty"}</span><span class="mono">${draw.laps} lap${draw.laps > 1 ? "s" : ""} / window</span>`;
    if (nowEl._h !== html) { nowEl._h = html; nowEl.innerHTML = html; }
  }

  function renderBoard() {
    for (const m of arena.models) {
      const r = mrows[m.id], probs = m.probs, pred = m.ready ? m.pred : -1, truth = arena.source.labelled && arena.lastResult ? arena.lastResult.label : -1;
      r.bars.forEach((b, i) => {
        const p = m.ready ? probs[i] : 0, h = Math.round(p * 100), key2 = h + ":" + (i === pred) + ":" + (i === truth);
        if (r.last[i] === key2) return;
        r.last[i] = key2;
        b.firstChild.textContent = i === pred ? Math.round(p * 100) + "" : "";
        b.children[1].style.height = h + "%";
        b.classList.toggle("pred", i === pred); b.classList.toggle("truth", i === truth);
      });
      const acc = arena.accuracy(m), a = acc == null ? "--" : pct(acc) + (m.recent.length < 10 ? "*" : "");
      if (r.accEl.textContent !== a) r.accEl.textContent = a;
      const ms = m.emaMs ? (m.emaMs < 1 ? m.emaMs.toFixed(2) : m.emaMs.toFixed(1)) + " ms" : "--";
      if (r.msEl.textContent !== ms) r.msEl.textContent = ms;
    }
  }

  // ------------------------------------------------------------------------------------------------ main loop
  foot.innerHTML = `Accuracy counters use the last 60 windows that lie entirely inside one labelled segment (the cycling stream); * means fewer than 10 so far. Key, mic and draw streams are unlabelled. <b>Energy</b> is a digital-equivalent arithmetic estimate (${arena.energy ? "45 nm, Horowitz ISSCC 2014 per-operation costs, hover for caveats" : "pending: data/energy_refs.json not found"}); <b>time</b> is measured JavaScript compute in this browser, not hardware latency. Trained test accuracies (4000 held-out windows, mixed SNR): ${models.map((m) => `${m.label} ${pct(m.acc)}${m.provisional ? " (provisional)" : ""}`).join(", ")}. <b>on MCU</b> uses the verified STM32G474 run current (datasheet DS12288 Table 21: ${mc ? mc.mA + " mA at " + mc.MHz + " MHz" : "n/a"}) with an assumed ${mc ? mc.vdd + " V" : "supply"} and an assumed cycle count per operation, so it is a floor. <b>Reading the race honestly:</b> on accuracy the 1-D CNN and the cumulant classifier beat the width-8 DLNN. The DLNN's case is cost, not accuracy: ternary taps need adds only (see the multiplies in each ops count) and the structure maps onto a physical delay line. <b>DLNN-B</b> is the board-trained net run on the simulated board line (line plus input RC only, from the M1a SPICE deck: not the op-amps, noise or mismatch). The <b>feature MLP</b> is the fair conventional competitor at matched size: it is given hand-built phase-invariant features, which the raw-I/Q MLP is not. All models are simulated.`;
  setSource("cycle");
  buildInspector(); themeColors();
  new ResizeObserver(() => { const w = Math.round(dlPanel.getBoundingClientRect().width); if (buildInspector._w !== w) { buildInspector._w = w; buildInspector(); lastTraceN = -1; } }).observe(dlPanel);
  addEventListener("dlnn-theme", () => { themeColors(); lastTraceN = -1; });
  let last = performance.now(), fpsT = last, fpsN = 0, uiT = 0;
  const hv = document.getElementById("ar-fps");
  function frame(t) {
    const dt = t - last; last = t;
    if (inView && !document.hidden) {
      if (dt > 40) stats.longFrames++;
      const w0 = performance.now();
      if (stats.source === "mic") mic.poll();
      arena.tick(dt);
      renderScope();
      if (panelInput._pad) panelInput._pad._render();
      renderBoard();
      updateInspector();
      uiT += dt;
      if (uiT > 120) {
        uiT = 0; renderNow(); renderChallenge();
        if (stats.source === "mic") {
          const l = document.getElementById("ar-mic-lvl"), rd = document.getElementById("ar-mic-read");
          if (l) l.style.width = Math.min(100, mic.level * mic.sens * 400) + "%";
          if (rd) rd.textContent = mic.state === "running" ? `level ${mic.level.toFixed(3)} ${mic.on ? "(on)" : "(gate)"}  pitch ${mic.pitch ? Math.round(mic.pitch) + " Hz" : "--"}` : "";
          if (panelInput._setStatus && mic.state !== panelInput._lastMicState) { panelInput._lastMicState = mic.state; panelInput._setStatus(); }
        }
      }
      const wm = performance.now() - w0; stats.workMsAvg = (stats.workMsAvg || wm) * 0.95 + wm * 0.05; stats.workMsMax = Math.max(stats.workMsMax || 0, wm);
      fpsN++; stats.frames++;
      if (t - fpsT >= 1000) { stats.fps = Math.round((fpsN * 1000) / (t - fpsT)); fpsN = 0; fpsT = t; if (hv && new URLSearchParams(location.search).has("debug")) hv.textContent = stats.fps + " fps"; }
    } else { fpsT = t; fpsN = 0; }
    requestAnimationFrame(frame);
  }
  syncPlay();
  requestAnimationFrame(frame);
  stats.ready = true;
}
