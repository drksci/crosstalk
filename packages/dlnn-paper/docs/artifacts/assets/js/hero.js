// hero.js: p5.js sketches (instance mode, vendored p5 1.11.3, LGPL-2.1).
//  mountHero:     pulses travel down a tapped delay line; each tap is wired +1 / -1 / 0 into a summing node; the sum is plotted live.
//  mountMercury:  the lineage nod: a mercury delay line recirculating a pulse train (transducer, tube, amplifier, back to the start).
// Light: pauses off-screen and in hidden tabs, one still frame under prefers-reduced-motion, colours come from the CSS tokens.
import { loadP5, cssVar, reducedMotion } from "./common.js";

const W = [1, 0, -1, 1, 1, 0, -1, 1]; // example ternary weights (decorative; the real ones are in W1)
const col = (n, fb) => cssVar(n) || fb;

function visibility(host, p) {
  const io = new IntersectionObserver((es) => { for (const e of es) { host._vis = e.isIntersecting; } }, { rootMargin: "80px" });
  io.observe(host); host._vis = true;
  const upd = () => { const on = host._vis && !document.hidden && !reducedMotion(); on ? p.loop() : p.noLoop(); };
  setInterval(upd, 400);
  window.addEventListener("dlnn-theme", () => p.redraw());
}

export async function mountHero(host) {
  const P5 = await loadP5();
  host.replaceChildren();
  host.setAttribute("role", "img");
  host.setAttribute("aria-label", "Animation: pulses travel down a delay line, each tap is wired plus, minus or off into a summing node, and the summed output is traced on a scope.");
  new P5((p) => {
    const pulses = [], hist = new Array(240).fill(0);
    let t = 0, nextAt = 0, w = 800, h = 210;
    const rnd = (() => { let s = 12345; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); })();
    const tapX = (k) => 0.05 * w + (k + 0.5) * (0.56 * w / 8);
    const lineY = () => h * 0.3, busY = () => h * 0.74;
    function step(dt) {
      t += dt;
      if (t >= nextAt) { pulses.push({ x: 0.03 * w, a: [1, -1, 1, 0.6, -0.6][Math.floor(rnd() * 5)] }); nextAt = t + 0.6 + rnd() * 0.9; }
      const v = 0.18 * w; // px per second
      for (const q of pulses) q.x += v * dt;
      while (pulses.length && pulses[0].x > 0.66 * w) pulses.shift();
      let y = 0;
      for (const q of pulses) for (let k = 0; k < 8; k++) y += W[k] * q.a * Math.exp(-(((q.x - tapX(k)) / (0.018 * w)) ** 2));
      hist.push(y); hist.shift();
    }
    p.setup = () => {
      w = host.clientWidth || 800; h = w < 560 ? 190 : 220;
      p.createCanvas(w, h).parent(host); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1)); p.frameRate(40);
      for (let i = 0; i < 260; i++) step(1 / 40); // warm-up so the first (and reduced-motion) frame is populated
    };
    p.windowResized = () => { w = host.clientWidth || w; h = w < 560 ? 190 : 220; p.resizeCanvas(w, h); };
    p.draw = () => {
      step(Math.min(0.05, p.deltaTime / 1000 || 0.025));
      p.clear();
      const ink = col("--ink", "#222"), muted = col("--muted", "#777"), pos = col("--pos", "#d9480f"), neg = col("--neg", "#0e7490"), zero = col("--zero", "#999"), acc = col("--accent", "#0a7a5c"), rule = col("--rule", "#ccc");
      const ly = lineY(), by = busY(), nodeX = 0.66 * w + 20;
      // delay line: LC ladder (alternating coil bumps)
      p.noFill(); p.stroke(rule); p.strokeWeight(2);
      p.beginShape(); for (let x = 0.03 * w; x <= 0.64 * w; x += 3) p.vertex(x, ly + 5 * Math.sin(x * 0.32)); p.endShape();
      p.fill(muted); p.noStroke(); p.textFont("Hanken Grotesk, system-ui, sans-serif"); p.textSize(11);
      p.text("input", 0.03 * w, ly - 24); p.text("delay line, tapped", 0.2 * w, ly - 24);
      // taps -> bus
      for (let k = 0; k < 8; k++) {
        const x = tapX(k), c = W[k] > 0 ? pos : W[k] < 0 ? neg : zero;
        p.stroke(c); p.strokeWeight(W[k] === 0 ? 1 : 2.2); p.drawingContext.setLineDash(W[k] === 0 ? [3, 4] : []);
        p.line(x, ly + 7, x, by); p.drawingContext.setLineDash([]);
        p.noStroke(); p.fill(c); p.circle(x, ly, 7);
        p.fill(c); p.textAlign(p.CENTER); p.textSize(12); p.text(W[k] > 0 ? "+" : W[k] < 0 ? "−" : "0", x, ly + 24); p.textAlign(p.LEFT);
      }
      p.stroke(ink); p.strokeWeight(2); p.line(tapX(0), by, nodeX - 12, by);
      p.noFill(); p.circle(nodeX, by, 24); p.noStroke(); p.fill(ink); p.textAlign(p.CENTER, p.CENTER); p.textSize(15); p.text("Σ", nodeX, by); p.textAlign(p.LEFT, p.BASELINE);
      // pulses on the line; a bead drops down every active tap while the pulse sits on it
      for (const q of pulses) {
        const c = q.a >= 0 ? pos : neg;
        p.noStroke(); p.fill(c); p.drawingContext.shadowBlur = 0;
        p.circle(q.x, ly, 6 + 7 * Math.abs(q.a)); p.fill(p.color(c + "40")); p.circle(q.x, ly, 16 + 10 * Math.abs(q.a));
        for (let k = 0; k < 8; k++) {
          const d = Math.abs(q.x - tapX(k)); if (!W[k] || d > 0.05 * w) continue;
          const f = 1 - d / (0.05 * w);
          const sgn = W[k] * q.a, cc = sgn >= 0 ? pos : neg, yy = ly + (by - ly) * (1 - f) * 0.9 + 6;
          p.fill(cc); p.circle(tapX(k), yy, 3 + 4 * f * Math.abs(q.a));
        }
      }
      // scope of the summed output
      const sx = 0.72 * w, sw = 0.25 * w, sy0 = h * 0.5, sh = h * 0.34;
      p.noFill(); p.stroke(rule); p.strokeWeight(1); p.rect(sx, h * 0.12, sw, h * 0.76, 6); p.line(sx, sy0, sx + sw, sy0);
      p.stroke(acc); p.strokeWeight(2); p.beginShape();
      for (let i = 0; i < hist.length; i++) p.vertex(sx + (i / (hist.length - 1)) * sw, sy0 - Math.max(-1.4, Math.min(1.4, hist[i])) * sh * 0.62);
      p.endShape();
      p.noStroke(); p.fill(muted); p.textSize(11); p.text("summed output", sx + 4, h * 0.12 - 5);
      p.stroke(acc); p.strokeWeight(1.2); p.line(nodeX + 12, by, sx, by); p.line(sx, by, sx, sy0 + 0); // sum feeds the scope
    };
    if (reducedMotion()) p.noLoop(); else visibility(host, p);
  });
}

export async function mountMercury(host) {
  const P5 = await loadP5();
  host.replaceChildren();
  host.setAttribute("role", "img");
  host.setAttribute("aria-label", "Animation: a mercury delay line. A pulse train is launched by a quartz transducer, crosses a tube of mercury as a sound wave, is picked up, amplified and re-injected, so the bits recirculate.");
  new P5((p) => {
    let w = 700, h = 150, t = 0;
    p.setup = () => { w = host.clientWidth || 700; p.createCanvas(w, h).parent(host); p.pixelDensity(Math.min(2, window.devicePixelRatio || 1)); p.frameRate(40); t = 3; };
    p.windowResized = () => { w = host.clientWidth || w; p.resizeCanvas(w, h); };
    p.draw = () => {
      t += Math.min(0.05, p.deltaTime / 1000 || 0.025);
      p.clear();
      const ink = col("--ink", "#222"), muted = col("--muted", "#777"), acc = col("--accent", "#0a7a5c"), rule = col("--rule", "#ccc"), tier = col("--tier-C", "#eda100");
      const x0 = 0.12 * w, x1 = 0.88 * w, y = 52, hh = 26;
      p.noStroke(); p.fill(p.color(rule + "80")); p.rect(x0, y - hh / 2, x1 - x0, hh, 13);
      p.fill(muted); p.rect(x0 - 8, y - hh / 2 - 3, 8, hh + 6, 2); p.rect(x1, y - hh / 2 - 3, 8, hh + 6, 2); // transducers
      const bits = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1];
      const cyc = 6, ph = (t % cyc) / cyc; // one trip
      bits.forEach((b, i) => {
        if (!b) return;
        const x = x0 + (((ph + i / bits.length * 0.8) % 1)) * (x1 - x0);
        p.noFill(); p.stroke(tier); p.strokeWeight(2);
        for (let r = 0; r < 3; r++) p.arc(x, y, 8 + r * 7, hh * 0.9, -p.HALF_PI, p.HALF_PI);
      });
      p.stroke(muted); p.strokeWeight(1.4); p.noFill();
      p.beginShape(); p.vertex(x1 + 8, y); p.vertex(x1 + 30, y); p.vertex(x1 + 30, 118); p.vertex(x0 - 30, 118); p.vertex(x0 - 30, y); p.vertex(x0 - 8, y); p.endShape();
      p.fill(acc); p.noStroke(); p.triangle(0.48 * w, 108, 0.48 * w, 128, 0.48 * w + 18, 118);
      p.fill(muted); p.textSize(11); p.textFont("Hanken Grotesk, system-ui, sans-serif");
      p.text("mercury tube (sound wave)", x0 + 8, y - hh / 2 - 8); p.text("amplify and re-time", 0.48 * w + 24, 123); p.text("transducers", x1 - 56, y + hh / 2 + 18);
    };
    if (reducedMotion()) p.noLoop(); else visibility(host, p);
  });
}
