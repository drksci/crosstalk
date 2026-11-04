// scale.js: W7 stitching diagram and W9 scale story (D3). Data: data/scale_story.json (from hw/board/cost.json and the model reports).
// Tile counts follow the formula in cost.json: tiles = (D-1)(W/8)^2 + W/8, inductors = 448 (D-1)(W/8)^2 + 112 (W/8). Cost is an ESTIMATE.
import { loadJSON, loadD3, el, cssVar } from "./common.js";

const fmt = (n) => (n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e4 ? Math.round(n / 1e3) + "k" : n >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(Math.round(n)));
const usd = (n) => "$" + fmt(n);
const colors = () => ({ ink: cssVar("--ink"), muted: cssVar("--muted"), rule: cssVar("--rule"), acc: cssVar("--accent"), l1: cssVar("--tier-C"), full: cssVar("--tier-V"), pos: cssVar("--pos"), neg: cssVar("--neg"), surf: cssVar("--surface-2") });

export async function mountStitch(host) {
  const [story] = await Promise.all([loadJSON("scale_story.json"), loadD3()]);
  const d3 = window.d3;
  const st = { W: 16, D: 4 };
  const wSeg = el("div", { class: "seg", role: "group", "aria-label": "Layer width" }), dSeg = el("div", { class: "seg", role: "group", "aria-label": "Depth" });
  const mk = (seg, key, vals, lab) => vals.forEach((v) => { const b = el("button", { type: "button", "aria-pressed": st[key] === v, text: lab(v) }); b.addEventListener("click", () => { st[key] = v; seg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b)); draw(); }); seg.append(b); });
  mk(wSeg, "W", [8, 16, 32, 64], (v) => "width " + v); mk(dSeg, "D", [3, 4, 6], (v) => "depth " + v);
  const stats = el("div", { class: "dg-stats st-stats" }), svgBox = el("div", { class: "st-svg" });
  host.replaceChildren(
    el("div", { class: "w-head" }, el("h3", { class: "w-title", text: "Stitching tiles into wider and deeper nets" }), el("span", { class: "w-tag", text: "W7 · simulation + cost estimate" })),
    el("p", { class: "w-sub", text: "One tile is 8 delay lines by 8 taps. A layer of width W needs a grid of (W/8) by (W/8) tiles: each row produces 8 output channels as the sum of the partial results from every input group, so the tiles in a row are stitched together by summing their outputs. The first layer only has 2 input channels, so it needs a single row." }),
    el("div", { class: "ar-opts" }, wSeg, dSeg), stats, svgBox,
    el("p", { class: "w-foot", text: "Counts from the formula in hw/board/cost.json (scaling_formula). Area and cost are ESTIMATES (parts prices read from JLC pages on 2026-11-03, PCB prices assumed; needs a live quote). Not built; the wall at large widths is the number of inductors and the loss through a long chain of tiles." }));
  function draw() {
    const { W, D } = st, n = W / 8, c = colors();
    const key = `W${W}_D${D}`, sc = story.scaling[key];
    const tiles = (D - 1) * n * n + n, ind = 448 * (D - 1) * n * n + 112 * n;
    stats.replaceChildren(...[["tiles", sc ? sc.tiles_total : tiles], ["inductors", fmt(sc ? sc.inductors : ind)], ["board area", sc ? fmt(sc.board_area_cm2) + " cm²" : "n/a"], ["cost (estimate)", sc ? usd(sc.cost_usd_estimate) : "n/a"]]
      .map(([k, v]) => el("div", { class: "dg-stat" }, el("b", { text: v }), el("span", { text: k }))));
    svgBox.replaceChildren();
    const colW = 150, rowH = 14, cell = Math.min(18, (n * rowH) > 0 ? 15 : 15), padX = 26, H = Math.max(n, 1) * (cell + 3) + 70;
    const Wd = padX * 2 + D * colW;
    const svg = d3.select(svgBox).append("svg").attr("viewBox", `0 0 ${Wd} ${H}`).attr("role", "img").attr("aria-label", `Tile grid for width ${W}, depth ${D}: ${tiles} tiles`);
    for (let l = 0; l < D; l++) {
      const cols = l === 0 ? 1 : n, x0 = padX + l * colW + (colW - 24 - cols * (cell + 3)) / 2;
      const g = svg.append("g");
      g.append("text").attr("x", padX + l * colW + (colW - 24) / 2).attr("y", 14).attr("text-anchor", "middle").attr("class", "dg-t").attr("fill", c.ink).text(`layer ${l + 1}`);
      for (let r = 0; r < n; r++) for (let q = 0; q < cols; q++) {
        g.append("rect").attr("x", x0 + q * (cell + 3)).attr("y", 28 + r * (cell + 3)).attr("width", cell).attr("height", cell).attr("rx", 2).attr("fill", l === 0 ? c.l1 : c.full).attr("opacity", 0.9);
        if (l > 0 && q === cols - 1) g.append("path").attr("d", `M${x0 + cols * (cell + 3)} ${28 + r * (cell + 3) + cell / 2}h8`).attr("stroke", c.ink).attr("stroke-width", 1.2); // row sum
      }
      if (l < D - 1) { // wires to the next layer: every output row feeds all columns of the next layer
        const xr = x0 + cols * (cell + 3) + (l > 0 ? 8 : 0), xn = padX + (l + 1) * colW + (colW - 24 - n * (cell + 3)) / 2 - 3;
        for (let r = 0; r < n; r++) svg.append("path").attr("d", `M${xr} ${28 + r * (cell + 3) + cell / 2} C${xr + 30} ${28 + r * (cell + 3) + cell / 2}, ${xn - 30} ${28 + r * (cell + 3) + cell / 2}, ${xn} ${28 + r * (cell + 3) + cell / 2}`).attr("fill", "none").attr("stroke", c.muted).attr("stroke-width", 0.8).attr("opacity", 0.7);
      }
    }
    const yL = H - 20;
    svg.append("rect").attr("x", padX).attr("y", yL - 10).attr("width", 11).attr("height", 11).attr("rx", 2).attr("fill", c.l1);
    svg.append("text").attr("x", padX + 16).attr("y", yL).attr("class", "dg-ax").attr("fill", c.ink).text("layer-1 tile (2 lines)");
    svg.append("rect").attr("x", padX + 150).attr("y", yL - 10).attr("width", 11).attr("height", 11).attr("rx", 2).attr("fill", c.full);
    svg.append("text").attr("x", padX + 166).attr("y", yL).attr("class", "dg-ax").attr("fill", c.ink).text("full tile (8 lines); rows sum to 8 output channels");
  }
  window.addEventListener("dlnn-theme", draw);
  draw();
}

export async function mountScale(host) {
  const [story] = await Promise.all([loadJSON("scale_story.json"), loadD3()]);
  const d3 = window.d3;
  const sc = story.scaling, keys = Object.keys(sc);
  const widths = [8, 16, 32, 64], depths = [3, 4, 6];
  const MET = { tiles: ["tiles", (k) => sc[k].tiles_total, fmt], inductors: ["inductors", (k) => sc[k].inductors, fmt], area: ["board area (cm²)", (k) => sc[k].board_area_cm2, fmt], cost: ["cost, estimate (USD)", (k) => sc[k].cost_usd_estimate, usd] };
  let metric = "inductors";
  const seg = el("div", { class: "seg", role: "group", "aria-label": "Metric" });
  Object.entries(MET).forEach(([id, m]) => { const b = el("button", { type: "button", "aria-pressed": id === metric, text: m[0].replace(/ \(.*/, "") }); b.addEventListener("click", () => { metric = id; seg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b)); draw(); }); seg.append(b); });
  const box = el("div", { class: "sc-grid" }), a = el("div", { class: "sc-a" }), b = el("div", { class: "sc-b" });
  box.append(a, b);
  host.replaceChildren(
    el("div", { class: "w-head" }, el("h3", { class: "w-title", text: "The scale story: what size costs, and what it buys" }), el("span", { class: "w-tag", text: "W9 · simulation + cost estimate" })),
    el("p", { class: "w-sub", text: "Left: how the hardware grows with width and depth (log scale). Right: test accuracy of the ternary nets we trained at three sizes, against the estimated cost of building that size. Accuracy comes from the model reports; cost is an estimate for boards that have not been built." }),
    el("div", { class: "ar-opts" }, seg), box,
    el("p", { class: "w-foot", html: `Accuracy points: ${story.accuracy_points.map((p) => `width ${p.width}, depth ${p.depth}: ${(p.acc * 100).toFixed(1)}% (${p.n_ternary_weights.toLocaleString()} ternary weights)`).join("; ")}. Same synthetic 8-class set, ideal-line simulation, not board-trained, so these are not the numbers of the board net. Cost: <b>estimate</b>, ${story.label}` }));
  function draw() {
    const c = colors(), [lab, get, f] = MET[metric];
    const dcol = { 3: c.acc, 4: cssVar("--lane-1"), 6: cssVar("--lane-2") };
    a.replaceChildren(); b.replaceChildren();
    // (a) metric vs width, one line per depth
    let svg = d3.select(a).append("svg").attr("viewBox", "0 0 380 250").attr("role", "img").attr("aria-label", `${lab} against layer width for depths 3, 4 and 6`);
    const all = keys.map(get), xs = d3.scaleLog([7, 70], [52, 362]), ys = d3.scaleLog([d3.min(all) * 0.7, d3.max(all) * 1.4], [208, 22]);
    for (const w of widths) { svg.append("text").attr("x", xs(w)).attr("y", 226).attr("text-anchor", "middle").attr("class", "dg-ax").attr("fill", c.muted).text(w); }
    svg.append("text").attr("x", 207).attr("y", 244).attr("text-anchor", "middle").attr("class", "dg-ax").attr("fill", c.muted).text("layer width W");
    svg.append("g").attr("transform", "translate(52,0)").call(d3.axisLeft(ys).ticks(5, "~s")).call((g) => g.selectAll("text").attr("fill", c.muted).attr("class", "dg-ax")).call((g) => g.selectAll("path,line").attr("stroke", c.rule));
    svg.append("text").attr("x", 4).attr("y", 12).attr("class", "dg-ax").attr("fill", c.muted).text(lab);
    for (const D of depths) {
      const pts = widths.map((w) => [w, get(`W${w}_D${D}`)]);
      svg.append("path").attr("d", d3.line().x((p) => xs(p[0])).y((p) => ys(p[1]))(pts)).attr("fill", "none").attr("stroke", dcol[D]).attr("stroke-width", 2);
      svg.selectAll(`circle.d${D}`).data(pts).join("circle").attr("cx", (p) => xs(p[0])).attr("cy", (p) => ys(p[1])).attr("r", 3.5).attr("fill", dcol[D]).append("title").text((p) => `W${p[0]} D${D}: ${f(p[1])}`);
      const last = pts[pts.length - 1]; svg.append("text").attr("x", xs(last[0]) - 6).attr("y", ys(last[1]) - 7).attr("text-anchor", "end").attr("class", "dg-ax").attr("fill", dcol[D]).text(`depth ${D}: ${f(last[1])}`);
    }
    // (b) accuracy vs estimated cost (or chosen metric) for the three trained sizes
    svg = d3.select(b).append("svg").attr("viewBox", "0 0 380 250").attr("role", "img").attr("aria-label", `Accuracy against ${lab} for three trained sizes`);
    const pts = story.accuracy_points.map((p) => ({ ...p, x: get(p.scaling_key) }));
    const xb = d3.scaleLog([d3.min(pts, (p) => p.x) * 0.5, d3.max(pts, (p) => p.x) * 2.5], [52, 362]), yb = d3.scaleLinear([0.7, 1], [208, 22]);
    svg.append("g").attr("transform", "translate(0,208)").call(d3.axisBottom(xb).ticks(4, "~s")).call((g) => g.selectAll("text").attr("fill", c.muted).attr("class", "dg-ax")).call((g) => g.selectAll("path,line").attr("stroke", c.rule));
    svg.append("g").attr("transform", "translate(52,0)").call(d3.axisLeft(yb).ticks(4, "%")).call((g) => g.selectAll("text").attr("fill", c.muted).attr("class", "dg-ax")).call((g) => g.selectAll("path,line").attr("stroke", c.rule));
    svg.append("text").attr("x", 207).attr("y", 244).attr("text-anchor", "middle").attr("class", "dg-ax").attr("fill", c.muted).text(lab);
    svg.append("text").attr("x", 4).attr("y", 12).attr("class", "dg-ax").attr("fill", c.muted).text("test accuracy (ternary, ideal line)");
    svg.append("path").attr("d", d3.line().x((p) => xb(p.x)).y((p) => yb(p.acc))(pts)).attr("fill", "none").attr("stroke", c.muted).attr("stroke-dasharray", "4 3");
    svg.selectAll("circle").data(pts).join("circle").attr("cx", (p) => xb(p.x)).attr("cy", (p) => yb(p.acc)).attr("r", 5).attr("fill", c.acc).attr("stroke", c.ink);
    svg.selectAll("text.l").data(pts).join("text").attr("x", (p) => xb(p.x) + 8).attr("y", (p) => yb(p.acc) + 14).attr("class", "dg-ax").attr("fill", c.ink).text((p) => `w${p.width} d${p.depth}: ${(p.acc * 100).toFixed(1)}%`);
  }
  window.addEventListener("dlnn-theme", draw);
  draw();
}
