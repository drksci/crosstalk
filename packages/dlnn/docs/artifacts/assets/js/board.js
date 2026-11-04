// board.js: W6 board viewer. three.js (vendored r187) renders the KiCad-exported GLBs with PBR materials from board/materials.json
// (exposed copper = ENIG gold: metalness 1, roughness 0.25, #d4a74a) under a small procedural studio environment; orbit controls and a slow
// tumble that turns the board over to show the underside (the gold artwork on the carrier). The carrier GLB (16 MB) loads when the widget
// scrolls into view; tile GLBs on demand. D3 + SVG layer viewer (copper, mask, silk, edge, drill) from the KiCad plots. BOM/cost from
// data/board_cost.json (ESTIMATE). Render of the design, not a photo: nothing here is fabricated.
import { loadJSON, loadD3, el, reducedMotion, DATA } from "./common.js";

const BOARD_URL = new URL("../../board/", import.meta.url);
const BOARDS = [
  { id: "carrier", label: "Carrier", glb: "carrier/carrier.glb", prefix: "carrier_art-", drill: "carrier-Drill.svg", dir: "carrier", bom: "carrier/carrier_bom_jlc.csv", mb: 16 },
  { id: "tile_L1", label: "Tile, layer 1", glb: "tile_L1/tile_L1.glb", prefix: "tile_L1-", drill: "tile_L1-Drill.svg", dir: "tile_L1", bom: "tile_L1/tile_L1_bom_jlc.csv", mb: 3 },
  { id: "tile_L2", label: "Tile, layer 2", glb: "tile_L2/tile_L2.glb", prefix: "tile_L2-", drill: "tile_L2-Drill.svg", dir: "tile_L2", bom: "tile_L2/tile_L2_bom_jlc.csv", mb: 4 },
  { id: "tile_L3", label: "Tile, layer 3", glb: "tile_L3/tile_L3.glb", prefix: "tile_L3-", drill: "tile_L3-Drill.svg", dir: "tile_L3", bom: "tile_L3/tile_L3_bom_jlc.csv", mb: 4 },
];

function statusOf(b, m) {
  const d = m.drc, clean = d.violations === 0 && d.unconnected === 0;
  if (b.id === "carrier") return { text: "designed, DRC-clean, not fabricated", cls: clean ? "des" : "bad", note: "DRC 0 violations, 0 unconnected. ERC clean." };
  if (m.routed && clean) return { text: "routed, DRC-clean", cls: "des", note: "Reduced slice: 2 of 8 delay lines populated. Not fabricated." };
  return { text: "placed, not routed, not DRC-clean", cls: "bad", note: `DRC: ${d.violations} violations, ${d.unconnected} unconnected items. Placement only; not fabricated.` };
}

export async function mountBoard(host) {
  const [manifest, materials, cost] = await Promise.all([fetch(new URL("board_manifest.json", BOARD_URL)).then((r) => r.json()), fetch(new URL("materials.json", BOARD_URL)).then((r) => r.json()), loadJSON("board_cost.json"), loadD3()]);
  const d3 = window.d3, mById = Object.fromEntries(manifest.boards.map((b) => [b.board, b]));
  let cur = BOARDS[0], mode = reducedMotion() ? "top" : "auto";

  // ------------------------------------------------------------------ DOM
  const tabs = el("div", { class: "seg", role: "tablist", "aria-label": "Board" }), sideSeg = el("div", { class: "seg", role: "group", "aria-label": "Side" });
  const canvasWrap = el("div", { class: "bd-canvas" }), prog = el("div", { class: "bd-prog", role: "status" });
  const chips = el("div", { class: "bd-chips" }), note = el("p", { class: "w-foot bd-note" });
  const layerBox = el("div", { class: "bd-layers" });
  for (const b of BOARDS) { const btn = el("button", { role: "tab", type: "button", id: "bd-tab-" + b.id, "aria-selected": b.id === cur.id, text: b.label }); btn.addEventListener("click", () => select(b)); tabs.append(btn); }
  const sideBtns = {};
  for (const [id, label] of [["auto", "Auto-turn"], ["top", "Top"], ["bottom", "Bottom"]]) { const btn = el("button", { type: "button", "aria-pressed": id === mode, text: label }); btn.addEventListener("click", () => setMode(id)); sideBtns[id] = btn; sideSeg.append(btn); }
  host.replaceChildren(
    el("div", { class: "w-head" }, el("h3", { class: "w-title", text: "The boards, rendered from the design files" }), el("span", { class: "w-tag", text: "W6 · designed, not fabricated · a render, not a photo" })),
    el("p", { class: "w-sub", text: "Drag to orbit, scroll or pinch to zoom. The carrier carries the Nucleo, the converters and a figure in exposed gold copper on its underside, so the view slowly turns over. Copper and pads are shown as ENIG gold." }),
    el("div", { class: "bd-top" }, tabs, sideSeg), el("div", { class: "bd-main" }, canvasWrap, el("div", { class: "bd-side" }, chips, layerBox)), note);
  canvasWrap.append(prog);

  // ------------------------------------------------------------------ three.js
  let three = null;
  async function initThree() {
    const THREE = await import("three");
    const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
    const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
    const canvas = renderer.domElement; canvas.setAttribute("role", "img"); canvas.setAttribute("aria-label", "3-D render of the selected board; drag to orbit"); canvas.tabIndex = 0;
    canvasWrap.prepend(canvas);
    const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(32, 1.6, 0.01, 50);
    // procedural studio environment: dark gradient dome + bright soft boxes so a polished metal has something to reflect
    const envScene = new THREE.Scene();
    const dome = new THREE.Mesh(new THREE.SphereGeometry(10, 32, 16), new THREE.ShaderMaterial({ side: THREE.BackSide, depthWrite: false, uniforms: {},
      vertexShader: "varying vec3 p; void main(){ p = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }",
      fragmentShader: "varying vec3 p; void main(){ float h = p.y*0.5+0.5; vec3 c = mix(vec3(0.05,0.05,0.07), vec3(0.45,0.47,0.52), pow(h,1.4)); gl_FragColor = vec4(c,1.0); }" }));
    envScene.add(dome);
    const box = (w, h, pos, inten, tint = [1, 1, 1]) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(tint[0], tint[1], tint[2]).multiplyScalar(inten), side: THREE.DoubleSide, toneMapped: false })); m.position.set(...pos); m.lookAt(0, 0, 0); envScene.add(m); };
    box(7, 4, [0, 8, 2], 7); box(3, 6, [-8, 3, 2], 5, [1, 0.96, 0.9]); box(3, 6, [8, 2, -2], 4, [0.9, 0.95, 1]); box(8, 2, [0, 1.5, -9], 3); box(5, 3, [0, -7, 3], 2.2);
    const pm = new THREE.PMREMGenerator(renderer); scene.environment = pm.fromScene(envScene, 0.02).texture; pm.dispose();
    const key = new THREE.DirectionalLight(0xffffff, 1.4); key.position.set(-1, 2, 1.5); scene.add(key); scene.add(new THREE.HemisphereLight(0xffffff, 0x333344, 0.35));
    const pivot = new THREE.Group(); scene.add(pivot);
    const controls = new OrbitControls(camera, canvas); controls.enableDamping = !reducedMotion(); controls.dampingFactor = 0.08; controls.minDistance = 0.4; controls.maxDistance = 4;
    const matCache = {};
    const matFor = (boardId, name) => {
      const role = materials.glb[boardId] && materials.glb[boardId][name], spec = role && materials.roles[role];
      if (!spec) return null;
      if (!matCache[role]) matCache[role] = new THREE.MeshStandardMaterial({ color: spec.color, metalness: spec.metalness, roughness: spec.roughness, transparent: spec.opacity != null && spec.opacity < 1, opacity: spec.opacity ?? 1, envMapIntensity: role === "copper_enig_gold" ? 2.6 : 0.8 });
      return matCache[role];
    };
    const cache = {}, loader = new GLTFLoader();
    let model = null, angle = 0, tClock = 0, interacting = false, dirty = true, vis = true, last = performance.now();
    const loadBoard = (b) => cache[b.id] || (cache[b.id] = new Promise((res, rej) => loader.load(new URL(b.glb, BOARD_URL).href, (g) => {
      const root = g.scene;
      root.traverse((o) => { if (o.isMesh) { const m = matFor(b.id, o.material && o.material.name); if (m) o.material = m; } });
      const bb = new THREE.Box3().setFromObject(root), c = bb.getCenter(new THREE.Vector3()), s = bb.getSize(new THREE.Vector3()), k = 1 / Math.max(s.x, s.z);
      root.position.sub(c); const g2 = new THREE.Group(); g2.add(root); g2.scale.setScalar(k); g2.userData = { size: s.clone().multiplyScalar(k) }; res(g2);
    }, (e) => { if (e.lengthComputable) prog.textContent = `Loading ${b.label} 3-D model: ${(e.loaded / 1048576).toFixed(1)} of ${(e.total / 1048576).toFixed(1)} MB`; else prog.textContent = `Loading ${b.label} 3-D model (about ${b.mb} MB)...`; }, rej)));
    function resize() { const w = canvasWrap.clientWidth, h = canvasWrap.clientHeight; if (!w || !h) return; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); dirty = true; }
    new ResizeObserver(resize).observe(canvasWrap);
    controls.addEventListener("start", () => { interacting = true; }); controls.addEventListener("end", () => { interacting = false; }); controls.addEventListener("change", () => { dirty = true; });
    const targetAngle = () => {
      if (mode === "top") return 0; if (mode === "bottom") return Math.PI;
      const P = 16, t = tClock % P; // auto: dwell on top 3.5 s, turn 4.5 s, dwell on the underside 3.5 s, turn back 4.5 s
      const sm = (x) => x * x * (3 - 2 * x);
      if (t < 3.5) return 0; if (t < 8) return Math.PI * sm((t - 3.5) / 4.5); if (t < 11.5) return Math.PI; return Math.PI * (1 - sm((t - 11.5) / 4.5));
    };
    function frame(now) {
      const dt = Math.min(0.1, (now - last) / 1000); last = now;
      if (!vis) return;
      let animating = false;
      if (mode === "auto" && !interacting && !reducedMotion()) { tClock += dt; animating = true; }
      const ta = targetAngle(), prev = angle;
      angle = mode === "auto" ? ta : angle + (ta - angle) * (reducedMotion() ? 1 : 1 - Math.exp(-dt * 6));
      if (Math.abs(angle - prev) > 1e-4) { animating = true; }
      pivot.rotation.x = angle;
      if (controls.enableDamping) { if (controls.update()) dirty = true; } else controls.update();
      if (dirty || animating) { renderer.render(scene, camera); dirty = false; }
    }
    renderer.setAnimationLoop(frame);
    new IntersectionObserver((es) => { vis = es[0].isIntersecting; if (vis) { last = performance.now(); dirty = true; } }, { rootMargin: "100px" }).observe(canvasWrap);
    window.addEventListener("dlnn-theme", () => { dirty = true; });
    return {
      THREE, async show(b) {
        prog.hidden = false; prog.textContent = `Loading ${b.label} 3-D model (about ${b.mb} MB)...`;
        const g = await loadBoard(b);
        if (cur.id !== b.id) return;
        if (model) pivot.remove(model);
        model = g; pivot.add(g);
        const s = g.userData.size, d = 0.6 / (Math.tan((camera.fov * Math.PI) / 360) * Math.min(camera.aspect, 1.6)) + s.y;
        camera.position.set(0, d * 0.78, d * 0.62); controls.target.set(0, 0, 0); controls.update(); tClock = 0; dirty = true; prog.hidden = true;
      }, markDirty() { dirty = true; },
    };
  }
  async function select(b) {
    cur = b; tabs.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", x.id === "bd-tab-" + b.id));
    renderStatus(); drawLayers();
    if (three) try { await three.show(b); } catch (e) { prog.hidden = false; prog.textContent = "Could not load the 3-D model: " + e.message; console.error("board", e); }
  }
  function setMode(m) { mode = m; Object.entries(sideBtns).forEach(([k, v]) => v.setAttribute("aria-pressed", k === m)); if (three) three.markDirty(); }
  function renderStatus() {
    const m = mById[cur.id], s = statusOf(cur, m);
    chips.replaceChildren(el("span", { class: "status " + (s.cls === "des" ? "des" : "bad"), "data-board": cur.id }, el("i", { text: s.cls === "des" ? "▲" : "■" }), " ", s.text), el("span", { class: "status none", title: "Nothing has been fabricated or measured" }, el("i", { text: "■" }), " measured: none"));
    const bom = cost.tiles.find((t) => (cur.id === "tile_L1" && /layer-1/.test(t.tag)) || (cur.id === "tile_L2" && /layer-2/.test(t.tag)) || (cur.id === "tile_L3" && /layer-3/.test(t.tag)));
    note.innerHTML = "";
    note.append(`${m.alt} `, el("b", { text: s.note }), ` ${m.cpl_rows} placement rows` + (m.cpl_is_model ? `; the crossbar is populated from the trained model (${m.cpl_is_model.n_cpl_crossbar} of ${m.cpl_is_model.n_nonzero_weights} non-zero weights placed, match: ${m.cpl_is_model.match}).` : "."), bom ? ` Parts ≈ $${bom.parts_usd.toFixed(0)} (estimate).` : "",
      cur.id === "carrier" ? " Artwork supplied by the project owner." : "");
  }

  // ------------------------------------------------------------------ layer viewer (D3 + SVG)
  const LAYERS = [["copper", "Copper"], ["mask", "Mask"], ["silk", "Silk"], ["edge", "Edge"], ["drill", "Drill"]];
  const lv = { on: { copper: true, mask: false, silk: false, edge: true, drill: false }, inner: false, side: "F" };
  const svgCache = new Map();
  const getSvg = (url) => { if (!svgCache.has(url)) svgCache.set(url, d3.xml(url).then((x) => x.documentElement)); return svgCache.get(url); };
  const layerFiles = (b) => {
    const base = new URL(b.dir + "/svg/", BOARD_URL).href, p = b.prefix, s = lv.side;
    const f = [];
    if (lv.on.copper) { if (lv.inner) f.push([base + p + "In2_Cu.svg", "inner", 0.55], [base + p + "In1_Cu.svg", "inner", 0.55]); f.push([base + p + s + "_Cu.svg", "copper", 0.95]); }
    if (lv.on.mask) f.push([base + p + s + "_Mask.svg", "mask", 0.5]);
    if (lv.on.silk) f.push([base + p + s + "_Silkscreen.svg", "silk", 0.95]);
    if (lv.on.edge) f.push([base + p + "Edge_Cuts.svg", "edge", 1]);
    if (lv.on.drill) f.push([base + b.drill, "drill", 1]);
    return f;
  };
  let layerToken = 0;
  async function drawLayers() {
    const tok = ++layerToken, b = cur;
    const tg = el("div", { class: "bd-toggles", role: "group", "aria-label": "Layers" });
    for (const [id, label] of LAYERS) { const cb = el("input", { type: "checkbox", checked: lv.on[id], id: "lv-" + id }); cb.addEventListener("change", () => { lv.on[id] = cb.checked; drawLayers(); }); tg.append(el("label", { class: "bd-tg" }, cb, label)); }
    const inner = el("input", { type: "checkbox", checked: lv.inner, id: "lv-inner" }); inner.addEventListener("change", () => { lv.inner = inner.checked; drawLayers(); });
    const sd = el("div", { class: "seg", role: "group", "aria-label": "Layer side" }, ...[["F", "Front"], ["B", "Back"]].map(([k, l]) => { const bt = el("button", { type: "button", "aria-pressed": lv.side === k, text: l }); bt.addEventListener("click", () => { lv.side = k; drawLayers(); }); return bt; }));
    const stage = el("div", { class: "bd-stage" }), msg = el("div", { class: "bd-lmsg", text: "Loading layers..." });
    layerBox.replaceChildren(el("h4", { class: "ar-h", text: `Layers: ${b.label}` }), tg, el("div", { class: "ar-opts" }, sd, el("label", { class: "bd-tg" }, inner, "inner copper")), stage, msg);
    const files = layerFiles(b);
    const docs = await Promise.all(files.map(([u]) => getSvg(u).catch(() => null)));
    if (tok !== layerToken) return;
    const first = docs.find(Boolean);
    if (!first) { msg.textContent = "No layers selected."; return; }
    const vb = (first.getAttribute("viewBox") || "0 0 100 100").split(/\s+/).map(Number);
    const svg = d3.select(stage).append("svg").attr("viewBox", `0 0 ${vb[2]} ${vb[3]}`).attr("role", "img").attr("aria-label", `${b.label} ${lv.side === "F" ? "front" : "back"} layers: ${files.map((f) => f[1]).join(", ")}`);
    const g = svg.append("g").attr("transform", lv.side === "B" ? `translate(${vb[2]},0) scale(-1,1)` : null);
    docs.forEach((d, i) => { if (!d) return; const n = d.cloneNode(true); n.removeAttribute("width"); n.removeAttribute("height"); n.setAttribute("x", 0); n.setAttribute("y", 0); n.setAttribute("width", vb[2]); n.setAttribute("height", vb[3]); n.setAttribute("opacity", files[i][2]); n.classList.add("lv-" + files[i][1]); g.node().appendChild(document.importNode(n, true)); });
    msg.textContent = `${files.length} layer${files.length === 1 ? "" : "s"} shown, ${lv.side === "F" ? "front" : "back (mirrored)"}. KiCad plot colours.`;
  }

  renderStatus();
  const costBox = el("div", { class: "bd-cost" });
  host.append(costBox);
  (function drawCost() {
    const rows = cost.tiles.map((t) => el("tr", {}, el("td", { text: t.tag }), el("td", { class: "num", text: "$" + t.parts_usd.toFixed(2) }), el("td", { class: "num", text: t.joints.toLocaleString() }), el("td", { class: "num", text: t.unique_parts })));
    const tot = cost.system_total_usd, tc = cost.system_total_usd_with_25pct_contingency;
    costBox.append(el("h4", { class: "ar-h" }, "Bill of materials and cost summary ", el("span", { class: "status des", title: cost.label }, el("i", { text: "▲" }), " estimate")),
      el("div", { class: "bd-tablewrap" }, el("table", { class: "bd-table" }, el("thead", {}, el("tr", {}, ["Board", "Parts", "Solder joints", "Unique parts"].map((t) => el("th", { text: t })))), el("tbody", {}, rows,
        el("tr", {}, el("td", { text: "Tile PCBs (5 boards ordered, 3 used)" }), el("td", { class: "num", text: "$" + cost.tile_pcb_usd.toFixed(2) }), el("td"), el("td")),
        el("tr", {}, el("td", { text: "Tile assembly fees (economic / standard)" }), el("td", { class: "num", text: `$${cost.tile_assembly_fees_usd.economic_single_sided.toFixed(0)} / $${cost.tile_assembly_fees_usd.standard_double_sided.toFixed(0)}` }), el("td"), el("td")),
        el("tr", {}, el("td", { text: "Nucleo, cables, supply" }), el("td", { class: "num", text: "$" + cost.nucleo_cables_supply_usd.toFixed(2) }), el("td"), el("td")),
        el("tr", { class: "tot" }, el("td", { text: "Total, 3 tiles + carrier (economic / standard assembly)" }), el("td", { class: "num", text: `$${Math.round(tot.economic_single_sided_tiles)} / $${Math.round(tot.standard_double_sided_tiles)}` }), el("td"), el("td")),
        el("tr", {}, el("td", { text: "With 25% contingency" }), el("td", { class: "num", text: `$${Math.round(tc.low)} to $${Math.round(tc.high)}` }), el("td"), el("td"))))),
      el("p", { class: "w-foot", text: `ESTIMATE, not a quote. ${cost.label}. ${cost.pcb_assumptions} Contingency covers: ${tc.contingency_covers}` }));
  })();
  drawLayers();

  try { three = await initThree(); await three.show(cur); }
  catch (e) { prog.hidden = false; prog.textContent = "3-D view unavailable (" + (e.message || e) + "). The layer viewer still works."; console.warn("board 3d", e); }
}
