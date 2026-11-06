// Datenansicht: Seiten über daten/ (results, folds, ablations, external, hyperparameters, sources, details.zip) und ausgabe/.
import {
  h, csv, json, text, daten, zip, Table, fmtNum, fix, signed, thousands, bytes, middle, mean, sd, niceMax,
  diverging, inkOn, divergingBar, resetColors, cssVar, tipTitle, tipRow, tipSub, hoverTip, tip, untip,
  seg, select, chip, card, legend, drawer, closeDrawer, kv, VirtualTable, lineChart, markdown, jsonTree,
} from "/core.js";

const main = document.getElementById("main");

/* ---------- Namen ---------- */
const DS = { cars: "Cars196", cub: "CUB", inshop: "In-Shop", inshop10: "In-Shop · labeled classes", semiinat: "Semi-iNat-2021" };
const METHODS = [["sup", "ÜBl."], ["iscen", "LP"], ["lrml", "LRML"], ["hoffer", "Hoffer"], ["slade", "SLADE"]];
const LOSSES = [["arc", "ArcFace"], ["cir", "Circle"], ["ms", "MS"], ["nt", "NT-Xent"], ["prx", "Proxy-Anchor"]];
const MNAME = { ...Object.fromEntries(METHODS), lp: "LP" };
const LNAME = Object.fromEntries(LOSSES);
const normM = m => (m === "lp" ? "iscen" : m);
const SLOT = { sup: "--s1", iscen: "--s2", lrml: "--s3", hoffer: "--s4", slade: "--s5" };
const LSLOT = { arc: "--s1", cir: "--s2", ms: "--s3", nt: "--s4", prx: "--s5" };
const mcolor = m => cssVar(SLOT[normM(m)] || "--muted");
const lcolor = l => cssVar(LSLOT[l] || "--muted");
const METRICS = [["mean_average_precision_at_r", "MAP@R"], ["r_precision", "R-Precision"], ["precision_at_1", "P@1"], ["NMI", "NMI"], ["AMI", "AMI"],
  ...[1, 2, 4, 8, 10, 16, 20, 32, 40, 100].map(k => [`recall_at_${k}`, `Recall@${k}`])];
const MET = Object.fromEntries(METRICS);
const GROUPS = { "2x2": "2×2 decomposition", Kontrolle: "Control", "Graph-Kontrolle": "Graph control", Oracle: "Oracle" };
const CELLS = ["ssl_hp/with_data", "ssl_hp/no_data", "sup_hp/with_data", "sup_hp/no_data"];
const K4V = [["ue", "ÜBl."], ["defaults", "LRML, regular"], ["pool_labeled", "LRML, SSL HPs without unlabeled images"],
  ["sup_all_shared", "LRML, ÜBl. HPs with unlabeled images"], ["zero_weight", "LRML, weight of unlabeled term = 0"]];
const uniq = a => [...new Set(a.filter(x => x != null))];
const blabel = (ds, b) => (ds === "inshop10" ? `${fmtNum(b)} %` : `k = ${fmtNum(b)}`);
const pct = x => (x == null ? null : 100 * x);
const swatch = c => h("span", { class: "swatch", style: { background: c } });

/* ---------- Daten ---------- */
const R = () => csv(daten("results.csv"));
const F = () => csv(daten("folds.csv"));
const A = () => csv(daten("ablations.csv"));
const E = () => csv(daten("external.csv"));
const HP = () => csv(daten("hyperparameters.csv"));
const SRC = () => csv(daten("sources.csv"));
const MAN = () => json(daten("manifest.json"));
const INFO = () => json("/info");
const ZIPL = () => json("/zip");
const OUTL = () => json("/ausgabe").catch(() => []);
const memo = new Map();
const once = (k, f) => { if (!memo.has(k)) memo.set(k, f()); return memo.get(k); };
const srcIndex = () => once("src", async () => {
  const o = (await SRC()).objects();
  return { byId: new Map(o.map(x => [x.source_id, x])), byPath: new Map(o.map(x => [x.path, x])), all: o };
});
const parseJV = v => {
  if (v == null) return null;
  try { const x = JSON.parse(v); return x === null ? "null" : typeof x === "object" ? JSON.stringify(x) : typeof x === "boolean" ? String(x) : x; } catch { return v; }
};
const hpIndex = () => once("hp", async () => {
  const m = new Map();
  for (const o of (await HP()).objects()) {
    if (!m.has(o.config_id)) m.set(o.config_id, { source_id: o.source_id, params: [] });
    m.get(o.config_id).params.push([o.parameter, parseJV(o.value_json)]);
  }
  return m;
});
const usage = () => once("use", async () => {
  const [r, f] = await Promise.all([R(), F()]), u = new Map();
  const add = (c, v) => { if (!c) return; if (!u.has(c)) u.set(c, new Set()); u.get(c).add(v); };
  for (const o of r.objects()) add(o.config_id, o.run_id);
  for (const o of f.objects()) add(o.config_id, o.run_id || `${o.experiment}:${o.dataset_key}`);
  return u;
});
const foldsByCv = () => once("fcv", async () => {
  const m = new Map();
  for (const o of (await F()).objects()) {
    const cv = String(o.source || "").split("/fold_")[0];
    if (!m.has(cv)) m.set(cv, []);
    m.get(cv).push(o);
  }
  return m;
});

/* ---------- Router ---------- */
let current = { page: "", params: {} }, token = 0;
function parseHash() {
  const raw = location.hash.replace(/^#\/?/, ""), [page, q] = raw.split("?");
  return { page: page || "", params: Object.fromEntries(new URLSearchParams(q || "")) };
}
const hashFor = (page, params) => "#/" + page + (params && Object.keys(params).length ? "?" + new URLSearchParams(params) : "");
function go(page, params = {}) { location.hash = hashFor(page, params); }
function update(changes) {
  const p = { ...current.params, ...changes };
  for (const k of Object.keys(p)) if (p[k] == null || p[k] === "") delete p[k];
  history.replaceState(null, "", hashFor(current.page, p));
  route(true);
}
const PAGES = { "": overview, phase1, phase2, diagnostics, external, figures, hp, details, explore };
async function route(keepScroll) {
  current = parseHash();
  closeDrawer(); untip();
  document.querySelectorAll(".popover").forEach(p => p.remove());
  document.querySelectorAll("#nav a").forEach(a => a.setAttribute("aria-current", a.getAttribute("href") === "#/" + current.page ? "page" : "false"));
  const my = ++token, y = scrollY;
  if (!main.childElementCount) main.replaceChildren(h("div", { class: "loading" }, "Loading…"));
  main.classList.add("busy");
  try {
    const out = await (PAGES[current.page] || overview)(current.params);
    if (my !== token) return;
    const { nodes, after } = Array.isArray(out) ? { nodes: out } : out;
    main.replaceChildren(...nodes.filter(Boolean));
    scrollTo(0, keepScroll ? y : 0);
    if (after) after();
  } catch (e) {
    if (my === token) main.replaceChildren(h("div", { class: "card error" }, String(e && e.stack ? e.stack : e)));
  }
  if (my === token) main.classList.remove("busy");
}
addEventListener("hashchange", () => route(false));

/* ---------- Verweise zwischen Tabellen ---------- */
function link(col, v) {
  const sv = String(v);
  if (/^hp_[0-9a-f]{6,}$/.test(sv)) return () => configDrawer(sv);
  if (/^s_[0-9a-f]{6,}$/.test(sv)) return () => sourceDrawer({ id: sv });
  if (/^p1:[a-z0-9]+:[a-z]+_[a-z]+:[\d.]+:\d+$/.test(sv)) return () => { const [, ds, m, b] = sv.split(":"); runDrawer(ds, m, +b); };
  if (sv.startsWith("/") && /(source|path|cv_dir|study|fold_dir)$/.test(col)) return () => sourceDrawer({ path: sv });
  return null;
}
const a = (label, fn) => h("a", { href: "#", onclick: e => { e.preventDefault(); fn(); } }, label);
const linkOrText = (col, v) => {
  const l = v == null ? null : link(col, v);
  if (!l) return v == null ? "–" : String(v);
  const el = a(String(v).length > 60 ? middle(v, 56) : String(v), () => { closeDrawer(); l(); });
  el.title = String(v);
  return el;
};

async function sourceDrawer({ id, path }) {
  const ix = await srcIndex();
  let rec = id ? ix.byId.get(id) : ix.byPath.get(path);
  let note = null;
  if (!rec && path) {
    const cv = path.split("/fold_")[0].replace(/\/cv_summary\.json$/, "");
    rec = ix.all.find(x => x.path.startsWith(cv + "/"));
    note = rec ? "This exact path is not a recorded source file; showing the recorded file of the same CV folder." : "No recorded source file for this path.";
  }
  const sections = [];
  if (path) sections.push(h("section", {}, h("h3", {}, "Referenced path"), h("div", { class: "path" }, path)));
  if (note) sections.push(h("p", { class: "ink2" }, note));
  if (rec) {
    sections.push(h("section", {}, h("h3", {}, "Recorded source"), kv([
      ["source_id", rec.source_id], ["path", h("span", { class: "path" }, rec.path)], ["available at import", String(rec.available)],
      ["size", bytes(rec.bytes)], ["sha256", h("span", { class: "mono" }, rec.sha256 || "–")], ["selection", rec.selection]])));
    sections.push(h("p", {}, a("Show in sources table", () => go("explore", { src: "file:sources.csv", "f.source_id": "=" + rec.source_id }))));
  }
  drawer("Provenance", rec ? rec.path.split("/").slice(-2).join("/") : "Source", ...sections);
}

/* ---------- Bausteine ---------- */
const head = (title, textEl) => h("div", { class: "page-head" }, h("h2", {}, title), textEl ? h("p", {}, textEl) : null);
function cellEl({ main: v, sub, n, bg, best, onClick, tipBuild, kind }) {
  const el = h("div", { class: "cell" + (kind ? " " + kind : bg ? "" : " plainc"), tabindex: onClick ? "0" : null, role: onClick ? "button" : null },
    h("div", { class: "v" + (best ? " best" : "") }, v), sub ? h("div", { class: "sd" }, sub) : null, n ? h("div", { class: "n" }, n) : null);
  if (bg) { el.style.background = bg; el.style.color = inkOn(bg); }
  if (onClick) {
    el.addEventListener("click", onClick);
    el.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick(); } });
  }
  if (tipBuild) hoverTip(el, tipBuild);
  return el;
}
function divLegend(dmax, neg, pos) {
  return h("div", { class: "legend-div" }, h("span", {}, `−${fmtNum(dmax)} pp · ${neg}`),
    h("span", { class: "bar", style: { background: divergingBar() }, role: "img", "aria-label": "diverging color scale" }),
    h("span", {}, `+${fmtNum(dmax)} pp · ${pos}`), h("span", { class: "muted" }, "Differences beyond the range use the end color."));
}
function quantileAbs(vals, q = 0.95) {
  const v = vals.filter(x => x != null && !Number.isNaN(x)).map(Math.abs).sort((p, r) => p - r);
  return v.length ? v[Math.min(v.length - 1, Math.floor(q * (v.length - 1)))] : 1;
}
function multiples(panels, opts) {
  const all = panels.flatMap(pn => [...pn.series.flatMap(se => se.values), ...(pn.marks || []).map(k => k.y)]).filter(v => v != null);
  if (!all.length) return h("div", { class: "empty-note" }, "No values for this selection.");
  let lo = Math.min(...all), hi = Math.max(...all);
  if (opts.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  for (const r of panels.flatMap(pn => pn.refs || [])) { lo = Math.min(lo, r.y); hi = Math.max(hi, r.y); }
  const pad = (hi - lo) * 0.06 || 1;
  return h("div", { class: "multiples" }, panels.map(pn => h("div", { class: "panel" },
    h("div", { class: "ptitle" }, h("h4", {}, pn.title), pn.note ? h("span", { class: "muted", style: { fontSize: "12px" } }, pn.note) : null),
    lineChart({ ...opts, ...pn, ydomain: pn.ydomain || [lo - pad, hi + pad], label: pn.title }))));
}
const smallTable = (cols, rows, opts = {}) => h("div", { class: "scrollx" }, h("table", { class: "plain" },
  h("thead", {}, h("tr", {}, cols.map(([t, num]) => h("th", { class: num ? "num" : null }, t)))),
  h("tbody", {}, rows.map(r => h("tr", { class: opts.onRow ? "click" : null, onclick: opts.onRow ? () => opts.onRow(r) : null },
    r.cells.map((c, i) => h("td", { class: cols[i][1] ? "num" : null }, c)))))));

/* ---------- Diagnostics ---------- */
const DIAG = "/diagnostics/";
const isNumeric = c => /^(?:[+\-–]?[\d.]+(?: → [\d.]+)?|–|[\d/]+)$/.test(String(c).replaceAll("**", ""));
const diagCell = c => {
  const m = String(c).match(/^\*\*(.*)\*\*$/);
  return m ? h("strong", {}, m[1]) : String(c);
};
function diagTable(t) {
  const cols = t.columns.map((name, i) => [name, t.rows.every(r => isNumeric(r[i]))]);
  const rows = t.rows.map((r, n) => ({
    cells: r.map((c, i) => h("span", { class: i === 0 && n && t.rows[n - 1][0] === c ? "muted" : null, style: { whiteSpace: "nowrap" } }, diagCell(c))),
  }));
  return card(t.title, `${t.rows.length} rows`, smallTable(cols, rows), t.note ? h("p", { class: "foot" }, t.note) : null);
}
async function diagnostics(p) {
  const doc = await json(DIAG + "tables.json");
  const sec = doc.sections.find(s => s.id === p.part) || doc.sections[0];
  const titles = sec.tables.map(t => t.title);
  const chosen = sec.tables.find(t => t.title === p.t) || sec.tables[0];
  const filters = h("div", { class: "filters" },
    seg("Part", doc.sections.map(s => [s.id, s.title]), sec.id, v => update({ part: v, t: null })),
    seg("Table", titles.map(t => [t, t]), chosen.title, v => update({ t: v })));
  const files = sec.file.split(", ").map(f => h("a", { href: DIAG + f, download: f, style: { marginRight: "1em" } }, f));
  return [
    head(doc.title, "Condensed view of the diagnostic measurements behind Chapters 5 and 6 of the thesis, per dataset, budget and loss."),
    filters,
    h("div", { class: "stack" },
      card(sec.title, null, h("p", {}, sec.intro), h("p", { class: "foot" }, "Download: ", files)),
      diagTable(chosen),
      card("Keep in mind", null, h("ul", {}, doc.notes.map(n => h("li", {}, n))),
        h("p", { class: "foot" }, "ÜBl. is the supervised baseline, LP is label propagation."))),
  ];
}

/* ---------- Übersicht ---------- */
async function overview() {
  const [info, man, readme, check, outl, zl] = await Promise.all([INFO(), MAN(), text(daten("README.md")).catch(() => ""),
    json("/ausgabe/pruefung.json").catch(() => null), OUTL(), ZIPL()]);
  const c = (check && check.counts) || {};
  const tile = (label, value, note, href) => h(href ? "a" : "div", { class: "tile", href }, h("div", { class: "label" }, label),
    h("div", { class: "value" }, value), note ? h("div", { class: "note" }, note) : null);
  const previews = outl.filter(f => f.path.startsWith("vorschau/") && f.path.endsWith(".png")).length;
  const changed = info.files.filter(f => f.matches === false), unknown = info.files.filter(f => f.matches == null);
  const files = smallTable([["File"], ["Size", 1], ["Modified"], ["SHA-256"], ["Manifest"]], info.files.map(f => ({
    f, cells: [h("strong", {}, f.name), bytes(f.bytes), new Date(f.mtime * 1000).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" }),
      h("span", { class: "mono", title: f.sha256 }, f.sha256.slice(0, 12) + "…"),
      f.matches === true ? chip("matches", "good") : f.matches === false ? chip("edited after import", "warn") : h("span", { class: "muted" }, "not listed")],
  })), { onRow: r => openFile(r.f.name) });
  const valid = check ? card("Validator", "ausgabe/pruefung.json, written by pruefe_daten.py",
    h("p", {}, check.passed ? chip("passed", "good") : chip("failed", "bad"), " ",
      h("span", { class: "ink2" }, `${check.errors.length} errors, ${check.notes.length} notes`)),
    check.errors.length ? h("ul", {}, check.errors.map(e => h("li", { class: "error" }, String(e)))) : null,
    check.notes.length ? h("ul", {}, check.notes.map(e => h("li", {}, String(e)))) : null,
    kv(Object.entries(c).map(([k, v]) => [k, thousands(v)])),
    h("p", { class: "foot" }, "Checks unique keys, links between configurations and sources, five-seed coverage, fold counts, metric ranges, epochs, "
      + "agreement of Phase-1 Mean values with their folds, and the patience-40 exclusion outside CUB."))
    : card("Validator", null, h("p", { class: "ink2" }, "No ausgabe/pruefung.json yet. Run python3 abbildungen_src/pruefe_daten.py."));
  const start = card("Start here", "ausgabe/uebersicht.csv",
    h("p", {}, "One row per dataset, method, loss and budget (450 settings): Mean and Concat MAP@R with seed standard deviations, paired gains over ÜBl. and selected / trained epochs. It is derived from the six source tables."),
    h("p", {}, a("Open the 450 settings", () => go("explore", { src: "out:uebersicht.csv" })), " · ", a("Phase 1 matrices", () => go("phase1")), " · ", a("Geometry, pseudo-labels and neighborhoods", () => go("diagnostics")), " · ",
      a("All tables and manifest views", () => go("explore"))));
  return [
    head("Condensed thesis data", "Everything the figure and table generators read, in one portable bundle. This page reads daten/ and ausgabe/ directly and never changes them."),
    h("div", { class: "stack" },
      h("div", { class: "tiles" },
        tile("Phase-1 result rows", thousands(c["results.csv"] ?? NaN), "Mean and Concat per seed", "#/phase1"),
        tile("Fold records", thousands(c["folds.csv"] ?? NaN), "Phase 1, Phase 2, external", "#/explore?src=file:folds.csv"),
        tile("Ablation rows", thousands(c["ablations.csv"] ?? NaN), "variants, sweeps, k = 4", "#/phase2"),
        tile("External-image runs", thousands(c["external.csv"] ?? NaN), "CompCars, Fashion200k", "#/external"),
        tile("Configurations", thousands(c.configurations ?? NaN), `${thousands(c.hyperparameters ?? NaN)} values`, "#/hp"),
        tile("Sources", thousands(c.sources ?? NaN), "paths with SHA-256", "#/explore?src=file:sources.csv"),
        tile("details.zip", String(zl.length), "members", "#/details"),
        tile("Figures", String(previews), "generated previews", "#/figures")),
      h("div", { class: "grid2" }, start, valid),
      card("Files in daten/", changed.length ? `${changed.length} file(s) differ from the import checksums in manifest.json (expected after intentional edits)`
        : `all ${info.files.length - unknown.length} listed files match manifest.json`, files),
      readme ? card("README", "daten/README.md", markdown(readme)) : null),
  ];
}
function openFile(name) {
  if (name.endsWith(".csv")) go("explore", { src: "file:" + name });
  else if (name === "details.zip") go("details");
  else if (name.endsWith(".json")) json(daten(name)).then(j => drawer("daten/", name, jsonTree(j)));
  else text(daten(name)).then(t => drawer("daten/", name, name.endsWith(".md") ? markdown(t) : h("pre", {}, t)));
}

/* ---------- Phase 1 ---------- */
function phase1Index(rows, bcol) {
  const m = new Map();
  for (const o of rows) { const k = `${o.method}|${o[bcol]}|${o.scope}`; if (!m.has(k)) m.set(k, []); m.get(k).push(o); }
  const get = (method, b, scope) => (m.get(`${method}|${b}|${scope}`) || []).slice().sort((x, y) => x.seed - y.seed);
  const gain = (method, b, scope, metric) => {
    const base = new Map(get("sup_" + method.split("_")[1], b, scope).map(o => [o.seed, o[metric]]));
    return get(method, b, scope).filter(o => base.get(o.seed) != null && o[metric] != null).map(o => ({ seed: o.seed, d: 100 * (o[metric] - base.get(o.seed)) }));
  };
  return { get, gain };
}
/* Mittel über fünf Seeds und fünf Verluste: die Werte, die die Arbeit berichtet */
const ACROSS_SCOPES = [["mean", "Mean (single model)"], ["concat", "Concat (ensemble)"]];
const THESIS_TABLES = {
  cars: "Figure 5.1 and Table B.6, the negative counts are Table B.8", cub: "Figure 5.1 and Table B.6, the negative counts are Table B.8",
  inshop: "Figure 5.1 and Table B.6, the negative counts are Table B.8", semiinat: "Figure 5.1 and Table B.6, the negative counts are Table B.8",
  inshop10: "Figure 5.3 and Table B.7",
};
const acrossValues = (P, mk, b, scope, metric, gain) => LOSSES.map(([lk, ln]) => ({
  lk, ln, seeds: gain ? P.gain(`${mk}_${lk}`, b, scope, metric).map(x => [x.seed, x.d])
    : P.get(`${mk}_${lk}`, b, scope).filter(o => o[metric] != null).map(o => [o.seed, pct(o[metric])]),
}));
function across(P, mk, b, scope, metric, gain) {
  const v = acrossValues(P, mk, b, scope, metric, gain).flatMap(l => l.seeds.map(x => x[1]));
  return { m: mean(v), s: sd(v), n: v.length, neg: v.filter(x => x < 0).length };
}
// Eine Nachkommastelle wie in der Arbeit; was auf null rundet, trägt kein Vorzeichen
const oneDecimal = (x, gain) => (x == null ? "–" : !gain ? fix(x, 1) : +x.toFixed(1) === 0 ? "0.0" : signed(x, 1));
function acrossCard(P, ds, view, budgets, metric, on) {
  const gain = view === "gain", ssl = METHODS.filter(([mk]) => mk !== "sup");
  const gainOf = (mk, b, scope) => across(P, mk, b, scope, metric, true);
  const vmax = !gain ? 1 : Math.max(0.5, ...ACROSS_SCOPES.flatMap(([sc]) => ssl.flatMap(([mk]) => budgets.map(b => Math.abs(gainOf(mk, b, sc).m ?? 0)))));
  // Fett wie in der Arbeit: über alle vier SSL-Verfahren, auch wenn einzelne ausgeblendet sind
  const top = (b, scope) => Math.max(...ssl.map(([mk]) => +(gainOf(mk, b, scope).m ?? -Infinity).toFixed(1)));
  const tbody = h("tbody", {});
  for (const [scope, name] of ACROSS_SCOPES) {
    tbody.append(h("tr", { class: "section" }, h("td", { colspan: budgets.length + 1 }, name)));
    for (const [mk, mn] of on) {
      const asGain = gain && mk !== "sup";
      tbody.append(h("tr", {}, h("td", { class: "rowh" }, swatch(mcolor(mk)), mn), budgets.map(b => {
        const x = across(P, mk, b, scope, metric, asGain), title = `${mn} · ${scope === "mean" ? "Mean" : "Concat"} · ${blabel(ds, b)}`;
        if (!x.n) return h("td", {}, cellEl({ main: "–", kind: "empty" }));
        const open = () => acrossDrawer(ds, mk, b, metric, asGain);
        const unit = asGain ? " pp" : " %", three = v => (asGain ? signed(v, 3) : fix(v, 3));
        const tipBuild = () => [tipTitle(title), tipRow(`Mean of ${x.n} ${asGain ? "differences" : "runs"}`, three(x.m) + unit), tipRow("SD", fix(x.s, 3) + " pp"),
          ...acrossValues(P, mk, b, scope, metric, asGain).map(l => tipRow(l.ln, three(mean(l.seeds.map(v => v[1]))) + unit)), tipSub(`Click for all ${x.n} values.`)];
        if (gain && mk === "sup") return h("td", {}, cellEl({ main: fix(x.m, 1), sub: "baseline", kind: "ref", onClick: open, tipBuild }));
        const rounded = +x.m.toFixed(1), t = gain ? top(b, scope) : 0;
        return h("td", {}, cellEl({ main: oneDecimal(x.m, gain), sub: x.s != null ? "± " + fix(x.s, 1) : null, n: gain ? `${x.neg}/${x.n} negative` : null,
          bg: gain ? diverging(x.m / vmax) : null, best: gain && t > 0 && rounded === t, onClick: open, tipBuild }));
      })));
    }
  }
  const isMap = metric === "mean_average_precision_at_r";
  return card(gain ? `Thesis values · gain over ÜBl. · ${MET[metric]}` : `Mean over seeds and losses · ${MET[metric]} [%]`,
    (gain ? "arithmetic mean ± SD of the 25 paired differences (five seeds × five losses), in percentage points, rounded as in the thesis; ▲ bold: largest rounded gain per scope and budget over all four SSL methods, if positive"
      : "arithmetic mean ± SD of the 25 runs (five seeds × five losses)"),
    h("div", { class: "scrollx" }, h("table", { class: "matrix" },
      h("thead", {}, h("tr", {}, h("th", { class: "rowh" }, "Method"), budgets.map(b => h("th", {}, blabel(ds, b))))), tbody)),
    gain ? divLegend(+vmax.toFixed(1), "worse than ÜBl.", "better than ÜBl.") : null,
    h("p", { class: "foot" }, gain && isMap ? `In the thesis these are ${THESIS_TABLES[ds]}. They average the per-loss values in the next table. Click a cell for all 25 differences.`
      : gain ? "The thesis reports these gains for MAP@R. They average the per-loss values in the next table. Click a cell for all values."
        : "They average the per-loss values in the next table. Click a cell for all values."));
}
async function acrossDrawer(ds, mk, b, metric, gain) {
  const r = await R();
  const bcol = ds === "inshop10" ? "budget_percent" : "k";
  const P = phase1Index(r.objects().filter(o => o.dataset_key === ds && o[bcol] === b), bcol);
  const fmt = gain ? signed : fix, unit = gain ? "pp" : "%";
  const block = ([scope, name]) => {
    const losses = acrossValues(P, mk, b, scope, metric, gain), maps = losses.map(l => new Map(l.seeds));
    const seeds = uniq(losses.flatMap(l => l.seeds.map(x => x[0]))).sort((x, y) => x - y);
    const x = across(P, mk, b, scope, metric, gain);
    return h("section", {}, h("h3", {}, `${name} · ${MET[metric]} [${unit}]`),
      kv([[`Mean of the ${x.n} values`, fmt(x.m, 3)], ["As printed", `${oneDecimal(x.m, gain)} ± ${fix(x.s, 1)}`], ["SD of the values", fix(x.s, 3)], ...(gain ? [["Negative differences", `${x.neg} of ${x.n}`]] : [])]),
      smallTable([["Loss"], ...seeds.map(sd_ => [`Seed ${sd_}`, 1]), ["Mean", 1]], [
        ...losses.map((l, i) => ({ cells: [a(l.ln, () => runDrawer(ds, `${mk}_${l.lk}`, b, metric)), ...seeds.map(sd_ => fmt(maps[i].get(sd_))), fmt(mean(l.seeds.map(v => v[1])))] })),
        { cells: [h("strong", {}, "Mean"), ...seeds.map(sd_ => h("strong", {}, fmt(mean(maps.map(m => m.get(sd_)))))), h("strong", {}, fmt(x.m))] }]));
  };
  drawer(DS[ds], `${MNAME[mk]} · all five losses · ${blabel(ds, b)}`,
    h("section", {}, h("p", {}, gain ? `Each value is a run of ${MNAME[mk]} minus the ÜBl. run with the same loss, budget, seed and scope (results.csv), in percentage points. The thesis reports their arithmetic mean.`
      : `Each value is one run of ${MNAME[mk]} (results.csv). The summary is their arithmetic mean.`)),
    ...ACROSS_SCOPES.map(block),
    h("p", {}, a("Show these rows in results.csv", () => go("explore", { src: "file:results.csv", "f.dataset_key": "=" + ds, "f.method": mk + "_", ["f." + bcol]: "=" + b }))));
}
function methodToggles(on, shown) {
  const set = next => update({ show: next.length === METHODS.length ? null : METHODS.filter(([mk]) => next.includes(mk)).map(([mk]) => mk).join(",") });
  return h("div", { class: "field" }, h("span", {}, "Methods"),
    h("div", { class: "seg", role: "group", "aria-label": "Methods shown" },
      METHODS.map(([mk, mn]) => h("button", {
        type: "button", "aria-pressed": String(shown(mk)), title: shown(mk) ? `Hide ${mn}` : `Show ${mn}`,
        onclick: () => {
          const now = on.map(([k]) => k), next = shown(mk) ? now.filter(k => k !== mk) : [...now, mk];
          if (next.length) set(next);  // mindestens ein Verfahren bleibt sichtbar
        },
      }, swatch(mcolor(mk)), mn)),
      h("button", { type: "button", "aria-pressed": String(on.length === METHODS.length), onclick: () => set(METHODS.map(([mk]) => mk)) }, "All")));
}
async function phase1(p) {
  const r = await R();
  const ds = DS[p.ds] ? p.ds : "cars", scope = p.scope === "concat" ? "concat" : "mean";
  const view = p.view === "gain" ? "gain" : "abs", by = p.by === "loss" ? "loss" : "method", bestBy = p.best === "method" ? "method" : "loss";
  const rows = r.objects().filter(o => o.dataset_key === ds);
  const bcol = ds === "inshop10" ? "budget_percent" : "k";
  const budgets = uniq(rows.map(o => o[bcol])).sort((x, y) => x - y);
  const metrics = METRICS.filter(([m]) => rows.some(o => o[m] != null));
  const metric = metrics.some(([m]) => m === p.metric) ? p.metric : "mean_average_precision_at_r";
  const P = phase1Index(rows, bcol);
  const picked = String(p.show || "").split(",").filter(m => METHODS.some(([mk]) => mk === m));
  const on = picked.length ? METHODS.filter(([mk]) => picked.includes(mk)) : METHODS;
  const shown = mk => on.some(([k]) => k === mk);
  const stat = (method, b) => {
    if (view === "gain") { const d = P.gain(method, b, scope, metric).map(x => x.d); return { m: mean(d), s: sd(d), n: d.length, pos: d.filter(x => x > 0).length, d }; }
    const v = P.get(method, b, scope).map(o => pct(o[metric])); return { m: mean(v), s: sd(v), n: v.length, v };
  };
  const sections = by === "method"
    ? on.map(([mk, mn]) => ({ title: mn, color: mcolor(mk), rows: LOSSES.map(([lk, ln]) => ({ label: ln, mk, lk })) }))
    : LOSSES.map(([lk, ln]) => ({ title: ln, rows: on.map(([mk, mn]) => ({ label: mn, color: mcolor(mk), mk, lk })) }));
  const stats = new Map();
  for (const [mk] of METHODS) for (const [lk] of LOSSES) for (const b of budgets) stats.set(`${mk}_${lk}|${b}`, stat(`${mk}_${lk}`, b));
  // Hervorhebung je Verlust (bestes Verfahren) oder je Verfahren (bester Verlust), getrennt für jedes Budget
  const best = new Map();
  for (const b of budgets) {
    const groups = bestBy === "loss" ? LOSSES.map(([lk]) => on.filter(([mk]) => !(view === "gain" && mk === "sup")).map(([mk]) => `${mk}_${lk}`))
      : on.filter(([mk]) => !(view === "gain" && mk === "sup")).map(([mk]) => LOSSES.map(([lk]) => `${mk}_${lk}`));
    for (const g of groups) {
      if (g.length < 2) continue;  // nur ein gezeigtes Verfahren: nichts zu vergleichen
      let bm = -Infinity, arg = null;
      for (const c of g) { const x = stats.get(`${c}|${b}`).m; if (x != null && x > bm) { bm = x; arg = c; } }
      if (arg) best.set(`${arg}|${b}`, true);
    }
  }
  const dmax = Math.max(0.5, niceMax(quantileAbs([...stats.entries()].filter(([k]) => !k.startsWith("sup_")).map(([, x]) => x.m))));
  const thead = h("thead", {}, h("tr", {}, h("th", { class: "rowh" }, by === "method" ? "Method · loss" : "Loss · method"), budgets.map(b => h("th", {}, blabel(ds, b)))));
  const tbody = h("tbody", {});
  for (const sec of sections) {
    tbody.append(h("tr", { class: "section" }, h("td", { colspan: budgets.length + 1 }, sec.color ? swatch(sec.color) : null, sec.title)));
    for (const row of sec.rows) {
      const method = `${row.mk}_${row.lk}`;
      tbody.append(h("tr", {}, h("td", { class: "rowh" }, row.color ? swatch(row.color) : null, row.label), budgets.map(b => {
        const st = stats.get(`${method}|${b}`), title = `${MNAME[row.mk]} · ${LNAME[row.lk]} · ${blabel(ds, b)}`;
        const open = () => runDrawer(ds, method, b, metric);
        if (st.n === 0 && !(view === "gain" && row.mk === "sup")) return h("td", {}, cellEl({ main: "–", kind: "empty" }));
        if (view === "gain" && row.mk === "sup") {
          const v = P.get(method, b, scope).map(o => pct(o[metric]));
          return h("td", {}, cellEl({ main: fix(mean(v)), sub: "baseline", kind: "ref", onClick: open,
            tipBuild: () => [tipTitle(title), tipRow("ÜBl. value", fix(mean(v)) + " %"), tipSub("Gains in this column are paired against these runs.")] }));
        }
        const bg = view === "gain" && st.m != null ? diverging(st.m / dmax) : null;
        return h("td", {}, cellEl({
          main: view === "gain" ? signed(st.m) : fix(st.m), sub: st.s != null ? "± " + fix(st.s) : null,
          n: view === "gain" ? `${st.pos}/${st.n} seeds > 0` : null, bg, best: best.has(`${method}|${b}`), onClick: open,
          tipBuild: () => [tipTitle(title),
            view === "gain" ? tipRow("Paired gain over ÜBl.", `${signed(st.m)} ± ${fix(st.s)} pp`) : tipRow(`${MET[metric]} (${scope})`, `${fix(st.m)} ± ${fix(st.s)} %`),
            ...(view === "gain" ? P.gain(method, b, scope, metric).map(x => tipRow(`seed ${x.seed}`, signed(x.d) + " pp"))
              : P.get(method, b, scope).map(o => tipRow(`seed ${o.seed}`, fix(pct(o[metric])) + " %"))),
            tipSub("Click for seeds, folds, settings and sources.")],
        }));
      })));
    }
  }
  const panels = LOSSES.map(([lk, ln]) => ({
    title: ln, series: on.filter(([mk]) => view === "abs" || mk !== "sup").map(([mk, mn]) => ({
      name: mn, color: mcolor(mk), values: budgets.map(b => stats.get(`${mk}_${lk}|${b}`).m),
    })),
  }));
  const legendItems = on.filter(([mk]) => view === "abs" || mk !== "sup").map(([mk, mn]) => ({ name: mn, color: mcolor(mk) }));
  const xt = ds === "inshop10" ? "labeled classes (%)" : "labeled images per class (k)";
  return {
    nodes: [
      head("Phase 1", `${DS[ds]}: all five methods × five losses × ${budgets.length} budgets, five seeds each (results.csv).`),
      h("div", { class: "filters" },
        seg("Dataset", Object.entries(DS), ds, v => update({ ds: v })),
        seg("Scope", [["mean", "Mean"], ["concat", "Concat"]], scope, v => update({ scope: v })),
        select("Metric", metrics, metric, v => update({ metric: v })),
        seg("Show", [["abs", "Values"], ["gain", "Gain over ÜBl."]], view, v => update({ view: v })),
        seg("Rows", [["method", "By method"], ["loss", "By loss"]], by, v => update({ by: v })),
        seg("Bold", [["loss", "Best method per loss"], ["method", "Best loss per method"]], bestBy, v => update({ best: v })),
        methodToggles(on, shown)),
      h("div", { class: "stack" },
        acrossCard(P, ds, view, budgets, metric, on),
        card(view === "gain" ? `Per loss · paired gain over ÜBl. · ${MET[metric]} · ${scope === "mean" ? "Mean" : "Concat"}` : `Per loss · ${MET[metric]} · ${scope === "mean" ? "Mean" : "Concat"} [%]`,
          (view === "gain" ? "mean ± SD of the five seed-wise differences to ÜBl. with the same loss, budget and seed" : "mean ± sample SD over five seeds")
            + (bestBy === "loss" ? `; ▲ bold: best ${on.length < METHODS.length ? "shown " : ""}method for this loss and budget` : "; ▲ bold: best loss for this method and budget"),
          h("div", { class: "scrollx" }, h("table", { class: "matrix" }, thead, tbody)),
          view === "gain" ? divLegend(dmax, "worse than ÜBl.", "better than ÜBl.") : null),
        card(view === "gain" ? "Gain by budget" : `${MET[metric]} by budget`, "one panel per loss, shared y-axis; hover for every method at a budget",
          legend(legendItems), multiples(panels, { xs: budgets, log: true, xlabel: b => fmtNum(b), xtitle: xt, zero: view === "gain",
            tipfmt: v => (view === "gain" ? signed(v) + " pp" : fix(v) + " %"), yfmt: v => fmtNum(v) }))),
    ],
    after: p.open ? () => { const [m, b] = p.open.split("|"); runDrawer(ds, m, +b, metric); } : null,
  };
}
async function runDrawer(ds, method, b, metric = "mean_average_precision_at_r") {
  const [r, f, hpx, src] = await Promise.all([R(), F(), hpIndex(), srcIndex()]);
  const bcol = ds === "inshop10" ? "budget_percent" : "k";
  const rows = r.objects().filter(o => o.dataset_key === ds && o.method === method && o[bcol] === b);
  const [mk, lk] = method.split("_");
  const P = phase1Index(r.objects().filter(o => o.dataset_key === ds && o[bcol] === b), bcol);
  const seeds = uniq(rows.map(o => o.seed)).sort((x, y) => x - y);
  const by = (seed, scope) => rows.find(o => o.seed === seed && o.scope === scope);
  const g = scope => new Map(P.gain(method, b, scope, metric).map(x => [x.seed, x.d]));
  const gm = g("mean"), gc = g("concat");
  const runIds = new Set(rows.map(o => o.run_id));
  const folds = f.objects().filter(o => o.experiment === "phase1" && runIds.has(o.run_id)).sort((x, y) => x.seed - y.seed || x.fold - y.fold);
  const cfgs = uniq(rows.map(o => o.config_id));
  const first = hpx.get(cfgs[0]);
  const differing = new Set();
  if (first) for (const c of cfgs.slice(1)) {
    const other = new Map((hpx.get(c) || { params: [] }).params);
    for (const [k, v] of first.params) if (String(other.get(k)) !== String(v)) differing.add(k);
  }
  drawer(DS[ds], `${MNAME[mk]} · ${LNAME[lk]} · ${blabel(ds, b)}`,
    h("section", {}, kv([["Seeds", seeds.join(", ")], ["Epoch cap", uniq(rows.map(o => o.epochs)).join(", ")], ["Patience", uniq(rows.map(o => o.patience)).join(", ")],
      ["Series (import)", uniq(rows.map(o => o.series)).join(", ") || "–"], ["Run IDs", h("span", { class: "mono" }, [...runIds][0].replace(/:\d+$/, ":<seed>"))]])),
    h("section", {}, h("h3", {}, `Per seed · ${MET[metric]} [%]`), smallTable(
      [["Seed", 1], ["Mean", 1], ["Concat", 1], ["Δ ÜBl. Mean", 1], ["Δ ÜBl. Concat", 1], ["Configuration"]],
      seeds.map(sd_ => ({ cells: [sd_, fix(pct(by(sd_, "mean")?.[metric])), fix(pct(by(sd_, "concat")?.[metric])),
        mk === "sup" ? "–" : signed(gm.get(sd_)), mk === "sup" ? "–" : signed(gc.get(sd_)), linkOrText("config_id", by(sd_, "mean")?.config_id)] })))),
    h("section", {}, h("h3", {}, `Folds (${folds.length})`), folds.length ? smallTable(
      [["Seed", 1], ["Fold", 1], ["Test MAP@R", 1], ["Best valid MAP@R", 1], ["Selected epoch", 1], ["Trained epochs", 1], ["Warm-up", 1]],
      folds.map(o => ({ cells: [o.seed, o.fold, fix(pct(o.mean_average_precision_at_r)), fix(pct(o.best_valid_map)), o.selected_epoch, o.epochs_trained, fmtNum(o.warmup_epochs)] })))
      : h("p", { class: "ink2" }, "No fold records."), h("p", { class: "foot" }, "Selected epoch is zero-based; −1 marks the untrained head.")),
    first ? h("section", {}, h("h3", {}, "Settings of seed " + seeds[0]), differing.size ? h("p", { class: "ink2" }, "Differs between seeds: " + [...differing].join(", ")) : null,
      kv(first.params.map(([k, v]) => [k, v == null ? "–" : String(v)])), h("p", {}, a("Open configuration " + cfgs[0], () => configDrawer(cfgs[0])))) : null,
    h("section", {}, h("h3", {}, "Sources"), smallTable([["Seed", 1], ["cv_summary.json"], ["SHA-256"]], seeds.map(sd_ => {
      const o = by(sd_, "mean"), rec = o && src.byPath.get(o.source);
      return { cells: [sd_, h("span", { class: "path", title: o?.source }, middle(o?.source, 70)), rec ? h("span", { class: "mono" }, rec.sha256.slice(0, 12) + "…") : "–"] };
    }), { onRow: row => sourceDrawer({ path: by(row.cells[0], "mean").source }) })),
    h("p", {}, a("Show these rows in results.csv", () => go("explore", { src: "file:results.csv", "f.run_id": [...runIds][0].replace(/:\d+$/, ":") }))));
}

/* ---------- Phase 2 ---------- */
const P2T = [["phase2|Cars196", "Cars196 · k = 16"], ["phase2|DeepFashionInShop", "In-Shop · k = 2"], ["phase2_k8|Cars196", "Cars196 · k = 8"]];
const LOSSKEY = { ArcFaceLoss: "arc", CircleLoss: "cir", MultiSimilarityLoss: "ms", NTXentLoss: "nt", ProxyAnchorLoss: "prx" };
const HPO_BUDGET = { Cars196: "k = 16", DeepFashionInShop: "k = 2" };
// Bester Trial jeder Phase-1-Studie: "Datensatz|Methode" -> Parameter -> Verlust -> Wert. Ohne die Serie mit gelabelten Klassen (class_subset_label_*)
const hpoChosen = () => once("hpoChosen", () => csv(zip("tables/hpo_sensitivitaet/best_params.csv")).then(t => {
  const m = new Map();
  for (const o of t.objects()) {
    if (o.phase !== 1 || !o.study_id.includes("k_shot_label_1_k_")) continue;
    const key = `${o.dataset}|${o.method}`;
    if (!m.has(key)) m.set(key, new Map());
    if (!m.get(key).has(o.short)) m.get(key).set(o.short, {});
    m.get(key).get(o.short)[LOSSKEY[o.loss]] = o.best_raw;
  }
  return m;
}).catch(() => new Map()));
const fmtChosen = v => (v == null ? "–" : Number.isFinite(+v) ? fmtNum(+v) : v);
const chosenLine = vals => tipSub("Chosen by HPO: " + LOSSES.map(([lk, ln]) => `${ln} ${fmtChosen(vals[lk])}`).join(", "));
// Rauten: x = Wert der HPO, y = regulärer Lauf in der gewählten Ansicht. result(Verlust) liefert [y, Wert in %]
const chosenMarks = (vals, result) => LOSSES.flatMap(([lk, ln]) => {
  const x = vals[lk] == null ? NaN : +vals[lk], [y, value] = result(lk);
  return Number.isFinite(x) && y != null && value != null ? [{ x, y, value, name: ln, color: lcolor(lk) }] : [];
});
const markTip = hit => [tipTitle("Chosen by HPO → regular run"),
  ...hit.map(k => tipRow(k.name, `${fmtNum(k.x)} → ${fix(k.value)} %`, h("span", { class: "linekey", style: { background: k.color } })))];
const logAxis = (xs, marks) => { const all = [...xs, ...marks.map(k => k.x)], lo = Math.min(...all); return lo > 0 && Math.max(...all) / lo >= 8; };
function lossLegend() {
  const el = legend(LOSSES.map(([lk, ln]) => ({ name: ln, color: lcolor(lk) })));
  el.append(h("span", { class: "item" }, h("span", { class: "diamondkey" }), "value chosen by HPO, at the result of the regular run"));
  return el;
}
// Tabelle über den Diagrammen: gewählte Werte je Verlust neben den Werten des Sweeps, darunter die übrigen Parameter der Suche
function chosenBlock(params, swept, note) {
  if (!params) return h("p", { class: "ink2" }, "The values chosen by HPO could not be loaded.");
  const name = n => h("span", { class: "mono" }, n);
  const lossCols = () => LOSSES.map(([lk, ln]) => [h("span", {}, swatch(lcolor(lk)), ln), 1]);
  const values = n => LOSSES.map(([lk]) => fmtChosen(params.get(n)?.[lk]));
  const rest = [...params.keys()].filter(n => !swept.has(n)).sort((x, y) => /^(loss|miner):/.test(x) - /^(loss|miner):/.test(y) || x.localeCompare(y));
  return h("div", { class: "chosen" }, h("h4", {}, "Chosen by HPO"), h("p", { class: "note" }, note),
    smallTable([["Parameter"], ...lossCols(), ["Swept values"]], [...swept].map(([n, xs]) => ({ cells: [name(n), ...values(n), xs.map(fmtNum).join(", ")] }))),
    rest.length ? h("details", {}, h("summary", {}, `${rest.length} more parameters chosen by HPO and not swept`),
      smallTable([["Parameter"], ...lossCols()], rest.map(n => ({ cells: [name(n), ...values(n)] })))) : null);
}
async function phase2(p) {
  const ab = await A();
  const tab = ["variants", "sweeps", "slade", "k4"].includes(p.tab) ? p.tab : "variants";
  const scope = p.scope === "concat" ? "concat" : "mean";
  const tabs = seg("Part", [["variants", "Variants"], ["sweeps", "Hyperparameter sweeps"], ["slade", "SLADE sweeps"], ["k4", "k = 4 ensemble"]], tab, v => update({ tab: v }));
  const scopeSeg = seg("Scope", [["mean", "Mean"], ["concat", "Concat"]], scope, v => update({ scope: v }));
  const intro = head("Phase 2", "Ablations on Cars196 (k = 16 and k = 8) and In-Shop (k = 2), seed 2, all five losses (ablations.csv). Values in %, differences in percentage points.");
  if (tab === "k4") return [intro, h("div", { class: "filters" }, tabs), ...k4View(ab, p)];
  if (tab === "slade") return [intro, ...await sladeView(ab, p, tabs, scopeSeg, scope)];
  const tsel = P2T.some(([k]) => k === p.t) ? p.t : P2T[0][0], [tk, dsn] = tsel.split("|");
  const view = ["base", "sup", "abs"].includes(p.view) ? p.view : "base";
  const filters = h("div", { class: "filters" }, tabs, seg("Table", P2T, tsel, v => update({ t: v })), scopeSeg,
    seg("Show", [["base", "Δ regular run"], ["sup", "Δ ÜBl."], ["abs", "Values"]], view, v => update({ view: v })));
  const rows = ab.objects().filter(o => o.table_key === tk && o.dataset === dsn);
  const val = o => (o == null ? null : view === "abs" ? o[`${scope}_value`] : view === "base" ? o[`${scope}_diff_baseline`] : o[`${scope}_diff_supervised`]);
  if (tab === "sweeps") return [intro, filters, await sweepsView(rows.filter(o => o.group === "Sweep"), rows, dsn, tk, val, view, scope)];
  const vrows = rows.filter(o => o.group && o.group !== "Sweep");
  const methods = ["lp", "lrml", "hoffer", "slade", "sup"].filter(m => vrows.some(o => o.method === m));
  const shown = vrows.map(val).filter(v => v != null);
  const dmax = Math.max(0.5, niceMax(quantileAbs(shown)));
  const isRegular = o => o.cell === "ssl_hp/with_data" || (o.method === "sup" && o.cell === "sup_hp/no_data");
  const tbody = h("tbody", {});
  for (const m of methods) {
    const mine = vrows.filter(o => o.method === m);
    const vkey = o => o.variant || "defaults";  // die Circle-Läufe der Ausgangsserie haben keine Variante
    const keys = uniq(mine.map(vkey));
    const meta = k => mine.find(o => vkey(o) === k);
    keys.sort((x, y) => {
      const a1 = meta(x), b1 = meta(y), go_ = g => { const i = Object.keys(GROUPS).indexOf(g); return i < 0 ? 9 : i; };
      return go_(a1.group) - go_(b1.group) || CELLS.indexOf(a1.cell) - CELLS.indexOf(b1.cell) || String(a1.label_de).localeCompare(String(b1.label_de));
    });
    tbody.append(h("tr", { class: "section" }, h("td", { colspan: 6 }, swatch(mcolor(m)), MNAME[m])));
    for (const k of keys) {
      const mm = meta(k), hidden = String(mm.uses_hidden_labels) === "True";
      tbody.append(h("tr", {}, h("td", { class: "rowh" }, mm.label_de || k, h("span", { class: "tag" }, GROUPS[mm.group] || mm.group), hidden ? h("span", { class: "tag" }, "hidden labels") : null),
        LOSSES.map(([lk, ln]) => {
          const o = mine.find(x => vkey(x) === k && x.loss === lk);
          if (!o) return h("td", {}, cellEl({ main: "–", kind: "empty" }));
          const v = val(o), ref = view !== "abs" && (v == null || (view === "base" && isRegular(o)) || (view === "sup" && o.method === "sup"));
          const tipBuild = () => [tipTitle(`${MNAME[m]} · ${ln} · ${mm.label_de || k}`), tipRow("Value", fix(o[`${scope}_value`]) + " %"),
            tipRow("Δ regular run", signed(o[`${scope}_diff_baseline`]) + " pp"), tipRow("Δ ÜBl.", signed(o[`${scope}_diff_supervised`]) + " pp"),
            tipSub(`variant ${o.variant || "(regular)"}${o.cell ? " · " + o.cell : ""}`)];
          if (ref) return h("td", {}, cellEl({ main: fix(o[`${scope}_value`]), sub: view === "base" ? "regular" : "reference", kind: "ref", onClick: () => ablationDrawer(o), tipBuild }));
          return h("td", {}, cellEl({ main: view === "abs" ? fix(v) : signed(v), bg: view === "abs" ? null : diverging(v / dmax), onClick: () => ablationDrawer(o), tipBuild }));
        })));
    }
  }
  return [intro, filters, h("div", { class: "stack" }, card(`${P2T.find(([k2]) => k2 === tsel)[1]} · ${scope === "mean" ? "Mean" : "Concat"} MAP@R`,
    view === "base" ? "difference to the regular run of the same method and loss" : view === "sup" ? "difference to ÜBl. with the same loss" : "values in %",
    h("div", { class: "scrollx" }, h("table", { class: "matrix" }, h("thead", {}, h("tr", {}, h("th", { class: "rowh" }, "Variant"), LOSSES.map(([, ln]) => h("th", {}, ln)))), tbody)),
    view === "abs" ? null : divLegend(dmax, view === "base" ? "worse than regular" : "worse than ÜBl.", view === "base" ? "better than regular" : "better than ÜBl."),
    h("p", { class: "foot" }, "Each cell is one run with seed 2. The 2×2 decomposition separates the hyperparameters (SSL or ÜBl.) from the use of unlabeled images. Oracle variants use labels that are hidden from the method.")))];
}
const parseSweep = v => { const m = String(v || "").match(/^(.*?)_(-?\d+(?:p\d+)?)$/); return m ? [m[1], +m[2].replace("p", ".")] : null; };
async function sweepsView(rows, all, dsn, tk, val, view, scope) {
  const chosen = await hpoChosen();
  const regular = (m, l) => all.find(o => o.method === m && o.loss === l && o.cell === "ssl_hp/with_data");
  const byMethod = new Map();
  for (const o of rows) {
    const ps = parseSweep(o.variant); if (!ps) continue;
    const key = o.method;
    if (!byMethod.has(key)) byMethod.set(key, new Map());
    const pm = byMethod.get(key);
    if (!pm.has(ps[0])) pm.set(ps[0], []);
    pm.get(ps[0]).push({ ...o, x: ps[1] });
  }
  const out = [];
  for (const [m, params] of byMethod) {
    const picks = chosen.get(`${dsn}|${m}`), swept = new Map();
    const panels = [...params].map(([param, list]) => {
      const xs = uniq(list.map(o => o.x)).sort((x, y) => x - y), vals = picks?.get(param) || {};
      swept.set(param, xs);
      const marks = chosenMarks(vals, lk => { const o = regular(m, lk); return [view === "base" ? 0 : val(o), o?.[`${scope}_value`]]; });
      return {
        title: param, note: xs.length === 1 ? "1 value" : `${xs.length} values`, xs, marks, marktip: markTip, log: logAxis(xs, marks),
        series: LOSSES.map(([lk, ln]) => ({ name: ln, color: lcolor(lk), values: xs.map(x => val(list.find(o => o.loss === lk && o.x === x))) })),
        extra: () => [chosenLine(vals),
          view === "abs" ? tipSub("Regular run: " + LOSSES.map(([lk, ln]) => `${ln} ${fix(regular(m, lk)?.[`${scope}_value`])}`).join(", ")) : null],
      };
    });
    out.push(card(`${MNAME[m]} sweeps`, view === "abs" ? "values in %" : "difference in pp; 0 = regular run (Phase-1 value of the parameter)",
      chosenBlock(picks, swept, `Best trial of each Phase-1 study, tuned at ${HPO_BUDGET[dsn]}${tk === "phase2_k8" ? "; the k = 8 runs reuse these values" : ""}.`),
      lossLegend(),
      multiples(panels, { zero: view !== "abs", xlabel: x => fmtNum(x), tipfmt: v => (view === "abs" ? fix(v) + " %" : signed(v) + " pp"), yfmt: v => fmtNum(v) })));
  }
  return h("div", { class: "stack" }, out.length ? out : [card("No sweeps", null, h("p", { class: "ink2" }, "This table has no sweep variants."))]);
}
async function sladeView(ab, p, tabs, scopeSeg, scope) {
  const dsn = p.sds === "DeepFashionInShop" ? "DeepFashionInShop" : "Cars196";
  const view = p.view === "abs" ? "abs" : "diff";
  const rows = ab.objects().filter(o => o.table_key === "slade_sweeps" && o.dataset === dsn);
  const params = uniq(rows.map(o => o.param));
  const picks = (await hpoChosen()).get(`${dsn}|slade`), swept = new Map();
  const panels = params.map(param => {
    const list = rows.filter(o => o.param === param), xs = uniq(list.map(o => o.value)).sort((x, y) => x - y);
    const pick = (lk, x) => list.find(o => o.loss === lk && o.value === x), vals = picks?.get(param) || {};
    swept.set(param, xs);
    const marks = chosenMarks(vals, lk => { const reg = list.find(o => o.loss === lk)?.[`regular_${scope}`]; return [view === "abs" ? reg : 0, reg]; });
    return {
      title: param, xs, marks, marktip: markTip, log: logAxis(xs, marks),
      series: LOSSES.map(([lk, ln]) => ({
        name: ln, color: lcolor(lk),
        values: xs.map(x => { const o = pick(lk, x); return o ? (view === "abs" ? o[`${scope}_value`] : o[`diff_${scope}`]) : null; }),
        hollow: xs.map(x => { const o = pick(lk, x); return o ? Math.abs(o[`diff_${scope}`]) <= o[`threshold_${scope}`] : false; }),
        ring: xs.map(x => { const o = pick(lk, x); return o ? String(o.is_tuned) === "True" : false; }),
      })),
      extra: i => {
        const os = LOSSES.map(([lk]) => pick(lk, xs[i])).filter(Boolean);
        return [chosenLine(vals), tipSub("Threshold: " + os.map(o => `${LNAME[o.loss]} ${fix(o[`threshold_${scope}`])}`).join(", ") + " pp"),
          os.some(o => String(o.is_tuned) === "True") ? tipSub("Large marker: the Phase-1 value of this parameter") : null];
      },
    };
  });
  return [h("div", { class: "filters" }, tabs, seg("Dataset", [["Cars196", "Cars196 · k = 16"], ["DeepFashionInShop", "In-Shop · k = 2"]], dsn, v => update({ sds: v })), scopeSeg,
    seg("Show", [["diff", "Δ regular run"], ["abs", "Values"]], view, v => update({ view: v }))),
  h("div", { class: "stack" }, card(`SLADE sweeps · ${dsn === "Cars196" ? "Cars196" : "In-Shop"}`, `${rows.length} runs; one parameter per panel, the others at their Phase-1 values`,
    chosenBlock(picks, swept, `Best trial of each Phase-1 study, tuned at ${HPO_BUDGET[dsn]}.`),
    lossLegend(),
    multiples(panels, { zero: view !== "abs", xlabel: x => fmtNum(x), tipfmt: v => (view === "abs" ? fix(v) + " %" : signed(v) + " pp"), yfmt: v => fmtNum(v) }),
    h("p", { class: "foot" }, "Filled markers: the difference exceeds the seed-noise threshold of that loss (threshold_mean / threshold_concat). Hollow markers: within the threshold. Large markers: sweep values equal to the value chosen by HPO."),
    h("p", {}, a("Open as table", () => go("explore", { src: "file:ablations.csv", "f.table_key": "=slade_sweeps", "f.dataset": "=" + dsn })))))];
}
function k4View(ab, p) {
  const rows = ab.objects().filter(o => o.table_key === "k4_ensemble");
  const meas = [["mean", "Mean MAP@R"], ["concat", "Concat MAP@R"], ["gap", "Concat − Mean"], ["topR_jaccard", "Top-R Jaccard"], ["fp_also_in_other_fold_topR", "Shared false positives"]];
  const mk = meas.some(([k]) => k === p.measure) ? p.measure : "gap";
  const variants = K4V.filter(([v]) => rows.some(o => o.variant === v));
  const tbody = h("tbody", {}, variants.map(([v, label]) => h("tr", {}, h("td", { class: "rowh" }, label), LOSSES.map(([lk, ln]) => {
    const list = rows.filter(o => o.variant === v && o.loss === lk), xs = list.map(o => o[mk]);
    return h("td", {}, cellEl({ main: fix(mean(xs), mk.includes("jaccard") || mk.startsWith("fp") ? 3 : 2), sub: sd(xs) != null ? "± " + fix(sd(xs), mk.includes("jaccard") || mk.startsWith("fp") ? 3 : 2) : null, n: `${xs.length} seeds`,
      tipBuild: () => [tipTitle(`${label} · ${ln}`), ...list.sort((x, y) => x.seed - y.seed).map(o => tipRow(`seed ${o.seed}`, fmtNum(o[mk])))],
      onClick: () => drawer("k = 4 ensemble", `${label} · ${ln}`, smallTable([["Seed", 1], ["Mean", 1], ["Concat", 1], ["Gap", 1], ["Jaccard", 1], ["Shared FP", 1], ["CV folder"]],
        list.map(o => ({ cells: [o.seed, fix(o.mean), fix(o.concat), fix(o.gap), fmtNum(o.topR_jaccard), fmtNum(o.fp_also_in_other_fold_topR), linkOrText("cv_dir", o.cv_dir)] })))) }));
  }))));
  return [h("div", { class: "filters" }, seg("Measure", meas, mk, v => update({ measure: v }))),
    card("Cars196 · k = 4 · fold-model ensemble", "mean ± SD over five seeds",
      h("div", { class: "scrollx" }, h("table", { class: "matrix" }, h("thead", {}, h("tr", {}, h("th", { class: "rowh" }, "Setting"), LOSSES.map(([, ln]) => h("th", {}, ln)))), tbody)),
      h("p", { class: "foot" }, "Concat − Mean is the gain from concatenating the four fold models. Top-R Jaccard measures how much the fold models' top-R lists overlap; shared false positives is the share of a fold model's false positives that another fold model also ranks in its top R."))];
}
async function ablationDrawer(o) {
  const fcv = await foldsByCv(), folds = (fcv.get(o.source) || []).sort((x, y) => x.fold - y.fold);
  const fields = Object.entries(o).filter(([, v]) => v != null && v !== "");
  drawer(`${o.dataset} · ${o.table_key}`, `${MNAME[o.method] || o.method} · ${LNAME[o.loss] || o.loss} · ${o.label_de || o.variant || ""}`,
    h("section", {}, h("h3", {}, "Row"), kv(fields.map(([k, v]) => [k, linkOrText(k, v)]))),
    h("section", {}, h("h3", {}, `Folds (${folds.length})`), folds.length ? smallTable([["Fold", 1], ["Test MAP@R", 1], ["Best valid MAP@R", 1], ["Selected epoch", 1], ["Trained epochs", 1], ["Configuration"]],
      folds.map(f => ({ cells: [f.fold, fix(pct(f.mean_average_precision_at_r)), fix(pct(f.best_valid_map)), f.selected_epoch, f.epochs_trained, linkOrText("config_id", f.config_id)] })))
      : h("p", { class: "ink2" }, "No fold records for this CV folder.")));
}

/* ---------- Externe Bilder ---------- */
async function external(p) {
  const e = await E();
  const dsn = p.ds === "In-Shop" ? "In-Shop" : "Cars196", scope = p.scope === "concat" ? "concat" : "mean";
  const metric = ["map", "p1", "rp"].includes(p.metric) ? p.metric : "map";
  const view = ["abs", "ctrl", "ref"].includes(p.view) ? p.view : "abs";
  const col = `${scope}_${metric}`;
  const all = e.objects().filter(o => o.dataset === dsn).map(o => ({ ...o, m: normM(o.method) }));
  const of = kind => all.filter(o => o.kind === kind);
  const reg = of("regulär"), ctrl = of("kontrolle"), ref = of("referenz"), sw = of("sweep");
  const pcts = uniq(reg.map(o => o.pct)).sort((x, y) => x - y);
  const methods = METHODS.map(([k]) => k).filter(k => reg.some(o => o.m === k));
  const st = list => { const v = list.map(o => o[col]); return { m: mean(v), s: sd(v), n: v.length, list }; };
  const ctrlOf = (m, l) => st(ctrl.filter(o => o.m === m && o.loss === l)), refOf = l => st(ref.filter(o => o.loss === l));
  const shown = (s0, m, l) => (s0.m == null ? null : view === "abs" ? s0.m : view === "ctrl" ? (ctrlOf(m, l).m == null ? null : s0.m - ctrlOf(m, l).m) : refOf(l).m == null ? null : s0.m - refOf(l).m);
  const cellsFor = (list, m, l) => {
    const s0 = st(list), v = shown(s0, m, l);
    return { s0, v };
  };
  const allV = [];
  for (const m of methods) for (const [l] of LOSSES) { for (const pc of pcts) allV.push(cellsFor(reg.filter(o => o.m === m && o.loss === l && o.pct === pc), m, l).v); allV.push(cellsFor(ctrl.filter(o => o.m === m && o.loss === l), m, l).v); }
  const dmax = Math.max(0.5, niceMax(quantileAbs(allV)));
  const mkCell = (list, m, l, title, isCtrl) => {
    const { s0, v } = cellsFor(list, m, l);
    if (!s0.n) return cellEl({ main: "–", kind: "empty" });
    const tipBuild = () => [tipTitle(title), tipRow(`${scope === "mean" ? "Mean" : "Concat"} ${metric.toUpperCase()}`, `${fix(s0.m)} %${s0.n > 1 ? ` (n = ${s0.n})` : ""}`),
      view !== "abs" ? tipRow(view === "ctrl" ? "Δ control" : "Δ ÜBl. reference", signed(v) + " pp") : null,
      tipSub(`${s0.list[0].n_external} external images${s0.list[0].external_source ? " from " + s0.list[0].external_source : ""} · values from ${uniq(s0.list.map(o => o.werte)).join(", ")}`)];
    const onClick = () => externalDrawer(s0.list, title);
    if (view === "ctrl" && isCtrl) return cellEl({ main: fix(s0.m), sub: "control", kind: "ref", onClick, tipBuild });
    return cellEl({ main: view === "abs" ? fix(v) : signed(v), sub: s0.s != null ? "± " + fix(s0.s) : null, n: s0.n > 1 ? `n = ${s0.n}` : null,
      bg: view === "abs" ? null : diverging(v / dmax), onClick, tipBuild });
  };
  const matrix = (rowsDef, colsDef) => h("div", { class: "scrollx" }, h("table", { class: "matrix" },
    h("thead", {}, h("tr", {}, h("th", { class: "rowh" }, "Method · loss"), colsDef.map(c => h("th", {}, c.label)))),
    h("tbody", {}, rowsDef.flatMap(sec => [h("tr", { class: "section" }, h("td", { colspan: colsDef.length + 1 }, swatch(mcolor(sec.m)), MNAME[sec.m])),
      ...sec.rows.map(r => h("tr", {}, h("td", { class: "rowh" }, r.label), colsDef.map(c => h("td", {}, c.cell(r)))))]))));
  const refCol = { label: "ÜBl., all labels", cell: r => { const s0 = refOf(r.l); return s0.n ? cellEl({ main: fix(s0.m), sub: "reference", kind: "ref", onClick: () => externalDrawer(s0.list, `ÜBl. reference · ${LNAME[r.l]}`) }) : cellEl({ main: "–", kind: "empty" }); } };
  const regCols = [{ label: "0 % (control)", cell: r => mkCell(ctrl.filter(o => o.m === r.m && o.loss === r.l), r.m, r.l, `${MNAME[r.m]} · ${LNAME[r.l]} · control`, true) },
    ...pcts.map(pc => ({ label: `${pc} %`, cell: r => mkCell(reg.filter(o => o.m === r.m && o.loss === r.l && o.pct === pc), r.m, r.l, `${MNAME[r.m]} · ${LNAME[r.l]} · ${pc} % external`) })),
    ...(view === "abs" ? [refCol] : [])];
  const regRows = methods.map(m => ({ m, rows: LOSSES.map(([l, ln]) => ({ m, l, label: ln })) }));
  const swMethods = METHODS.map(([k]) => k).filter(k => sw.some(o => o.m === k)), swPcts = uniq(sw.map(o => o.pct)).sort((x, y) => x - y);
  const swRows = swMethods.map(m => ({ m, rows: LOSSES.flatMap(([l, ln]) => uniq(sw.filter(o => o.m === m && o.loss === l).map(o => o.sweep_value)).sort((x, y) => x - y).map(v => ({ m, l, v, label: `${ln} · ${fmtNum(v)}` }))) }));
  const swCols = swPcts.map(pc => ({ label: `${pc} %`, cell: r => mkCell(sw.filter(o => o.m === r.m && o.loss === r.l && o.pct === pc && o.sweep_value === r.v), r.m, r.l, `${MNAME[r.m]} · ${LNAME[r.l]} · sweep ${fmtNum(r.v)} · ${pc} %`) }));
  const xs = [0, ...pcts];
  const panels = LOSSES.map(([l, ln]) => ({
    title: ln, xs, categorical: true,
    refs: view === "abs" && refOf(l).n ? [{ y: refOf(l).m, label: "ÜBl., all labels" }] : view === "ref" ? [{ y: 0, label: "ÜBl., all labels" }] : [],
    series: methods.map(m => ({ name: MNAME[m], color: mcolor(m), values: xs.map(x => shown(st(x === 0 ? ctrl.filter(o => o.m === m && o.loss === l) : reg.filter(o => o.m === m && o.loss === l && o.pct === x)), m, l)) })),
  }));
  const ext = dsn === "Cars196" ? "CompCars" : "Fashion200k";
  return [
    head("External unlabeled images", `${dsn} with all labels plus ${ext} images as the unlabeled pool (external.csv). Control: the same method without external images. Reference: ÜBl. with all labels.`),
    h("div", { class: "filters" },
      seg("Dataset", [["Cars196", "Cars196 + CompCars"], ["In-Shop", "In-Shop + Fashion200k"]], dsn, v => update({ ds: v })),
      seg("Scope", [["mean", "Mean"], ["concat", "Concat"]], scope, v => update({ scope: v })),
      seg("Metric", [["map", "MAP@R"], ["p1", "P@1"], ["rp", "R-Precision"]], metric, v => update({ metric: v })),
      seg("Show", [["abs", "Values"], ["ctrl", "Δ control"], ["ref", "Δ ÜBl. reference"]], view, v => update({ view: v }))),
    h("div", { class: "stack" },
      card(`Share of external images · ${scope === "mean" ? "Mean" : "Concat"} ${metric === "map" ? "MAP@R" : metric === "p1" ? "P@1" : "R-Precision"}`,
        view === "abs" ? "values in %; mean over the available seeds" : view === "ctrl" ? "difference to the same method without external images (pp)" : "difference to ÜBl. with all labels (pp)",
        matrix(regRows, regCols), view === "abs" ? null : divLegend(dmax, view === "ctrl" ? "worse than control" : "worse than ÜBl.", view === "ctrl" ? "better than control" : "better than ÜBl.")),
      card("By share of external images", "one panel per loss; 0 % is the control run", legend(methods.map(m => ({ name: MNAME[m], color: mcolor(m) }))),
        multiples(panels, { xlabel: x => (x === 0 ? "0" : `${x}`), xtitle: "external images (%)", zero: view !== "abs", tipfmt: v => (view === "abs" ? fix(v) + " %" : signed(v) + " pp"), yfmt: v => fmtNum(v) })),
      sw.length ? card("Sweeps with external images", "rows: loss · sweep value (neighbours for LP and LRML, the swept setting of the method otherwise)", matrix(swRows, swCols)) : null),
  ];
}
async function externalDrawer(list, title) {
  const f = await F();
  const keyOf = o => [o.dataset, normM(o.method), o.loss, o.pct, o.seed, o.kind, o.sweep_value ?? ""].join("|");
  const keys = new Set(list.map(keyOf));
  const folds = f.objects().filter(o => o.experiment === "external" && keys.has(keyOf(o))).sort((x, y) => x.seed - y.seed || x.fold - y.fold);
  drawer("External images", title,
    h("section", {}, h("h3", {}, "Runs"), smallTable([["Seed", 1], ["Mean MAP@R", 1], ["Concat MAP@R", 1], ["P@1", 1], ["External", 1], ["Selected epoch (median)", 1], ["Values from"], ["Source"]],
      list.map(o => ({ cells: [o.seed, fix(o.mean_map), fix(o.concat_map), fix(o.mean_p1), thousands(o.n_external), fmtNum(o.selected_epoch_median), o.werte, linkOrText("source", o.source)] })))),
    h("section", {}, h("h3", {}, `Folds (${folds.length})`), folds.length ? smallTable([["Seed", 1], ["Fold", 1], ["Test MAP@R", 1], ["P@1", 1], ["Selected epoch", 1], ["Trained epochs", 1]],
      folds.map(o => ({ cells: [o.seed, o.fold, fix(o.test_map), fix(o.test_p1), o.selected_epoch, o.epochs_trained] }))) : h("p", { class: "ink2" }, "No fold records.")),
    h("section", {}, h("h3", {}, "Settings"), kv(Object.entries(list[0]).filter(([k, v]) => v != null && !["m"].includes(k) && !/^(mean|concat)_/.test(k)).map(([k, v]) => [k, linkOrText(k, v)]))));
}

/* ---------- Abbildungen ---------- */
async function figures() {
  const list = await OUTL();
  if (!list.length) return [head("Figures"), card("No generated outputs", null, h("p", { class: "ink2" }, "Run python3 abbildungen_src/erzeuge_aus_daten.py to fill ausgabe/."))];
  const has = new Set(list.map(f => f.path));
  const previews = list.filter(f => f.path.startsWith("vorschau/") && f.path.endsWith(".png"));
  const tex = list.filter(f => f.path.startsWith("kapitel/") && f.path.endsWith(".tex"));
  const gen = list.filter(f => f.path.startsWith("generiert/"));
  const logs = list.filter(f => f.path.startsWith("protokoll/"));
  const gallery = h("div", { class: "multiples", style: { gridTemplateColumns: "repeat(auto-fill, minmax(min(100%, 340px), 1fr))" } }, previews.map(f => {
    const name = f.path.slice("vorschau/".length, -4), pdf = `bilder/neu/${name}.pdf`;
    return h("figure", { class: "card", style: { margin: 0, padding: "12px" } },
      h("img", { src: "/ausgabe/" + f.path, alt: name, loading: "lazy", style: { width: "100%", borderRadius: "8px", background: "#fff", cursor: "zoom-in", display: "block" }, onclick: () => lightbox("/ausgabe/" + f.path, name) }),
      h("figcaption", { style: { display: "flex", gap: "10px", marginTop: "8px", alignItems: "baseline", flexWrap: "wrap" } }, h("strong", {}, name),
        has.has(pdf) ? h("a", { href: "/ausgabe/" + pdf, target: "_blank" }, "PDF") : null,
        h("span", { class: "muted", style: { marginLeft: "auto", fontSize: "12px" } }, new Date(f.mtime * 1000).toLocaleString("en-GB", { dateStyle: "short", timeStyle: "short" }))));
  }));
  const srcList = (files, label) => h("div", {}, files.map(f => {
    const d = h("details", { class: "card", style: { padding: "8px 12px", marginBottom: "6px" } }, h("summary", { style: { cursor: "pointer" } }, h("span", { class: "mono" }, f.path), " ", h("span", { class: "muted" }, bytes(f.bytes))));
    d.addEventListener("toggle", () => { if (d.open && !d.dataset.done) { d.dataset.done = 1; text("/ausgabe/" + f.path).then(t => d.append(h("pre", { style: { overflow: "auto", maxHeight: "420px", fontSize: "12px" } }, t))); } });
    return d;
  }), files.length ? null : h("p", { class: "ink2" }, `No ${label}.`));
  return [head("Figures and generated tables", "Output of erzeuge_aus_daten.py in ausgabe/: PNG previews of every figure, the PDFs for the thesis, LaTeX tables, derived CSVs and execution logs."),
    h("div", { class: "stack" },
      card(`Figures (${previews.length})`, "click a preview to enlarge", gallery),
      card(`LaTeX tables (${tex.length})`, "ausgabe/kapitel/", srcList(tex, "tables")),
      card(`Derived CSVs (${gen.length})`, "ausgabe/generiert/", h("ul", {}, gen.map(f => h("li", {}, f.path.endsWith(".csv") ? a(f.path, () => go("explore", { src: "out:" + f.path })) : f.path)))),
      card(`Logs (${logs.length})`, "ausgabe/protokoll/", srcList(logs, "logs")))];
}
function lightbox(src, name) {
  const box = h("div", { style: { position: "fixed", inset: 0, background: "rgba(0,0,0,0.72)", zIndex: 60, display: "grid", placeItems: "center", padding: "24px", cursor: "zoom-out" },
    onclick: () => box.remove(), role: "dialog", "aria-label": name },
  h("img", { src, alt: name, style: { maxWidth: "100%", maxHeight: "100%", background: "#fff", borderRadius: "8px", boxShadow: "0 12px 48px rgba(0,0,0,0.5)" } }));
  addEventListener("keydown", function esc(e) { if (e.key === "Escape") { box.remove(); removeEventListener("keydown", esc); } });
  document.body.append(box);
}

/* ---------- Hyperparameter ---------- */
const hpWide = () => once("hpw", async () => {
  const [hpx, use] = await Promise.all([hpIndex(), usage()]);
  const distinct = new Map();
  for (const { params } of hpx.values()) for (const [k, v] of params) { if (!distinct.has(k)) distinct.set(k, new Map()); const d = distinct.get(k); d.set(String(v), (d.get(String(v)) || 0) + 1); }
  const params = [...distinct.keys()].sort((x, y) => distinct.get(y).size - distinct.get(x).size || x.localeCompare(y));
  const objs = [...hpx].map(([id, { params: ps }]) => ({ config_id: id, runs: (use.get(id) || new Set()).size, ...Object.fromEntries(ps) }));
  return { table: Table.fromObjects(objs, ["config_id", "runs", ...params]), distinct, params };
});
async function hp(p) {
  const { table, distinct, params } = await hpWide();
  const filters = Object.fromEntries(Object.entries(p).filter(([k]) => k.startsWith("f.")).map(([k, v]) => [k.slice(2), v]));
  const vt = new VirtualTable(table, { filters, link, name: "configurations", onRow: o => configDrawer(o.config_id), title: "Configuration" });
  const overview_ = smallTable([["Parameter"], ["Distinct", 1], ["Most common values"]], params.map(k => {
    const d = [...distinct.get(k)].sort((x, y) => y[1] - x[1]);
    return { cells: [h("span", { class: "mono" }, k), d.length, h("span", {}, d.slice(0, 5).map(([v, n]) => h("a", { href: "#", class: "tag", style: { marginLeft: 0, marginRight: "6px" },
      onclick: e => { e.preventDefault(); update({ ["f." + k]: "=" + v }); } }, `${middle(v, 28)} ×${n}`)), d.length > 5 ? h("span", { class: "muted" }, `+${d.length - 5} more`) : null)] };
  }));
  return {
    nodes: [head("Hyperparameters", `${thousands(table.length)} resolved training configurations, ${params.length} parameters (hyperparameters.csv). "runs" counts the result rows and fold records that refer to a configuration.`),
      h("div", { class: "stack" },
        card("Configurations", Object.keys(filters).length ? "filtered from the parameter list below" : "one row per configuration; click a row for all values, the runs that use it and its source", vt.el,
          Object.keys(filters).length ? h("p", {}, a("Clear filters", () => go("hp"))) : null),
        card("Parameters", "values across all configurations; click a value to filter", overview_))],
    after: p.config ? () => configDrawer(p.config) : null,
  };
}
async function configDrawer(id) {
  const [hpx, use, src] = await Promise.all([hpIndex(), usage(), srcIndex()]);
  const c = hpx.get(id);
  if (!c) return drawer("Configuration", id, h("p", { class: "ink2" }, "Unknown configuration."));
  const runs = [...(use.get(id) || [])].sort();
  const rec = src.byId.get(c.source_id);
  const cmp = h("div", {});
  const input = h("input", { type: "text", list: "cfg-ids", placeholder: "hp_…", style: { width: "100%" }, onchange: e => {
    const other = hpx.get(e.target.value.trim());
    if (!other) { cmp.replaceChildren(h("p", { class: "ink2" }, "Unknown configuration.")); return; }
    const om = new Map(other.params), mine = new Map(c.params), keys = uniq([...mine.keys(), ...om.keys()]).filter(k => String(mine.get(k)) !== String(om.get(k)));
    cmp.replaceChildren(keys.length ? smallTable([["Parameter"], ["This"], ["Other"]], keys.map(k => ({ cells: [h("span", { class: "mono" }, k), String(mine.get(k) ?? "–"), String(om.get(k) ?? "–")] })))
      : h("p", { class: "ink2" }, "Identical parameters."));
  } });
  drawer("Configuration", id,
    h("section", {}, h("h3", {}, `Used by ${runs.length}`), runs.length ? h("div", {}, runs.slice(0, 60).map(r => h("div", { class: "mono", style: { fontSize: "12px" } }, linkOrText("run_id", r))),
      runs.length > 60 ? h("div", { class: "muted" }, `… ${runs.length - 60} more`) : null) : h("p", { class: "ink2" }, "No result or fold rows refer to it.")),
    h("section", {}, h("h3", {}, `Parameters (${c.params.length})`), kv(c.params.map(([k, v]) => [k, v == null ? "–" : String(v)]))),
    h("section", {}, h("h3", {}, "Source"), rec ? kv([["path", h("span", { class: "path" }, rec.path)], ["sha256", h("span", { class: "mono" }, rec.sha256)], ["selection", rec.selection]]) : h("p", {}, c.source_id)),
    h("section", {}, h("h3", {}, "Compare with another configuration"), input, h("datalist", { id: "cfg-ids" }, [...hpx.keys()].slice(0, 2000).map(k => h("option", { value: k }))), cmp));
}

/* ---------- details.zip ---------- */
async function details(p) {
  const [list, man] = await Promise.all([ZIPL(), MAN()]);
  const viewsFor = name => Object.entries(man.views).filter(([, v]) => v.member === name);
  const groups = new Map();
  for (const f of list) {
    const parts = f.name.split("/"), g = parts.length > 2 ? parts.slice(0, 2).join("/") : parts.length > 1 ? parts[0] : "(top level)";
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(f);
  }
  const sel = p.m && list.some(f => f.name === p.m) ? p.m : null;
  const side = h("nav", { class: "card side", "aria-label": "details.zip members" }, [...groups].map(([g, fs]) => [h("h3", {}, g),
    fs.filter(f => !(g === "qualitative/images" && fs.length > 12 && f !== fs[0] && f.name !== sel)).map(f => h("a", { href: hashFor("details", { m: f.name }), "aria-current": String(f.name === sel) },
      h("span", {}, f.name.split("/").pop()), h("span", { class: "n" }, bytes(f.bytes)))),
    g === "qualitative/images" && fs.length > 12 ? h("div", { class: "muted", style: { padding: "2px 8px", fontSize: "12px" } }, `${fs.length} thumbnails, shown in the retrieval view`) : null]));
  let body;
  if (!sel) {
    body = h("div", { class: "stack" },
      card("details.zip", `${list.length} members, ${bytes(list.reduce((s0, f) => s0 + f.bytes, 0))} unpacked`, h("p", {}, "Geometry measurements and validation curves, LP diagnostics, baselines, timings, search spaces, run-selection records, two LP t-SNE frames, the published retrieval neighbours with thumbnails, and the plotting fonts. Pick a member on the left.")),
      await retrievalCard(), await lpCard(man));
  } else body = await memberView(sel, viewsFor(sel), man);
  return [head("details.zip", "Detailed inputs, stored once in compressed form. Each member can be read in full here."), h("div", { class: "split" }, side, body)];
}
async function memberView(name, views, man) {
  const info = (await ZIPL()).find(f => f.name === name);
  const meta = card(name.split("/").pop(), `${name} · ${bytes(info.bytes)} (packed ${bytes(info.packed)})`,
    views.length ? smallTable([["Manifest view"], ["Selection"], ["Rows", 1], ["Source"]], views.map(([vn, v]) => ({ cells: [h("span", { class: "mono" }, vn), v.selection || "–", v.rows != null ? thousands(v.rows) : "–", v.source_id ? linkOrText("source_id", v.source_id) : "–"] })))
      : h("p", { class: "ink2" }, "Not referenced by a manifest view."));
  const ext = name.split(".").pop().toLowerCase();
  let view;
  if (name === "qualitative/retrieval_neighbors.csv") view = await retrievalCard();
  else if (/^qualitative\/lp_\d+\.csv$/.test(name)) view = await lpCard(man);
  else if (ext === "csv") { const t = await csv(zip(name)); view = card("Table", `${thousands(t.length)} rows × ${t.columns.length} columns`, new VirtualTable(t, { link, name: name.split("/").pop() }).el); }
  else if (ext === "json") view = card("JSON", null, jsonTree(await json(zip(name))));
  else if (ext === "jsonl") {
    const lines = (await text(zip(name))).split("\n").filter(Boolean).map(l => { try { return JSON.parse(l); } catch { return l; } });
    const flat = lines.length && lines.every(o => o && typeof o === "object" && !Array.isArray(o));
    view = flat ? card("Records", `${lines.length} lines`, new VirtualTable(Table.fromObjects(lines), { link, name }).el) : card("Records", `${lines.length} lines`, jsonTree(lines));
  } else if (ext === "png") view = card("Image", null, h("img", { src: zip(name), alt: name, style: { maxWidth: "100%", borderRadius: "8px" } }));
  else if (ext === "otf") view = fontCard(name);
  else view = card("Text", null, h("pre", { style: { whiteSpace: "pre-wrap" } }, await text(zip(name))));
  return h("div", { class: "stack" }, meta, view);
}
function fontCard(name) {
  const fam = "spec-" + name.replace(/\W+/g, "");
  const sample = h("div", { style: { fontFamily: fam + ", serif", fontSize: "30px", lineHeight: 1.3 } }, "Semi-supervised Metric Learning",
    h("div", { style: { fontSize: "18px" } }, "MAP@R 0123456789 · Überwachte Referenz · ÄÖÜ äöü ß"));
  new FontFace(fam, `url(${zip(name)})`).load().then(f => document.fonts.add(f)).catch(() => sample.append(h("p", { class: "error" }, "Font could not be loaded.")));
  return card("Font specimen", "TeX Gyre Pagella, the font of the thesis figures (license in fonts/GUST-FONT-LICENSE.txt)", sample);
}
async function retrievalCard() {
  const [nb, test] = await Promise.all([csv(zip("qualitative/retrieval_neighbors.csv")), csv(zip("qualitative/cars_test.csv"))]);
  const tst = test.objects(), cls = i => { const pth = tst[i]?.path || ""; const parts = pth.split("/"); return parts[parts.length - 2] || "?"; };
  const rows = nb.objects(), queries = uniq(rows.map(o => o.query));
  const names = { dino: "DINOv2 (frozen)", sup: "ÜBl.", lp: "LP" };
  const thumb = (idx, kind, badge, tipLines) => {
    const el = h("div", { class: "thumb " + kind, tabindex: "0" }, h("img", { src: zip(`qualitative/images/${idx}.png`), alt: `test image ${idx}`, loading: "lazy" }), badge ? h("span", { class: "badge" }, badge) : null);
    hoverTip(el, tipLines);
    return el;
  };
  return card("Retrieval examples", "Cars196 test queries 758 and 5348, six nearest neighbours per model; frames: ✓ same class, ✗ other class",
    h("div", { class: "retr" }, queries.map(q => {
      const ql = tst[q]?.label;
      return h("div", { class: "retr-q" }, h("h3", {}, `Query ${q} · ${cls(q)}`),
        uniq(rows.filter(o => o.query === q).map(o => o.method)).map(m => {
          const nn = rows.filter(o => o.query === q && o.method === m).sort((x, y) => x.rank - y.rank), ok = nn.filter(o => o.label === ql).length;
          return h("div", { class: "retr-row" }, h("div", { class: "rl" }, names[m] || m, h("div", { class: "muted", style: { fontWeight: 400, fontSize: "12px" } }, `${ok}/${nn.length} correct`)),
            thumb(q, "q", "Q", () => [tipTitle(`Query ${q}`), tipSub(cls(q))]),
            nn.map(o => thumb(o.index, o.label === ql ? "ok" : "bad", `${o.rank} ${o.label === ql ? "✓" : "✗"}`,
              () => [tipTitle(`Rank ${o.rank} · test image ${o.index}`), tipRow(o.label === ql ? "✓ same class" : "✗ other class", `label ${o.label}`), tipSub(cls(o.index))])));
        }));
    })));
}
async function lpCard(man) {
  const epochs = await json(zip("qualitative/lp_epochs.json")).catch(() => [20, 140]);
  const frames = await Promise.all(epochs.map(e => csv(zip(`qualitative/lp_${e}.csv`)).then(t => [e, t.objects()])));
  const view0 = man.views[`qualitative/lp_${epochs[0]}.csv`] || {};
  let thr = view0.threshold ?? 0.8;
  const stats = h("div", { class: "grid2" });
  const canvases = frames.map(([e, pts]) => ({ e, pts, c: h("canvas", { class: "scatter", "aria-label": `t-SNE of the training pool at epoch ${e}` }), s: h("div", { class: "ink2", style: { fontSize: "12.5px", marginTop: "6px" } }) }));
  const colors = () => ({ ok: cssVar("--s1"), bad: cssVar("--s2"), rej: cssVar("--axis"), lab: cssVar("--ink") });
  // Wie im Generator: lp_cert ist float16 (array_dtypes im Manifest), verglichen wird in float16
  const f16 = Math.f16round || (x => { if (!x || !isFinite(x)) return x; const st = 2 ** (Math.max(Math.floor(Math.log2(Math.abs(x))), -14) - 10); return Math.round(x / st) * st; });
  const q = (view0.array_dtypes || {}).lp_cert === "float16" ? f16 : (x => x);
  const kindOf = (o, t) => (o.kind !== "unlabeled" ? "lab" : q(o.lp_cert) >= q(t) ? (o.lp_pred === o.label ? "ok" : "bad") : "rej");
  const draw = () => {
    const col = colors();
    for (const cv of canvases) {
      const w = cv.c.clientWidth || 480, hh = Math.round(w * 0.78), dpr = devicePixelRatio || 1;
      cv.c.width = w * dpr; cv.c.height = hh * dpr; cv.c.style.height = hh + "px";
      const g = cv.c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, hh);
      const xs = cv.pts.map(o => o.x), ys = cv.pts.map(o => o.y), x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), pad = 10;
      const X = x => pad + ((x - x0) / (x1 - x0)) * (w - 2 * pad), Y = y => pad + (1 - (y - y0) / (y1 - y0)) * (hh - 2 * pad);
      cv.proj = cv.pts.map(o => [X(o.x), Y(o.y)]);
      const n = { lab: 0, ok: 0, bad: 0, rej: 0 };
      for (const k of ["rej", "ok", "lab", "bad"]) cv.pts.forEach((o, i) => {
        if (kindOf(o, thr) !== k) return;
        const [px, py] = cv.proj[i];
        n[k]++;
        if (k === "bad") { g.strokeStyle = col.bad; g.lineWidth = 1.6; g.beginPath(); g.moveTo(px - 2.6, py - 2.6); g.lineTo(px + 2.6, py + 2.6); g.moveTo(px + 2.6, py - 2.6); g.lineTo(px - 2.6, py + 2.6); g.stroke(); }
        else { g.fillStyle = col[k]; g.beginPath(); g.arc(px, py, k === "rej" ? 1.5 : 1.8, 0, 2 * Math.PI); g.fill(); }
      });
      const unl = n.ok + n.bad + n.rej, acc = n.ok + n.bad;
      cv.s.textContent = `Epoch ${cv.e}: ${thousands(acc)} of ${thousands(unl)} unlabeled images above the threshold (coverage ${fix((100 * acc) / unl, 1)} %), `
        + `${fix((100 * n.ok) / Math.max(1, acc), 1)} % of them correct; ${thousands(n.lab)} labeled images.`;
    }
  };
  const slider = h("input", { type: "range", min: "0", max: "1", step: "0.01", value: String(thr), "aria-label": "confidence threshold",
    oninput: e => { thr = +e.target.value; lab.textContent = fix(thr, 2) + (Math.abs(thr - (view0.threshold ?? -1)) < 0.005 ? " (Phase-1 value)" : ""); draw(); } });
  const lab = h("span", { class: "num" }, fix(thr, 2) + " (Phase-1 value)");
  for (const cv of canvases) {
    cv.c.addEventListener("pointermove", ev => {
      if (!cv.proj) return;
      const r = cv.c.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
      let best = -1, bd = 144;
      cv.proj.forEach(([px, py], i) => { const d = (px - mx) ** 2 + (py - my) ** 2; if (d < bd) { bd = d; best = i; } });
      if (best < 0) return untip();
      const o = cv.pts[best], k = kindOf(o, thr);
      tip(ev, tipTitle({ lab: "Labeled image", ok: "Pseudo-label correct", bad: "Pseudo-label wrong", rej: "Below the threshold" }[k]),
        tipRow("class", String(o.label)), o.kind === "unlabeled" ? tipRow("LP prediction", `${o.lp_pred} (certainty ${fix(o.lp_cert, 3)})`) : null);
    });
    cv.c.addEventListener("pointerleave", untip);
  }
  const col = colors();
  const ro = new ResizeObserver(() => draw());
  setTimeout(() => { canvases.forEach(cv => ro.observe(cv.c)); draw(); }, 0);
  stats.append(...canvases.map(cv => h("div", {}, h("h4", { style: { margin: "0 0 6px" } }, `Epoch ${cv.e}`), cv.c, cv.s)));
  return card("LP pseudo-labels on the Cars196 training pool", "t-SNE of 4,000 tracked images, LP with Circle, k = 16, seed 2, fold 0",
    h("div", { class: "filters", style: { marginBottom: "8px" } }, h("label", { class: "field" }, h("span", {}, "Confidence threshold"), h("span", { style: { display: "flex", gap: "8px", alignItems: "center" } }, slider, lab))),
    h("div", { class: "legend" }, [["Labeled", col.lab], ["Pseudo-label correct", col.ok], ["Below the threshold", col.rej]].map(([n, c]) => h("span", { class: "item" }, swatch(c), n)),
      h("span", { class: "item" }, h("span", { style: { color: col.bad, fontWeight: 700, marginRight: "6px" } }, "✕"), "Pseudo-label wrong")),
    stats);
}

/* ---------- Alle Tabellen ---------- */
function pairedDifferences(results, ds, columns) {
  const rows = results.objects().filter(o => o.dataset_key === ds);
  const budget = ds === "inshop10" ? "budget_percent" : "k", metrics = ["r_precision", "mean_average_precision_at_r", "precision_at_1"];
  const base = new Map(rows.filter(o => o.method.startsWith("sup_")).map(o => [[o[budget], o.scope, o.seed, o.method.split("_").pop()].join("|"), o]));
  const groups = new Map();
  for (const o of rows) {
    if (o.method.startsWith("sup_")) continue;
    const b = base.get([o[budget], o.scope, o.seed, o.method.split("_").pop()].join("|"));
    if (!b) continue;
    const k = [o.method, o[budget], o.scope].join("|");
    if (!groups.has(k)) groups.set(k, { method: o.method, b: o[budget], scope: o.scope, pairs: [] });
    groups.get(k).pairs.push([o, b]);
  }
  const out = [];
  for (const g of groups.values()) for (const m of metrics) {
    const d = g.pairs.map(([o, b]) => 100 * (o[m] - b[m])).filter(x => !Number.isNaN(x));
    out.push({ method: g.method, [budget]: g.b, scope: g.scope, metric: m, mean: mean(d), std: sd(d), positive_seeds: d.filter(x => x > 0).length,
      min: Math.min(...d), max: Math.max(...d), n_seeds: d.length });
  }
  return Table.fromObjects(out, columns);
}
async function explore(p) {
  const [man, outl, zl, info] = await Promise.all([MAN(), OUTL(), ZIPL(), INFO()]);
  const src = p.src || "file:results.csv";
  const filters = Object.fromEntries(Object.entries(p).filter(([k]) => k.startsWith("f.")).map(([k, v]) => [k.slice(2), v]));
  const item = (key, label, n) => h("a", { href: hashFor("explore", { src: key }), "aria-current": String(key === src), title: label }, h("span", {}, label), n != null ? h("span", { class: "n" }, n) : null);
  const views = Object.entries(man.views);
  const vk = v => (v.derived && !v.member ? "derived" : v.file ? "file" : "member");
  const side = h("nav", { class: "card side", "aria-label": "Tables" },
    h("h3", {}, "Bundle · daten/"), info.files.filter(f => f.name.endsWith(".csv")).map(f => item("file:" + f.name, f.name, bytes(f.bytes))),
    outl.some(f => f.path.endsWith(".csv")) ? [h("h3", {}, "Generated · ausgabe/"), outl.filter(f => f.path.endsWith(".csv")).map(f => item("out:" + f.path, f.path, bytes(f.bytes)))] : null,
    h("h3", {}, "details.zip"), zl.filter(f => /\.(csv|jsonl)$/.test(f.name)).map(f => item("zip:" + f.name, f.name.replace(/^tables\//, ""), bytes(f.bytes))),
    h("h3", {}, `Manifest views (${views.length})`),
    ["file", "member", "derived"].map(kind => h("details", { open: src.startsWith("view:") }, h("summary", {}, { file: "Filtered source tables", member: "Members of details.zip", derived: "Derived at generation time" }[kind]),
      views.filter(([, v]) => vk(v) === kind).map(([n, v]) => item("view:" + n, n, v.rows != null ? thousands(v.rows) : null)))));
  let t, title = src.replace(/^\w+:/, ""), notes = [];
  const [kind, ...rest] = src.split(":"), name = rest.join(":");
  if (kind === "file") t = await csv(daten(name));
  else if (kind === "out") t = await csv("/ausgabe/" + name);
  else if (kind === "zip") {
    if (name.endsWith(".jsonl")) t = Table.fromObjects((await text(zip(name))).split("\n").filter(Boolean).map(l => JSON.parse(l)));
    else t = await csv(zip(name));
  } else if (kind === "view") {
    const v = man.views[name];
    if (!v) throw new Error("Unknown manifest view " + name);
    if (v.file) { t = (await csv(daten(v.file))).view(v.where, v.columns); notes.push(`${v.file} where ${Object.entries(v.where || {}).map(([c, x]) => `${c} = ${x}`).join(", ") || "(all rows)"}, ${v.columns.length} columns`); }
    else if (v.derived === "paired_differences") { t = pairedDifferences(await R(), v.dataset, v.columns); notes.push("Derived from results.csv exactly like datensammlung.py: 100 × (method − ÜBl. with the same loss, budget, scope and seed)."); }
    else if (v.member && /\.(csv)$/.test(v.member)) { t = await csv(zip(v.member)); if (v.columns) t = t.view(null, v.columns); notes.push(`details.zip member ${v.member}${v.selection ? " · selection: " + v.selection : ""}`); }
    else return [head("All tables"), h("div", { class: "split" }, side, card(name, null, kv(Object.entries(v).map(([k, x]) => [k, typeof x === "object" ? JSON.stringify(x) : String(x)])),
      v.member ? h("p", {}, a("Open in details.zip", () => go("details", { m: v.member }))) : null))];
    if (v.rows != null) notes.push(t.length === v.rows ? chip(`${thousands(v.rows)} rows, as recorded in the manifest`, "good") : chip(`${thousands(t.length)} rows, manifest records ${thousands(v.rows)}`, "warn"));
    if (v.source_id) notes.push(h("span", {}, "Imported from ", linkOrText("source_id", v.source_id)));
  }
  const vt = new VirtualTable(t, { filters, link, name: title.split("/").pop(), title });
  return [head("All tables", "Every table of the bundle, the generated outputs and the manifest views, in full. Filter, sort, open a row, or download the current view."),
    h("div", { class: "split" }, side, card(title, `${thousands(t.length)} rows × ${t.columns.length} columns`,
      notes.length ? h("div", { style: { display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "center", marginBottom: "10px", color: "var(--ink-2)", fontSize: "13px" } }, notes.map(x => (typeof x === "string" ? h("span", {}, x) : x))) : null,
      vt.el))];
}

/* ---------- Start ---------- */
const themeBtn = document.getElementById("theme");
function showTheme() { themeBtn.textContent = "Theme: " + (document.documentElement.dataset.theme || "system"); }
themeBtn.addEventListener("click", () => {
  const order = [undefined, "light", "dark"], cur = document.documentElement.dataset.theme, next = order[(order.indexOf(cur) + 1) % 3];
  if (next) document.documentElement.dataset.theme = next; else delete document.documentElement.dataset.theme;
  try { next ? localStorage.setItem("datenansicht-theme", next) : localStorage.removeItem("datenansicht-theme"); } catch (e) { /* ohne Speicher */ }
  resetColors(); showTheme(); route(true);
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { resetColors(); route(true); });
showTheme();
INFO().then(info => { document.getElementById("sub").textContent = info.data_dir; }).catch(() => {});
json("/ausgabe/pruefung.json").then(c => document.getElementById("status").replaceChildren(c.passed ? chip("validator passed", "good") : chip("validator failed", "bad"))).catch(() => {});
route(false);
// Hintergrund: große Tabellen schon parsen, während die erste Seite steht
setTimeout(() => { F().then(() => HP()).then(() => SRC()).catch(() => {}); }, 1500);
