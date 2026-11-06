// Datenansicht: Laden, Formatieren, Farben, Tabellen, Diagramme. Keine externen Bibliotheken.

/* ---------- DOM ---------- */
export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  setAttrs(el, attrs);
  append(el, kids);
  return el;
}
const SVGNS = "http://www.w3.org/2000/svg";
export function s(tag, attrs, ...kids) {
  const el = document.createElementNS(SVGNS, tag);
  setAttrs(el, attrs);
  append(el, kids);
  return el;
}
function setAttrs(el, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.setAttribute("class", v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
}
function append(el, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k == null || k === false) continue;
    el.append(k instanceof Node ? k : String(k));
  }
}

/* ---------- Laden ---------- */
const WORKER_SRC = `
const NUM = /^[-+]?(\\d+\\.?\\d*|\\.\\d+)([eE][-+]?\\d+)?$/;
function parse(text) {
  if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);
  const out = []; let row = [], f = "", i = 0, q = false; const n = text.length;
  while (i < n) {
    const c = text.charCodeAt(i);
    if (q) {
      if (c === 34) { if (text.charCodeAt(i + 1) === 34) { f += '"'; i += 2; continue; } q = false; i++; continue; }
      const j = text.indexOf('"', i); if (j < 0) { f += text.slice(i); i = n; continue; }
      f += text.slice(i, j); i = j; continue;
    }
    if (c === 34) { q = true; i++; continue; }
    if (c === 44) { row.push(f); f = ""; i++; continue; }
    if (c === 10 || c === 13) { row.push(f); f = ""; out.push(row); row = []; if (c === 13 && text.charCodeAt(i + 1) === 10) i++; i++; continue; }
    let j = i; while (j < n) { const d = text.charCodeAt(j); if (d === 44 || d === 34 || d === 10 || d === 13) break; j++; }
    f += text.slice(i, j); i = j;
  }
  if (f !== "" || row.length) { row.push(f); out.push(row); }
  const columns = out.shift() || [];
  const rows = out.filter(r => !(r.length === 1 && r[0] === ""));
  const m = columns.length, types = [];
  for (let c = 0; c < m; c++) {
    let num = true, any = false;
    for (const r of rows) { const v = r[c]; if (v === undefined || v === "") continue; any = true; if (!NUM.test(v) && v !== "nan" && v !== "NaN") { num = false; break; } }
    types.push(num && any ? "num" : "str");
  }
  for (const r of rows) for (let c = 0; c < m; c++) {
    const v = r[c];
    if (v === undefined || v === "") r[c] = null;
    else if (types[c] === "num") r[c] = (v === "nan" || v === "NaN") ? null : +v;
  }
  return { columns, rows, types };
}
self.onmessage = e => { try { self.postMessage({ id: e.data.id, result: parse(e.data.text) }); } catch (err) { self.postMessage({ id: e.data.id, error: String(err) }); } };`;
// Kleine Dateien gleich hier, große in bis zu drei Workern, damit eine kleine Datei nicht hinter einer großen wartet
const parseLocal = new Function(WORKER_SRC.replace(/self\.onmessage[\s\S]*$/, "") + "; return parse;")();
const pool = [];
let seq = 0;
const waiting = new Map();
function parseInWorker(text) {
  if (text.length < 400000) return Promise.resolve(parseLocal(text));
  let w = pool.find(x => x.busy === 0);
  if (!w && pool.length < 3) {
    w = { worker: new Worker(URL.createObjectURL(new Blob([WORKER_SRC], { type: "text/javascript" }))), busy: 0 };
    w.worker.onmessage = e => {
      const p = waiting.get(e.data.id); waiting.delete(e.data.id); p.w.busy--;
      e.data.error ? p.reject(new Error(e.data.error)) : p.resolve(e.data.result);
    };
    pool.push(w);
  }
  if (!w) w = pool.reduce((x, y) => (y.busy < x.busy ? y : x));
  const id = ++seq;
  w.busy++;
  return new Promise((resolve, reject) => { waiting.set(id, { resolve, reject, w }); w.worker.postMessage({ id, text }); });
}
const cache = new Map();
function once(key, make) {
  if (!cache.has(key)) cache.set(key, make().catch(e => { cache.delete(key); throw e; }));
  return cache.get(key);
}
async function get(url, as) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${r.statusText}`);
  return as === "json" ? r.json() : r.text();
}
export const text = url => once("t:" + url, () => get(url));
export const json = url => once("j:" + url, () => get(url, "json"));
export const csv = url => once("c:" + url, () => text(url).then(parseInWorker).then(t => new Table(t)));
export const daten = name => `/daten/${encodeURIComponent(name)}`;
export const zip = name => `/zip/${name.split("/").map(encodeURIComponent).join("/")}`;

export class Table {
  constructor({ columns, rows, types }) {
    this.columns = columns; this.rows = rows; this.types = types;
    this.ix = Object.fromEntries(columns.map((c, i) => [c, i]));
  }
  get length() { return this.rows.length; }
  objects() {
    if (!this._obj) this._obj = this.rows.map(r => { const o = {}; this.columns.forEach((c, i) => { o[c] = r[i]; }); return o; });
    return this._obj;
  }
  static fromObjects(objs, columns) {
    columns = columns || [...new Set(objs.flatMap(o => Object.keys(o)))];
    const rows = objs.map(o => columns.map(c => (o[c] === undefined || o[c] === "" ? null : o[c])));
    const types = columns.map((c, i) => rows.every(r => r[i] == null || typeof r[i] === "number") && rows.some(r => r[i] != null) ? "num" : "str");
    rows.forEach(r => types.forEach((t, i) => { if (t === "str" && r[i] != null && typeof r[i] !== "string") r[i] = typeof r[i] === "object" ? JSON.stringify(r[i]) : String(r[i]); }));
    return new Table({ columns, rows, types });
  }
  // Ansicht wie datensammlung.read_csv: Gleichheit als Text, dann Spaltenauswahl
  view(where, columns) {
    const conds = Object.entries(where || {}).map(([c, v]) => [this.ix[c], String(v)]);
    let rows = this.rows.filter(r => conds.every(([i, v]) => r[i] != null && String(r[i]) === v));
    let cols = this.columns;
    if (columns) { const idx = columns.map(c => this.ix[c]); rows = rows.map(r => idx.map(i => (i === undefined ? null : r[i]))); cols = columns; }
    const t = new Table({ columns: cols, rows, types: cols.map(c => this.types[this.ix[c]] || "str") });
    return t;
  }
}

/* ---------- Formatieren ---------- */
const trimz = x => x.replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
export function fmtNum(x) {
  if (x == null || Number.isNaN(x)) return "";
  if (typeof x !== "number") return String(x);
  if (Number.isInteger(x)) return Math.abs(x) >= 10000 ? x.toLocaleString("en-US") : String(x);
  const a = Math.abs(x);
  if (a >= 1000) return x.toLocaleString("en-US", { maximumFractionDigits: 1 });
  if (a >= 1) return trimz(x.toFixed(3));
  if (a < 1e-4) return x.toExponential(2);
  return trimz(x.toPrecision(4));
}
export const fix = (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : x.toFixed(d));
export const signed = (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : (x > 0 ? "+" : x < 0 ? "−" : "±") + Math.abs(x).toFixed(d));
export const thousands = x => (x == null ? "–" : Number(x).toLocaleString("en-US"));
export function bytes(n) {
  if (n == null) return "";
  const u = ["B", "KB", "MB", "GB"]; let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (i ? n.toFixed(n < 10 ? 1 : 0) : n) + " " + u[i];
}
export function middle(sv, max = 64) {
  if (sv == null) return "";
  sv = String(sv);
  if (sv.length <= max) return sv;
  const head = Math.round(max * 0.3);
  return sv.slice(0, head) + "…" + sv.slice(sv.length - (max - head - 1));
}
export function mean(a) { const v = a.filter(x => x != null && !Number.isNaN(x)); return v.length ? v.reduce((p, c) => p + c, 0) / v.length : null; }
export function sd(a) {
  const v = a.filter(x => x != null && !Number.isNaN(x));
  if (v.length < 2) return null;
  const m = mean(v);
  return Math.sqrt(v.reduce((p, c) => p + (c - m) ** 2, 0) / (v.length - 1));
}
export function niceStep(span, target = 5) {
  const raw = span / target, p = 10 ** Math.floor(Math.log10(raw)), f = raw / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
}
export function niceMax(x) {
  if (!(x > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(x)), f = x / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
}

/* ---------- Farben ---------- */
export const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function hexRgb(hex) {
  hex = hex.replace("#", "");
  if (hex.length === 3) hex = [...hex].map(c => c + c).join("");
  return [0, 2, 4].map(i => parseInt(hex.slice(i, i + 2), 16) / 255);
}
const lin = c => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const delin = c => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
function oklab(hex) {
  const [r, g, b] = hexRgb(hex).map(lin);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const q = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * q, 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * q,
    0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * q];
}
function fromOklab([L, A, B]) {
  const l = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3, m = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3,
    q = (L - 0.0894841775 * A - 1.2914855480 * B) ** 3;
  const rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * q, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * q,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * q];
  return "#" + rgb.map(c => Math.round(Math.min(1, Math.max(0, delin(c))) * 255).toString(16).padStart(2, "0")).join("");
}
function luminance(hex) { const [r, g, b] = hexRgb(hex).map(lin); return 0.2126 * r + 0.7152 * g + 0.0722 * b; }
export function contrast(a, b) { const [x, y] = [luminance(a), luminance(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); }
let pal = null;
export function resetColors() { pal = null; }
function palette() {
  if (!pal) pal = { mid: oklab(cssVar("--div-mid")), pos: oklab(cssVar("--div-pos")), neg: oklab(cssVar("--div-neg")), ink: cssVar("--ink") };
  return pal;
}
// t in [-1, 1]: negativ rot, positiv blau, 0 grau (Mitte der Referenzpalette)
export function diverging(t) {
  const p = palette(), u = Math.min(1, Math.abs(t)) ** 0.9, e = t >= 0 ? p.pos : p.neg;
  return fromOklab(p.mid.map((v, i) => v + (e[i] - v) * u));
}
export function inkOn(bg) { return contrast(bg, "#0b0b0b") >= contrast(bg, "#ffffff") ? "#0b0b0b" : "#ffffff"; }
export function divergingBar(steps = 24) {
  return "linear-gradient(90deg," + Array.from({ length: steps + 1 }, (_, i) => diverging(-1 + (2 * i) / steps)).join(",") + ")";
}

/* ---------- Tooltip ---------- */
let tipEl = null;
export function tip(ev, ...content) {
  if (!tipEl) tipEl = document.body.appendChild(h("div", { class: "tip", role: "tooltip" }));
  tipEl.replaceChildren(...content.flat().filter(Boolean));
  tipEl.hidden = false;
  moveTip(ev);
}
export function moveTip(ev) {
  if (!tipEl || tipEl.hidden) return;
  const r = tipEl.getBoundingClientRect(), pad = 14;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - pad;
  tipEl.style.left = Math.max(8, x) + "px";
  tipEl.style.top = Math.max(8, y) + "px";
}
export function untip() { if (tipEl) tipEl.hidden = true; }
export const tipTitle = t => h("div", { class: "t" }, t);
export const tipRow = (k, v, key) => h("div", { class: "row" }, h("span", { class: "k" }, key || null, k), h("span", { class: "val" }, v));
export const tipSub = t => h("div", { class: "sub" }, t);
export function hoverTip(el, build) {
  el.addEventListener("pointerenter", ev => tip(ev, build()));
  el.addEventListener("pointermove", moveTip);
  el.addEventListener("pointerleave", untip);
  el.addEventListener("focus", () => { const r = el.getBoundingClientRect(); tip({ clientX: r.right, clientY: r.bottom }, build()); });
  el.addEventListener("blur", untip);
}

/* ---------- Bedienelemente ---------- */
export function seg(label, options, value, onChange) {
  return h("label", { class: "field" }, h("span", {}, label),
    h("div", { class: "seg", role: "group", "aria-label": label },
      options.map(([v, t]) => h("button", { type: "button", "aria-pressed": String(String(v) === String(value)), onclick: () => onChange(v) }, t))));
}
export function select(label, options, value, onChange) {
  const el = h("select", { onchange: e => onChange(e.target.value) },
    options.map(([v, t]) => h("option", { value: v, selected: String(v) === String(value) }, t)));
  return h("label", { class: "field" }, h("span", {}, label), el);
}
export function chip(text, status) {
  return h("span", { class: "chip " + (status ? "status-" + status : "") }, status ? h("span", { class: "dot" }) : null, text);
}
export function card(title, hint, ...body) {
  return h("section", { class: "card" }, h("div", { class: "card-head" }, h("h2", {}, title), hint ? h("span", { class: "hint" }, hint) : null), ...body);
}
export function legend(items, kind = "line") {
  return h("div", { class: "legend" }, items.map(it => h("span", { class: "item" },
    kind === "line" ? h("span", { class: "linekey", style: { background: it.color } }) : h("span", { class: "swatch", style: { background: it.color } }),
    it.name)));
}

/* ---------- Schublade ---------- */
let drawerEls = null;
export function drawer(kicker, title, ...sections) {
  closeDrawer();
  const back = h("div", { class: "drawer-back", onclick: closeDrawer });
  const panel = h("aside", { class: "drawer", role: "dialog", "aria-label": title },
    h("div", { class: "drawer-head" },
      h("div", { style: { flex: 1, minWidth: 0 } }, kicker ? h("div", { class: "kicker" }, kicker) : null, h("h2", {}, title)),
      h("button", { class: "iconbtn", onclick: closeDrawer, "aria-label": "Close" }, "Close ✕")),
    h("div", { class: "drawer-body" }, ...sections));
  document.body.append(back, panel);
  drawerEls = [back, panel];
  panel.querySelector("button").focus();
  return panel.querySelector(".drawer-body");
}
export function closeDrawer() { if (drawerEls) drawerEls.forEach(e => e.remove()); drawerEls = null; untip(); }
addEventListener("keydown", e => { if (e.key === "Escape") closeDrawer(); });
export function kv(pairs) {
  return h("dl", { class: "kv" }, pairs.filter(Boolean).map(([k, v]) => [h("dt", {}, k), h("dd", {}, v instanceof Node ? v : v == null ? "–" : String(v))]));
}

/* ---------- Virtuelle Tabelle ---------- */
function predicate(type, raw) {
  const t = raw.trim();
  if (!t) return null;
  if (t === "empty") return v => v == null;
  if (t === "!empty") return v => v != null;
  if (type === "num") {
    let m = t.match(/^(>=|<=|>|<|=|!=)\s*(-?[\d.]+(?:e[-+]?\d+)?)$/i);
    if (m) {
      const x = +m[2];
      return { ">=": v => v != null && v >= x, "<=": v => v != null && v <= x, ">": v => v != null && v > x, "<": v => v != null && v < x,
        "=": v => v === x, "!=": v => v !== x }[m[1]];
    }
    m = t.match(/^(-?[\d.]+)\s*\.\.\s*(-?[\d.]+)$/);
    if (m) { const a = +m[1], b = +m[2]; return v => v != null && v >= a && v <= b; }
  }
  if (t.startsWith("!=")) { const x = t.slice(2); return v => String(v ?? "") !== x; }
  if (t.startsWith("=")) { const x = t.slice(1); return v => String(v ?? "") === x; }
  const neg = t.startsWith("!"), needle = (neg ? t.slice(1) : t).toLowerCase();
  return v => { const sv = v == null ? "" : String(v).toLowerCase(); return neg ? !sv.includes(needle) : sv.includes(needle); };
}
const collator = new Intl.Collator("en", { numeric: true, sensitivity: "base" });

export class VirtualTable {
  // opts: columns (Auswahl/Reihenfolge), filters {spalte: text}, search, onRow(obj, row), link(col, value) -> {href|onclick}, hideEmpty
  constructor(table, opts = {}) {
    this.t = table; this.o = opts; this.rh = 28;
    this.filters = { ...(opts.filters || {}) };
    this.search = opts.search || "";
    this.sort = opts.sort || null; // [index, dir]
    const all = opts.columns ? opts.columns.map(c => table.ix[c]).filter(i => i !== undefined) : table.columns.map((_, i) => i);
    const empty = i => table.rows.every(r => r[i] == null);
    this.visible = all.filter(i => !(opts.hideEmpty !== false && empty(i)));
    this.hiddenEmpty = all.length - this.visible.length;
    this.el = h("div", { class: "vt" });
    this.build();
  }
  widths() {
    const t = this.t, sample = t.rows.length > 400 ? Array.from({ length: 400 }, (_, k) => t.rows[Math.floor((k * t.rows.length) / 400)]) : t.rows;
    return this.visible.map(i => {
      let w = t.columns[i].length * 7.4 + 30;
      for (const r of sample) { const v = r[i]; if (v != null) w = Math.max(w, Math.min(52, this.text(i, v).length) * 7.1 + 18); }
      return Math.round(Math.min(420, Math.max(64, w)));
    });
  }
  text(i, v) { return v == null ? "" : this.t.types[i] === "num" ? fmtNum(v) : String(v); }
  build() {
    const t = this.t;
    this.searchEl = h("input", { type: "search", placeholder: "Search all columns…", value: this.search, oninput: e => { this.search = e.target.value; this.later(); } });
    this.countEl = h("span", { class: "count" });
    const colsBtn = h("button", { class: "iconbtn", type: "button", onclick: e => this.chooser(e.currentTarget) }, "Columns");
    const dl = h("button", { class: "iconbtn", type: "button", onclick: () => this.download() }, "Download CSV");
    this.head = h("div", { class: "vt-head" });
    this.rowsEl = h("div", { class: "vt-rows", role: "rowgroup" });
    this.body = h("div", { class: "vt-body" }, this.rowsEl);
    this.scroll = h("div", { class: "vt-scroll", role: "table", "aria-rowcount": String(t.rows.length) }, this.head, this.body);
    if (this.o.height) this.scroll.style.height = this.o.height;
    this.scroll.addEventListener("scroll", () => this.paint());
    this.el.replaceChildren(h("div", { class: "vt-bar" }, this.searchEl, this.countEl, colsBtn, dl), this.scroll,
      h("div", { class: "foot" }, "Filters per column: text matches substrings, ", h("code", {}, "=x"), " exact, ", h("code", {}, "!x"), " excludes, ",
        h("code", {}, ">0.5"), ", ", h("code", {}, "2..8"), ", ", h("code", {}, "empty"), ". Click a header to sort, a row for details.",
        this.hiddenEmpty ? ` ${this.hiddenEmpty} empty column${this.hiddenEmpty > 1 ? "s" : ""} hidden (Columns).` : ""));
    this.header();
    this.apply();
    new ResizeObserver(() => this.paint()).observe(this.scroll);
  }
  header() {
    const t = this.t, w = this.widths();
    this.el.style.setProperty("--cols", w.map(x => x + "px").join(" "));
    const hr = h("div", { class: "vt-row", role: "row" }, this.visible.map(i => {
      const dir = this.sort && this.sort[0] === i ? this.sort[1] : 0;
      return h("div", { class: "vt-hcell", role: "columnheader", title: t.columns[i] + (t.types[i] === "num" ? " (number)" : ""),
        "aria-sort": dir ? (dir > 0 ? "ascending" : "descending") : "none",
        onclick: () => { this.sort = !dir ? [i, 1] : dir > 0 ? [i, -1] : null; this.header(); this.apply(); } },
      t.columns[i], dir ? h("span", { class: "arrow" }, dir > 0 ? "▲" : "▼") : null);
    }));
    const fr = h("div", { class: "vt-row" }, this.visible.map(i => h("div", { class: "vt-fcell" },
      h("input", { type: "text", value: this.filters[t.columns[i]] || "", "aria-label": "Filter " + t.columns[i], placeholder: t.types[i] === "num" ? ">, <, a..b" : "filter",
        oninput: e => { this.filters[t.columns[i]] = e.target.value; this.later(); } }))));
    this.head.replaceChildren(hr, fr);
  }
  later() { clearTimeout(this.timer); this.timer = setTimeout(() => this.apply(), 120); }
  apply() {
    const t = this.t;
    const preds = Object.entries(this.filters).map(([c, v]) => [t.ix[c], predicate(t.types[t.ix[c]], v || "")]).filter(([i, p]) => i !== undefined && p);
    const q = this.search.trim().toLowerCase();
    if (q && !this.hay) this.hay = t.rows.map(r => r.map(v => (v == null ? "" : String(v))).join("\u0001").toLowerCase());
    let idx = [];
    for (let k = 0; k < t.rows.length; k++) {
      const r = t.rows[k];
      if (preds.every(([i, p]) => p(r[i])) && (!q || this.hay[k].includes(q))) idx.push(k);
    }
    if (this.sort) {
      const [i, dir] = this.sort, num = t.types[i] === "num";
      idx.sort((a, b) => {
        const x = t.rows[a][i], y = t.rows[b][i];
        if (x == null) return y == null ? 0 : 1;
        if (y == null) return -1;
        return dir * (num ? x - y : collator.compare(String(x), String(y)));
      });
    }
    this.idx = idx;
    this.countEl.textContent = `${thousands(idx.length)} of ${thousands(t.rows.length)} rows`;
    this.body.style.height = idx.length * this.rh + "px";
    this.last = null;
    this.paint();
  }
  paint() {
    const top = this.scroll.scrollTop, hgt = this.scroll.clientHeight || 600, headH = this.head.offsetHeight;
    const first = Math.max(0, Math.floor((top - headH) / this.rh) - 8), last = Math.min(this.idx.length, Math.ceil((top + hgt) / this.rh) + 8);
    if (this.last && this.last[0] === first && this.last[1] === last) return;
    this.last = [first, last];
    const t = this.t, link = this.o.link;
    const rows = [];
    for (let k = first; k < last; k++) {
      const r = t.rows[this.idx[k]];
      rows.push(h("div", { class: "vt-row", role: "row", onclick: () => this.rowClick(this.idx[k]) }, this.visible.map(i => {
        const v = r[i], txt = this.text(i, v), num = t.types[i] === "num";
        const l = v != null && link ? link(t.columns[i], v) : null;
        const content = l ? h("span", { class: "lnk", onclick: e => { e.stopPropagation(); l(); } }, txt.length > 70 ? middle(txt, 64) : txt)
          : txt.length > 70 ? middle(txt, 64) : txt;
        return h("div", { class: "vt-cell" + (num ? " n" : "") + (v == null ? " null" : ""), role: "cell", title: v == null ? null : String(v) }, content);
      })));
    }
    this.rowsEl.style.transform = `translateY(${first * this.rh}px)`;
    this.rowsEl.replaceChildren(...rows);
  }
  rowClick(k) {
    const t = this.t, r = t.rows[k], obj = Object.fromEntries(t.columns.map((c, i) => [c, r[i]]));
    if (this.o.onRow) return this.o.onRow(obj, k);
    const link = this.o.link;
    drawer(this.o.title || "Row", `Row ${k + 1}`, h("section", {}, kv(t.columns.map((c, i) => {
      if (r[i] == null) return null;
      const l = link ? link(c, r[i]) : null;
      return [c, l ? h("a", { href: "#", onclick: e => { e.preventDefault(); closeDrawer(); l(); } }, String(r[i])) : t.types[i] === "num" ? String(r[i]) : String(r[i])];
    }))));
  }
  chooser(btn) {
    const t = this.t;
    document.querySelectorAll(".popover").forEach(p => p.remove());
    const set = new Set(this.visible);
    const box = h("div", { class: "popover" },
      h("div", { style: { display: "flex", gap: "8px", marginBottom: "6px" } },
        h("button", { class: "iconbtn", onclick: () => { this.visible = t.columns.map((_, i) => i); done(); } }, "All"),
        h("button", { class: "iconbtn", onclick: () => { this.visible = t.columns.map((_, i) => i).filter(i => t.rows.some(r => r[i] != null)); done(); } }, "Non-empty")),
      t.columns.map((c, i) => h("label", {}, h("input", { type: "checkbox", checked: set.has(i), onchange: e => {
        if (e.target.checked) set.add(i); else set.delete(i);
        this.visible = t.columns.map((_, j) => j).filter(j => set.has(j)); this.header(); this.last = null; this.paint();
      } }), c)));
    const done = () => { box.remove(); this.header(); this.last = null; this.paint(); };
    const r = btn.getBoundingClientRect();
    box.style.left = Math.max(8, Math.min(innerWidth - 340, r.left + scrollX)) + "px";
    box.style.top = r.bottom + scrollY + 6 + "px";
    document.body.append(box);
    setTimeout(() => addEventListener("pointerdown", function off(e) { if (!box.contains(e.target)) { box.remove(); removeEventListener("pointerdown", off); } }), 0);
  }
  download() {
    const t = this.t, q = v => { if (v == null) return ""; const sv = String(v); return /[",\n\r]/.test(sv) ? '"' + sv.replace(/"/g, '""') + '"' : sv; };
    const lines = [this.visible.map(i => q(t.columns[i])).join(",")];
    for (const k of this.idx) lines.push(this.visible.map(i => q(t.rows[k][i])).join(","));
    const a = h("a", { href: URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "text/csv" })), download: (this.o.name || "table").replace(/[^\w.-]+/g, "_") + ".csv" });
    document.body.append(a); a.click(); a.remove();
  }
}

/* ---------- Liniendiagramm ---------- */
// opts: xs (Zahlen), xlabel(x), log, series [{name, color, values, hollow?: [bool], ring?: [bool]}], yfmt, ydomain, zero, refs [{y, label}], height, ytitle, xtitle, extra(i) -> tooltip rows, marks [{x, y, color}] (Rauten), marktip(getroffene marks) -> tooltip rows
export function lineChart(opts) {
  const W = opts.width || 320, H = opts.height || 220, m = { l: 44, r: 12, t: 10, b: opts.xtitle ? 38 : 26 };
  const xs = opts.xs;
  const xv = opts.categorical ? xs.map((_, i) => i) : opts.log ? xs.map(Math.log) : xs;
  // Rauten (opts.marks: x, y, color) liegen meist zwischen den Rasterwerten, die Achse reicht bis zu ihnen
  const marks = opts.marks || [], xm = marks.map(k => (opts.log ? Math.log(k.x) : k.x));
  const x0 = Math.min(...xv, ...xm), x1 = Math.max(...xv, ...xm);
  const X = v => m.l + (x1 === x0 ? (W - m.l - m.r) / 2 : ((v - x0) / (x1 - x0)) * (W - m.l - m.r));
  let [y0, y1] = opts.ydomain;
  const step = niceStep(y1 - y0, 4);
  y0 = Math.floor(y0 / step) * step; y1 = Math.ceil(y1 / step) * step;
  if (y1 === y0) { y1 += step; y0 -= step; }
  const Y = v => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const yfmt = opts.yfmt || (v => fmtNum(+v.toFixed(6)));
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label || "line chart" });
  const grid = s("g", { class: "grid" });
  for (let v = y0; v <= y1 + step / 2; v += step) {
    grid.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
    svg.append(s("text", { x: m.l - 6, y: Y(v) + 3.5, "text-anchor": "end", class: "num" }, yfmt(+v.toFixed(10))));
  }
  svg.prepend(grid);
  svg.append(s("line", { class: "baseline", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
  if (opts.zero && y0 < 0 && y1 > 0) svg.append(s("line", { class: "zero", x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0) }));
  for (const r of opts.refs || []) {
    if (r.y < y0 || r.y > y1) continue;
    svg.append(s("line", { class: "ref", x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y) }));
    if (r.label) svg.append(s("text", { x: W - m.r, y: Y(r.y) - 4, "text-anchor": "end" }, r.label));
  }
  // Achsenbeschriftung ohne Überlappung: zu nahe Marken entfallen, die letzte bleibt immer
  const ticks = xs.map((x, i) => [x, xv[i]]);
  for (const v of [x0, x1]) if (xm.includes(v) && !xv.includes(v)) ticks.push([marks[xm.indexOf(v)].x, v]);
  ticks.sort((p, q) => p[1] - q[1]);
  let lastX = -Infinity, lastEl = null;
  ticks.forEach(([x, v], i) => {
    const px = X(v), close = px - lastX < 22;
    if (close && i < ticks.length - 1) return;
    if (close && lastEl) lastEl.remove();
    lastEl = s("text", { x: px, y: H - m.b + 15, "text-anchor": "middle", class: "num" }, opts.xlabel ? opts.xlabel(x) : fmtNum(x));
    svg.append(lastEl); lastX = px;
  });
  if (opts.xtitle) svg.append(s("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "axis-title" }, opts.xtitle));
  if (opts.ytitle) svg.append(s("text", { x: 4, y: m.t - 0, class: "axis-title" }, ""));
  for (const se of opts.series) {
    let d = "", pen = false;
    se.values.forEach((v, i) => { if (v == null) { pen = false; return; } d += (pen ? "L" : "M") + X(xv[i]).toFixed(1) + " " + Y(v).toFixed(1); pen = true; });
    svg.append(s("path", { class: "series", d, stroke: se.color }));
    se.values.forEach((v, i) => {
      if (v == null) return;
      const hollow = se.hollow && se.hollow[i];
      svg.append(s("circle", { class: hollow ? "hollow" : "dot", cx: X(xv[i]), cy: Y(v), r: se.ring && se.ring[i] ? 6 : 4, fill: hollow ? null : se.color, stroke: hollow ? se.color : null }));
    });
  }
  marks.forEach((k, i) => {
    const px = X(xm[i]), py = Y(k.y), r = 6;
    svg.append(s("path", { class: "mark", fill: k.color, d: `M${px} ${py - r}L${px + r} ${py}L${px} ${py + r}L${px - r} ${py}Z` }));
  });
  const hair = s("line", { class: "hair", y1: m.t, y2: H - m.b, visibility: "hidden" });
  const over = s("rect", { x: m.l - 10, y: 0, width: W - m.l - m.r + 20, height: H, fill: "transparent" });
  svg.append(hair, over);
  const pointer = ev => {
    const p = svg.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY;
    return p.matrixTransform(svg.getScreenCTM().inverse());
  };
  const nearest = q => {
    let best = 0, bd = Infinity;
    xv.forEach((v, i) => { const dd = Math.abs(X(v) - q.x); if (dd < bd) { bd = dd; best = i; } });
    return best;
  };
  over.addEventListener("pointermove", ev => {
    const q = pointer(ev), i = nearest(q);
    // Über einer Raute deren Werte statt der ganzen Spalte
    const hit = opts.marktip ? marks.filter((k, j) => Math.hypot(X(xm[j]) - q.x, Y(k.y) - q.y) <= 9) : [];
    const hx = hit.length ? X(xm[marks.indexOf(hit[0])]) : X(xv[i]);
    hair.setAttribute("x1", hx); hair.setAttribute("x2", hx); hair.setAttribute("visibility", "visible");
    if (hit.length) { tip(ev, opts.marktip(hit)); return; }
    const rows = opts.series.map(se => [se, se.values[i]]).filter(([, v]) => v != null).sort((a, b) => b[1] - a[1])
      .map(([se, v]) => tipRow(se.name, (opts.tipfmt || yfmt)(v), h("span", { class: "linekey", style: { background: se.color } })));
    tip(ev, tipTitle(opts.xlabel ? opts.xlabel(xs[i]) : fmtNum(xs[i])), rows, opts.extra ? opts.extra(i) : null);
  });
  over.addEventListener("pointerleave", () => { hair.setAttribute("visibility", "hidden"); untip(); });
  return svg;
}

/* ---------- Markdown (kleiner Ausschnitt) ---------- */
function inline(t) {
  const out = []; let rest = t;
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\[[^\]]+\]\([^)]+\))/;
  while (rest) {
    const m = rest.match(re);
    if (!m) { out.push(rest); break; }
    if (m.index) out.push(rest.slice(0, m.index));
    const tok = m[0];
    if (tok.startsWith("`")) out.push(h("code", {}, tok.slice(1, -1)));
    else if (tok.startsWith("**")) out.push(h("strong", {}, inline(tok.slice(2, -2))));
    else { const mm = tok.match(/^\[([^\]]+)\]\(([^)]+)\)$/); out.push(h("a", { href: /^https?:/.test(mm[2]) ? mm[2] : "#", target: "_blank", rel: "noopener" }, mm[1])); }
    rest = rest.slice(m.index + tok.length);
  }
  return out;
}
export function markdown(src) {
  const root = h("div", { class: "md" }), lines = src.replace(/\r/g, "").split("\n");
  for (let i = 0; i < lines.length;) {
    const l = lines[i];
    if (l.startsWith("```")) { const buf = []; i++; while (i < lines.length && !lines[i].startsWith("```")) buf.push(lines[i++]); i++; root.append(h("pre", {}, h("code", {}, buf.join("\n")))); continue; }
    const hm = l.match(/^(#{1,4})\s+(.*)$/);
    if (hm) { root.append(h("h" + Math.min(4, hm[1].length), {}, inline(hm[2]))); i++; continue; }
    if (l.startsWith("|")) {
      const rows = []; while (i < lines.length && lines[i].startsWith("|")) rows.push(lines[i++]);
      const cells = r => r.replace(/^\||\|$/g, "").split("|").map(c => c.trim());
      const body = rows.slice(2);
      root.append(h("table", {}, h("thead", {}, h("tr", {}, cells(rows[0]).map(c => h("th", {}, inline(c))))),
        h("tbody", {}, body.map(r => h("tr", {}, cells(r).map(c => h("td", {}, inline(c))))))));
      continue;
    }
    if (/^\s*([-*]|\d+\.)\s+/.test(l)) {
      const ordered = /^\s*\d+\./.test(l), list = h(ordered ? "ol" : "ul", {});
      while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) {
        let item = lines[i++].replace(/^\s*([-*]|\d+\.)\s+/, "");
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*]|\d+\.)\s+/.test(lines[i])) item += " " + lines[i++].trim();
        list.append(h("li", {}, inline(item)));
      }
      root.append(list); continue;
    }
    if (!l.trim()) { i++; continue; }
    const buf = [l]; i++;
    while (i < lines.length && lines[i].trim() && !/^(#|\||```|\s*([-*]|\d+\.)\s)/.test(lines[i])) buf.push(lines[i++]);
    root.append(h("p", {}, inline(buf.join(" "))));
  }
  return root;
}

/* ---------- JSON ---------- */
export function jsonTree(value, depth = 0, key = null) {
  const k = key != null ? [h("span", { class: "k" }, JSON.stringify(key)), ": "] : [];
  if (value === null || typeof value !== "object") {
    const cls = value === null ? "nl" : typeof value === "string" ? "s" : "nm";
    return h("div", {}, k, h("span", { class: cls }, JSON.stringify(value)));
  }
  const arr = Array.isArray(value), entries = arr ? value.map((v, i) => [i, v]) : Object.entries(value);
  const label = arr ? `[${entries.length}]` : `{${entries.length}}`;
  const det = h("details", { open: depth < 2 && entries.length <= 60 }, h("summary", {}, k, h("span", { class: "nl" }, label)));
  det.addEventListener("toggle", function fill() {
    if (!det.open || det.dataset.filled) return;
    det.dataset.filled = "1";
    const shown = entries.slice(0, 2000);
    det.append(...shown.map(([kk, v]) => jsonTree(v, depth + 1, arr ? null : kk)));
    if (entries.length > shown.length) det.append(h("div", { class: "nl" }, `… ${entries.length - shown.length} more`));
  });
  if (det.open) det.dispatchEvent(new Event("toggle"));
  return depth === 0 ? h("div", { class: "json" }, det) : det;
}
