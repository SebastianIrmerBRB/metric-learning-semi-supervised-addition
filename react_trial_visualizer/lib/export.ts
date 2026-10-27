import { slugify } from "@/lib/trials";

export type ExportTheme = "dark" | "print";

export interface ExportLegendItem {
  name: string;
  color: string;
  dashed?: boolean;
}

export interface ExportOptions {
  /** "print": white background, neutral inks darkened -- for the thesis. Default "dark" (the app's look). */
  theme?: ExportTheme;
  /** Drawn into the image below the plot, so an exported figure is self-contained. */
  legend?: ExportLegendItem[];
  /** PNG pixel density; print defaults to 4 (about 300 dpi at a text-width figure). */
  scale?: number;
}

const SVG_NS = "http://www.w3.org/2000/svg";

function parseColor(value: string): [number, number, number, number] | null {
  const hex = value.trim().match(/^#([0-9a-f]{3,8})$/i);
  if (hex) {
    let digits = hex[1];
    if (digits.length === 3 || digits.length === 4) digits = digits.split("").map((c) => c + c).join("");
    if (digits.length !== 6 && digits.length !== 8) return null;
    const channel = (i: number) => parseInt(digits.slice(i, i + 2), 16);
    return [channel(0), channel(2), channel(4), digits.length === 8 ? channel(6) / 255 : 1];
  }
  const rgb = value.trim().match(/^rgba?\(([^)]+)\)$/i);
  if (rgb) {
    const parts = rgb[1].split(",").map((part) => parseFloat(part));
    if (parts.length < 3 || parts.some((part) => Number.isNaN(part))) return null;
    return [parts[0], parts[1], parts[2], parts[3] ?? 1];
  }
  if (value.trim().toLowerCase() === "white") return [255, 255, 255, 1];
  return null;
}

/**
 * Print ink for a dark-theme color: light neutral grays (text, axes, grid,
 * white reference lines) flip their lightness so they read on white paper;
 * saturated series colors are kept, so methods keep their colors.
 */
export function printColor(value: string): string {
  const parsed = parseColor(value);
  if (!parsed) return value;
  const [r, g, b, alpha] = parsed;
  const max = Math.max(r, g, b) / 255;
  const min = Math.min(r, g, b) / 255;
  const lightness = (max + min) / 2;
  const saturation = max === min ? 0 : (max - min) / (1 - Math.abs(2 * lightness - 1));
  if (saturation > 0.2 || lightness <= 0.4) return value;
  const shift = Math.round((1 - lightness) * 255 - lightness * 255);
  const ink = (channel: number) => Math.max(0, Math.min(255, channel + shift));
  return `rgba(${ink(r)}, ${ink(g)}, ${ink(b)}, ${alpha})`;
}

function applyPrintTheme(root: SVGSVGElement) {
  const elements = [root, ...Array.from(root.querySelectorAll<SVGElement>("*"))];
  for (const element of elements) {
    for (const attribute of ["fill", "stroke", "stop-color"]) {
      const value = element.getAttribute(attribute);
      if (value && value !== "none") element.setAttribute(attribute, printColor(value));
    }
  }
}

function appendLegend(root: SVGSVGElement, items: ExportLegendItem[], width: number, top: number, ink: string): number {
  const group = document.createElementNS(SVG_NS, "g");
  const rowHeight = 16;
  let x = 12;
  let y = top + 12;
  for (const item of items) {
    const itemWidth = 30 + item.name.length * 6;
    if (x + itemWidth > width - 8 && x > 12) {
      x = 12;
      y += rowHeight;
    }
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", String(x));
    line.setAttribute("x2", String(x + 18));
    line.setAttribute("y1", String(y));
    line.setAttribute("y2", String(y));
    line.setAttribute("stroke", item.color);
    line.setAttribute("stroke-width", "2");
    if (item.dashed) line.setAttribute("stroke-dasharray", "4 3");
    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("x", String(x + 23));
    label.setAttribute("y", String(y + 3.5));
    label.setAttribute("font-size", "10");
    label.setAttribute("fill", ink);
    label.textContent = item.name;
    group.append(line, label);
    x += itemWidth;
  }
  root.appendChild(group);
  return y + rowHeight - 4 - top;
}

function prepareSvg(container: HTMLElement, options: ExportOptions = {}): { source: string; width: number; height: number } {
  const original = container.querySelector<SVGSVGElement>("svg[data-export-root]");
  if (!original) throw new Error("No diagram was found to export.");

  const print = options.theme === "print";
  const clone = original.cloneNode(true) as SVGSVGElement;
  clone.querySelectorAll("[data-export-ignore]").forEach((element) => element.remove());
  const viewBox = clone.viewBox.baseVal;
  const width = viewBox?.width || original.clientWidth || 960;
  let height = viewBox?.height || original.clientHeight || 360;
  if (print) applyPrintTheme(clone);
  if (options.legend?.length) {
    const legendColors = options.legend.map((item) => ({ ...item, color: print ? printColor(item.color) : item.color }));
    height += appendLegend(clone, legendColors, width, height, print ? "#1f2937" : "#c8d2e4");
  }

  clone.setAttribute("xmlns", SVG_NS);
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));
  clone.setAttribute("viewBox", `0 0 ${width} ${height}`);
  clone.setAttribute("style", "font-family: Inter, ui-sans-serif, system-ui, sans-serif");

  const background = document.createElementNS(SVG_NS, "rect");
  background.setAttribute("x", "0");
  background.setAttribute("y", "0");
  background.setAttribute("width", String(width));
  background.setAttribute("height", String(height));
  background.setAttribute("fill", print ? "#ffffff" : "#10182a");
  clone.insertBefore(background, clone.firstChild);

  return {
    source: new XMLSerializer().serializeToString(clone),
    width,
    height,
  };
}

export function downloadBlob(blob: Blob, fileName: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
}

export function exportSvg(container: HTMLElement, name: string, options: ExportOptions = {}) {
  const { source } = prepareSvg(container, options);
  downloadBlob(new Blob([source], { type: "image/svg+xml;charset=utf-8" }), `${slugify(name)}.svg`);
}

export async function exportPng(container: HTMLElement, name: string, options: ExportOptions = {}) {
  const { source, width, height } = prepareSvg(container, options);
  const svgUrl = URL.createObjectURL(new Blob([source], { type: "image/svg+xml;charset=utf-8" }));

  try {
    const image = new Image();
    image.decoding = "async";
    image.src = svgUrl;
    await image.decode();

    const scale = options.scale ?? (options.theme === "print" ? 4 : 2);
    const canvas = document.createElement("canvas");
    canvas.width = width * scale;
    canvas.height = height * scale;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("PNG export is not supported by this browser.");
    context.scale(scale, scale);
    context.drawImage(image, 0, 0, width, height);

    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((result) => (result ? resolve(result) : reject(new Error("Could not create the PNG."))), "image/png");
    });
    downloadBlob(blob, `${slugify(name)}.png`);
  } finally {
    URL.revokeObjectURL(svgUrl);
  }
}

export function downloadText(text: string, name: string, type = "text/csv;charset=utf-8") {
  downloadBlob(new Blob([text], { type }), name);
}

export type ExportRow = Record<string, string | number | boolean | null | undefined>;

function csvCell(value: ExportRow[string]): string {
  if (value == null || (typeof value === "number" && !Number.isFinite(value))) return "";
  const text = String(value);
  return /[",\n;]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** CSV with the union of all row keys as header, in first-seen order. */
export function toCsv(rows: ExportRow[]): string {
  const columns: string[] = [];
  for (const row of rows) for (const key of Object.keys(row)) if (!columns.includes(key)) columns.push(key);
  return [columns.join(","), ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(","))].join("\n") + "\n";
}

export function exportCsv(rows: ExportRow[], name: string) {
  downloadText(toCsv(rows), `${slugify(name)}.csv`);
}

const LATEX_SYMBOLS: Record<string, string> = {
  "±": "$\\pm$",
  "Δ": "$\\Delta$",
  "−": "$-$",
  "×": "$\\times$",
  "ρ": "$\\rho$",
  "π": "$\\pi$",
  "κ": "$\\kappa$",
  "γ": "$\\gamma$",
  "α": "$\\alpha$",
  "≈": "$\\approx$",
  "≥": "$\\geq$",
  "≤": "$\\leq$",
  "→": "$\\rightarrow$",
};

/** Escape plain text for pdflatex; the Greek letters and math signs the app uses become math. */
function latexEscape(text: string): string {
  return Array.from(text)
    .map((char) => {
      if (char in LATEX_SYMBOLS) return LATEX_SYMBOLS[char];
      if (char === "\\") return "\\textbackslash{}";
      if (char === "~") return "\\textasciitilde{}";
      if (char === "^") return "\\textasciicircum{}";
      if ("&%$#_{}".includes(char)) return `\\${char}`;
      return char;
    })
    .join("");
}

export interface LatexColumn {
  header: string;
  align?: "l" | "r" | "c";
}

/** A booktabs table (\\usepackage{booktabs}); cells are plain text and escaped here. */
export function toLatexTable(columns: LatexColumn[], rows: string[][], caption?: string, label?: string): string {
  const lines = [
    "\\begin{table}[htbp]",
    "  \\centering",
    "  \\small",
    `  \\begin{tabular}{${columns.map((c) => c.align ?? "l").join("")}}`,
    "    \\toprule",
    `    ${columns.map((c) => latexEscape(c.header)).join(" & ")} \\\\`,
    "    \\midrule",
    ...rows.map((row) => `    ${row.map(latexEscape).join(" & ")} \\\\`),
    "    \\bottomrule",
    "  \\end{tabular}",
  ];
  if (caption) lines.push(`  \\caption{${latexEscape(caption)}}`);
  if (label) lines.push(`  \\label{${label}}`);
  lines.push("\\end{table}", "");
  return lines.join("\n");
}

export function exportLatex(source: string, name: string) {
  downloadText(source, `${slugify(name)}.tex`, "application/x-tex;charset=utf-8");
}
