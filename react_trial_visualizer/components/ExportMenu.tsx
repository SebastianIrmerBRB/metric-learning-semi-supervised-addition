"use client";

import { useRef, useState, type ReactNode, type RefObject } from "react";
import { Download } from "lucide-react";
import {
  exportCsv,
  exportLatex,
  exportPng,
  exportSvg,
  type ExportLegendItem,
  type ExportRow,
} from "@/lib/export";

type Format = "svg" | "png" | "csv" | "tex";

/**
 * Thesis export for one figure or table: vector (SVG) and 300-dpi raster (PNG)
 * in print colors with the legend drawn in, the plotted numbers as CSV, and a
 * booktabs table as LaTeX. Only the formats whose source is given are offered.
 */
export function ExportMenu({
  name,
  figure,
  legend,
  rows,
  latex,
  png,
}: {
  name: string;
  /** Container of an `svg[data-export-root]`. */
  figure?: RefObject<HTMLElement | null>;
  legend?: ExportLegendItem[];
  rows?: () => ExportRow[];
  latex?: () => string;
  /** Raster export for canvas views, which have no SVG. */
  png?: () => Promise<void> | void;
}) {
  const [busy, setBusy] = useState<Format | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async (format: Format) => {
    setBusy(format);
    setError(null);
    try {
      if (format === "svg" && figure?.current) exportSvg(figure.current, name, { theme: "print", legend });
      if (format === "png") {
        if (png) await png();
        else if (figure?.current) await exportPng(figure.current, name, { theme: "print", legend });
      }
      if (format === "csv" && rows) exportCsv(rows(), name);
      if (format === "tex" && latex) exportLatex(latex(), name);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(null);
    }
  };

  const formats: { format: Format; label: string; title: string }[] = [];
  if (figure) formats.push({ format: "svg", label: "SVG", title: "Vektorgrafik, helles Druckschema, Legende im Bild" });
  if (figure || png) formats.push({ format: "png", label: "PNG", title: "Rastergrafik in 4-facher Auflösung (~300 dpi), helles Druckschema" });
  if (rows) formats.push({ format: "csv", label: "CSV", title: "Die dargestellten Werte als Tabelle" });
  if (latex) formats.push({ format: "tex", label: "LaTeX", title: "booktabs-Tabelle (\\usepackage{booktabs})" });

  return (
    <div className="flex items-center gap-1" data-export-ignore>
      <Download size={12} className="text-[#7f8ba2]" aria-hidden />
      {formats.map(({ format, label, title }) => (
        <button
          key={format}
          type="button"
          title={title}
          disabled={busy != null}
          onClick={() => void run(format)}
          className="rounded-md border border-white/[0.08] px-1.5 py-0.5 text-[10px] font-semibold text-[#98a4ba] transition hover:border-white/20 hover:text-white disabled:opacity-50"
        >
          {busy === format ? "…" : label}
        </button>
      ))}
      {error && <span className="ml-1 text-[10px] text-[#f5a39c]">{error}</span>}
    </div>
  );
}

/** A titled card whose figure (the first `svg[data-export-root]` inside) can be exported. */
export function FigureCard({
  title,
  description,
  name,
  legend,
  rows,
  latex,
  controls,
  className = "",
  exportFigure = true,
  children,
}: {
  title: string;
  description?: ReactNode;
  name: string;
  legend?: ExportLegendItem[];
  rows?: () => ExportRow[];
  latex?: () => string;
  controls?: ReactNode;
  className?: string;
  /** False for a card that holds only a table: then only CSV / LaTeX are offered. */
  exportFigure?: boolean;
  children: ReactNode;
}) {
  const figure = useRef<HTMLDivElement>(null);
  return (
    <section className={`rounded-2xl border border-white/[0.07] bg-white/[0.02] p-4 ${className}`}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-white">{title}</h3>
          {description && <p className="mt-1 text-[11px] text-[#7f8ba2]">{description}</p>}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {controls}
          <ExportMenu name={name} figure={exportFigure ? figure : undefined} legend={legend} rows={rows} latex={latex} />
        </div>
      </div>
      <div ref={figure}>{children}</div>
    </section>
  );
}
