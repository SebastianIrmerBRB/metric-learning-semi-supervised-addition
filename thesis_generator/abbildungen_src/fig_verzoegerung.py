"""Patience-Verzögerung als zweiter Wert (Anhang, Entscheidung des Autors 23.10.2026): je Lauf mit Kriterium A die
Test-MAP@R mit verzögerter Patience minus der ohne Verzögerung (Hauptwert der Arbeit), (a) je Versuch, (b) gegen die
Differenz der Validierungs-MAP@R. Daten: Sicht patience_delay/paare.csv des Datenpakets (sammle_verzoegerung_paare.py).
Ohne die 18 Läufe von Ü/Proxy-Anchor der In-Shop-Serie, deren Hauptwert schon mit Verzögerung ist (Spalte hauptwert), sie
stehen nur im Protokoll.
Ausgabe: bilder/neu/verzoegerung.pdf, kapitel/anhang_tabellen/verzoegerung.tex, im Protokoll die Zahlen für den Text."""
import datensammlung as sammlung
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import stil

stil.apply()
ALLE = sammlung.read_csv("patience_delay/paare.csv")
AUSNAHME = ALLE[ALLE.hauptwert == "verzoegert"]
P = ALLE[ALLE.hauptwert == "ohne"].copy()
TAB = sammlung.output_dir("kapitel/anhang_tabellen") + "verzoegerung.tex"
INSHOP = "DeepFashionInShop"
# (Bezeichnung, Quellen, Datensatz oder None)
GRUPPEN = [("Phase 1, Cars196", ["phase1_cars"], None),
           ("Phase 1, In-Shop", ["phase1_inshop"], None),
           ("Phase 1, In-Shop-Serie", ["phase1_inshop10"], None),
           ("Phase 1, Semi-iNat-2021", ["phase1_semiinat"], None),
           ("Phase 2, Cars196", ["phase2_variants", "phase2_slade_sweeps"], "Cars196"),
           ("Phase 2, In-Shop", ["phase2_variants", "phase2_slade_sweeps"], INSHOP),
           ("Zerlegung bei $k=4$, Cars196", ["k4_ensemble"], None),
           ("Externe Bilder, Cars196", ["extern", "alle_labels"], "Cars196"),
           ("Externe Bilder, In-Shop", ["extern"], INSHOP),
           ("Geometrie, Cars196", ["geometrie_cars"], None),
           ("Geometrie, In-Shop-Serie", ["geometrie_inshop10"], None),
           ("Geometrie, Semi-iNat-2021", ["geometrie_semiinat"], None)]


def gruppe(r):
    for i, (_, quellen, ds) in enumerate(GRUPPEN):
        if r.quelle in quellen and (ds is None or r.dataset == ds):
            return i
    raise ValueError((r.quelle, r.dataset))


P["gruppe"] = [gruppe(r) for r in P.itertuples()]
G = [(i, lab, P[P.gruppe == i]) for i, (lab, _, _) in enumerate(GRUPPEN) if (P.gruppe == i).any()]

fig, axes = plt.subplots(1, 2, figsize=(stil.TEXTWIDTH, 3.2), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
rng = np.random.default_rng(0)
for y, (i, lab, d) in enumerate(G):
    for m in stil.METHOD_ORDER:
        dm = d[d.method == m]
        ax.scatter(dm.d_mean, y + rng.uniform(-0.22, 0.22, len(dm)), s=10, color=stil.METHOD_COLORS[m],
                   marker=stil.METHOD_MARKERS[m], edgecolor="white", linewidth=0.3, zorder=3)
ax.axvline(0, color=stil.INK, lw=0.6)
ax.set_yticks(range(len(G)))
ax.set_yticklabels([f"{lab} ({len(d)})" for _, lab, d in G], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("Differenz im Test (Mean) [PP]", fontsize=8)
ax.margins(x=0.04)
stil.panel(ax, "a", "je Versuch")
ax = axes[1]
for m in stil.METHOD_ORDER:
    dm = P[P.method == m]
    ax.scatter(dm.d_val, dm.d_mean, s=10, color=stil.METHOD_COLORS[m], marker=stil.METHOD_MARKERS[m],
               edgecolor="white", linewidth=0.3, zorder=3)
lo = min(P.d_val.min(), P.d_mean.min()) - 0.3
hi = max(P.d_val.max(), P.d_mean.max()) + 0.3
ax.plot([lo, hi], [lo, hi], color=stil.MUTED, lw=0.7, ls="--", zorder=1)
ax.axhline(0, color=stil.INK, lw=0.6)
ax.axvline(0, color=stil.INK, lw=0.6)
ax.set_xlabel("Differenz in der Validierung [PP]", fontsize=8)
ax.set_ylabel("Differenz im Test (Mean) [PP]", fontsize=8)
stil.panel(ax, "b", "Validierung und Test")
methoden = [m for m in stil.METHOD_ORDER if (P.method == m).any()]
fig.legend([Line2D([], [], ls="", marker=stil.METHOD_MARKERS[m], color=stil.METHOD_COLORS[m]) for m in methoden],
           [stil.METHOD_LABELS[m] for m in methoden], loc="lower center", ncol=len(methoden), fontsize=7,
           bbox_to_anchor=(0.5, 0), handletextpad=0.2)
fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.2)
stil.save(fig, "verzoegerung")


def num(x, nd=1):
    return f"\\num{{{x:.{nd}f}}}" if round(x, nd) != 0 else "\\num{0.0}"


L = ["\\begin{table}[tbp]\n\\centering",
     "\\caption[Wirkung der verzögerten Patience]{Läufe mit einem zu früh beendeten Fold (Abschnitt~\\ref{sec:training}): "
     "Differenz der Test-MAP@R mit verzögerter Patience minus der Test-MAP@R ohne Verzögerung, die die Arbeit berichtet, in "
     "Prozentpunkten, Mittel und Spanne über die Läufe, und Zahl der Läufe, die mit Verzögerung besser validieren.}",
     "\\label{tab:anhang:verzoegerung}",
     "\\small\n\\begin{tabular}{@{}l S[table-format=3.0] S[table-format=-1.2] c S[table-format=-1.2] S[table-format=3.0]@{}}"
     "\n\\toprule",
     "{Versuch} & {Läufe} & {Mean} & {\\shortstack{Spanne\\\\(Mean)}} & {Concat} & {\\shortstack{Validierung\\\\besser}} "
     "\\\\\n\\midrule"]
for _, lab, d in G:
    L.append(f"{lab} & {len(d)} & {d.d_mean.mean():.2f} & {num(d.d_mean.min())} bis {num(d.d_mean.max())} & "
             f"{d.d_concat.mean():.2f} & {int((d.d_val > 1e-9).sum())} \\\\")
L.append(f"\\midrule\nalle & {len(P)} & {P.d_mean.mean():.2f} & {num(P.d_mean.min())} bis {num(P.d_mean.max())} & "
         f"{P.d_concat.mean():.2f} & {int((P.d_val > 1e-9).sum())} \\\\")
L.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
open(TAB, "w", encoding="utf-8").write("\n".join(L))

print("Hauptwert mit Verzögerung (nicht im Bild):", len(AUSNAHME), "Läufe,", AUSNAHME.quelle.value_counts().to_dict(),
      f"ohne Verzögerung Mean {AUSNAHME.mean_plain.min():.2f} bis {AUSNAHME.mean_plain.max():.2f}, mit {AUSNAHME.mean_verz.min():.2f} "
      f"bis {AUSNAHME.mean_verz.max():.2f}")
print("andere Validierungskurve vor dem früheren Ende:", int((~P.gleiche_kurve).sum()), "Läufe")
print(P[~P.gleiche_kurve][["quelle", "dataset", "method", "loss", "seed", "d_mean", "d_val", "original"]].round(2).to_string())
print(len(P), "Paare, Validierung besser:", int((P.d_val > 1e-9).sum()), "gleich:", int((P.d_val.abs() <= 1e-9).sum()))
for _, lab, d in G:
    print(f"{lab}: n={len(d)}, Mean {d.d_mean.mean():+.2f} ({d.d_mean.min():+.2f} bis {d.d_mean.max():+.2f}), "
          f"Concat {d.d_concat.mean():+.2f} ({d.d_concat.min():+.2f} bis {d.d_concat.max():+.2f}), "
          f"Test steigt {int((d.d_mean > 1e-9).sum())}, sinkt {int((d.d_mean < -1e-9).sum())}, "
          f"Validierung besser {int((d.d_val > 1e-9).sum())}")
besser = P[P.d_val > 1e-9]
print("Validierung besser:", len(besser), "davon Test sinkt", int((besser.d_mean < -1e-9).sum()),
      f"Korrelation dVal/dTest {np.corrcoef(P.d_val, P.d_mean)[0, 1]:.2f}")
print("größte Änderungen:")
print(P.reindex(P.d_mean.abs().sort_values(ascending=False).index)[
    ["quelle", "dataset", "method", "loss", "seed", "d_mean", "d_concat", "d_val", "original"]].head(8).round(2).to_string())
