"""Geometrieläufe der Phase 2 (sammle_geometrie.py) laden und die Maße am per Validierung gewählten Stand bestimmen.

Gewählter Stand: letzte verfolgte Epoche, die nicht nach der per Validierung gewählten Epoche liegt. In den Geometrieläufen
(alle Datensätze seit 09.10.2026) wird jede Epoche verfolgt, also die gewählte Epoche selbst. Abgebrochene Verfolgung
(laeufe_*.csv: geo_last_epoch < selected_epoch) filtert fig_geometrie_weitere.py vorher aus.
Test-MAP@R in Prozent: einzelnes Fold-Modell (cv_results.csv).
"""
import datensammlung as sammlung
import pandas as pd

RD = "geometrie/"


def load(ds):
    E = sammlung.read_csv(RD + f"epochen_{ds}.csv", low_memory=False)
    R = sammlung.read_csv(RD + f"laeufe_{ds}.csv")
    V = sammlung.read_csv(RD + f"validierung_{ds}.csv")
    return E, R, V


def at_selected(E, R):
    """Eine Zeile je Fold mit Testwert: Maße an der letzten verfolgten Epoche <= gewählte Epoche."""
    rows = []
    for _, r in R.dropna(subset=["test_map"]).iterrows():
        e = E[(E.method == r.method) & (E.loss == r.loss) & (E.k == r.k) & (E.fold == r.fold)]
        s = e[e.epoch <= r.selected_epoch]
        x = s.loc[s.epoch.idxmax()].copy()
        x["selected_epoch"], x["test_map"] = r.selected_epoch, 100 * r.test_map
        rows.append(x)
    return pd.DataFrame(rows)


def complete(S):
    """Nur Kombinationen aus Verfahren, Verlustfunktion und Budget mit allen vier Folds."""
    n = S.groupby(["method", "loss", "k"]).fold.transform("count")
    return S[n == 4]


def curve(x, col):
    """Mittel über die Folds an den Epochen, an denen alle vier Folds einen Wert haben."""
    x = x.dropna(subset=[col])
    n = x.groupby("epoch").fold.nunique()
    g = x.groupby("epoch")[col].mean()
    return g[n == 4]
