# Datenansicht für die Abgabe

`ergebnisuebersicht.csv` bündelt 2728 Ergebniszeilen über Phase 1,
Ablationen (beide Budgets und gelabelte Klassen), SLADE-Sweeps, Fold-Ensemble,
externe Bilder und die Referenz mit allen Labels. Sie ist ein Leseeinstieg;
Geometrie, Laufzeiten, HPO und Diagnosen bleiben kompakt in `../daten/details.zip`.
`spalten.csv` erklärt die Übersichtsspalten und Einheiten.

`dateninventar.csv` erschließt **alle 87 dokumentierten Datensichten** und
37 zusätzliche Bild-/Schriftdateien. Jede Zeile nennt Datei, Umfang,
Auswahl und vorhandene Quellreferenz. `file` nennt die Quelldatei, `member` den
Eintrag in `details.zip`, sofern vorhanden. `export_sha256` beschreibt den bei
Bedarf erzeugten Ansichtsinhalt, nicht den ZIP-Container. Abgeleitete Sichten
werden mit dem mitgelieferten Code neu berechnet.

Die Abgabe speichert jeden Detaildatensatz einmal komprimiert. Optional erzeugt
`python export_data.py --expand` alle Ansichten unter `ansichten/` und die
Bild-/Schriftdateien unter `assets/`. Der Index nennt dann auch `expanded_file`.

| Bereich | Logische Datensicht (siehe Inventar) |
| --- | --- |
| Phase 1, fünf Datenserien, fünf Seeds, alle Metriken/Folds/Settings | `*_ergebnisse/data/` |
| Ablationen, 2×2-Zerlegung, Sweeps, LP-Diagnostik | `phase2/`, `phase2_k8/`, `phase2_klassen/`, `phase2_lp_diagnostik.csv` |
| Ensemble und geteilte Retrieval-Fehler | `phase2_k4_ensemble/` |
| Externe ungelabelte Bilder, Kontrollen und Sweeps | `extern/` |
| Geometrie, Epochen, Validierung, Pseudo-Label-Qualität, Korrelationen | `geometrie/` |
| Unveränderte Merkmale, untrainierter Head, alle Labels | `ausgangsleistung/` |
| Laufzeitmessungen | `timings/faelle.csv` |
| Verzögerte Patience und Auswahlentscheidungen | `patience_delay/`, `cvsum_rerun/` |
| HPO-Suchräume und Protokolle (235 Studien) | `studies.json` |
| HPO-Versuche, Parameter, Validierungs-Folds, komplette Studienkonfigurationen | `hpo/` |
| Sensitivität und flache Optima; Textzahlen | `hpo_sensitivitaet/`, zusätzlich `hpo_plateau.csv` im Ansichtsordner |
| Vorexperimente und Verteilung guter Versuche | `phase0_vergleich/` |
| Klassen- und Bildzahlen | `datensaetze/statistik.json` |
| Retrieval-Nachbarn, t-SNE und übernommene Pseudo-Labels | `qualitative/` |

Die sechs CSVs und `details.zip` unter `../daten/` sind die unveränderten
Quellen dieses Exports. Sie erhalten auch die ursprüngliche Genauigkeit und
Provenienz. Die Ansicht schreibt Fließkommazahlen mit 17 signifikanten Stellen,
die kondensierte Ergebnisübersicht mit zwölf. Einheiten unterscheiden sich je
Quellsicht; die Quellenbeschreibung `../daten/README.md` ist maßgeblich.
CSV: UTF-8, Komma als Trennzeichen, Punkt als Dezimalzeichen. In Excel über
„Daten → Aus Text/CSV“ mit diesen Einstellungen öffnen.

Die Abgabe enthält die Messungen, die für die aktuelle Arbeit benötigt werden,
mit den im Manifest dokumentierten Ausschnitten. Bei Cars196 sind alle
Geometrie-Epochen enthalten; bei den anderen Geometrieserien die ausgewerteten
Anfangs-/Endstände und Validierungsverläufe. Dies ist kein vollständiges Archiv
aller Trainingslogs. Die ergänzte Abgabefassung enthält die kompakten Trial-Daten
der 235 ausgewerteten HPO-Studien und die Retrieval-Rangfolgen aller 8131 Querys.
Vollständige Bilddatensätze,
Modellgewichte und fremde Literaturdatensätze gehören nicht zu diesem Paket.
Die kleinen Abbildungs-Thumbnails sind enthalten. Synthetische Lehrabbildungen
werden aus dem mitgelieferten Code erzeugt.

Nach Datenänderungen: `python generate.py`, dann `python export_data.py` im
Paketstamm. Das regeneriert die Ausgaben und diese Ansichten ohne Trainingslogs.
