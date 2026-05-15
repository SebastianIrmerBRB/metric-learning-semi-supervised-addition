# metric-learning-semi-supervised-addition

Original repository: https://github.com/gmberton/image-retrieval

For CUDA support on Windows, install PyTorch wheels with (done automatically now, in requirements.txt):
`--extra-index-url https://download.pytorch.org/whl/cu128`

Sind nicht unbedingt in richtiger Reihenfolge erledigt (Skript erledigt einige der Sachen schon). 

| Status | Phase | Meilenstein                             |
|--------|-------|-----------------------------------------|
| ☑/2    | 1     | Forschungsfragen                        |
| ☑/2    | 2     | Literaturreview aufbauen                |
| ☑/2    | 3     | Hypothesen und Variablen definieren     |
| ☑      | 4     | Evaluationsdesign festlegen             |
| ☑/2    | 5     | Datensätze auswählen                    |
| ☑      | 6     | Baseline-Methoden implementieren        |
| ☑      | 7     | ViT-Models finden (CLIP/DinoV2)         |
| ☑      | 8     | Pilotexperimente durchführen            |
| ☑      | 9     | Log erweitern                           |
| ☑/2    | 10    | SSL Approaches finden                   |
| ☐      | 11    | SSL Approaches implementieren           |
| ☐      | 12    | Baseline Runs - Hyperparameter tuning?  |
| ☐      | 13    | Baseline Runs - Darstellungen           |
| ☐      | 14    | SSDML Runs - Parameter to tune?         |
| ☐      | 15    | SSDML Runs - Manual / automatic tuning? |
| ☐      | 16    | SSDML Runs - Gegenüberstellung Arten    |


...
## Problem zur Zeit:
- SSDML-Methoden sind nicht offen implementiert (LMNR, SERAPH sind bekannt, aber nicht direkt auf GitHub veröffentlicht)
- SSDML-Methoden sind oft nur transductiv (SSDML-Papers beschreiben Affinity Propagation, Label Propagation, ...), weil sonst Class-Predictions notwendig sind
- Evtl. Classifier-Head dazu machen? 

## Documentation (Work in progress, Daten dazu werden noch nicht mit hochgeladen)

- Sampler epoch length: [docs/length_before_new_iter.md](docs/length_before_new_iter.md)
- Semi-supervised FixMatch training: [docs/semi_supervised_fixmatch.md](docs/semi_supervised_fixmatch.md)
- Example config values: [docs/example_config.yaml](docs/example_config.yaml)


- graph / fixmatch propagation repository: https://github.com/thomasbohm/semi-supervised-dml https://github.com/google-research/fixmatch
