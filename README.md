# metric-learning-semi-supervised-addition

Original repository: https://github.com/gmberton/image-retrieval

For CUDA support on Windows, install PyTorch wheels with (done automatically now, in requirements.txt):
`--extra-index-url https://download.pytorch.org/whl/cu128`

Sind nicht unbedingt in richtiger Reihenfolge erledigt (Skript erledigt einige der Sachen schon). 

| Status | Phase | Meilenstein                             |
| ------ | ----- | --------------------------------------- |
| ☑/2    | 1     | Forschungsfragen                        |
| ☑/2    | 2     | Literaturreview aufbauen                |
| ☑/2    | 3     | Hypothesen und Variablen definieren     |
| ☑      | 4     | Evaluationsdesign festlegen             |
| ☑/2    | 5     | Datensätze auswählen    |
| ☑      | 6     | Baseline-Methoden implementieren        |
| ☐      | 7     | SSML-Methoden implementieren            |
| ☑      | 8     | Foundation-Model-Strategien integrieren |
| ☑/2    | 9     | Pilotexperimente durchführen            |
...

## Documentation (Work in progress, Daten dazu werden noch nicht mit hochgeladen)

- Sampler epoch length: [docs/length_before_new_iter.md](docs/length_before_new_iter.md)
- Semi-supervised FixMatch training: [docs/semi_supervised_fixmatch.md](docs/semi_supervised_fixmatch.md)
- Example config values: [docs/example_config.yaml](docs/example_config.yaml)


- graph / fixmatch propagation repository: https://github.com/thomasbohm/semi-supervised-dml https://github.com/google-research/fixmatch
