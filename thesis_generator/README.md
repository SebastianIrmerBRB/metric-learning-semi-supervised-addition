# Thesis figure and table generator

The generator code and compact source data are bundled here. It reads `daten/`
and regenerates the thesis figures and tables without training logs, image
datasets, model checkpoints, or network access.

After installing the repository's root `requirements.txt` or
`requirements-cuda.txt`, run this from the repository root:

```bash
python thesis_generator/generate.py
```

Output goes to `thesis_generator/ausgabe/`: PDF figures in `bilder/neu/`, LaTeX
tables in `kapitel/`, PNG previews in `vorschau/`, and script logs in `protokoll/`.

```bash
python thesis_generator/generate.py --output /path/to/output
python thesis_generator/generate.py --only fig_phase2 tab_extern
python thesis_generator/abbildungen_src/pruefe_daten.py
```

The generator validates the bundled data before producing output. Its data
tables, provenance manifest, detailed measurements, fonts, and published
retrieval examples are included in `daten/`; detailed inputs and the font
license are stored in `daten/details.zip`. The source CSV files are not modified
by generation.

`requirements-getestet.txt` records the generator's exact reference versions;
the root training requirements use the same NumPy, pandas, Matplotlib, SciPy,
and Pillow versions, so one environment can run both workflows.
