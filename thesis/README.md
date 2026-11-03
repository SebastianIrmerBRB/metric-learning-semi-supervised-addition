# Thesis LaTeX source

LaTeX source of the master's thesis as submitted (state of 3 November 2026), with every figure it includes.

Build with a TeX distribution that has biblatex/biber (tested with MiKTeX):

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

| Path | Contents |
| --- | --- |
| `main.tex`, `einstellungen.tex`, `gestaltung.tex` | main document, packages and settings, layout |
| `kapitel/` | chapters, front and back matter, appendix |
| `kapitel/*_tabellen/` | tables, written by the generators in `../thesis_generator` |
| `kapitel/abb/` | the three schematic figures (TikZ) |
| `bilder/neu/` | the data figures, written by the generators in `../thesis_generator` |
| `literatur.bib` | bibliography |

Every data figure and table can be regenerated from the compact data with `python ../thesis_generator/generate.py`. `../thesis_generator/abdeckung.csv` maps each output to its script and its place in the thesis. The folder contains only the files `main.tex` uses. Editorial `%` comments were removed from the chapter files, which does not change the PDF.
