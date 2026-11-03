# Data viewer

Website over the compact thesis data in `../thesis_generator/daten`: Phase-1 results, ablations, external images, the
generated figures and tables, all hyperparameter configurations, every member of `details.zip` and every table with
filter, sorting and CSV download. Plain JavaScript without external libraries, read-only.

**Online:** https://sebastianirmerbrb.github.io/metric-learning-semi-supervised-addition/ (rebuilt by
`.github/workflows/datenansicht.yml` on every push to `main` that touches the data, the generators or the viewer).

The same workflow publishes the Trial Atlas (`../react_trial_visualizer`) at
https://sebastianirmerbrb.github.io/metric-learning-semi-supervised-addition/trial-atlas/, linked from the viewer's
navigation.

**Locally:**

```bash
python thesis_generator/generate.py --output thesis_generator/ausgabe   # optional, for the Figures page
python viewer/serve.py                                                  # http://127.0.0.1:8770
```

`build_site.py` writes the same pages as a static site: it copies `daten/`, unpacks `details.zip`, writes the three
file listings that `serve.py` answers as JSON and makes the paths in `web/` relative.
