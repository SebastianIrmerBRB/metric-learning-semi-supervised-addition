"""Build the data viewer as a static site (GitHub Pages): the same pages as viewer/serve.py, without a server.

    python viewer/build_site.py --data thesis_generator/daten --ausgabe <generator output> --out _site

serve.py answers three listing routes (/info, /zip, /ausgabe) and serves files from daten/, details.zip and the generator
output. Here the files are copied or unpacked and the listings are written as JSON files. The page code in viewer/web is
copied with relative paths, so the site also works below a project path such as /<repository>/.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

WEB = Path(__file__).resolve().parent / "web"
LISTS = 'const STATIC_LISTS = { "/info": "info.json", "/zip": "zip/index.json", "/ausgabe": "ausgabe/index.json" };\n'
# (file, old, new, expected count): every change must match exactly, otherwise the page code has changed and the build stops
PATCHES = [
    ("index.html", 'href="/style.css"', 'href="style.css"', 1),
    ("index.html", 'src="/app.js"', 'src="app.js"', 1),
    ("core.js", "async function get(url, as) {\n  const r = await fetch(url);",
     LISTS + "async function get(url, as) {\n  const r = await fetch(STATIC_LISTS[url] || url);", 1),
    ("core.js", "`/daten/${", "`daten/${", 1),
    ("core.js", "`/zip/${", "`zip/${", 1),
    ("app.js", 'from "/core.js"', 'from "./core.js"', 1),
    ("app.js", '"/ausgabe/', '"ausgabe/', 7),
    ("app.js", 'const DIAG = "/diagnostics/"', 'const DIAG = "diagnostics/"', 1),
]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--ausgabe", type=Path, help="output folder of thesis_generator/generate.py (Figures page)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", default=os.environ.get("SITE_LABEL", ""), help="text shown as the data source")
    p.add_argument("--atlas", help="relative URL of the Trial Atlas, adds a link to the navigation")
    a = p.parse_args()
    data, out = a.data.resolve(), a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)

    for name in ("index.html", "app.js", "core.js", "style.css"):
        text = (WEB / name).read_text(encoding="utf-8")
        for f, old, new, n in PATCHES:
            if f == name:
                if text.count(old) != n:
                    raise SystemExit(f"{name}: expected {n}x {old!r}, found {text.count(old)}")
                text = text.replace(old, new)
        if name == "index.html" and a.atlas:
            nav = '<a href="#/explore">All tables</a>'
            if text.count(nav) != 1:
                raise SystemExit("index.html: navigation anchor for the Trial Atlas link not found")
            text = text.replace(nav, nav + f'\n        <a href="{a.atlas}">Trial Atlas ↗</a>')
        (out / name).write_text(text, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")

    # condensed tables of the Diagnostics page
    shutil.copytree(data.parent / "datenansicht" / "diagnostics", out / "diagnostics")

    # daten/ and /info
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8")).get("files", {})
    files = []
    (out / "daten").mkdir()
    for f in sorted(data.iterdir()):
        if f.is_file():
            shutil.copy2(f, out / "daten" / f.name)
            e, s = manifest.get(f.name) or {}, sha256(f)
            files.append(dict(name=f.name, bytes=f.stat().st_size, mtime=f.stat().st_mtime, sha256=s,
                              manifest_sha256=e.get("sha256"), matches=(e.get("sha256") == s) if e else None))
    write_json(out / "info.json", dict(data_dir=a.label or str(data), files=files))

    # details.zip unpacked, /zip
    with zipfile.ZipFile(data / "details.zip") as z:
        members = [i for i in z.infolist() if not i.is_dir()]
        if any(i.filename == "index.json" for i in members):
            raise SystemExit("details.zip has a member index.json, which the listing would overwrite")
        for i in members:
            target = (out / "zip" / i.filename).resolve()
            if out / "zip" not in target.parents:
                raise SystemExit(f"unsafe member name {i.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(i))
        write_json(out / "zip" / "index.json", [dict(name=i.filename, bytes=i.file_size, packed=i.compress_size)
                                                for i in members])

    # generator output and /ausgabe (without the matplotlib cache)
    listing = []
    if a.ausgabe:
        src = a.ausgabe.resolve()
        for f in sorted(src.rglob("*")):
            rel = f.relative_to(src)
            if f.is_file() and rel.parts[0] != ".matplotlib":
                dst = out / "ausgabe" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                listing.append(dict(path=rel.as_posix(), bytes=f.stat().st_size, mtime=f.stat().st_mtime))
        if any(x["path"] == "index.json" for x in listing):
            raise SystemExit("the generator output has a file index.json, which the listing would overwrite")
    write_json(out / "ausgabe" / "index.json", listing)

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{out}: {len(files)} data files, {len(members)} details.zip members, {len(listing)} generator outputs, "
          f"{size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
