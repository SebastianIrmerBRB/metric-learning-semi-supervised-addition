"""Datenansicht (21.10.2026): lokale Website über die kompakte Datensammlung daten/.

    python3 viewer/serve.py            # http://127.0.0.1:8770
    python3 viewer/serve.py --port 8771 --data /anderer/ordner/daten

Liest daten/ nur, schreibt nichts (wie abbildungen_src/datensammlung.py auch THESIS_DATA_DIR). Routen:
  /                  Seite (index.html, app.js, style.css aus diesem Ordner)
  /daten/<datei>     eine Datei aus daten/ (nur Namen, die dort liegen)
  /zip               Inhalt von details.zip (Name, Größe, gepackte Größe)
  /zip/<mitglied>    eine Datei aus details.zip
  /info              Dateien in daten/ mit Größe, Änderungszeit und SHA-256, verglichen mit manifest.json
  /ausgabe           Dateiliste der erzeugten Ausgaben (erzeuge_aus_daten.py, Ordner ausgabe/ oder --ausgabe)
  /ausgabe/<pfad>    eine Datei daraus (Vorschau-PNG, PDF, Tabellen, uebersicht.csv, pruefung.json)
  /diagnostics/<datei>  eine Datei aus datenansicht/diagnostics (kondensierte Tabellen, export_diagnostics.py)
"""
import argparse
import hashlib
import json
import mimetypes
import os
import zipfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HIER = Path(__file__).resolve().parent
DATEN = Path(os.environ.get("THESIS_DATA_DIR", HIER.parent / "thesis_generator" / "daten")).resolve()
AUSGABE = (HIER.parent / "thesis_generator" / "ausgabe").resolve()
DIAGNOSE = DATEN.parent / "datenansicht" / "diagnostics"
SEITE = {"/": "index.html", "/app.js": "app.js", "/core.js": "core.js", "/style.css": "style.css"}
TYPEN = {".csv": "text/csv; charset=utf-8", ".json": "application/json; charset=utf-8", ".jsonl": "text/plain; charset=utf-8",
         ".md": "text/markdown; charset=utf-8", ".txt": "text/plain; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".html": "text/html; charset=utf-8", ".png": "image/png", ".otf": "font/otf",
         ".pdf": "application/pdf", ".tex": "text/plain; charset=utf-8", ".log": "text/plain; charset=utf-8"}
_SHA = {}  # Pfad -> ((mtime, Größe), sha256)


def sha256(p):
    stand = (p.stat().st_mtime_ns, p.stat().st_size)
    if _SHA.get(p, (None,))[0] != stand:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        _SHA[p] = (stand, h.hexdigest())
    return _SHA[p][1]


def info():
    try:
        erwartet = json.loads((DATEN / "manifest.json").read_text(encoding="utf-8")).get("files", {})
    except (OSError, ValueError):
        erwartet = {}
    dateien = []
    for p in sorted(DATEN.iterdir()):
        if p.is_file():
            e = erwartet.get(p.name) or {}
            s = sha256(p)
            dateien.append(dict(name=p.name, bytes=p.stat().st_size, mtime=p.stat().st_mtime, sha256=s,
                                manifest_sha256=e.get("sha256"), matches=(e.get("sha256") == s) if e else None))
    return dict(data_dir=str(DATEN), files=dateien)


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):  # ruhig, nur Fehler
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)

    def senden(self, body, typ, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", typ)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def datei(self, p):
        stand = f'"{p.stat().st_mtime_ns:x}-{p.stat().st_size:x}"'
        if self.headers.get("If-None-Match") == stand:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.end_headers()
            return
        body = p.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", TYPEN.get(p.suffix, mimetypes.guess_type(p.name)[0] or "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("ETag", stand)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        pfad = unquote(urlparse(self.path).path)
        if pfad in SEITE:
            return self.datei(HIER / "web" / SEITE[pfad])
        if pfad.startswith("/daten/"):
            name = pfad[len("/daten/"):]
            if "/" not in name and name in os.listdir(DATEN):
                return self.datei(DATEN / name)
        if pfad == "/ausgabe":
            liste = [dict(path=p.relative_to(AUSGABE).as_posix(), bytes=p.stat().st_size, mtime=p.stat().st_mtime)
                     for p in sorted(AUSGABE.rglob("*")) if p.is_file()] if AUSGABE.is_dir() else []
            return self.senden(json.dumps(liste).encode(), TYPEN[".json"])
        if pfad.startswith("/ausgabe/"):
            p = (AUSGABE / pfad[len("/ausgabe/"):]).resolve()
            if p.is_file() and AUSGABE in p.parents:
                return self.datei(p)
        if pfad.startswith("/diagnostics/"):
            name = pfad[len("/diagnostics/"):]
            if "/" not in name and (DIAGNOSE / name).is_file():
                return self.datei(DIAGNOSE / name)
        if pfad == "/info":
            return self.senden(json.dumps(info()).encode(), TYPEN[".json"])
        if pfad == "/zip" or pfad.startswith("/zip/"):
            with zipfile.ZipFile(DATEN / "details.zip") as z:
                if pfad == "/zip":
                    liste = [dict(name=i.filename, bytes=i.file_size, packed=i.compress_size)
                             for i in z.infolist() if not i.is_dir()]
                    return self.senden(json.dumps(liste).encode(), TYPEN[".json"])
                name = pfad[len("/zip/"):]
                if name in z.namelist():
                    return self.senden(z.read(name), TYPEN.get(Path(name).suffix, "application/octet-stream"))
        self.senden(b"not found", "text/plain", HTTPStatus.NOT_FOUND)


def main():
    global DATEN, AUSGABE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--data", type=Path, default=DATEN)
    ap.add_argument("--ausgabe", type=Path, default=AUSGABE)
    a = ap.parse_args()
    DATEN, AUSGABE = a.data.resolve(), a.ausgabe.resolve()
    print(f"Datenansicht: http://127.0.0.1:{a.port}  (daten: {DATEN}, ausgabe: {AUSGABE})")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
