"""Generate all thesis figures and tables from the included compact data (see README.md)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "abbildungen_src"))
from erzeuge_aus_daten import main

if __name__ == "__main__":
    main()
