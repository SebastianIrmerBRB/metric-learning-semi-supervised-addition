"""Write the condensed view of the geometry, pseudo-label and neighborhood data (datenansicht/diagnostics)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "abbildungen_src"))
from exportiere_diagnostik import main

if __name__ == "__main__":
    main()
