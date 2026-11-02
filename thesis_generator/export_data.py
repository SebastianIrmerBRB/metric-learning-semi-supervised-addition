"""Export every compact thesis data view and the cross-experiment overview."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "abbildungen_src"))
from exportiere_daten import main

if __name__ == "__main__":
    main()
