from pathlib import Path
import sys


_SRC_ROOT = Path(__file__).resolve().parent / "src"
if _SRC_ROOT.exists() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from moppit.mogdfm import main  # noqa: E402


if __name__ == "__main__":
    main()