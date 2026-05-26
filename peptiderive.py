from pathlib import Path
import sys


_SRC_ROOT = Path(__file__).resolve().parent / "src"
if _SRC_ROOT.exists() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from moppit.peptiderive import build_parser, main, main_cli, run_peptiderive  # noqa: E402

__all__ = ["build_parser", "main", "main_cli", "run_peptiderive"]


if __name__ == "__main__":
    main_cli()