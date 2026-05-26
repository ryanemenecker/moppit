from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parent
_SRC_ROOT = _REPO_ROOT / "src"
_PACKAGE_ROOT = _SRC_ROOT / "moppit"

if _SRC_ROOT.exists() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

if __name__ == "moppit" and _PACKAGE_ROOT.exists():
    __path__ = [str(_PACKAGE_ROOT)]

from moppit.bindevaluator import PeptideModel, calculate_score, parse_motif  # noqa: E402
from moppit.generation import Binder, cal_score, generate_random_seq, main, main_cli  # noqa: E402

__all__ = [
    "Binder",
    "PeptideModel",
    "cal_score",
    "calculate_score",
    "generate_random_seq",
    "main",
    "main_cli",
    "parse_motif",
]


if __name__ == "__main__":
    main_cli()