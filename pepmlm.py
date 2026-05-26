from pathlib import Path
import sys


_SRC_ROOT = Path(__file__).resolve().parent / "src"
if _SRC_ROOT.exists() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from moppit.pepmlm import (  # noqa: E402
    PEPMLM_MODEL_NAME,
    compute_pseudo_perplexity,
    generate_peptide,
    generate_peptide_for_single_sequence,
    get_default_pepmlm,
    main,
)

__all__ = [
    "PEPMLM_MODEL_NAME",
    "compute_pseudo_perplexity",
    "generate_peptide",
    "generate_peptide_for_single_sequence",
    "get_default_pepmlm",
    "main",
]


if __name__ == "__main__":
    main()