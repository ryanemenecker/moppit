from pathlib import Path
import sys


_SRC_ROOT = Path(__file__).resolve().parent / "src"
if _SRC_ROOT.exists() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from moppit.predict_motifs import (  # noqa: E402
    CHECKPOINT_PRESETS,
    PeptideModel,
    build_parser,
    calculate_score,
    calculate_scores,
    compute_metrics,
    load_bindevaluator_checkpoint,
    main,
    parse_motif,
    predict_binding_sites,
    resolve_checkpoint_path,
    resolve_model_config,
    summarize_motif_scores,
    write_prediction_output,
)

__all__ = [
    "CHECKPOINT_PRESETS",
    "PeptideModel",
    "build_parser",
    "calculate_score",
    "calculate_scores",
    "compute_metrics",
    "load_bindevaluator_checkpoint",
    "main",
    "parse_motif",
    "predict_binding_sites",
    "resolve_checkpoint_path",
    "resolve_model_config",
    "summarize_motif_scores",
    "write_prediction_output",
]


if __name__ == "__main__":
    main()