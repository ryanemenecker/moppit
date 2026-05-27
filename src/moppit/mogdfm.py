from argparse import ArgumentParser
import os
from pathlib import Path
import runpy
import sys

from .bindevaluator import is_git_lfs_pointer


HF_ROOT_ENV_VAR = "MOPPIT_HF_ROOT"
DEFAULT_HF_ROOT = Path("moPPIt")
DEFAULT_SOLVER_CKPT = Path("ckpt/peptide/cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt")
DEFAULT_BINDEVALUATOR_CKPT = Path("classifier_ckpt/finetuned_BindEvaluator.ckpt")
KNOWN_CLASSIFIER_ASSETS = (
    Path("classifier_ckpt/best_model_half_life.pth"),
    Path("classifier_ckpt/best_model_nonfouling.json"),
    Path("classifier_ckpt/best_model_solubility.json"),
    Path("classifier_ckpt/binding_affinity_pooled.pt"),
    Path("classifier_ckpt/binding_affinity_unpooled.pt"),
    Path("classifier_ckpt/wt_affinity.pt"),
    Path("classifier_ckpt/wt_halflife.pt"),
    Path("classifier_ckpt/wt_hemolysis.json"),
    Path("classifier_ckpt/wt_nonfouling.pt"),
)


def build_parser():
    parser = ArgumentParser(
        description=(
            "Run the Hugging Face moPPIt Multi-Objective-Guided Discrete Flow Matching "
            "workflow from a local HF clone. Unrecognized arguments are passed through "
            "to the Hugging Face script, including --cyclic, --offtarget, --fixed_positions, "
            "and --starting_sequence."
        )
    )
    parser.add_argument("--hf-root", default=None,
                        help="Path to the Hugging Face moPPIt clone. Defaults to MOPPIT_HF_ROOT or ./moPPIt.")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip local asset and dependency checks before launching the HF script.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate paths and print the normalized HF command without running generation.")
    return parser


def resolve_hf_root(hf_root=None):
    root = hf_root or os.environ.get(HF_ROOT_ENV_VAR) or DEFAULT_HF_ROOT
    return Path(root).expanduser().resolve()


def normalize_hf_args(argv):
    args = list(argv)
    if args and args[0] == "--":
        args = args[1:]

    add_specificity = False
    normalized = []
    for arg in args:
        if arg in {"--motif_penalty", "--motif-penalty"}:
            add_specificity = True
        else:
            normalized.append(arg)

    if add_specificity:
        normalized = _ensure_specificity_objective(normalized)

    return normalized


def _ensure_specificity_objective(args):
    if "--objectives" not in args:
        return args + ["--objectives", "Motif", "Specificity"]

    objective_index = args.index("--objectives") + 1
    next_option_index = objective_index
    while next_option_index < len(args) and not args[next_option_index].startswith("--"):
        next_option_index += 1

    objectives = args[objective_index:next_option_index]
    if "Specificity" in objectives:
        return args

    return args[:next_option_index] + ["Specificity"] + args[next_option_index:]


def parse_launcher_args(argv=None):
    parser = build_parser()
    args, hf_args = parser.parse_known_args(argv)
    return parser, args, normalize_hf_args(hf_args)


def validate_hf_environment(hf_root):
    errors = []
    script_path = hf_root / "moppit.py"
    if not script_path.exists():
        errors.append(f"Missing Hugging Face generator script: {script_path}")

    required_assets = (DEFAULT_SOLVER_CKPT, DEFAULT_BINDEVALUATOR_CKPT)
    for relative_asset_path in required_assets:
        asset_path = hf_root / relative_asset_path
        if not asset_path.exists():
            errors.append(f"Missing required MOG-DFM asset: {asset_path}")
        elif is_git_lfs_pointer(asset_path):
            errors.append(
                f"Required MOG-DFM asset is a Git LFS pointer, not real weights: {asset_path}. "
                "Run `git lfs pull` in the Hugging Face clone."
            )

    for relative_asset_path in KNOWN_CLASSIFIER_ASSETS:
        asset_path = hf_root / relative_asset_path
        if asset_path.exists() and is_git_lfs_pointer(asset_path):
            errors.append(
                f"Bundled Hugging Face classifier asset is a Git LFS pointer, not real weights: {asset_path}. "
                "Run `git lfs pull` in the Hugging Face clone."
            )

    peptiverse_root = hf_root / "PeptiVerse"
    if not peptiverse_root.exists():
        errors.append(
            f"Missing PeptiVerse checkout: {peptiverse_root}. The HF MOG-DFM script imports "
            "./PeptiVerse/inference.py and ./PeptiVerse/best_models.txt at startup."
        )
    else:
        for relative_path in (Path("inference.py"), Path("best_models.txt")):
            if not (peptiverse_root / relative_path).exists():
                errors.append(f"Missing PeptiVerse file: {peptiverse_root / relative_path}")

    if errors:
        raise RuntimeError("\n".join(errors))


def run_hf_script(hf_root, hf_args):
    script_path = hf_root / "moppit.py"
    original_cwd = Path.cwd()
    original_argv = sys.argv[:]
    original_path = sys.path[:]

    try:
        os.chdir(hf_root)
        sys.path.insert(0, str(hf_root))
        sys.argv = [str(script_path)] + hf_args
        runpy.run_path(str(script_path), run_name="__main__")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"Missing optional MOG-DFM dependency: {error.name}. Install optional dependencies with "
            "`python -m pip install -e '.[mogdfm]'` and ensure PeptiVerse is available under the HF clone."
        ) from error
    finally:
        os.chdir(original_cwd)
        sys.argv = original_argv
        sys.path[:] = original_path


def main(argv=None):
    parser, args, hf_args = parse_launcher_args(argv)
    hf_root = resolve_hf_root(args.hf_root)

    if not args.skip_validation:
        try:
            validate_hf_environment(hf_root)
        except RuntimeError as error:
            parser.error(str(error))

    if args.dry_run:
        print("HF root:", hf_root)
        print("HF command:", "python -u moppit.py", " ".join(hf_args))
        return None

    try:
        run_hf_script(hf_root, hf_args)
    except RuntimeError as error:
        parser.error(str(error))

    return None


if __name__ == "__main__":
    main()