from argparse import ArgumentParser, SUPPRESS
import json
import os
from pathlib import Path
import warnings

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import AutoTokenizer, EsmModel

from .models import FFN, MultiHeadAttentionSequence, RepeatedModule3


ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DEFAULT_MAX_LENGTH = 40000
DEFAULT_MODEL_ENV_VAR = "MOPPIT_BINDEVALUATOR_CKPT"
DEFAULT_MODEL_DIR_ENV_VAR = "MOPPIT_MODEL_DIR"
MODEL_DIR_ENV_VARS = (DEFAULT_MODEL_DIR_ENV_VAR, "MOPPIT_MODEL_WEIGHTS_DIR")
HF_ROOT_ENV_VAR = "MOPPIT_HF_ROOT"
FROZEN_ESM_COMPATIBILITY_KEYS = (
    "esm_model.embeddings.position_embeddings.weight",
)
DEFAULT_MODEL_FILENAMES = (
    "finetuned_BindEvaluator.ckpt",
    "pretrained_BindEvaluator.ckpt",
)
PUBLISHED_BINDEVALUATOR_CONFIG = {
    "n_layers": 8,
    "d_model": 128,
    "d_hidden": 128,
    "n_head": 8,
    "d_k": 64,
    "d_v": 128,
    "d_inner": 64,
}
LEGACY_BINDEVALUATOR_CONFIG = {
    "n_layers": 6,
    "d_model": 64,
    "d_hidden": 128,
    "n_head": 6,
    "d_k": 64,
    "d_v": 128,
    "d_inner": 64,
}
CHECKPOINT_PRESETS = {
    "published": PUBLISHED_BINDEVALUATOR_CONFIG,
    "legacy": LEGACY_BINDEVALUATOR_CONFIG,
}
DEFAULT_MODEL_CANDIDATES = (
    Path("~/model_weights/moppit/finetuned_BindEvaluator.ckpt"),
    Path("~/model_weights/moppit/pretrained_BindEvaluator.ckpt"),
    Path("model_path/finetuned_BindEvaluator.ckpt"),
    Path("model_path/pretrained_BindEvaluator.ckpt"),
    Path("classifier_ckpt/finetuned_BindEvaluator.ckpt"),
    Path("moPPIt/classifier_ckpt/finetuned_BindEvaluator.ckpt"),
)
DEFAULT_MODEL_DIR_CANDIDATES = (
    Path("~/model_weights/moppit"),
    Path("model_path"),
    Path("classifier_ckpt"),
    Path("moPPIt/classifier_ckpt"),
)


def drop_frozen_esm_compatibility_keys(state_dict, model_state_keys):
    removed_keys = []
    for key in FROZEN_ESM_COMPATIBILITY_KEYS:
        if key in state_dict and key not in model_state_keys:
            state_dict.pop(key)
            removed_keys.append(key)
    return removed_keys


def parse_motif(motif: str, index_base: int = 0) -> list[int]:
    motif = motif.strip().strip("[]")
    if not motif:
        return []

    parts = motif.split(",")
    residues = []

    for part in parts:
        part = part.strip()
        if "-" in part:
            start, end = map(int, part.split("-"))
            residues.extend(range(start, end + 1))
        elif part:
            residues.append(int(part))

    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1")
    if index_base == 1:
        residues = [residue - 1 for residue in residues]

    return residues


class PeptideModel(pl.LightningModule):
    def __init__(self, n_layers, d_model, d_hidden, n_head,
                 d_k, d_v, d_inner, dropout=0.2,
                 learning_rate=0.00001, max_epochs=15, kl_weight=1):
        super(PeptideModel, self).__init__()

        self.esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME)
        for param in self.esm_model.parameters():
            param.requires_grad = False

        self.repeated_module = RepeatedModule3(n_layers, d_model, d_hidden,
                                               n_head, d_k, d_v, d_inner, dropout=dropout)

        self.final_attention_layer = MultiHeadAttentionSequence(n_head, d_model,
                                                                d_k, d_v, dropout=dropout)

        self.final_ffn = FFN(d_model, d_inner, dropout=dropout)

        self.output_projection_prot = nn.Linear(d_model, 1)

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.kl_weight = kl_weight

        self.classification_threshold = nn.Parameter(torch.tensor(0.5))
        self.historical_memory = 0.9
        self.class_weights = torch.tensor([3.000471363174231, 0.5999811490272925])

    def forward(self, binder_tokens, target_tokens):
        peptide_sequence = self.esm_model(**binder_tokens).last_hidden_state
        protein_sequence = self.esm_model(**target_tokens).last_hidden_state

        prot_enc, sequence_enc, sequence_attention_list, prot_attention_list, \
            seq_prot_attention_list, seq_prot_attention_list = self.repeated_module(peptide_sequence,
                                                                                    protein_sequence)

        prot_enc, final_prot_seq_attention = self.final_attention_layer(prot_enc, sequence_enc, sequence_enc)

        prot_enc = self.final_ffn(prot_enc)

        prot_enc = self.output_projection_prot(prot_enc)

        return prot_enc

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint.get("state_dict")
        if not state_dict:
            return

        removed_keys = drop_frozen_esm_compatibility_keys(state_dict, set(self.state_dict()))
        if removed_keys:
            warnings.warn(
                "Ignoring frozen ESM checkpoint key(s) that are not present in the installed "
                f"transformers EsmModel: {', '.join(removed_keys)}. This preserves BindEvaluator "
                "checkpoint loading across transformers versions without changing the moPPIt model head.",
                RuntimeWarning,
                stacklevel=2,
            )

    def get_probs(self, binder_input_ids, target_input_ids):
        target_input_ids = target_input_ids.repeat(binder_input_ids.shape[0], 1)
        binder_attention_mask = torch.ones_like(binder_input_ids)
        target_attention_mask = torch.ones_like(target_input_ids)

        binder_attention_mask[:, 0] = binder_attention_mask[:, -1] = 0
        target_attention_mask[:, 0] = target_attention_mask[:, -1] = 0

        binder_tokens = {
            "input_ids": binder_input_ids,
            "attention_mask": binder_attention_mask.to(binder_input_ids.device),
        }
        target_tokens = {
            "input_ids": target_input_ids,
            "attention_mask": target_attention_mask.to(target_input_ids.device),
        }

        logits = self.forward(binder_tokens, target_tokens).squeeze(-1)
        logits[:, 0] = logits[:, -1] = -100
        return torch.sigmoid(logits)

    def motif_score(self, binder_input_ids, target_input_ids, motifs):
        probs = self.get_probs(binder_input_ids, target_input_ids)
        motif_probs = probs[:, motifs]
        return motif_probs.sum(dim=-1) / len(motifs)

    def non_motif_score(self, binder_input_ids, target_input_ids, motifs, threshold=0.5):
        probs = self.get_probs(binder_input_ids, target_input_ids)
        non_motif_positions = [index for index in range(probs.shape[1]) if index not in set(motifs)]
        non_motif_probs = probs[:, non_motif_positions]
        mask = non_motif_probs >= threshold
        count = mask.sum(dim=-1)
        return torch.where(count > 0, (non_motif_probs * mask).sum(dim=-1) / count, torch.zeros_like(count))

    def scoring(self, binder_input_ids, target_input_ids, motifs, penalty=False, threshold=0.5):
        probs = self.get_probs(binder_input_ids, target_input_ids)
        motif_probs = probs[:, motifs]
        motif_score = motif_probs.sum(dim=-1) / len(motifs)

        if penalty:
            non_motif_positions = [index for index in range(probs.shape[1]) if index not in set(motifs)]
            non_motif_probs = probs[:, non_motif_positions]
            mask = non_motif_probs >= threshold
            count = mask.sum(dim=-1)
            specificity_score = 1 - count / target_input_ids.shape[1]
            return motif_score, specificity_score

        return motif_score


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _tokenize_sequence(tokenizer, sequence: str, device: torch.device, max_length: int) -> dict[str, torch.Tensor]:
    tokens = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    if tokens["attention_mask"].shape[-1] > 1:
        tokens["attention_mask"][0][0] = 0
        tokens["attention_mask"][0][-1] = 0
    return {
        "input_ids": tokens["input_ids"].to(device),
        "attention_mask": tokens["attention_mask"].to(device),
    }


def calculate_score(target_sequence, binder_sequence, model, args=None, tokenizer=None, max_length=DEFAULT_MAX_LENGTH):
    device = _model_device(model)
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(ESM_MODEL_NAME)

    target_tokens = _tokenize_sequence(tokenizer, target_sequence, device, max_length)
    binder_tokens = _tokenize_sequence(tokenizer, binder_sequence, device, max_length)

    model.eval()
    with torch.inference_mode():
        prediction = model(binder_tokens, target_tokens).squeeze(-1)[0][1:-1]
        prediction = torch.sigmoid(prediction)

    return prediction, model.classification_threshold


def compute_metrics(true_residues, predicted_residues, length):
    true_list = [0] * length
    predicted_list = [0] * length

    for index in true_residues:
        true_list[index] = 1
    for index in predicted_residues:
        predicted_list[index] = 1

    accuracy = accuracy_score(true_list, predicted_list)
    f1 = f1_score(true_list, predicted_list)
    mcc = matthews_corrcoef(true_list, predicted_list)

    return accuracy, f1, mcc


def resolve_device(device="auto"):
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _expand_path(path):
    return Path(path).expanduser()


def _iter_configured_model_directories(model_dir=None):
    if model_dir:
        yield "--model-dir", _expand_path(model_dir)

    for env_var in MODEL_DIR_ENV_VARS:
        env_model_dir = os.environ.get(env_var)
        if env_model_dir:
            yield env_var, _expand_path(env_model_dir)


def _iter_default_model_directories():
    hf_root = os.environ.get(HF_ROOT_ENV_VAR)
    if hf_root:
        yield f"{HF_ROOT_ENV_VAR}/classifier_ckpt", _expand_path(hf_root) / "classifier_ckpt"


    for candidate in DEFAULT_MODEL_DIR_CANDIDATES:
        yield "default", _expand_path(candidate)


def _dedupe_directories(directories):
    seen = set()
    for source, directory in directories:
        directory_key = str(directory)
        if directory_key in seen:
            continue
        seen.add(directory_key)
        yield source, directory


def _find_checkpoint_in_directories(directories):
    searched = []
    ambiguous_directories = []
    for source, directory in _dedupe_directories(directories):
        known_candidates = [directory / filename for filename in DEFAULT_MODEL_FILENAMES]
        for candidate in known_candidates:
            searched.append(candidate)
            if candidate.exists():
                return str(candidate), searched, ambiguous_directories

        if not directory.exists() or not directory.is_dir():
            continue

        fallback_candidates = sorted(
            candidate for candidate in directory.glob("*.ckpt")
            if candidate.name not in DEFAULT_MODEL_FILENAMES
        )
        if len(fallback_candidates) == 1:
            return str(fallback_candidates[0]), searched, ambiguous_directories
        if len(fallback_candidates) > 1:
            ambiguous_directories.append((source, directory, fallback_candidates))

    return None, searched, ambiguous_directories


def _raise_checkpoint_discovery_error(searched, ambiguous_directories):
    if ambiguous_directories:
        details = "; ".join(
            f"{directory} ({', '.join(candidate.name for candidate in candidates)})"
            for source, directory, candidates in ambiguous_directories
        )
        raise FileNotFoundError(
            "Found multiple BindEvaluator checkpoint candidates. Pass --model/-sm explicitly, "
            "or rename the desired checkpoint to finetuned_BindEvaluator.ckpt or "
            f"pretrained_BindEvaluator.ckpt. Ambiguous directories: {details}"
        )

    candidates = ", ".join(str(candidate) for candidate in searched)
    model_dir_env_vars = "/".join(MODEL_DIR_ENV_VARS)
    raise FileNotFoundError(
        "No BindEvaluator checkpoint was provided or discovered. Use --model/-sm, set "
        f"{DEFAULT_MODEL_ENV_VAR} to an exact checkpoint path, set {model_dir_env_vars} "
        "or --model-dir to a directory such as ~/model_weights/moppit, or place a checkpoint at one of: "
        f"{candidates}"
    )


def resolve_checkpoint_path(model_path=None, model_dir=None):
    if model_path:
        return str(_expand_path(model_path))

    env_model_path = os.environ.get(DEFAULT_MODEL_ENV_VAR)
    if env_model_path:
        return str(_expand_path(env_model_path))

    configured_directories = list(_dedupe_directories(_iter_configured_model_directories(model_dir)))
    if configured_directories:
        checkpoint_path, searched, ambiguous_directories = _find_checkpoint_in_directories(configured_directories)
        if checkpoint_path:
            return checkpoint_path
        _raise_checkpoint_discovery_error(searched, ambiguous_directories)

    checkpoint_path, searched, ambiguous_directories = _find_checkpoint_in_directories(_iter_default_model_directories())
    if checkpoint_path:
        return checkpoint_path
    _raise_checkpoint_discovery_error(searched, ambiguous_directories)


def is_git_lfs_pointer(path):
    path = Path(path)
    try:
        header = path.read_bytes()[:128].decode("utf-8", errors="ignore")
    except OSError:
        return False
    return header.startswith("version https://git-lfs.github.com/spec/v1")


def validate_checkpoint_path(checkpoint_path):
    checkpoint_path = _expand_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"BindEvaluator checkpoint does not exist: {checkpoint_path}")
    if is_git_lfs_pointer(checkpoint_path):
        raise ValueError(
            f"BindEvaluator checkpoint is a Git LFS pointer, not the actual weights: {checkpoint_path}. "
            "Run `git lfs pull` in the Hugging Face clone or provide a real checkpoint with --model."
        )
    return str(checkpoint_path)


def resolve_model_config(args):
    preset_name = getattr(args, "checkpoint_preset", "published")
    config = dict(CHECKPOINT_PRESETS[preset_name])

    for key in ("n_layers", "d_model", "d_hidden", "n_head", "d_inner"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    return config


def load_bindevaluator_checkpoint(args, device=None):
    device = device or resolve_device(getattr(args, "device", "auto"))
    checkpoint_path = validate_checkpoint_path(
        resolve_checkpoint_path(getattr(args, "sm", None), getattr(args, "model_dir", None))
    )
    config = resolve_model_config(args)
    model = PeptideModel.load_from_checkpoint(
        checkpoint_path,
        weights_only=False,
        n_layers=config["n_layers"],
        d_model=config["d_model"],
        d_hidden=config["d_hidden"],
        n_head=config["n_head"],
        d_k=config["d_k"],
        d_v=config["d_v"],
        d_inner=config["d_inner"],
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def predict_binding_sites(target_sequence, binder_sequence, model, threshold=0.5):
    prediction, _ = calculate_score(target_sequence, binder_sequence, model)
    binding_site = []
    for index in range(len(prediction)):
        if prediction[index] >= threshold:
            binding_site.append(index)
    return binding_site, prediction


def write_prediction_output(output_path, target_sequence, binder_sequence, binding_site, prediction, threshold,
                            motif_summary=None):
    payload = {
        "target_length": len(target_sequence),
        "binder_length": len(binder_sequence),
        "threshold": threshold,
        "binding_site": binding_site,
        "scores": [float(score) for score in prediction.detach().cpu().tolist()],
    }
    if motif_summary is not None:
        payload["motif_summary"] = motif_summary
    Path(output_path).write_text(json.dumps(payload, indent=2) + "\n")


def summarize_motif_scores(prediction, motifs, threshold=0.5):
    motifs = list(motifs)
    if not motifs:
        raise ValueError("At least one motif residue is required")

    prediction = prediction.detach().cpu()
    length = prediction.shape[0]
    out_of_range = [motif for motif in motifs if motif < 0 or motif >= length]
    if out_of_range:
        raise ValueError(f"Motif indices out of range for target length {length}: {out_of_range}")

    motif_tensor = torch.tensor(motifs, dtype=torch.long)
    motif_score = prediction[motif_tensor].mean().item()
    non_motif_indices = [index for index in range(length) if index not in set(motifs)]

    if non_motif_indices:
        non_motif_scores = prediction[torch.tensor(non_motif_indices, dtype=torch.long)]
        non_motif_hits = int((non_motif_scores >= threshold).sum().item())
        if non_motif_hits:
            off_motif_score = non_motif_scores[non_motif_scores >= threshold].mean().item()
        else:
            off_motif_score = 0.0
    else:
        non_motif_hits = 0
        off_motif_score = 0.0

    return {
        "motifs": motifs,
        "motif_score": motif_score,
        "off_motif_score": off_motif_score,
        "off_motif_hits": non_motif_hits,
        "specificity_score": 1 - (non_motif_hits / length),
    }


def add_checkpoint_arguments(parser):
    parser.add_argument("-sm", "--model", dest="sm", default=None,
                        help="BindEvaluator checkpoint path. Overrides checkpoint directory discovery.")
    parser.add_argument("--model-dir", default=None,
                        help="Directory containing BindEvaluator checkpoints. Defaults to MOPPIT_MODEL_DIR, "
                            "MOPPIT_MODEL_WEIGHTS_DIR, MOPPIT_HF_ROOT/classifier_ckpt, "
                            "~/model_weights/moppit, and repo-local fallback paths.")
    parser.add_argument("--checkpoint-preset", choices=sorted(CHECKPOINT_PRESETS), default="published",
                        help="Architecture preset for loading checkpoints. Use 'published' for ChatterjeeLab/moPPIt weights.")
    parser.add_argument("-n_layers", "--n-layers", dest="n_layers", type=int, default=None,
                        help="Override checkpoint preset layer count.")
    parser.add_argument("-d_model", "--d-model", dest="d_model", type=int, default=None,
                        help="Override checkpoint preset model dimension.")
    parser.add_argument("-d_hidden", "--d-hidden", dest="d_hidden", type=int, default=None,
                        help="Override checkpoint preset CNN hidden dimension.")
    parser.add_argument("-n_head", "--n-head", dest="n_head", type=int, default=None,
                        help="Override checkpoint preset attention head count.")
    parser.add_argument("-d_inner", "--d-inner", dest="d_inner", type=int, default=None,
                        help="Override checkpoint preset feed-forward dimension.")
    parser.add_argument("--device", default="auto",
                        help="Torch device to use, such as auto, cpu, cuda, or cuda:0.")
    return parser


def build_parser():
    parser = ArgumentParser()
    add_checkpoint_arguments(parser)
    parser.add_argument("-target", "--target", required=True, type=str)
    parser.add_argument("-binder", "--binder", required=True, type=str)
    parser.add_argument("-gt", "--ground-truth", dest="gt", type=str, default=None,
                        help="Ground truth binding motifs, for example '1,4-6'.")
    parser.add_argument("-motifs", "--motifs", "--motif", dest="motifs", type=str, default=None,
                        help="Motif residues to score, for example '18,23,59-61'.")
    parser.add_argument("--motif-index-base", type=int, choices=[0, 1], default=0,
                        help="Index base for --motifs and --ground-truth. Defaults to 0 for original moPPIt behavior.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binding-site probability threshold.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON file for scores and predicted residues.")
    parser.add_argument("--print-scores", action="store_true", help="Print per-residue prediction probabilities.")
    parser.add_argument("-batch_size", type=int, default=None, help=SUPPRESS)
    parser.add_argument("-lr", type=float, default=None, help=SUPPRESS)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        model = load_bindevaluator_checkpoint(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    binding_site, prediction = predict_binding_sites(args.target, args.binder, model, threshold=args.threshold)

    print("Prediction: ", binding_site)

    if args.print_scores:
        scores = [round(float(score), 4) for score in prediction.detach().cpu().tolist()]
        print("Scores: ", scores)

    motif_summary = None
    if args.motifs is not None:
        try:
            motifs = parse_motif(args.motifs, index_base=args.motif_index_base)
            motif_summary = summarize_motif_scores(prediction, motifs, threshold=args.threshold)
        except ValueError as error:
            parser.error(str(error))
        print(f"Motif Score: {motif_summary['motif_score']:.4f}")
        print(f"Specificity Score: {motif_summary['specificity_score']:.4f}")
        print(f"Off-Motif Hits: {motif_summary['off_motif_hits']}")

    if args.output is not None:
        write_prediction_output(args.output, args.target, args.binder, binding_site, prediction, args.threshold,
                                motif_summary=motif_summary)

    if args.gt is not None:
        length = len(args.target)
        ground_truth = parse_motif(args.gt, index_base=args.motif_index_base)
        print("Ground Truth: ", ground_truth)

        accuracy, f1, mcc = compute_metrics(ground_truth, binding_site, length)
        print(f"Accuracy={accuracy}\tF1={f1}\tMCC={mcc}")


if __name__ == "__main__":
    main()