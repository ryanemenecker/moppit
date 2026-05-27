from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions.categorical import Categorical
from transformers import AutoModelForMaskedLM, AutoTokenizer


PEPMLM_MODEL_NAME = "ChatterjeeLab/PepMLM-650M"
PEPMLM_MAX_RESIDUES = 1024

_DEFAULT_TOKENIZER = None
_DEFAULT_MODEL = None


def validate_pepmlm_context(protein_seq, peptide_length):
    peptide_length = int(peptide_length)
    if peptide_length < 1:
        raise ValueError("peptide_length must be at least 1.")

    context_length = len(protein_seq) + peptide_length
    if context_length > PEPMLM_MAX_RESIDUES:
        raise ValueError(
            f"PepMLM sees the target and peptide as one sequence. The submitted target length "
            f"({len(protein_seq)}) plus peptide length ({peptide_length}) is {context_length}, "
            f"but {PEPMLM_MODEL_NAME} supports about {PEPMLM_MAX_RESIDUES} residues. Use a "
            "shorter target domain or local region containing the motif, then reindex motif residues "
            "to that submitted sequence."
        )


def get_default_pepmlm(device=None):
    global _DEFAULT_TOKENIZER, _DEFAULT_MODEL

    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if _DEFAULT_TOKENIZER is None:
        _DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained(PEPMLM_MODEL_NAME)
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = AutoModelForMaskedLM.from_pretrained(PEPMLM_MODEL_NAME).to(device)
    elif _DEFAULT_MODEL.device != device:
        _DEFAULT_MODEL = _DEFAULT_MODEL.to(device)

    return _DEFAULT_MODEL, _DEFAULT_TOKENIZER


def _flush_pseudo_perplexity_batch(model, tokenizer, batch_inputs, batch_positions, batch_targets,
                                   batch_binder_indices, loss_sums, token_counts):
    if not batch_inputs:
        return

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 1

    input_ids = pad_sequence(batch_inputs, batch_first=True, padding_value=pad_token_id).to(model.device)
    attention_mask = (input_ids != pad_token_id).long().to(model.device)
    positions = torch.tensor(batch_positions, dtype=torch.long, device=model.device)
    targets = torch.tensor(batch_targets, dtype=torch.long, device=model.device)

    with torch.inference_mode():
        logits = model(input_ids, attention_mask=attention_mask).logits
        row_indices = torch.arange(len(batch_positions), device=model.device)
        losses = F.cross_entropy(logits[row_indices, positions], targets, reduction="none")

    for binder_index, loss in zip(batch_binder_indices, losses.detach().cpu().tolist()):
        loss_sums[binder_index] += loss
        token_counts[binder_index] += 1


def compute_pseudo_perplexities(model, tokenizer, protein_seq, binder_seqs, batch_size=256, progress_callback=None):
    binder_seqs = list(binder_seqs)
    if not binder_seqs:
        return []
    for binder_seq in binder_seqs:
        validate_pepmlm_context(protein_seq, len(binder_seq))

    batch_size = max(int(batch_size or 1), 1)
    loss_sums = [0.0 for binder_seq in binder_seqs]
    token_counts = [0 for binder_seq in binder_seqs]
    total_positions = sum(len(binder_seq) for binder_seq in binder_seqs)
    processed_positions = 0

    batch_inputs = []
    batch_positions = []
    batch_targets = []
    batch_binder_indices = []

    def flush_batch():
        nonlocal processed_positions, batch_inputs, batch_positions, batch_targets, batch_binder_indices
        batch_length = len(batch_inputs)
        _flush_pseudo_perplexity_batch(model, tokenizer, batch_inputs, batch_positions, batch_targets,
                                       batch_binder_indices, loss_sums, token_counts)
        processed_positions += batch_length
        if progress_callback is not None and batch_length:
            progress_callback(processed_positions, total_positions)
        batch_inputs = []
        batch_positions = []
        batch_targets = []
        batch_binder_indices = []

    for binder_index, binder_seq in enumerate(binder_seqs):
        sequence = protein_seq + binder_seq
        tensor_input = tokenizer.encode(sequence, return_tensors="pt").squeeze(0).to(model.device)
        start_position = tensor_input.shape[0] - len(binder_seq) - 1
        end_position = tensor_input.shape[0] - 1

        for token_position in range(start_position, end_position):
            masked_input = tensor_input.clone()
            masked_input[token_position] = tokenizer.mask_token_id
            batch_inputs.append(masked_input.detach().cpu())
            batch_positions.append(token_position)
            batch_targets.append(int(tensor_input[token_position].detach().cpu().item()))
            batch_binder_indices.append(binder_index)

            if len(batch_inputs) >= batch_size:
                flush_batch()

    flush_batch()

    return [float(np.exp(loss_sum / max(token_count, 1)))
            for loss_sum, token_count in zip(loss_sums, token_counts)]


def compute_pseudo_perplexity(model, tokenizer, protein_seq, binder_seq, batch_size=256):
    return compute_pseudo_perplexities(model, tokenizer, protein_seq, [binder_seq], batch_size=batch_size)[0]


def generate_peptide_for_single_sequence(protein_seq, peptide_length=15, top_k=3, num_binders=4,
                                         model=None, tokenizer=None, compute_ppl=True, ppl_batch_size=256,
                                         progress_callback=None):
    peptide_length = int(peptide_length)
    top_k = int(top_k)
    num_binders = int(num_binders)
    validate_pepmlm_context(protein_seq, peptide_length)

    if model is None or tokenizer is None:
        model, tokenizer = get_default_pepmlm()

    masked_peptide = "<mask>" * peptide_length
    input_sequence = protein_seq + masked_peptide
    inputs = tokenizer(input_sequence, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        logits = model(**inputs).logits
    mask_token_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    logits_at_masks = logits[0, mask_token_indices]

    top_k_logits, top_k_indices = logits_at_masks.topk(top_k, dim=-1)
    probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
    predicted_indices = Categorical(probabilities).sample((num_binders,))
    expanded_top_k_indices = top_k_indices.unsqueeze(0).expand(num_binders, -1, -1)
    predicted_token_ids = expanded_top_k_indices.gather(-1, predicted_indices.unsqueeze(-1)).squeeze(-1)

    generated_binders = [tokenizer.decode(token_ids, skip_special_tokens=True).replace(" ", "")
                         for token_ids in predicted_token_ids]
    if compute_ppl:
        ppl_values = compute_pseudo_perplexities(model, tokenizer, protein_seq, generated_binders,
                                                batch_size=ppl_batch_size,
                                                progress_callback=progress_callback)
    else:
        ppl_values = [np.nan for generated_binder in generated_binders]

    binders_with_ppl = [[generated_binder, ppl_value]
                        for generated_binder, ppl_value in zip(generated_binders, ppl_values)]

    return binders_with_ppl


def generate_peptide(input_seqs, peptide_length=15, top_k=3, num_binders=4, model=None, tokenizer=None,
                     compute_ppl=True, ppl_batch_size=256, progress_callback=None):
    if model is None or tokenizer is None:
        model, tokenizer = get_default_pepmlm()

    if isinstance(input_seqs, str):
        binders = generate_peptide_for_single_sequence(input_seqs, peptide_length, top_k, num_binders,
                                                       model=model, tokenizer=tokenizer, compute_ppl=compute_ppl,
                                                       ppl_batch_size=ppl_batch_size,
                                                       progress_callback=progress_callback)
        return pd.DataFrame(binders, columns=["Binder", "Pseudo Perplexity"])

    if isinstance(input_seqs, list):
        results = []
        for sequence in input_seqs:
            binders = generate_peptide_for_single_sequence(sequence, peptide_length, top_k, num_binders,
                                                           model=model, tokenizer=tokenizer, compute_ppl=compute_ppl,
                                                           ppl_batch_size=ppl_batch_size,
                                                           progress_callback=progress_callback)
            for binder, ppl in binders:
                results.append([sequence, binder, ppl])
        return pd.DataFrame(results, columns=["Input Sequence", "Binder", "Pseudo Perplexity"])

    raise TypeError("input_seqs must be a protein sequence string or a list of sequence strings")


def resolve_device(device="auto"):
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("-s", "--sequence", dest="sequence", type=str, required=True)
    parser.add_argument("--peptide_length", "--peptide-length", dest="peptide_length", type=int, default=13)
    parser.add_argument("--top_k", "--top-k", dest="top_k", type=int, default=2)
    parser.add_argument("--num_binders", "--num-binders", dest="num_binders", type=int, default=50)
    parser.add_argument("--ppl-batch-size", type=int, default=256,
                        help="Masked-position batch size for pseudo-perplexity scoring. Increase on large GPUs.")
    parser.add_argument("--device", default="auto", help="Torch device to use, such as auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV file for generated binders.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    validate_pepmlm_context(args.sequence, args.peptide_length)
    model, tokenizer = get_default_pepmlm(device)
    peptide_df = generate_peptide(args.sequence, args.peptide_length, args.top_k, args.num_binders,
                                  model=model, tokenizer=tokenizer, ppl_batch_size=args.ppl_batch_size)
    peptide_df = peptide_df.drop_duplicates(subset="Binder")
    peptide_df = peptide_df.sort_values(by="Pseudo Perplexity")
    if args.output is not None:
        peptide_df.to_csv(args.output, index=False)
    print(peptide_df)


if __name__ == "__main__":
    main()