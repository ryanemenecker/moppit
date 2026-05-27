from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from torch.distributions.categorical import Categorical
from transformers import AutoModelForMaskedLM, AutoTokenizer


PEPMLM_MODEL_NAME = "ChatterjeeLab/PepMLM-650M"

_DEFAULT_TOKENIZER = None
_DEFAULT_MODEL = None


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


def compute_pseudo_perplexity(model, tokenizer, protein_seq, binder_seq):
    sequence = protein_seq + binder_seq
    tensor_input = tokenizer.encode(sequence, return_tensors="pt").to(model.device)
    total_loss = 0

    for token_index in range(-len(binder_seq) - 1, -1):
        masked_input = tensor_input.clone()
        masked_input[0, token_index] = tokenizer.mask_token_id

        labels = torch.full(tensor_input.shape, -100).to(model.device)
        labels[0, token_index] = tensor_input[0, token_index]

        with torch.no_grad():
            outputs = model(masked_input, labels=labels)
            total_loss += outputs.loss.item()

    average_loss = total_loss / len(binder_seq)
    pseudo_perplexity = np.exp(average_loss)
    return pseudo_perplexity


def generate_peptide_for_single_sequence(protein_seq, peptide_length=15, top_k=3, num_binders=4,
                                         model=None, tokenizer=None):
    peptide_length = int(peptide_length)
    top_k = int(top_k)
    num_binders = int(num_binders)

    if model is None or tokenizer is None:
        model, tokenizer = get_default_pepmlm()

    binders_with_ppl = []

    for binder_index in range(num_binders):
        masked_peptide = "<mask>" * peptide_length
        input_sequence = protein_seq + masked_peptide
        inputs = tokenizer(input_sequence, return_tensors="pt").to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits
        mask_token_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        logits_at_masks = logits[0, mask_token_indices]

        top_k_logits, top_k_indices = logits_at_masks.topk(top_k, dim=-1)
        probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
        predicted_indices = Categorical(probabilities).sample()
        predicted_token_ids = top_k_indices.gather(-1, predicted_indices.unsqueeze(-1)).squeeze(-1)

        generated_binder = tokenizer.decode(predicted_token_ids, skip_special_tokens=True).replace(" ", "")
        ppl_value = compute_pseudo_perplexity(model, tokenizer, protein_seq, generated_binder)
        binders_with_ppl.append([generated_binder, ppl_value])

    return binders_with_ppl


def generate_peptide(input_seqs, peptide_length=15, top_k=3, num_binders=4, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        model, tokenizer = get_default_pepmlm()

    if isinstance(input_seqs, str):
        binders = generate_peptide_for_single_sequence(input_seqs, peptide_length, top_k, num_binders,
                                                       model=model, tokenizer=tokenizer)
        return pd.DataFrame(binders, columns=["Binder", "Pseudo Perplexity"])

    if isinstance(input_seqs, list):
        results = []
        for sequence in input_seqs:
            binders = generate_peptide_for_single_sequence(sequence, peptide_length, top_k, num_binders,
                                                           model=model, tokenizer=tokenizer)
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
    parser.add_argument("--device", default="auto", help="Torch device to use, such as auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV file for generated binders.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    model, tokenizer = get_default_pepmlm(device)
    peptide_df = generate_peptide(args.sequence, args.peptide_length, args.top_k, args.num_binders,
                                  model=model, tokenizer=tokenizer)
    peptide_df = peptide_df.drop_duplicates(subset="Binder")
    peptide_df = peptide_df.sort_values(by="Pseudo Perplexity")
    if args.output is not None:
        peptide_df.to_csv(args.output, index=False)
    print(peptide_df)


if __name__ == "__main__":
    main()