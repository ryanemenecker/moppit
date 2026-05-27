from argparse import ArgumentParser, SUPPRESS
import csv
from pathlib import Path
import random

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .bindevaluator import (
    ESM_MODEL_NAME,
    add_checkpoint_arguments,
    calculate_score,
    load_bindevaluator_checkpoint,
    parse_motif,
    resolve_device,
)
from .pepmlm import PEPMLM_MODEL_NAME, compute_pseudo_perplexity, generate_peptide


AA = "ARNDCEQGHILKMFPSTWYV"


def generate_random_seq(length):
    return "".join(random.choice(AA) for residue_index in range(length))


def cal_score(binder_seq, protein_seq, motif, model, args, tokenizer=None):
    prediction, threshold = calculate_score(protein_seq, binder_seq, model, args, tokenizer=tokenizer)
    score = 0
    for position in motif:
        if prediction[position] < args.threshold:
            score += 1

    for position in range(len(prediction)):
        if position not in motif and prediction[position] >= args.threshold:
            score += 0.5

    return score


class Binder(object):
    def __init__(self, binder_seq, model, pepmlm, pepmlm_tokenizer, bindevaluator_tokenizer, args):
        self.binder_seq = binder_seq
        self.protein_seq = args.protein_seq
        self.motif = parse_motif(args.motif.strip("[]"))
        self.model = model
        self.args = args
        self.pepmlm = pepmlm
        self.pepmlm_tokenizer = pepmlm_tokenizer
        self.bindevaluator_tokenizer = bindevaluator_tokenizer
        self.score = cal_score(binder_seq, self.protein_seq, self.motif, self.model, self.args,
                               tokenizer=self.bindevaluator_tokenizer)
        self.ppl = compute_pseudo_perplexity(self.pepmlm, self.pepmlm_tokenizer, self.protein_seq, self.binder_seq)

    def mutated_aa(self):
        return random.choice(AA)

    def mutate_seq(self, binder_seq):
        position = random.randint(0, len(binder_seq) - 1)
        mutated_seq = binder_seq[:position] + self.mutated_aa() + binder_seq[position + 1:]
        return mutated_seq

    def mate(self, par2):
        assert len(par2.binder_seq) == len(self.binder_seq)

        child = ""
        for position in range(len(par2.binder_seq)):
            parent_one_aa = self.binder_seq[position]
            parent_two_aa = par2.binder_seq[position]

            probability = random.random()
            if probability < 0.45:
                child += parent_one_aa
            elif probability < 0.90:
                child += parent_two_aa
            else:
                child += self.mutated_aa()

        return Binder(child, self.model, self.pepmlm, self.pepmlm_tokenizer, self.bindevaluator_tokenizer, self.args)


def main(args):
    print(parse_motif(args.motif))
    random.seed(args.seed)
    binders = []
    generation = 0

    device = resolve_device(args.device)
    print(f"Device: {device}")
    model = load_bindevaluator_checkpoint(args, device=device)

    tokenizer = AutoTokenizer.from_pretrained(PEPMLM_MODEL_NAME)
    pepmlm = AutoModelForMaskedLM.from_pretrained(PEPMLM_MODEL_NAME).to(device)
    bindevaluator_tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)

    temp_binders = []
    count = 2
    while len(temp_binders) < args.num_binders and count >= 0:
        needed_binders = args.num_binders - len(temp_binders)
        temp_binders += generate_peptide(args.protein_seq, args.peptide_length, args.top_k, needed_binders,
                                         model=pepmlm, tokenizer=tokenizer)["Binder"].tolist()
        temp_binders = [sequence for sequence in temp_binders if "X" not in sequence]
        temp_binders = list(set(temp_binders))
        count -= 1

    for binder_index in range(0, args.num_binders - len(temp_binders)):
        temp_binders.append(generate_random_seq(args.peptide_length))
    temp_binders = list(set(temp_binders))

    print(f"Pool Size = {len(temp_binders)}")

    for binder_seq in temp_binders:
        binders.append(Binder(binder_seq, model, pepmlm, tokenizer, bindevaluator_tokenizer, args))

    binders = sorted(binders, key=lambda binder: (binder.score, binder.ppl))
    for display_index in range(min(args.num_display, len(binders))):
        print(f"Generation: -1\tBinder: {binders[display_index].binder_seq}\tScore: {binders[display_index].score}\tPPL: {binders[display_index].ppl}")

    for display_index in range(min(args.num_display, len(binders))):
        print(f"{binders[display_index].binder_seq}")

    no_improvement_generations = 0
    max_tolerance = args.max_iterations
    threshold = int(0.1 * len(parse_motif(args.motif)))

    print(f"Threshold: {threshold}")

    while no_improvement_generations < max_tolerance:
        previous_score = binders[0].score
        previous_ppl = binders[0].ppl

        new_binders = []

        selected_count = int((10 * len(binders)) / 100)
        new_binders.extend(binders[:selected_count])

        offspring_count = int((90 * len(binders)) / 100)
        half = int(len(binders) / 2)
        for offspring_index in range(offspring_count):
            par1 = random.choice(binders[:half])
            par2 = random.choice(binders[:half])
            new_binders.append(par1.mate(par2))

        for binder_index in range(1, len(new_binders)):
            if new_binders[binder_index].binder_seq == new_binders[binder_index - 1].binder_seq:
                new_seq = new_binders[binder_index].mutate_seq(new_binders[binder_index].binder_seq)
                new_binders[binder_index] = Binder(new_seq, model, pepmlm, tokenizer, bindevaluator_tokenizer, args)

        new_binders = sorted(new_binders, key=lambda binder: (binder.score, binder.ppl))

        for display_index in range(min(args.num_display, len(new_binders))):
            print(f"Generation: {generation}\tBinder: {new_binders[display_index].binder_seq}\tScore: {new_binders[display_index].score}\tPPL: {new_binders[display_index].ppl}")

        if new_binders[0].score < previous_score or new_binders[0].ppl < previous_ppl:
            no_improvement_generations = 0
            print(f"Generation: {generation}\tImproved!")
        else:
            no_improvement_generations += 1
            print(f"Generation: {generation}\tNo improvement {no_improvement_generations} generations")

        for display_index in range(min(args.num_display, len(new_binders))):
            print(f"{new_binders[display_index].binder_seq}")

        binders = new_binders
        generation += 1

    print(f"moPPIt Stopping\tBinder: {binders[0].binder_seq}\tScore: {binders[0].score}\tPPL: {binders[0].ppl}")

    if args.output is not None:
        write_generation_output(args.output, binders, args.num_display)

    return binders


def write_generation_output(output_path, binders, limit=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_binders = binders[:limit] if limit is not None else binders

    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["binder", "score", "pseudo_perplexity"])
        writer.writeheader()
        for binder in selected_binders:
            writer.writerow({
                "binder": binder.binder_seq,
                "score": binder.score,
                "pseudo_perplexity": binder.ppl,
            })


def build_parser():
    parser = ArgumentParser()
    add_checkpoint_arguments(parser)
    parser.add_argument("--protein_seq", "--protein-seq", dest="protein_seq", type=str, required=True)
    parser.add_argument("--peptide_length", "--peptide-length", dest="peptide_length", type=int, required=True)
    parser.add_argument("--motif", type=str, required=True)
    parser.add_argument("--top_k", "--top-k", dest="top_k", type=int, default=3)
    parser.add_argument("--num_binders", "--num-binders", dest="num_binders", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5, help="Binding-site probability threshold for scoring.")
    parser.add_argument("--num_display", "--num-display", dest="num_display", type=int, default=1)
    parser.add_argument("-max_iterations", "--max_iterations", "--max-iterations", dest="max_iterations",
                        type=int, default=20, help="Maximum no-improvement iterations")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV file for the final displayed binders.")
    parser.add_argument("-batch_size", type=int, default=None, help=SUPPRESS)
    parser.add_argument("-lr", type=float, default=None, help=SUPPRESS)
    return parser


def main_cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        main(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main_cli()