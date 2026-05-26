from argparse import ArgumentParser
import random

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .bindevaluator import calculate_score, load_bindevaluator_checkpoint, parse_motif
from .pepmlm import PEPMLM_MODEL_NAME, compute_pseudo_perplexity, generate_peptide


AA = "ARNDCEQGHILKMFPSTWYV"


def generate_random_seq(length):
    return "".join(random.choice(AA) for residue_index in range(length))


def cal_score(binder_seq, protein_seq, motif, model, args):
    prediction, threshold = calculate_score(protein_seq, binder_seq, model, args)
    score = 0
    for position in motif:
        if prediction[position] < 0.5:
            score += 1

    for position in range(len(prediction)):
        if position not in motif and prediction[position] >= 0.5:
            score += 0.5

    return score


class Binder(object):
    def __init__(self, binder_seq, model, pepmlm, tokenizer, args):
        self.binder_seq = binder_seq
        self.protein_seq = args.protein_seq
        self.motif = parse_motif(args.motif.strip("[]"))
        self.model = model
        self.args = args
        self.pepmlm = pepmlm
        self.tokenizer = tokenizer
        self.score = cal_score(binder_seq, self.protein_seq, self.motif, self.model, self.args)
        self.ppl = compute_pseudo_perplexity(self.pepmlm, self.tokenizer, self.protein_seq, self.binder_seq)

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

        return Binder(child, self.model, self.pepmlm, self.tokenizer, self.args)


def main(args):
    print(parse_motif(args.motif.strip("[]")))
    random.seed(args.seed)
    binders = []
    generation = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = load_bindevaluator_checkpoint(args, device=device)

    tokenizer = AutoTokenizer.from_pretrained(PEPMLM_MODEL_NAME)
    pepmlm = AutoModelForMaskedLM.from_pretrained(PEPMLM_MODEL_NAME).to(device)

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
        binders.append(Binder(binder_seq, model, pepmlm, tokenizer, args))

    binders = sorted(binders, key=lambda binder: (binder.score, binder.ppl))
    for display_index in range(min(args.num_display, len(binders))):
        print(f"Generation: -1\tBinder: {binders[display_index].binder_seq}\tScore: {binders[display_index].score}\tPPL: {binders[display_index].ppl}")

    for display_index in range(min(args.num_display, len(binders))):
        print(f"{binders[display_index].binder_seq}")

    no_improvement_generations = 0
    max_tolerance = args.max_iterations
    threshold = int(0.1 * len(parse_motif(args.motif.strip("[]"))))

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
                new_binders[binder_index] = Binder(new_seq, model, pepmlm, tokenizer, args)

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


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--protein_seq", type=str, required=True)
    parser.add_argument("--peptide_length", type=int, required=True)
    parser.add_argument("--motif", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--num_binders", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-sm", required=True, help="File containing initial params", type=str)
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("-d_model", type=int, default=64, help="Dimension of model")
    parser.add_argument("-d_hidden", type=int, default=128, help="Dimension of CNN block")
    parser.add_argument("-n_head", type=int, default=6, help="Number of heads")
    parser.add_argument("-d_inner", type=int, default=64)
    parser.add_argument("--num_display", type=int, default=1)
    parser.add_argument("-max_iterations", type=int, default=20, help="Maximum no improvement iterations")
    return parser


def main_cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    main(args)


if __name__ == "__main__":
    main_cli()