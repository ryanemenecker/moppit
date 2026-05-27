from argparse import ArgumentParser, SUPPRESS
import csv
from pathlib import Path
import random

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .bindevaluator import (
    ESM_MODEL_NAME,
    add_checkpoint_arguments,
    calculate_scores,
    load_bindevaluator_checkpoint,
    parse_motif,
    resolve_device,
)
from .pepmlm import PEPMLM_MODEL_NAME, compute_pseudo_perplexities, generate_peptide, validate_pepmlm_context


AA = "ARNDCEQGHILKMFPSTWYV"


def generate_random_seq(length):
    return "".join(random.choice(AA) for residue_index in range(length))


def log_progress(args, message):
    if not getattr(args, "quiet_progress", False):
        print(f"[moPPIt] {message}", flush=True)


def make_progress_callback(args, stage, unit):
    def progress_callback(done, total):
        if total <= 0:
            return
        percent = (100 * done) / total
        log_progress(args, f"{stage}: {done}/{total} {unit} ({percent:.1f}%)")
    return progress_callback


def score_prediction(prediction, motif, threshold):
    motif = list(motif)
    out_of_range = [position for position in motif if position < 0 or position >= len(prediction)]
    if out_of_range:
        raise ValueError(f"Motif indices out of range for target length {len(prediction)}: {out_of_range}")

    motif_set = set(motif)
    score = 0
    for position in motif:
        if prediction[position] < threshold:
            score += 1

    for position in range(len(prediction)):
        if position not in motif_set and prediction[position] >= threshold:
            score += 0.5

    return score


def cal_score(binder_seq, protein_seq, motif, model, args, tokenizer=None):
    prediction = calculate_scores(protein_seq, [binder_seq], model, args, tokenizer=tokenizer, batch_size=1)[0]
    return score_prediction(prediction, motif, args.threshold)


def validate_generation_inputs(args):
    motif = parse_motif(args.motif.strip("[]"))
    if not motif:
        raise ValueError("--motif must include at least one target residue.")
    out_of_range = [position for position in motif if position < 0 or position >= len(args.protein_seq)]
    if out_of_range:
        raise ValueError(f"Motif indices out of range for target length {len(args.protein_seq)}: {out_of_range}")
    if args.num_binders < 1:
        raise ValueError("--num-binders must be at least 1.")
    if args.num_display < 1:
        raise ValueError("--num-display must be at least 1.")
    if args.score_batch_size < 1:
        raise ValueError("--score-batch-size must be at least 1.")
    if args.ppl_batch_size < 1:
        raise ValueError("--ppl-batch-size must be at least 1.")
    validate_pepmlm_context(args.protein_seq, args.peptide_length)
    return motif


class Binder(object):
    def __init__(self, binder_seq, args, score=None, ppl=None):
        self.binder_seq = binder_seq
        self.protein_seq = args.protein_seq
        self.motif = parse_motif(args.motif.strip("[]"))
        self.args = args
        self.score = score
        self.ppl = ppl

    def is_evaluated(self):
        return self.score is not None and self.ppl is not None

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

        return Binder(child, self.args)


def evaluate_binders(binders, model, pepmlm, pepmlm_tokenizer, bindevaluator_tokenizer, args, stage):
    pending_binders = [binder for binder in binders if not binder.is_evaluated()]
    if not pending_binders:
        return

    binder_sequences = [binder.binder_seq for binder in pending_binders]
    log_progress(args, f"{stage}: scoring {len(pending_binders)} candidate binders with BindEvaluator "
                       f"(batch size {args.score_batch_size})")
    predictions = calculate_scores(args.protein_seq, binder_sequences, model, args,
                                   tokenizer=bindevaluator_tokenizer,
                                   batch_size=args.score_batch_size,
                                   progress_callback=make_progress_callback(args, stage, "BindEvaluator binders"))
    for binder, prediction in zip(pending_binders, predictions):
        binder.score = score_prediction(prediction, binder.motif, args.threshold)

    total_masked_positions = sum(len(binder.binder_seq) for binder in pending_binders)
    log_progress(args, f"{stage}: scoring pseudo-perplexity for {len(pending_binders)} binders "
                       f"({total_masked_positions} masked positions, batch size {args.ppl_batch_size})")
    ppl_values = compute_pseudo_perplexities(pepmlm, pepmlm_tokenizer, args.protein_seq, binder_sequences,
                                            batch_size=args.ppl_batch_size,
                                            progress_callback=make_progress_callback(args, stage,
                                                                                    "PepMLM masked positions"))
    for binder, ppl_value in zip(pending_binders, ppl_values):
        binder.ppl = ppl_value


def main(args):
    motif = validate_generation_inputs(args)
    random.seed(args.seed)
    binders = []
    generation = 0

    device = resolve_device(args.device)
    print(f"Device: {device}")
    log_progress(args, f"Starting generation: target length {len(args.protein_seq)}, peptide length "
                       f"{args.peptide_length}, requested pool {args.num_binders}, score batch size "
                       f"{args.score_batch_size}, PPL batch size {args.ppl_batch_size}")
    log_progress(args, f"Motif residues: {motif}")
    model = load_bindevaluator_checkpoint(args, device=device)

    tokenizer = AutoTokenizer.from_pretrained(PEPMLM_MODEL_NAME)
    pepmlm = AutoModelForMaskedLM.from_pretrained(PEPMLM_MODEL_NAME).to(device)
    bindevaluator_tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)

    temp_binders = []
    count = 2
    attempt = 1
    while len(temp_binders) < args.num_binders and count >= 0:
        needed_binders = args.num_binders - len(temp_binders)
        log_progress(args, f"Initial PepMLM sampling attempt {attempt}/3: requesting {needed_binders} binders")
        temp_binders += generate_peptide(args.protein_seq, args.peptide_length, args.top_k, needed_binders,
                                         model=pepmlm, tokenizer=tokenizer, compute_ppl=False)["Binder"].tolist()
        temp_binders = [sequence for sequence in temp_binders if "X" not in sequence]
        temp_binders = list(set(temp_binders))
        log_progress(args, f"Initial pool has {len(temp_binders)}/{args.num_binders} unique non-X binders")
        count -= 1
        attempt += 1

    for binder_index in range(0, args.num_binders - len(temp_binders)):
        temp_binders.append(generate_random_seq(args.peptide_length))
    temp_binders = list(set(temp_binders))

    print(f"Pool Size = {len(temp_binders)}")

    for binder_seq in temp_binders:
        binders.append(Binder(binder_seq, args))

    evaluate_binders(binders, model, pepmlm, tokenizer, bindevaluator_tokenizer, args, "Initial pool")

    binders = sorted(binders, key=lambda binder: (binder.score, binder.ppl))
    for display_index in range(min(args.num_display, len(binders))):
        print(f"Generation: -1\tBinder: {binders[display_index].binder_seq}\tScore: {binders[display_index].score}\tPPL: {binders[display_index].ppl}")

    for display_index in range(min(args.num_display, len(binders))):
        print(f"{binders[display_index].binder_seq}")

    no_improvement_generations = 0
    max_tolerance = args.max_iterations

    while no_improvement_generations < max_tolerance:
        log_progress(args, f"Generation {generation}: no-improvement counter "
                           f"{no_improvement_generations}/{max_tolerance}")
        previous_score = binders[0].score
        previous_ppl = binders[0].ppl

        new_binders = []

        selected_count = max(int((10 * len(binders)) / 100), 1)
        new_binders.extend(binders[:selected_count])

        offspring_count = max(len(binders) - selected_count, 0)
        half = max(int(len(binders) / 2), 1)
        log_progress(args, f"Generation {generation}: carrying {selected_count} elites and creating "
                           f"{offspring_count} offspring")
        for offspring_index in range(offspring_count):
            par1 = random.choice(binders[:half])
            par2 = random.choice(binders[:half])
            new_binders.append(par1.mate(par2))

        for binder_index in range(1, len(new_binders)):
            if new_binders[binder_index].binder_seq == new_binders[binder_index - 1].binder_seq:
                new_seq = new_binders[binder_index].mutate_seq(new_binders[binder_index].binder_seq)
                new_binders[binder_index] = Binder(new_seq, args)

        evaluate_binders(new_binders, model, pepmlm, tokenizer, bindevaluator_tokenizer, args,
                         f"Generation {generation}")

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
    parser.add_argument("--score-batch-size", type=int, default=8,
                        help="BindEvaluator candidate scoring batch size. Increase on large GPUs.")
    parser.add_argument("--ppl-batch-size", type=int, default=256,
                        help="PepMLM masked-position batch size for pseudo-perplexity. Increase on large GPUs.")
    parser.add_argument("--quiet-progress", action="store_true",
                        help="Disable progress updates during generation.")
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