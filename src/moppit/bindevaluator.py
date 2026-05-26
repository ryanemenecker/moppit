from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import AutoTokenizer, EsmModel

from .models import FFN, MultiHeadAttentionSequence, RepeatedModule3


ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DEFAULT_MAX_LENGTH = 40000


def parse_motif(motif: str) -> list[int]:
    parts = motif.split(",")
    residues = []

    for part in parts:
        part = part.strip()
        if "-" in part:
            start, end = map(int, part.split("-"))
            residues.extend(range(start, end + 1))
        elif part:
            residues.append(int(part))

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


def load_bindevaluator_checkpoint(args, device=None):
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return PeptideModel.load_from_checkpoint(
        args.sm,
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        n_head=args.n_head,
        d_k=64,
        d_v=128,
        d_inner=args.d_inner,
    ).to(device)


def predict_binding_sites(target_sequence, binder_sequence, model, threshold=0.5):
    prediction, _ = calculate_score(target_sequence, binder_sequence, model)
    binding_site = []
    for index in range(len(prediction)):
        if prediction[index] >= threshold:
            binding_site.append(index)
    return binding_site, prediction


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("-sm", required=True, help="File containing initial params", type=str)
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("-d_model", type=int, default=64, help="Dimension of model")
    parser.add_argument("-d_hidden", type=int, default=128, help="Dimension of CNN block")
    parser.add_argument("-n_head", type=int, default=6, help="Number of heads")
    parser.add_argument("-d_inner", type=int, default=64)
    parser.add_argument("-target", type=str)
    parser.add_argument("-binder", type=str)
    parser.add_argument("-gt", type=str, default=None, help="Ground Truth binding motifs")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    model = load_bindevaluator_checkpoint(args)
    binding_site, prediction = predict_binding_sites(args.target, args.binder, model)

    print("Prediction: ", binding_site)

    if args.gt is not None:
        length = len(args.target)
        ground_truth = parse_motif(args.gt)
        print("Ground Truth: ", ground_truth)

        accuracy, f1, mcc = compute_metrics(ground_truth, binding_site, length)
        print(f"Accuracy={accuracy}\tF1={f1}\tMCC={mcc}")

    return prediction


if __name__ == "__main__":
    main()