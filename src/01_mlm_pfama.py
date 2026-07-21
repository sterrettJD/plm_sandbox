import matplotlib.pyplot as plt
import numpy as np
import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

import tokenizer
from base_plm import ProteinLanguageModel


def read_fasta(path: str = "datasets/homeodomain_seed.fasta") -> list[SeqRecord]:
    with open(path) as p:
        records = list(SeqIO.parse(p, "fasta"))
    return records


def tokenize_seqrecords(seq_records: list[SeqRecord]) -> list[np.ndarray]:
    return [tokenizer.tokenize(seq.seq) for seq in seq_records]


def plot_loss(loss_vals: list[float], path: str = "figures/loss.png"):
    plt.plot(range(len(loss_vals)), loss_vals)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(path)


def main():
    path = "datasets/homeodomain_seed.fasta"
    with open(path) as p:
        records = list(SeqIO.parse(p, "fasta"))

    tokenized = tokenize_seqrecords(records)
    print(records[1])
    print(tokenized[1])

    vocab_size = len(tokenizer.KEYS)
    model = ProteinLanguageModel(
        vocab_size=vocab_size, embedding_dim=128, num_heads=4, num_layers=4
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    np.random.seed(42)
    num_epochs = 10
    loss_vals = [0 for n in range(num_epochs)]
    train_portion = 0.7
    train_n = int(len(records) * train_portion)

    np.random.shuffle(tokenized)
    train_subset = tokenized[0:train_n]
    test_subset = tokenized[train_n : len(tokenized)]

    train_subset_tensor = [torch.tensor(seq) for seq in train_subset]

    # Pad sequences (batch_first=True gives shape: (batch_size, seq_len))
    X_train = torch.nn.utils.rnn.pad_sequence(
        train_subset_tensor, batch_first=True, padding_value=tokenizer.PAD_TOKEN
    )

    print(f"Shape of X_train: {X_train.shape}")  # (train_n, max_len)

    for n in range(num_epochs):
        # 5.1 MLM Preparation: Mask 15% of the tokens
        mask_indices = np.array(
            [
                np.random.choice(
                    X_train.shape[1], size=int(0.15 * X_train.shape[1]), replace=False
                )
                for _ in range(X_train.shape[0])
            ]
        )

        row_indices = (
            torch.arange(X_train.shape[0])
            .unsqueeze(1)
            .expand_as(torch.tensor(mask_indices))
        )
        masked_X = X_train.clone()
        masked_X[row_indices, torch.tensor(mask_indices)] = tokenizer.MASK_TOKEN

        y_pred = model(masked_X)

        # 5.3 Calculate Loss
        loss = loss_fn(y_pred.view(-1, vocab_size), X_train.view(-1))

        # 5.4 Backpropagation and Optimization
        loss.backward()
        optimizer.step()

        # 5.5 Update Loss Values
        loss_vals[n] = loss.item()

        # 5.6 Reset Gradients
        optimizer.zero_grad()

    print(loss_vals)
    plot_loss(loss_vals, path="figures/loss.png")

    # 6. Evaluate
    test_subset_tensor = [torch.tensor(seq) for seq in test_subset]

    # Pad sequences (batch_first=True gives shape: (batch_size, seq_len))
    X_test = torch.nn.utils.rnn.pad_sequence(
        test_subset_tensor, batch_first=True, padding_value=tokenizer.PAD_TOKEN
    )
    mask_indices = np.array(
        [
            np.random.choice(
                X_test.shape[1], size=int(0.15 * X_test.shape[1]), replace=False
            )
            for _ in range(X_test.shape[0])
        ]
    )
    row_indices = (
        torch.arange(X_test.shape[0]).unsqueeze(1).expand_as(torch.tensor(mask_indices))
    )
    masked_X = X_test.clone()
    masked_X[row_indices, torch.tensor(mask_indices)] = tokenizer.MASK_TOKEN

    model.eval()
    with torch.no_grad():
        y_pred_test = model(masked_X)
        test_loss = loss_fn(y_pred_test.view(-1, vocab_size), X_test.view(-1))
    print(f"Test set loss: {test_loss}")


def test_tokenizer():
    sequence = "RARY"
    tokenized = tokenizer.tokenize(sequence)
    print(tokenized)
    detokenized = tokenizer.de_tokenize(tokenized)
    print(detokenized)


if __name__ == "__main__":
    test_tokenizer()
    main()
