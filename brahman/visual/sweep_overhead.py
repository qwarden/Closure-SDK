"""Minimum overhead sweep — what's the least neural architecture that generates closed walks?

Tests combinations of:
  - Embedding: full network (Linear→GELU→Linear) vs direct lookup (nn.Embedding)
  - Head: full network vs single Linear
  - Layers: 4, 2, 1, 0
  - All with pure next-token prediction (closure_weight=0.0)
  - Training data: closed walks only (the geometry finds the true shape)
"""

import sys
import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from brahman.model import S3Attention, S3Block, qmul

from data import (
    UP, DOWN, LEFT, RIGHT, EOS, VOCAB_SIZE,
    generate_closed_walk, generate_open_walk, generate_dataset,
    walk_to_positions, is_closed, walk_length, pad_batch, walk_to_string,
)


class S3Minimal(nn.Module):
    """S³ model with configurable overhead.

    embed_type:  "network" = Linear→GELU→Linear (original)
                 "lookup"  = nn.Embedding (direct, no hidden layer)
    head_type:   "network" = Linear→GELU→Linear (original)
                 "linear"  = single Linear
    n_layers:    number of S3Attention + quaternion composition layers
                 0 = embed → head only (no attention, no composition)
    """

    def __init__(self, vocab_size, m_factors=2, n_layers=4,
                 hidden=32, embed_type="network", head_type="network",
                 max_seq_len=34):
        super().__init__()
        self.m = m_factors
        self.dim = 4 * m_factors
        self.vocab_size = vocab_size

        # Embedding
        if embed_type == "network":
            self.embed = nn.Sequential(
                nn.Linear(vocab_size, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.dim),
            )
            self._embed_is_onehot = True
        else:  # "lookup"
            self.embed = nn.Embedding(vocab_size, self.dim)
            self._embed_is_onehot = False

        # Positional
        self.pos_embed = nn.Parameter(
            torch.randn(max_seq_len, self.dim) * 0.02
        )

        # Attention layers
        self.blocks = nn.ModuleList([
            S3Block(m_factors) for _ in range(n_layers)
        ])

        # Prediction head
        if head_type == "network":
            self.head = nn.Sequential(
                nn.Linear(self.dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, vocab_size),
            )
        else:  # "linear"
            self.head = nn.Linear(self.dim, vocab_size)

    def forward(self, tokens, targets=None, lengths=None):
        B, T = tokens.shape
        device = tokens.device

        if self._embed_is_onehot:
            x = F.one_hot(tokens, self.vocab_size).float()
            x = self.embed(x)
        else:
            x = self.embed(tokens)

        x = x + self.pos_embed[:T]
        x = x.view(B, T, self.m, 4)
        x = F.normalize(x, dim=-1)
        x = x.view(B, T, self.dim)

        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)

        logits = self.head(x)

        if targets is not None:
            pred_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
            )
            return logits, pred_loss
        return logits

    @torch.no_grad()
    def generate(self, start_token, max_length, temperature=0.8):
        device = next(self.parameters()).device
        tokens = [start_token]

        for _ in range(max_length - 1):
            tok_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = self(tok_tensor)
            logits = logits[0, -1:] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
            if next_token == EOS:
                break

        return tokens


def train_and_test(config, train_data, epochs=50, batch_size=128, lr=1e-3):
    """Train one configuration, return generation results."""
    device = torch.device("cpu")

    model = S3Minimal(
        vocab_size=VOCAB_SIZE,
        m_factors=config["m"],
        n_layers=config["layers"],
        hidden=config.get("hidden", 32),
        embed_type=config["embed"],
        head_type=config["head"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_data)
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            padded, lengths = pad_batch(batch)
            tokens = torch.tensor(padded, dtype=torch.long, device=device)
            _, loss = model(tokens, targets=tokens)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    train_time = time.perf_counter() - t0

    # Generation test
    model.eval()
    n_gen = 500
    valid = 0
    for _ in range(n_gen):
        start = random.choice([UP, DOWN, LEFT, RIGHT])
        walk = model.generate(start, max_length=32, temperature=0.8)
        if is_closed(walk):
            valid += 1

    return {
        "params": n_params,
        "valid": valid,
        "total": n_gen,
        "pct": valid / n_gen * 100,
        "time": train_time,
    }


def main():
    random.seed(42)
    torch.manual_seed(42)

    n_train = 20_000  # enough to learn, fast enough to sweep
    train_data = generate_dataset(n_train)

    print()
    print("  ╔═════════════════════════════════════════════════════════════╗")
    print("  ║  Minimum overhead sweep — grid walk closed loop generation ║")
    print("  ╚═════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Training: {n_train:,} closed walks, 50 epochs, closure_weight=0.0")
    print(f"  Test: 500 generated walks, check if closed")
    print(f"  Baseline: S3Transformer (4L, network embed+head) = 98.3%")
    print()

    configs = [
        # Baseline: full model
        {"name": "4L net+net",   "layers": 4, "embed": "network", "head": "network", "m": 2},
        # Strip head
        {"name": "4L net+lin",   "layers": 4, "embed": "network", "head": "linear",  "m": 2},
        # Strip embed
        {"name": "4L lookup+net","layers": 4, "embed": "lookup",  "head": "network", "m": 2},
        # Strip both
        {"name": "4L lookup+lin","layers": 4, "embed": "lookup",  "head": "linear",  "m": 2},
        # Reduce layers with minimal embed/head
        {"name": "2L lookup+lin","layers": 2, "embed": "lookup",  "head": "linear",  "m": 2},
        {"name": "1L lookup+lin","layers": 1, "embed": "lookup",  "head": "linear",  "m": 2},
        # Zero layers — embed → head, no attention at all
        {"name": "0L lookup+lin","layers": 0, "embed": "lookup",  "head": "linear",  "m": 2},
        {"name": "0L net+net",   "layers": 0, "embed": "network", "head": "network", "m": 2},
        {"name": "0L net+lin",   "layers": 0, "embed": "network", "head": "linear",  "m": 2},
    ]

    print(f"  {'Config':<18} {'Params':>8} {'Closed':>8} {'%':>8} {'Time':>8}")
    print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for cfg in configs:
        torch.manual_seed(42)
        random.seed(42)
        result = train_and_test(cfg, list(train_data), epochs=50)
        print(
            f"  {cfg['name']:<18} {result['params']:>8,} "
            f"{result['valid']:>5}/{result['total']:<3} "
            f"{result['pct']:>6.1f}% "
            f"{result['time']:>6.1f}s"
        )

    print()


if __name__ == "__main__":
    main()
