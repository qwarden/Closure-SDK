"""Bracket sequence generator for S³ RNN validation.

Vocabulary:
    0 = "("   (open)
    1 = ")"   (close)
    2 = EOS

The model must LEARN from data alone that "(" and ")" are
compositional inverses. This is not hardcoded anywhere.

Valid sequences: balanced brackets of lengths 4–32.
Corrupted sequences: valid sequences with two random positions swapped.
"""

import random

OPEN = 0
CLOSE = 1
EOS = 2
VOCAB_SIZE = 3


def generate_valid(max_depth=8, min_pairs=2, max_pairs=16, rng=None):
    """Generate a random valid bracket sequence.

    Uses a random walk: at each step, push "(" if we haven't hit max_depth
    and we still have pairs to place, or pop ")" if the stack is nonempty.
    The choice is random, biased by what's legal.
    """
    if rng is None:
        rng = random

    n_pairs = rng.randint(min_pairs, max_pairs)
    seq = []
    depth = 0
    opens_left = n_pairs

    for _ in range(2 * n_pairs):
        can_open = opens_left > 0 and depth < max_depth
        can_close = depth > 0

        if can_open and can_close:
            if rng.random() < 0.5:
                seq.append(OPEN)
                depth += 1
                opens_left -= 1
            else:
                seq.append(CLOSE)
                depth -= 1
        elif can_open:
            seq.append(OPEN)
            depth += 1
            opens_left -= 1
        elif can_close:
            seq.append(CLOSE)
            depth -= 1

    seq.append(EOS)
    return seq


def corrupt(seq, rng=None, n_flips=None):
    """Flip random brackets in the sequence (before EOS).

    More aggressive than swapping: flips ( → ) and ) → (
    at n_flips random positions. Default: flip ~20% of positions.
    Guarantees the result is actually invalid.
    """
    if rng is None:
        rng = random
    s = list(seq)
    bracket_part = len(s) - 1  # exclude EOS
    if bracket_part < 2:
        return s

    if n_flips is None:
        n_flips = max(1, bracket_part // 5)

    # Keep trying until we get an actually invalid sequence
    for _ in range(20):
        c = list(s)
        positions = rng.sample(range(bracket_part), min(n_flips, bracket_part))
        for p in positions:
            c[p] = CLOSE if c[p] == OPEN else OPEN
        if not is_valid(c):
            return c

    # Fallback: force invalid by flipping first close bracket
    c = list(s)
    c[0] = CLOSE
    return c


def is_valid(seq):
    """Check if a bracket sequence (without EOS) is balanced."""
    depth = 0
    for tok in seq:
        if tok == EOS:
            break
        if tok == OPEN:
            depth += 1
        elif tok == CLOSE:
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def make_dataset(n_samples, min_pairs=2, max_pairs=16, seed=42):
    """Generate n_samples valid bracket sequences.

    Returns list of token lists, each ending with EOS.
    """
    rng = random.Random(seed)
    dataset = []
    for _ in range(n_samples):
        seq = generate_valid(min_pairs=min_pairs, max_pairs=max_pairs, rng=rng)
        dataset.append(seq)
    return dataset


def bracket_length(seq):
    """Number of bracket tokens (before EOS)."""
    for i, tok in enumerate(seq):
        if tok == EOS:
            return i
    return len(seq)


def pad_batch(sequences, pad_value=EOS):
    """Pad sequences to the same length.

    Returns (padded_list, total_lengths, bracket_lengths).
    - total_lengths: full sequence length including EOS
    - bracket_lengths: number of bracket tokens before EOS
    """
    max_len = max(len(s) for s in sequences)
    padded = []
    total_lengths = []
    b_lengths = []
    for s in sequences:
        total_lengths.append(len(s))
        b_lengths.append(bracket_length(s))
        padded.append(s + [pad_value] * (max_len - len(s)))
    return padded, total_lengths, b_lengths
