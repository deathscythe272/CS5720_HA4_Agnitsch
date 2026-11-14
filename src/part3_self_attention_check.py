"""
CS5720 Home Assignment 4 - Part III: numerical self-attention, checked in code.

Reproduces the three-token self-attention example from the Chapter 9 (Transformers)
lecture using the lecture's recipe:

    * UNSCALED dot-product scores (no division by sqrt(d_k))
    * softmax first, then round the attention weights to ONE decimal place
    * weighted sum of the Value vectors using the rounded weights

The hand calculation for token 3 is written out in the README. This script exists
so the arithmetic can be verified and so the same inputs can be fed through the
scaled attention function from Part II for comparison.
"""

import sys
from pathlib import Path

import numpy as np

# Make the sibling Part II module importable whether this script is run from the
# repo root (python src/part3_...py) or from inside src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2_scaled_dot_product_attention import scaled_dot_product_attention, softmax  # noqa: E402

np.set_printoptions(precision=4, suppress=True)

# --- Inputs from the lecture slides -------------------------------------------
# Three input tokens, each a 4-dimensional embedding (one row per token).
X = np.array([[1, 0, 1, 0],   # input 1
              [0, 2, 0, 2],   # input 2
              [1, 1, 1, 1]])  # input 3

# Projection matrices (4 x 3): rows index the input dimensions, columns the
# query/key/value dimensions. These are the matrices shown on the slides.
W_Q = np.array([[1, 0, 1],
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 1]])
W_K = np.array([[0, 0, 1],
                [1, 1, 0],
                [0, 1, 0],
                [1, 1, 0]])
W_V = np.array([[0, 2, 0],
                [0, 3, 0],
                [1, 0, 3],
                [1, 1, 0]])


def lecture_self_attention(X, W_Q, W_K, W_V, token_index: int) -> np.ndarray:
    """Self-attention for one token, following the lecture's unscaled, rounded recipe."""
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    q = Q[token_index]

    # Unscaled scores: dot product of this token's query with every key.
    scores = K @ q
    print(f"Query q{token_index + 1} = {q}")
    for j, (k, s) in enumerate(zip(K, scores), start=1):
        terms = " + ".join(f"{a}*{b}" for a, b in zip(q, k))
        print(f"  score vs key {j}: {q} . {k} = {terms} = {s}")

    # Softmax, then round to one decimal place as done in class.
    weights_exact = softmax(scores.astype(float))
    weights = np.round(weights_exact, 1)
    print(f"softmax({scores}) = {weights_exact}  ->  rounded to 1 dp: {weights}")

    # Weighted values and their sum.
    weighted = weights[:, None] * V
    for j, row in enumerate(weighted, start=1):
        print(f"  {weights[j - 1]} * v{j} {V[j - 1]} = {row}")
    out = weighted.sum(axis=0)
    print(f"New representation of token {token_index + 1}: {out}\n")
    return out


def main() -> None:
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    print("Queries (rows):\n", Q)
    print("Keys (rows):\n", K)
    print("Values (rows):\n", V)
    print()

    print("=== Lecture recipe (unscaled, weights rounded to 1 dp) ===\n")
    results = [lecture_self_attention(X, W_Q, W_K, W_V, i) for i in range(3)]
    print("All three outputs:")
    print(np.array(results))
    print("Slide 41 shows: [2.0 7.0 1.5], [2.0 8.0 0.0], [2.0 7.8 0.3]")
    # Token 3 (the one the assignment asks for) matches the slide exactly. Token 1
    # differs slightly because the slide treats softmax([2, 4, 4]) as [0.0, 0.5, 0.5];
    # the true first weight is 0.063, which rounds to 0.1, not 0.0. The slide notes
    # this approximation itself ("can arise from rounding").
    print()

    # For comparison, run the same Q, K, V through the Part II implementation,
    # which scales by sqrt(d_k) and does not round. The ranking of the weights is
    # unchanged; the scaled version is just a little less peaked.
    print("=== Same inputs through the Part II scaled function (no rounding) ===\n")
    scaled_dot_product_attention(Q, K, V)


if __name__ == "__main__":
    main()
