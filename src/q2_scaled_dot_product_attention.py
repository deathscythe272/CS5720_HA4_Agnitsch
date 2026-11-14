"""
CS5720 Home Assignment 4 - Part II, Q2: scaled dot-product attention.

Implements the attention formula from "Attention Is All You Need":

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

and runs it on the assignment's test matrices. Each intermediate matrix is printed
so the four steps (dot product, scale, softmax, weighted sum) can be checked by hand.
"""

import numpy as np

# Print floats with four decimals and no scientific notation, so the matrices
# are easy to read and compare against hand calculations.
np.set_printoptions(precision=4, suppress=True)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along `axis`.

    Subtracting the row maximum before exponentiating does not change the result
    (softmax is shift-invariant) but keeps exp() from overflowing on large scores.
    """
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute scaled dot-product attention.

    Q: (n_queries, d_k)   one query vector per row
    K: (n_keys,    d_k)   one key vector per row, same width as Q
    V: (n_keys,    d_v)   one value vector per row, one per key

    Returns (attention_weights, output) where
        attention_weights has shape (n_queries, n_keys) and each row sums to 1
        output            has shape (n_queries, d_v)
    """
    Q, K, V = (np.asarray(m, dtype=float) for m in (Q, K, V))
    d_k = K.shape[1]  # the key dimension used for scaling

    # Step 1 - raw similarity between every query and every key.
    scores = Q @ K.T

    # Step 2 - scale by sqrt(d_k). Dot products grow with the vector length, and
    # large scores push softmax toward a one-hot output whose gradient is near zero.
    # Dividing by sqrt(d_k) keeps the scores at a size softmax handles well.
    scaled_scores = scores / np.sqrt(d_k)

    # Step 3 - softmax over the keys (axis=-1) turns each row into a probability
    # distribution: how much this query should attend to each key.
    weights = softmax(scaled_scores, axis=-1)

    # Step 4 - each output row is a weighted average of the value rows.
    output = weights @ V

    if verbose:
        print(f"Q K^T (raw scores):\n{scores}\n")
        print(f"Scaled by sqrt(d_k) = sqrt({d_k}) = {np.sqrt(d_k):.4f}:\n{scaled_scores}\n")
        print(f"Attention weights (softmax over each row):\n{weights}")
        print(f"Row sums: {weights.sum(axis=1)}\n")
        print(f"Output = weights @ V:\n{output}\n")

    return weights, output


def main() -> None:
    # Test inputs given in the assignment. Q and K are identical, and each query is
    # orthogonal to the other key, so the score matrix is diagonal.
    Q = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1]])
    K = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1]])
    V = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8]])

    print("=== Assignment test case ===\n")
    weights, output = scaled_dot_product_attention(Q, K, V)

    print("Final answer")
    print("Attention weights:\n", weights)
    print("Output:\n", output)


if __name__ == "__main__":
    main()
