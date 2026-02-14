"""
picogpt_commented.py — A Study Guide for picogpt.py
====================================================

This file is a heavily commented, educational expansion of picogpt.py — a complete
GPT-2 language model (training + inference) in <2000 characters of pure Python.

No external dependencies (no PyTorch, no NumPy). Everything — autograd, matrix math,
attention, backpropagation, Adam optimizer — is built from scratch.

Architecture (follows GPT-2 with minor changes):
  - Token + positional embeddings
  - 1 transformer block:
      - RMS LayerNorm (not standard LayerNorm — no bias, no learned params)
      - Multi-head causal self-attention (4 heads)
      - RMS LayerNorm
      - Feed-forward MLP with ReLU (not GELU)
  - Linear projection to vocabulary logits

Variable Name Glossary (minified -> meaning):
  V  = Value          (autograd node)
  Y  = coerce         (wraps raw numbers into V nodes if needed)
  O  = backward       (backpropagation through the computation graph)
  E  = n_embd    = 16 (embedding dimension, also block_size)
  R  = head_dim  = 4  (dimension per head, also n_head since E/R = 4)
  B  = BOS token id   (Beginning of Sequence, = len(unique_chars))
  Z  = vocab_size     (B + 1, includes BOS)
  U  = unique chars   (sorted list of characters in dataset)
  D  = docs           (list of training documents/names)
  S  = state_dict     (all weight matrices, as a list indexed 0..8)
  P  = params         (flat list of all Value parameters)
  F  = linear         (matrix-vector multiply)
  X  = softmax
  J  = rmsnorm        (Root Mean Square Layer Normalization)
  G  = gpt            (forward pass: one token through the transformer)
  A  = add            (element-wise vector addition)
  Q  = range          (alias)
  T  = sum            (alias)
  I  = len            (alias)
  K  = keys cache     (stored key projections for attention)
  C  = values cache   (stored value projections for attention)
  M  = adam_m         (Adam first moment estimates)
  N  = adam_v         (Adam second moment estimates)

S (state_dict) index mapping:
  S[0] = token embeddings   (vocab_size x 16)
  S[1] = position embeddings (16 x 16)
  S[2] = lm_head            (vocab_size x 16, output projection)
  S[3] = attn query weights (16 x 16)
  S[4] = attn key weights   (16 x 16)
  S[5] = attn value weights (16 x 16)
  S[6] = attn output proj   (16 x 16)
  S[7] = MLP fc1            (64 x 16, expand 4x)
  S[8] = MLP fc2            (16 x 64, compress back)
"""

import math as m, random as r

# ============================================================================
# PART 1: DATA LOADING
# ============================================================================
# Reproducible randomness
r.seed(42)

# Read training data: one word/name per line
D = open("input.txt").read().split()
r.shuffle(D)

# Build character-level vocabulary
# {*"".join(D)} is a set comprehension — unpacks all chars from all docs
U = sorted({*"".join(D)})

B = len(U)      # BOS token id (e.g., 26). Also used as EOS — same token.
Z = B + 1       # Total vocab size (e.g., 27)

# Tokenization example:
#   Name "cat" -> [26, 2, 0, 19, 26]
#                  BOS  c   a   t   BOS
#   (indices depend on actual vocab; BOS marks start AND end)


# ============================================================================
# PART 2: AUTOGRAD ENGINE
# ============================================================================
# A scalar-valued automatic differentiation engine (like Karpathy's micrograd).
# Each Value node stores:
#   .d = data (the forward-pass scalar value)
#   .g = grad (gradient, filled during backward pass)
#   .c = children (input nodes that produced this node)
#   .l = local_grads (d(this)/d(child) for each child)
#
# The computation graph is built implicitly during the forward pass.
# Each operation (+, *, etc.) creates a new Value node that remembers
# its inputs and how to propagate gradients back through them.
#
# Example computation graph for: L = (a * b) + c
#
#   a ──┐
#       ├── (a*b) ──┐
#   b ──┘           ├── L = (a*b) + c
#   c ──────────────┘
#
#   Forward:  a.d=2, b.d=3, c.d=1
#             (a*b).d = 6,  L.d = 7
#
#   Backward: L.g = 1 (seed)
#             (a*b).g = 1 * 1 = 1  (d(L)/d(a*b) = 1, addition local grad)
#             c.g     = 1 * 1 = 1
#             a.g     = 1 * b.d = 3 (d(a*b)/da = b)
#             b.g     = 1 * a.d = 2 (d(a*b)/db = a)


class V:
    # --- Constructor ---
    # Creates a new Value node.
    # Args: d=data value, c=children tuple, l=local gradients tuple
    def __init__(s, d, c=(), l=()):
        s.d = d       # scalar forward value
        s.g = 0       # gradient (accumulated during backward)
        s.c = c       # children in computation graph
        s.l = l       # local derivative d(this)/d(child) for each child

    # --- Addition: a + b ---
    # Forward:  result = a.d + b.d
    # Backward: da = 1 * grad_out,  db = 1 * grad_out
    # Local gradients are (1, 1) because d(a+b)/da = 1, d(a+b)/db = 1
    #
    # The walrus operator (:=) wraps raw numbers into Value nodes:
    #   V(3) + 2  ->  o becomes V(2), then V(3+2) = V(5)
    __add__ = lambda s, o: V(s.d + (o := Y(o)).d, (s, o), (1, 1))

    # --- Multiplication: a * b ---
    # Forward:  result = a.d * b.d
    # Backward: da = b.d * grad_out,  db = a.d * grad_out
    # Because d(a*b)/da = b and d(a*b)/db = a
    __mul__ = lambda s, o: V(s.d * (o := Y(o)).d, (s, o), (o.d, s.d))

    # --- Power: a ** n (n is a plain number, not a Value) ---
    # Forward:  result = a.d ^ n
    # Backward: da = n * a.d^(n-1) * grad_out  (power rule)
    # Used for: x**-1 (reciprocal, in division), x**-0.5 (in rmsnorm)
    __pow__ = lambda s, o: V(s.d**o, (s,), (o * s.d ** (o - 1),))

    # --- Reverse add: number + Value ---
    # Python calls this when the left operand doesn't know how to add.
    #   2 + V(3) -> V(3).__radd__(2) -> V(3) + 2
    __radd__ = lambda s, o: s + o

    # --- Division: a / b  ->  a * b^(-1) ---
    # Reuses __mul__ and __pow__ — no separate gradient formula needed.
    __truediv__ = lambda s, o: s * o**-1


# --- Coerce to Value ---
# Wraps raw numbers into Value nodes. If already a Value, returns as-is.
# Uses boolean indexing: type(o)==V is True (=1) -> returns o; False (=0) -> returns V(o)
Y = lambda o: [V(o), o][type(o) == V]


# --- Backward pass (reverse-mode autodiff / backpropagation) ---
# Standalone function (not a method on V) to save characters in the golfed version.
#
# Algorithm:
#   1. Topological sort: DFS to order nodes so parents come after children
#   2. Set output gradient to 1 (dL/dL = 1)
#   3. Walk in reverse order, accumulating gradients via chain rule:
#      child.grad += local_grad * parent.grad
#
# This is the ENTIRE backpropagation algorithm.
#
# Visualization of backward pass for L = (a * b) + c:
#   Topo order: [a, b, (a*b), c, L]
#   Reversed:   [L, c, (a*b), b, a]
#   L.g = 1
#   -> c.g += 1 * L.g = 1        (local_grad=1 for addition)
#   -> (a*b).g += 1 * L.g = 1    (local_grad=1 for addition)
#   -> b.g += a.d * (a*b).g       (local_grad=a.d for multiplication)
#   -> a.g += b.d * (a*b).g       (local_grad=b.d for multiplication)
def O(s):
    t = []           # topological order list
    v = set()        # visited set

    def b(n):        # DFS to build topological order
        if n not in v:
            v.add(n)
            [b(c) for c in n.c]  # visit children first
            t.append(n)          # post-order: append after children

    b(s)             # start DFS from the loss node
    s.g = 1          # seed the output gradient
    for n in t[::-1]:                    # reversed topo order
        for c, l in zip(n.c, n.l):       # each (child, local_grad)
            c.g += l * n.g               # chain rule: accumulate gradient


# ============================================================================
# PART 3: MODEL ARCHITECTURE & WEIGHT INITIALIZATION
# ============================================================================

# Short aliases (save characters in golfed version)
Q = range
T = sum
I = len

E = 16      # Embedding dim AND block_size (max sequence length). Both are 16.
R = 4       # Head dim AND num_heads. Both are 4. (E // R = R, since 16 // 4 = 4)

# Element-wise vector addition helper (used for residual connections)
A = lambda x, y: [a + b for a, b in zip(x, y)]

# State dict as a list of weight matrices (indexed 0..8).
# Each matrix is created with random Gaussian initialization (std=0.08).
# Dimensions are specified as parallel lists of (rows, cols) via zip:
#
#   Index  Name        Rows   Cols   Shape
#   -----  ----        ----   ----   -----
#   S[0]   tok_emb     Z      E      (vocab_size, 16)  — one row per token
#   S[1]   pos_emb     E      E      (16, 16)          — one row per position
#   S[2]   lm_head     Z      E      (vocab_size, 16)  — projects to logits
#   S[3]   attn_wq     E      E      (16, 16)          — query projection
#   S[4]   attn_wk     E      E      (16, 16)          — key projection
#   S[5]   attn_wv     E      E      (16, 16)          — value projection
#   S[6]   attn_wo     E      E      (16, 16)          — output projection
#   S[7]   mlp_fc1     4*E    E      (64, 16)          — MLP expand 4x
#   S[8]   mlp_fc2     E      4*E    (16, 64)          — MLP compress back
S = [
    [[V(r.gauss(0, 0.08)) for _ in Q(c)] for _ in Q(w)]
    for w, c in zip(
        [Z, E, Z] + [E] * 4 + [4 * E, E],   # rows: [Z,E,Z,E,E,E,E,64,E]
        [E] * 8 + [4 * E],                    # cols: [E,E,E,E,E,E,E,E,64]
    )
]

# Flatten all parameters into one list for the optimizer
P = [p for w in S for o in w for p in o]

# Adam optimizer moment buffers
M = [0.0] * I(P)    # First moment estimates  (momentum — smoothed gradient)
N = M[:]             # Second moment estimates (adaptive lr — smoothed grad^2)


# ============================================================================
# PART 4: NEURAL NETWORK BUILDING BLOCKS
# ============================================================================

# --- Linear layer (matrix-vector multiplication) ---
# Computes: output[i] = dot(weight_row[i], input)
#
# Example with 2x3 weight matrix and 3-element input:
#   W = [[w00, w01, w02],    x = [x0, x1, x2]
#        [w10, w11, w12]]
#
#   output = [w00*x0 + w01*x1 + w02*x2,
#             w10*x0 + w11*x1 + w12*x2]
#
# This IS the fundamental operation of neural networks.
F = lambda x, w: [T(a * b for a, b in zip(o, x)) for o in w]

# --- RMS LayerNorm (Root Mean Square Normalization) ---
# Formula: rmsnorm(x) = x / sqrt(mean(x^2) + eps)
#
# Normalizes vectors to have roughly unit magnitude.
# Simpler than standard LayerNorm (no mean subtraction, no learned params).
# Why? Keeps activations from growing/shrinking across layers.
#
# The walrus operator caches the scale factor:
#   c := 1/sqrt(mean(x^2) + eps)
# Returns index [1] of a tuple — the scaled vector.
J = lambda x: (
    c := (T(a * a for a in x) / I(x) + 1e-5) ** -0.5,
    [a * c for a in x],
)[1]

# --- Softmax (converts raw scores to probabilities) ---
# Formula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
#
# Subtracting max(x) is a numerical stability trick — it prevents
# exp() from overflowing while giving the same result mathematically.
#
# Example: softmax([2.0, 1.0, 0.1]) ≈ [0.659, 0.242, 0.099]
#   -> Largest input gets highest probability, all sum to 1.0
#
# This version inlines the exp() autograd operation (no separate ex function).
# For each value v in the input:
#   a := v + (-max)    — shifted value (a V node, preserving computation graph)
#   z := exp(a.d)      — raw exponential
#   V(z, (a,), (z,))   — exp node with child=a, local_grad=exp(a.d)
#                         (derivative of e^x is e^x)
X = lambda g: (
    e := [V((z := m.exp((a := v + -max(c.d for c in g)).d)), (a,), (z,)) for v in g],
    t := T(e),
    [i / t for i in e],
)[2]


# ============================================================================
# PART 5: THE TRANSFORMER FORWARD PASS
# ============================================================================
# G processes ONE token at a time (autoregressive generation).
# It uses a KV-cache (K, C) to remember past tokens' key/value projections.
#
# Args:
#   i = token_id (integer index into vocabulary)
#   p = position (0, 1, 2, ... up to E-1)
#   K = key cache   (list of key vectors from previous positions)
#   C = value cache (list of value vectors from previous positions)
#
# Returns: logits (list of Z values — unnormalized scores for each vocab token)
#
# The architecture for one transformer block:
#
#   Input token
#       |
#   [Token Embed + Position Embed]  -- look up learned vectors
#       |
#   [RMS Norm]
#       |──────────────────────┐
#   [RMS Norm]                 │ (residual connection)
#       |                      │
#   ┌───┴───┐                  │
#   [Q] [K] [V]  projections   │
#       |                      │
#   [Multi-Head Attention]     │
#       |                      │
#   [Output Projection]        │
#       |                      │
#       + ◄────────────────────┘ (add residual)
#       |
#       |──────────────────────┐
#       |                      │ (residual connection)
#   [RMS Norm]                 │
#       |                      │
#   [MLP: Linear -> ReLU]      │
#       |                      │
#   [MLP: Linear]              │
#       |                      │
#       + ◄────────────────────┘ (add residual)
#       |
#   [Linear -> vocab logits]
#       |
#   Output (Z scores)


def G(i, p, K, C):
    # --- Embedding ---
    # Look up token embedding + position embedding, add them, normalize
    # S[0][i] = 16-dim vector for token i
    # S[1][p] = 16-dim vector for position p
    x = J(A(S[0][i], S[1][p]))

    # --- Attention block ---
    # Walrus y:=x saves pre-attention state for residual connection
    x = J(y := x)   # RMS norm before attention (pre-norm architecture)

    # Project input to queries, keys, values (each is a 16-dim vector)
    q, k, v = F(x, S[3]), F(x, S[4]), F(x, S[5])

    # Append current k,v to the cache (building up context for attention)
    # K += k, is shorthand for K += (k,) which is K.append(k)
    K += k,
    C += v,

    # --- Multi-head attention ---
    # Split q/k/v into R=4 heads of R=4 dims each.
    # Each head independently attends to the full sequence,
    # then results are concatenated.
    #
    # For each head h:
    #   1. Extract head's slice: u = q[h*4 : h*4+4]  (4 dims)
    #   2. For each past position t, compute attention score:
    #      score_t = dot(u, k_t_head_slice) / sqrt(head_dim)
    #   3. Apply softmax to get attention weights (sum to 1)
    #   4. Weighted sum of value vectors = attention output
    #
    # Intuition: attention lets the model "look at" relevant past tokens.
    # Query = "what am I looking for?"
    # Key   = "what do I contain?"
    # Value = "what information should I pass along?"
    a = []
    for h in Q(R):
        e = h * R     # starting index for this head's slice
        u = q[e:e+R]  # query for this head (4 dims)

        # Attention logits: dot(query, key) / sqrt(head_dim)
        # /2 because sqrt(R) = sqrt(4) = 2
        w = X([T(u[j] * K[t][e+j] for j in Q(R)) / 2 for t in Q(I(K))])

        # Weighted sum of values for each dimension
        a += [T(w[t] * C[t][e+j] for t in Q(I(C))) for j in Q(R)]

    # Output projection + residual connection
    # Residual = "skip connection" — adds the original input back.
    # This prevents the vanishing gradient problem in deep networks.
    x = A(F(a, S[6]), y)

    # --- MLP block ---
    y = x  # save residual
    # FC1: 16->64 with ReLU activation (inline, no separate rl function)
    # V(max(0, a.d), (a,), (a.d > 0,)) is the ReLU autograd node:
    #   Forward: max(0, x)
    #   Backward: 1 if x > 0, else 0
    x = [V(max(0, a.d), (a,), (a.d > 0,)) for a in F(J(x), S[7])]

    # FC2: 64->16 + residual, then project to vocab logits
    return F(A(F(x, S[8]), y), S[2])


# ============================================================================
# PART 6: TRAINING LOOP
# ============================================================================
# For each training step:
#   1. Pick a document (name), tokenize it
#   2. For each position, predict the next token
#   3. Compute cross-entropy loss (how wrong were the predictions?)
#   4. Backpropagate to compute gradients
#   5. Update parameters with Adam optimizer
#
# Cross-entropy loss = -log(probability assigned to correct next token)
#   If model assigns 90% prob to correct token: loss = -log(0.9) ≈ 0.11 (low, good!)
#   If model assigns 10% prob to correct token: loss = -log(0.1) ≈ 2.30 (high, bad!)
#   If model assigns 1% prob to correct token:  loss = -log(0.01) ≈ 4.60 (very bad!)

for s in Q(1000):
    # Tokenize: wrap document in BOS tokens
    # "cat" -> [26, 2, 0, 19, 26]  (BOS c a t BOS)
    # Uses *map() for compact unpacking: [B, *map(U.index, chars), B]
    t = [B, *map(U.index, D[s % I(D)]), B]
    n = min(E, I(t) - 1)  # number of predictions to make

    K, C = [], []  # fresh KV cache for each document

    # Forward pass: for each position p, predict next token
    # The log is inlined as a V node: V(log(prob.d), (prob,), (1/prob.d,))
    #   - q := softmax(logits)[target]  — probability assigned to correct token
    #   - V(log(q.d), (q,), (1/q.d,))  — log node with gradient 1/x
    # Sum of logs / -n = negative log likelihood (average cross-entropy loss)
    L = T(
        V(m.log((q := X(G(t[p], p, K, C))[t[p + 1]]).d), (q,), (1 / q.d,))
        for p in Q(n)
    ) / -n
    # L is a Value node at the root of the computation graph.

    # Backpropagation: compute dL/d(param) for every parameter
    O(L)

    # --- Adam optimizer update ---
    # Adam = Adaptive Moment Estimation. Better than plain SGD because:
    #   1. Momentum (M): smooths gradient over time, prevents oscillation
    #   2. Adaptive rate (N): scales LR per-parameter based on gradient history
    #
    # Linear LR warmdown: lr decreases from 0.01 to 0 over 1000 steps
    # Constants inlined: beta1=0.85, beta2=0.99, eps=1e-8
    z = 0.01 * (1 - s / 1000)

    for i, p in enumerate(P):
        # Update first moment (momentum): M = beta1 * M + (1-beta1) * grad
        M[i] = 0.85 * M[i] + 0.15 * p.g
        # Update second moment (RMS of grads): N = beta2 * N + (1-beta2) * grad^2
        N[i] = 0.99 * N[i] + 0.01 * p.g**2
        # Bias-corrected update:
        #   m_hat = M[i] / (1 - beta1^step)     (correct for zero-initialization bias)
        #   v_hat = N[i] / (1 - beta2^step)
        #   param -= lr * m_hat / (sqrt(v_hat) + epsilon)
        p.d -= z * (M[i] / (1 - 0.85 ** (s + 1))) / ((N[i] / (1 - 0.99 ** (s + 1))) ** 0.5 + 1e-8)
        # Reset gradient for next step
        p.g = 0

    print(f"step {s+1:4d} / 1000 | loss {L.d:.4f}")


# ============================================================================
# PART 7: INFERENCE (TEXT GENERATION)
# ============================================================================
# Generate new names by sampling from the model's predictions.
#
# Process (autoregressive generation):
#   1. Start with BOS token
#   2. Feed token through transformer -> get probability distribution
#   3. Sample next token from distribution (temperature-scaled)
#   4. If sampled BOS (= end of sequence), stop
#   5. Otherwise, add token to output and repeat from step 2
#
# Temperature scaling (the `l*2` below, equivalent to temp=0.5):
#   - Dividing logits by temperature < 1 makes distribution sharper
#     (more confident, less random — model picks its top choices)
#   - Temperature = 1.0: sample from model's natural distribution
#   - Temperature = 0.5: logits * 2, sharper peaks, more deterministic
#   - Temperature -> 0: always picks most likely token (greedy)
#
# Example generation trace:
#   Token: BOS -> probs: {a:0.3, b:0.1, ..., z:0.05, BOS:0.02}
#   Sample: 'j' (token 9)
#   Token: 'j' -> probs: {a:0.4, e:0.2, o:0.15, ...}
#   Sample: 'a' (token 0)
#   Token: 'a' -> probs: {n:0.3, m:0.2, s:0.15, ...}
#   Sample: 'n' (token 13)
#   Token: 'n' -> probs: {e:0.3, BOS:0.25, ...}
#   Sample: BOS -> STOP
#   Result: "jan"

print("\n--- inference (new, hallucinated names) ---")
for i in Q(20):
    K, C = [], []    # fresh KV cache per sample
    t = B            # start with BOS token
    s = ""           # accumulate generated characters

    for p in Q(E):   # generate up to E=16 characters
        # Forward pass -> softmax with temperature=0.5 (logits * 2)
        t = r.choices(Q(Z), [a.d for a in X([l * 2 for l in G(t, p, K, C)])])[0]
        if t == B:   # sampled BOS = end of sequence
            break
        s += U[t]    # append decoded character

    print(f"sample {i+1:2d}: {s}")


# ============================================================================
# SUMMARY: HOW IT ALL FITS TOGETHER
# ============================================================================
#
# Training (1000 steps):
#   For each name in the dataset:
#     tokens = [BOS] + encode(name) + [BOS]
#     For each position:
#       logits = transformer(token, position, kv_cache)
#       probs  = softmax(logits)
#       loss  += -log(probs[next_correct_token])
#     O(loss)                <- autograd computes all gradients
#     adam_update(params)    <- update weights to reduce loss
#
# Inference:
#   Start with BOS, repeatedly:
#     logits = transformer(token, position, kv_cache)
#     probs  = softmax(logits / temperature)
#     next_token = sample(probs)
#     if next_token == BOS: stop
#
# The entire thing — forward pass, backward pass, optimization —
# operates on scalar Value nodes. There are no tensors, no batches,
# no GPU. Just <2000 characters of Python implementing a GPT from scratch.
