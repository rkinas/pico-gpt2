"""
picogpt_commented.py — A Study Guide for nanogpt.py
====================================================

This file is a heavily commented, educational expansion of nanogpt.py — a complete
GPT-2 language model (training + inference) in ~2400 characters of pure Python.

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
  E  = n_embd    = 16 (embedding dimension)
  H  = n_head    = 4  (number of attention heads)
  W  = block_size= 16 (max sequence length / context window)
  R  = head_dim  = 4  (dimension per head = E // H)
  B  = BOS token id   (Beginning of Sequence, = len(unique_chars))
  Z  = vocab_size     (B + 1, includes BOS)
  U  = unique chars   (sorted list of characters in dataset)
  D  = docs           (list of training documents/names)
  S  = state_dict     (all weight matrices, keyed by short names)
  P  = params         (flat list of all Value parameters)
  F  = linear         (matrix-vector multiply)
  X  = softmax
  J  = rmsnorm        (Root Mean Square Layer Normalization)
  G  = gpt            (forward pass: one token through the transformer)
  ex = exp            (exponential with autograd)
  lg = log            (natural log with autograd)
  rl = relu           (rectified linear unit with autograd)
  mx = matrix         (creates a random weight matrix)
  K  = keys cache     (stored key projections for attention)
  C  = values cache   (stored value projections for attention)
  M  = adam_m         (Adam first moment estimates)
  N  = adam_v         (Adam second moment estimates)
"""

import math as m, random as r

# ============================================================================
# PART 1: DATA LOADING
# ============================================================================
# Reproducible randomness
r.seed(42)

# Read training data: one word/name per line
# Example contents of input.txt
D = open("input.txt").read().split()
r.shuffle(D)

# Build character-level vocabulary
# Example: U = ['a', 'b', 'c', ..., 'z'] (26 unique chars)
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
# This is a scalar-valued automatic differentiation engine, similar to
# Karpathy's micrograd. Each Value node stores:
#   .d  = data (the forward-pass scalar value)
#   .g  = grad (gradient, filled during backward pass)
#   ._c = children (input nodes that produced this node)
#   ._l = local_grads (d(this)/d(child) for each child)
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

V = type(
    "V",       # class name
    (),        # base classes (none)
    {          # class body (dict of methods):

        # --- Constructor ---
        # Creates a new Value node.
        # Args: d=data value, c=children tuple, l=local gradients tuple
        "__init__": lambda s, d, c=(), l=(): vars(s).update(
            d=d, g=0, _c=c, _l=l
        ),

        # --- Addition: a + b ---
        # Forward:  result = a.d + b.d
        # Backward: da = 1 * grad_out,  db = 1 * grad_out
        # Local gradients are (1, 1) because d(a+b)/da = 1, d(a+b)/db = 1
        #
        # The walrus operator (:=) wraps raw numbers into Value nodes:
        #   V(3) + 2  ->  o becomes V(2), then V(3+2) = V(5)
        "__add__": lambda s, o: V(
            s.d + (o := V(o) if type(o) != V else o).d,
            (s, o),
            (1, 1),
        ),

        # --- Multiplication: a * b ---
        # Forward:  result = a.d * b.d
        # Backward: da = b.d * grad_out,  db = a.d * grad_out
        # Because d(a*b)/da = b and d(a*b)/db = a
        #
        # Example: a=V(3), b=V(4) -> result=V(12), local_grads=(4, 3)
        "__mul__": lambda s, o: V(
            s.d * (o := V(o) if type(o) != V else o).d,
            (s, o),
            (o.d, s.d),
        ),

        # --- Power: a ** n (n is a plain number, not a Value) ---
        # Forward:  result = a.d ^ n
        # Backward: da = n * a.d^(n-1) * grad_out  (power rule)
        #
        # Used for: x**-1 (reciprocal, in division), x**-0.5 (in rmsnorm)
        "__pow__": lambda s, o: V(s.d ** o, (s,), (o * s.d ** (o - 1),)),

        # --- Reverse add: number + Value ---
        # Python calls this when the left operand doesn't know how to add.
        #   2 + V(3) -> V(3).__radd__(2) -> V(3) + 2
        "__radd__": lambda s, o: s + o,

        # --- Division: a / b  ->  a * b^(-1) ---
        # Reuses __mul__ and __pow__ — no separate gradient formula needed.
        "__truediv__": lambda s, o: s * o ** -1,

        # --- Backward pass (reverse-mode autodiff / backpropagation) ---
        # 1. Topological sort: DFS to order nodes so parents come after children
        # 2. Set output gradient to 1 (dL/dL = 1)
        # 3. Walk in reverse order, accumulating gradients via chain rule:
        #    child.grad += local_grad * parent.grad
        #
        # This is the ENTIRE backpropagation algorithm in 5 lines.
        #
        # Visualization of backward pass for L = (a * b) + c:
        #   Topo order: [a, b, (a*b), c, L]
        #   Reversed:   [L, c, (a*b), b, a]
        #   L.g = 1
        #   -> c.g += 1 * L.g = 1        (local_grad=1 for addition)
        #   -> (a*b).g += 1 * L.g = 1    (local_grad=1 for addition)
        #   -> b.g += a.d * (a*b).g       (local_grad=a.d for multiplication)
        #   -> a.g += b.d * (a*b).g       (local_grad=b.d for multiplication)
        "backward": lambda s: (
            lambda t, v: (
                # Step 1: Build topological order via DFS
                (
                    b := lambda n: (
                        n not in v
                        and (v.add(n), [b(c) for c in n._c], t.append(n))
                    )
                ),
                b(s),  # start DFS from the loss node
                # Step 2: Seed the output gradient
                s.__setattr__("g", 1),
                # Step 3: Propagate gradients in reverse topological order
                [
                    c.__setattr__("g", c.g + l * n.g)
                    for n in t[::-1]               # reversed topo order
                    for c, l in zip(n._c, n._l)    # each (child, local_grad)
                ],
            )
        )([], set()),   # t=topo list, v=visited set
    },
)


# ============================================================================
# PART 3: AUTOGRAD-TRACKED MATH OPERATIONS
# ============================================================================
# These are standalone functions (not methods on V) to save characters.
# Each creates a new Value node with the correct forward value and local gradient.

# --- Exponential: e^x ---
# Forward:  result = e^(a.d)
# Backward: da = e^(a.d) * grad_out  (derivative of e^x is e^x)
# The walrus operator caches the exp value to avoid computing it twice.
ex = lambda a: V((e := m.exp(a.d)), (a,), (e,))

# --- Natural logarithm: ln(x) ---
# Forward:  result = ln(a.d)
# Backward: da = (1/a.d) * grad_out  (derivative of ln(x) is 1/x)
lg = lambda a: V(m.log(a.d), (a,), (1 / a.d,))

# --- ReLU: max(0, x) ---
# Forward:  result = max(0, a.d)
# Backward: da = 1 if a.d > 0 else 0  (gradient passes through if positive)
#
#   Output
#     |      /
#     |     /
#     |    /
#   --+---/-------> Input
#     |  0
#
rl = lambda a: V(max(0, a.d), (a,), (a.d > 0,))


# ============================================================================
# PART 4: MODEL ARCHITECTURE & WEIGHT INITIALIZATION
# ============================================================================

E = W = 16  # E=embedding dim, W=block_size (max sequence length)
H = R = 4   # H=num_heads, R=head_dim (E//H = 16//4 = 4)

# Create a random weight matrix of shape (rows, cols).
# Each entry is a Value node initialized from Gaussian(0, 0.08).
# Small std (0.08) prevents exploding activations at initialization.
mx = lambda a, b: [[V(r.gauss(0, 0.08)) for _ in range(b)] for _ in range(a)]

# State dictionary: all learnable weight matrices
# Shape annotations show (output_dim, input_dim):
S = {
    "e": mx(Z, E),   # Token embeddings:    (vocab_size, 16) — one row per token
    "p": mx(W, E),    # Position embeddings: (16, 16) — one row per position
    "h": mx(Z, E),    # LM head (output):    (vocab_size, 16) — projects to logits
    # Attention weights (all 16x16):
    **{k: mx(E, E) for k in "qkvo"},
    # "q" = query projection,  "k" = key projection,
    # "v" = value projection,  "o" = output projection
    "f": mx(4 * E, E),  # MLP first layer:  (64, 16) — expand 4x
    "b": mx(E, 4 * E),  # MLP second layer: (16, 64) — compress back
}

# Flatten all parameters into one list for the optimizer
# Example: S has 9 matrices -> flatten all rows -> flatten all Values
P = [p for w in S.values() for o in w for p in o]

# Adam optimizer hyperparameters
lr, b1, b2, ea = 0.01, 0.85, 0.99, 1e-8
# lr  = learning rate (how big each parameter update step is)
# b1  = beta1 = 0.85 (exponential decay rate for 1st moment / momentum)
# b2  = beta2 = 0.99 (exponential decay rate for 2nd moment / adaptive rate)
# ea  = epsilon = 1e-8 (prevents division by zero)

M = [0.0] * len(P)  # First moment estimates  (momentum — smoothed gradient)
N = M[:]             # Second moment estimates (adaptive lr — smoothed grad^2)

# Alias for setattr (saves characters in minified version)
a = setattr


# ============================================================================
# PART 5: NEURAL NETWORK BUILDING BLOCKS
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
# Every "layer" is just: output = linear(input, weights)
F = lambda x, w: [sum(a * b for a, b in zip(o, x)) for o in w]

# --- Softmax (converts raw scores to probabilities) ---
# Formula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
#
# Subtracting max(x) is a numerical stability trick — it prevents
# exp() from overflowing while giving the same result mathematically.
#
# Example: softmax([2.0, 1.0, 0.1]) ≈ [0.659, 0.242, 0.099]
#   -> Largest input gets highest probability, all sum to 1.0
#
# The walrus operators (:=) store intermediate results:
#   e := exponentials,  t := their sum
# Returns index [2] of a tuple — the normalized probabilities.
X = lambda g: (
    e := [ex(v + -max(a.d for a in g)) for v in g],  # exp(x - max)
    t := sum(e),                                       # sum of exponentials
    [i / t for i in e],                                # normalize to probabilities
)[2]

# --- RMS LayerNorm (Root Mean Square Normalization) ---
# Formula: rmsnorm(x) = x / sqrt(mean(x^2) + eps)
#
# Normalizes vectors to have roughly unit magnitude.
# Simpler than standard LayerNorm (no mean subtraction, no learned params).
#
# Example: rmsnorm([3, 4]) = [3, 4] / sqrt((9+16)/2 + 1e-5)
#                           = [3, 4] / sqrt(12.5)
#                           ≈ [0.849, 1.131]
#
# Why? Keeps activations from growing/shrinking across layers.
# Without normalization, values can explode or vanish as data flows
# through many matrix multiplications.
J = lambda x: (
    c := (sum(a * a for a in x) / len(x) + 1e-5) ** -0.5,  # 1/sqrt(rms + eps)
    [a * c for a in x],                                      # scale each element
)[1]


# ============================================================================
# PART 6: THE TRANSFORMER FORWARD PASS
# ============================================================================
# G processes ONE token at a time (autoregressive generation).
# It uses a KV-cache (K, C) to remember past tokens' key/value projections.
#
# Args:
#   i = token_id (integer index into vocabulary)
#   p = position (0, 1, 2, ... up to W-1)
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
    # S['e'][i] = 16-dim vector for token i
    # S['p'][p] = 16-dim vector for position p
    y = x = J([a + b for a, b in zip(S["e"][i], S["p"][p])])

    # --- Attention block ---
    x = J(x)  # RMS norm before attention (pre-norm architecture)

    # Project input to queries, keys, values (each is a 16-dim vector)
    q, k, v = F(x, S["q"]), F(x, S["k"]), F(x, S["v"])

    # Append current k,v to the cache (building up context for attention)
    K.append(k)
    C.append(v)

    # --- Multi-head attention ---
    # Split q/k/v into H=4 heads of R=4 dims each.
    # Each head independently attends to the full sequence,
    # then results are concatenated.
    #
    # For each head h:
    #   1. Extract head's slice: q_h = q[h*4 : (h+1)*4]  (4 dims)
    #   2. For each past position t, compute attention score:
    #      score_t = dot(q_h, k_t_h) / sqrt(head_dim)
    #   3. Apply softmax to get attention weights (sum to 1)
    #   4. Weighted sum of value vectors = attention output
    #
    # Intuition: attention lets the model "look at" relevant past tokens.
    # Query = "what am I looking for?"
    # Key   = "what do I contain?"
    # Value = "what information should I pass along?"
    # Score = "how relevant is this past token to the current query?"
    #
    # Example (simplified, 1 head, 2 dims):
    #   q = [1, 0]      (current token's query)
    #   K = [[1, 0],    (past token 0 — key matches query!)
    #        [0, 1]]    (past token 1 — key doesn't match)
    #   scores = [1*1 + 0*0, 1*0 + 0*1] / sqrt(2) = [0.71, 0]
    #   weights = softmax([0.71, 0]) ≈ [0.66, 0.34]
    #   output = 0.66 * V[0] + 0.34 * V[1]  (mostly reads from token 0)
    a = sum(
        [
            (
                lambda q, k, v: (
                    lambda w: [
                        # Weighted sum of values for each dimension j
                        sum(w[t] * v[t][j] for t in range(len(v)))
                        for j in range(R)
                    ]
                )(
                    # Attention weights = softmax of scaled dot-product scores
                    X(
                        [
                            sum(q[j] * k[t][j] for j in range(R)) / 2
                            #                         ^^^ / sqrt(R) = / sqrt(4) = / 2
                            for t in range(len(k))
                        ]
                    )
                )
            )(
                # Slice out this head's dimensions from q, K cache, C cache
                q[h * R : (h + 1) * R],                     # query head slice
                [i[h * R : (h + 1) * R] for i in K],        # all cached key slices
                [i[h * R : (h + 1) * R] for i in C],        # all cached value slices
            )
            for h in range(H)  # iterate over all 4 heads
        ],
        [],  # start with empty list (sum concatenates head outputs)
    )

    # Output projection + residual connection
    # Residual = "skip connection" — adds the original input back.
    # This prevents the vanishing gradient problem in deep networks.
    x = [a + b for a, b in zip(F(a, S["o"]), y)]

    # --- MLP block ---
    y = x  # save residual
    x = [rl(a) for a in F(J(x), S["f"])]   # FC1: 16->64 with ReLU activation
    x = [a + b for a, b in zip(F(x, S["b"]), y)]  # FC2: 64->16 + residual

    # --- Output head ---
    # Project from embedding dimension (16) to vocabulary size (Z)
    # Returns unnormalized logits — softmax converts these to probabilities
    return F(x, S["h"])


# ============================================================================
# PART 7: TRAINING LOOP
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

for s in range(1000):
    # Tokenize: wrap document in BOS tokens
    # "cat" -> [26, 2, 0, 19, 26]  (BOS c a t BOS)
    t = [B] + [U.index(c) for c in D[s % len(D)]] + [B]
    n = min(W, len(t) - 1)  # number of predictions to make

    K, C = [], []  # fresh KV cache for each document

    # Forward pass: for each position p, predict next token
    # G(t[p], p, K, C) returns logits -> softmax -> get prob of target
    # lg() = log of that probability -> sum -> divide by -n = negative log likelihood
    L = sum(lg(X(G(t[p], p, K, C))[t[p + 1]]) for p in range(n)) / -n
    # The loss L is a Value node at the root of the computation graph.
    # Every arithmetic operation above created Value nodes, forming a DAG
    # from parameters through embeddings, attention, MLP, softmax, to loss.

    # Backpropagation: compute dL/d(param) for every parameter
    L.backward()

    # --- Adam optimizer update ---
    # Adam = Adaptive Moment Estimation. Better than plain SGD because:
    #   1. Momentum (M): smooths gradient over time, prevents oscillation
    #   2. Adaptive rate (N): scales LR per-parameter based on gradient history
    #
    # Linear LR warmdown: lr decreases from 0.01 to 0 over 1000 steps
    z = lr * (1 - s / 1000)

    [
        (
            # Update first moment (momentum): M = beta1 * M + (1-beta1) * grad
            M.__setitem__(i, b1 * M[i] + (1 - b1) * p.g),
            # Update second moment (RMS of grads): N = beta2 * N + (1-beta2) * grad^2
            N.__setitem__(i, b2 * N[i] + (1 - b2) * p.g ** 2),
            # Bias-corrected update:
            #   m_hat = M[i] / (1 - beta1^step)     (correct for zero-initialization bias)
            #   v_hat = N[i] / (1 - beta2^step)
            #   param -= lr * m_hat / (sqrt(v_hat) + epsilon)
            a(
                p,
                "d",
                p.d
                - z
                * (M[i] / (1 - b1 ** (s + 1)))
                / ((N[i] / (1 - b2 ** (s + 1))) ** 0.5 + ea),
            ),
            # Reset gradient for next step
            a(p, "g", 0),
        )
        for i, p in enumerate(P)
    ]

    print(f"step {s + 1:4d} / 1000 | loss {L.d:.4f}")


# ============================================================================
# PART 8: INFERENCE (TEXT GENERATION)
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

# Recursive generation function:
#   K, C = KV cache (grows with each generated token)
#   t = current token id
#   p = current position
#   s = accumulated output tokens (list of chars)
g = lambda K, C, t, p, s: (
    s  # base case: return accumulated result
    if p >= W  # hit max sequence length
    or (
        n := r.choices(
            range(Z),
            [a.d for a in X([l * 2 for l in G(t, p, K, C)])],
            #                   ^^^ temperature=0.5 (logits * 2)
        )[0]
    )
    == B  # sampled BOS = end of sequence
    else g(K, C, n, p + 1, s + [U[n]])  # recursive case: continue generating
)

for i in range(20):
    # Each sample starts fresh with empty KV cache and BOS token
    print(f"sample {i + 1:2d}: {''.join(g([], [], B, 0, []))}")


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
#     loss.backward()         <- autograd computes all gradients
#     adam_update(params)      <- update weights to reduce loss
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
# no GPU. Just ~2400 characters of nested Python lambdas and list
# comprehensions implementing a GPT from scratch.
