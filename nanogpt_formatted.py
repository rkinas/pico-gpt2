import math as m, random as r

r.seed(42)
D = open("input.txt").read().split()
r.shuffle(D)
U = sorted({*"".join(D)})
B = len(U)
Z = B + 1
V = type(
    "V",
    (),
    {
        "__init__": lambda s, d, c=(), l=(): vars(s).update(d=d, g=0, _c=c, _l=l),
        "__add__": lambda s, o: V(
            s.d + (o := V(o) if type(o) != V else o).d, (s, o), (1, 1)
        ),
        "__mul__": lambda s, o: V(
            s.d * (o := V(o) if type(o) != V else o).d, (s, o), (o.d, s.d)
        ),
        "__pow__": lambda s, o: V(s.d**o, (s,), (o * s.d ** (o - 1),)),
        "__radd__": lambda s, o: s + o,
        "__truediv__": lambda s, o: s * o**-1,
        "backward": lambda s: (
            lambda t, v: (
                (
                    b := lambda n: (
                        n not in v and (v.add(n), [b(c) for c in n._c], t.append(n))
                    )
                ),
                b(s),
                s.__setattr__("g", 1),
                [
                    c.__setattr__("g", c.g + l * n.g)
                    for n in t[::-1]
                    for c, l in zip(n._c, n._l)
                ],
            )
        )([], set()),
    },
)
ex = lambda a: V((e := m.exp(a.d)), (a,), (e,))
lg = lambda a: V(m.log(a.d), (a,), (1 / a.d,))
rl = lambda a: V(max(0, a.d), (a,), (a.d > 0,))
E = W = 16
H = R = 4
mx = lambda a, b: [[V(r.gauss(0, 0.08)) for _ in range(b)] for _ in range(a)]
S = {
    "e": mx(Z, E),
    "p": mx(W, E),
    "h": mx(Z, E),
    **{k: mx(E, E) for k in "qkvo"},
    "f": mx(4 * E, E),
    "b": mx(E, 4 * E),
}
P = [p for w in S.values() for o in w for p in o]
lr, b1, b2, ea = 0.01, 0.85, 0.99, 1e-8
M = [0.0] * len(P)
N = M[:]
a = setattr
F = lambda x, w: [sum(a * b for a, b in zip(o, x)) for o in w]
X = lambda g: (
    e := [ex(v + -max(a.d for a in g)) for v in g],
    t := sum(e),
    [i / t for i in e],
)[2]
J = lambda x: (
    c := (sum(a * a for a in x) / len(x) + 1e-5) ** -0.5,
    [a * c for a in x],
)[1]


def G(i, p, K, C):
    x = J([a + b for a, b in zip(S["e"][i], S["p"][p])])
    y = x
    x = J(x)
    q, k, v = F(x, S["q"]), F(x, S["k"]), F(x, S["v"])
    K.append(k)
    C.append(v)
    a = sum(
        [
            (
                lambda q, k, v: (
                    lambda w: [
                        sum(w[t] * v[t][j] for t in range(len(v))) for j in range(R)
                    ]
                )(X([sum(q[j] * k[t][j] for j in range(R)) / 2 for t in range(len(k))]))
            )(
                q[h * R : (h + 1) * R],
                [i[h * R : (h + 1) * R] for i in K],
                [i[h * R : (h + 1) * R] for i in C],
            )
            for h in range(H)
        ],
        [],
    )
    x = [a + b for a, b in zip(F(a, S["o"]), y)]
    y = x
    x = [rl(a) for a in F(J(x), S["f"])]
    x = [a + b for a, b in zip(F(x, S["b"]), y)]
    return F(x, S["h"])


for s in range(1000):
    t = [B] + [U.index(c) for c in D[s % len(D)]] + [B]
    n = min(W, len(t) - 1)
    K, C = [], []
    L = sum(lg(X(G(t[p], p, K, C))[t[p + 1]]) for p in range(n)) / -n
    L.backward()
    z = lr * (1 - s / 1000)
    [
        (
            M.__setitem__(i, b1 * M[i] + (1 - b1) * p.g),
            N.__setitem__(i, b2 * N[i] + (1 - b2) * p.g**2),
            a(
                p,
                "d",
                p.d
                - z
                * (M[i] / (1 - b1 ** (s + 1)))
                / ((N[i] / (1 - b2 ** (s + 1))) ** 0.5 + ea),
            ),
            a(p, "g", 0),
        )
        for i, p in enumerate(P)
    ]
    print(f"step {s + 1:4d} / 1000 | loss {L.d:.4f}")
print("\n--- inference (new, hallucinated names) ---")
g = lambda K, C, t, p, s: (
    s
    if p >= W
    or (n := r.choices(range(Z), [a.d for a in X([l * 2 for l in G(t, p, K, C)])])[0])
    == B
    else g(K, C, n, p + 1, s + [U[n]])
)
for i in range(20):
    print(f"sample {i + 1:2d}: {''.join(g([], [], B, 0, []))}")
