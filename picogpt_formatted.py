import math as m, random as r

r.seed(42)
D = open("input.txt").read().split()
r.shuffle(D)
U = sorted({*"".join(D)})
B = len(U)
Z = B + 1


class V:
    def __init__(s, d, c=(), l=()):
        s.d = d
        s.g = 0
        s.c = c
        s.l = l

    __add__ = lambda s, o: V(s.d + (o := Y(o)).d, (s, o), (1, 1))
    __mul__ = lambda s, o: V(s.d * (o := Y(o)).d, (s, o), (o.d, s.d))
    __pow__ = lambda s, o: V(s.d**o, (s,), (o * s.d ** (o - 1),))
    __radd__ = lambda s, o: s + o
    __truediv__ = lambda s, o: s * o**-1


Y = lambda o: [V(o), o][type(o) == V]


def O(s):
    t = []
    v = set()

    def b(n):
        if n not in v:
            v.add(n)
            [b(c) for c in n.c]
            t.append(n)

    b(s)
    s.g = 1
    for n in t[::-1]:
        for c, l in zip(n.c, n.l):
            c.g += l * n.g


Q = range
T = sum
I = len
E = 16
R = 4
A = lambda x, y: [a + b for a, b in zip(x, y)]

S = [
    [[V(r.gauss(0, 0.08)) for _ in Q(c)] for _ in Q(w)]
    for w, c in zip(
        [Z, E, Z] + [E] * 4 + [4 * E, E],
        [E] * 8 + [4 * E],
    )
]
P = [p for w in S for o in w for p in o]
M = [0.0] * I(P)
N = M[:]

F = lambda x, w: [T(a * b for a, b in zip(o, x)) for o in w]
J = lambda x: (c := (T(a * a for a in x) / I(x) + 1e-5) ** -0.5, [a * c for a in x])[1]
X = lambda g: (
    e := [V((z := m.exp((a := v + -max(c.d for c in g)).d)), (a,), (z,)) for v in g],
    t := T(e),
    [i / t for i in e],
)[2]


def G(i, p, K, C):
    x = J(A(S[0][i], S[1][p]))
    x = J(y := x)
    q, k, v = F(x, S[3]), F(x, S[4]), F(x, S[5])
    K += (k,)
    C += (v,)
    a = []
    for h in Q(R):
        e = h * R
        u = q[e : e + R]
        w = X([T(u[j] * K[t][e + j] for j in Q(R)) / 2 for t in Q(I(K))])
        a += [T(w[t] * C[t][e + j] for t in Q(I(C))) for j in Q(R)]
    x = A(F(a, S[6]), y)
    y = x
    x = [V(max(0, a.d), (a,), (a.d > 0,)) for a in F(J(x), S[7])]
    return F(A(F(x, S[8]), y), S[2])


for s in Q(1000):
    t = [B, *map(U.index, D[s % I(D)]), B]
    n = min(E, I(t) - 1)
    K, C = [], []
    L = T(V(m.log((q := X(G(t[p], p, K, C))[t[p + 1]]).d), (q,), (1 / q.d,)) for p in Q(n)) / -n
    O(L)
    z = 0.01 * (1 - s / 1000)
    for i, p in enumerate(P):
        M[i] = 0.85 * M[i] + 0.15 * p.g
        N[i] = 0.99 * N[i] + 0.01 * p.g**2
        p.d -= z * (M[i] / (1 - 0.85 ** (s + 1))) / ((N[i] / (1 - 0.99 ** (s + 1))) ** 0.5 + 1e-8)
        p.g = 0
    print(f"step {s+1:4d} / 1000 | loss {L.d:.4f}")

print("\n--- inference (new, hallucinated names) ---")
for i in Q(20):
    K, C = [], []
    t = B
    s = ""
    for p in Q(E):
        t = r.choices(Q(Z), [a.d for a in X([l * 2 for l in G(t, p, K, C)])])[0]
        if t == B:
            break
        s += U[t]
    print(f"sample {i+1:2d}: {s}")
