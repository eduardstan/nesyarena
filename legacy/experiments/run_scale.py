"""NeSyArena scale experiments (protocol days 22-38).

Upgrades E6 to real perception: 8x8 images -> MLP -> fact probabilities ->
provenance -> BCE on sampled binary labels, trained end-to-end with manual
backprop chained through each provenance's own gradient semantics.

The generative model is known, so Bayes-optimal posteriors are closed-form:
  x = z * alpha * s + eps,  eps ~ N(0, I)  =>
  P(z=1|x) = sigmoid(alpha * s.x - alpha^2/2 + logit(pi))
which makes perception calibration *exactly* measurable.

Outputs: F7 (accuracy ties, fidelity diverges), F8 (depth-learning starvation),
gradient-liveness numbers, stratified-negation demo, results_scale.json.
"""

from __future__ import annotations
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nesyarena.oracle import wmc, wmc_with_grad
from nesyarena.arena import overlap_family, iterate, ground_tc, chain

OUT = os.environ.get("NESYARENA_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
RS: dict = {}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ------------------------------------------------------------ batch WMC ----

class BatchStructure:
    """Fixed proof structure over m facts; vectorized WMC value+grad over a
    batch of per-sample probability vectors. grad_f = (swp@bits - p*val)/p(1-p)."""

    def __init__(self, proofs, facts):
        self.facts = list(facts)
        m = len(facts)
        idx = {f: i for i, f in enumerate(facts)}
        self.pidx = [np.array(sorted(idx[f] for f in pr)) for pr in proofs]
        W = 1 << m
        self.bits = ((np.arange(W)[:, None] >> np.arange(m)) & 1).astype(bool)
        sat = np.zeros(W, bool)
        for pi in self.pidx:
            sat |= self.bits[:, pi].all(axis=1)
        self.sat = sat.astype(float)
        self.m = m

    def truth(self, Z):
        """Query truth on boolean states Z (B,m)."""
        out = np.zeros(Z.shape[0], bool)
        for pi in self.pidx:
            out |= Z[:, pi].all(axis=1)
        return out

    def wmc_batch(self, Pb):
        wp = np.prod(np.where(self.bits[None], Pb[:, None, :], 1 - Pb[:, None, :]), axis=2)
        swp = wp * self.sat[None]
        val = swp.sum(1)
        denom = np.clip(Pb * (1 - Pb), 1e-9, None)
        grad = (swp @ self.bits.astype(float) - Pb * val[:, None]) / denom
        return val, grad

    def scores(self, Pb):
        return np.stack([np.prod(Pb[:, pi], axis=1) for pi in self.pidx], axis=1)


# ----------------------------------------------------- batched provenances --

def prov_value_grad(name, S: BatchStructure, Pb, k=None):
    """Returns (val (B,), dval/dPb (B,m)) under each provenance's own
    differentiation semantics (clamp flat region, frozen top-k selection,
    one-hot min-max subgradient)."""
    B, m = Pb.shape
    if name == "exact":
        return S.wmc_batch(Pb)
    if name == "addmult":
        sc = S.scores(Pb)
        raw = sc.sum(1)
        val = np.minimum(1.0, raw)
        grad = np.zeros_like(Pb)
        for j, pi in enumerate(S.pidx):
            grad[:, pi] += (sc[:, j] / np.clip(Pb[:, pi].T, 1e-9, None)).T
        grad[raw >= 1.0] = 0.0  # clamp flat region
        return val, grad
    if name.startswith("top"):
        sc = S.scores(Pb)
        order = np.argsort(-sc, axis=1)[:, :k]
        val = np.zeros(B)
        grad = np.zeros_like(Pb)
        for b in range(B):
            sel = [frozenset(S.facts[i] for i in S.pidx[j]) for j in order[b]]
            probs = {f: float(Pb[b, i]) for i, f in enumerate(S.facts)}
            v, g = wmc_with_grad(sel, probs)
            val[b] = v
            for i, f in enumerate(S.facts):
                grad[b, i] = g.get(f, 0.0)
        return val, grad
    if name == "minmax":
        val = np.zeros(B)
        grad = np.zeros_like(Pb)
        for b in range(B):
            mins = [(Pb[b, pi].min(), pi[np.argmin(Pb[b, pi])]) for pi in S.pidx]
            j = int(np.argmax([mv for mv, _ in mins]))
            val[b] = mins[j][0]
            grad[b, mins[j][1]] = 1.0
        return val, grad
    raise ValueError(name)


# ------------------------------------------------------------------- MLP ----

class MLP:
    def __init__(self, din, dh, dout, rng):
        self.W1 = rng.standard_normal((din, dh)) * 0.15
        self.b1 = np.zeros(dh)
        self.W2 = rng.standard_normal((dh, dout)) * 0.15
        self.b2 = np.zeros(dout)
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.mom = [np.zeros_like(p) for p in self.params]
        self.vel = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def forward(self, X):
        self.X = X
        self.h = np.tanh(X @ self.W1 + self.b1)
        return self.h @ self.W2 + self.b2  # logits

    def backward(self, dlogits):
        gW2 = self.h.T @ dlogits
        gb2 = dlogits.sum(0)
        dh = (dlogits @ self.W2.T) * (1 - self.h ** 2)
        gW1 = self.X.T @ dh
        gb1 = dh.sum(0)
        return [gW1, gb1, gW2, gb2]

    def adam_step(self, grads, lr=1e-3):
        self.t += 1
        for p, g, mo, ve in zip(self.params, grads, self.mom, self.vel):
            mo *= 0.9; mo += 0.1 * g
            ve *= 0.999; ve += 0.001 * g * g
            mh = mo / (1 - 0.9 ** self.t)
            vh = ve / (1 - 0.999 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)


# ----------------------------------------------- treatment (Synth-path) ----

ALPHA, PI = 1.2, 0.55
DIN, NCELL = 64, 6


def gen_treatment(B, rng, sig):
    Z = (rng.random((B, NCELL)) < PI)
    X = Z[:, :, None] * ALPHA * sig[None, None, :] + rng.standard_normal((B, NCELL, DIN))
    bayes = sigmoid(ALPHA * (X @ sig) - ALPHA ** 2 / 2 + np.log(PI / (1 - PI)))
    return X, Z, bayes


def run_treatment(seeds=(0, 1, 2), epochs=20, N=4000, B=64):
    proofs, _ = overlap_family(4, 3, 2, 0.6)
    facts = sorted(set().union(*proofs))
    S_full = BatchStructure(proofs, facts)
    held = [BatchStructure([proofs[i] for i in idxs], facts)
            for idxs in [(0,), (1,), (2,), (3,), (0, 1, 2)]]
    suts = [("exact", None), ("addmult", None), ("top1", 1), ("top3", 3), ("minmax", None)]
    res = {n: dict(acc=[], cal=[], trans=[], fq=[]) for n, _ in suts}
    for seed in seeds:
        rng = np.random.default_rng(300 + seed)
        sig = rng.standard_normal(DIN); sig /= np.linalg.norm(sig)
        Xtr, Ztr, _ = gen_treatment(N, rng, sig)
        ytr = S_full.truth(Ztr).astype(float)
        Xte, Zte, bay_te = gen_treatment(1500, rng, sig)
        yte = S_full.truth(Zte).astype(float)
        for name, k in suts:
            net = MLP(DIN, 16, 1, np.random.default_rng(1000 + seed))
            steps = epochs * N // B
            for t in range(steps):
                ix = np.random.default_rng(t * 7 + seed).integers(0, N, B)
                xb = Xtr[ix].reshape(B * NCELL, DIN)
                logits = net.forward(xb)
                p = np.clip(sigmoid(logits).reshape(B, NCELL), 1e-4, 1 - 1e-4)
                v, dvdp = prov_value_grad(name, S_full, p, k)
                v = np.clip(v, 1e-4, 1 - 1e-4)
                dLdv = (v - ytr[ix]) / (v * (1 - v)) / B
                dLdp = dLdv[:, None] * dvdp
                dlogits = (dLdp * p * (1 - p)).reshape(B * NCELL, 1)
                net.adam_step(net.backward(dlogits), lr=2e-3)
            # evaluation
            pte = sigmoid(net.forward(Xte.reshape(-1, DIN))).reshape(-1, NCELL)
            pte = np.clip(pte, 1e-4, 1 - 1e-4)
            v_full, _ = prov_value_grad(name, S_full, pte, k)
            res[name]["acc"].append(float(np.mean((v_full >= 0.5) == (yte >= 0.5))))
            res[name]["cal"].append(float(np.mean(np.abs(pte - bay_te))))
            te, fq = [], []
            for H in held:
                truth, _ = H.wmc_batch(np.clip(bay_te, 1e-4, 1 - 1e-4))
                vh, _ = prov_value_grad(name, H, pte, k)
                wq, _ = H.wmc_batch(pte)
                te.append(float(np.mean(np.abs(vh - truth))))
                fq.append(float(np.mean(np.abs(wq - truth))))
            res[name]["trans"].append(float(np.mean(te)))
            res[name]["fq"].append(float(np.mean(fq)))
        print(f"  [treatment seed {seed}] done")
    RS["treatment"] = {n: {m: [float(np.mean(v)), float(np.std(v))]
                           for m, v in d.items()} for n, d in res.items()}


# -------------------------------------------------- control (Synth-sum) ----

def run_control(seeds=(0, 1, 2), epochs=20, N=4000, B=64):
    """Two cells, 3 classes each; train ONLY on query [zA+zB==2] (sum structure,
    mutually exclusive proofs). Transfer query: [zA==zB]."""
    pA_true = np.array([0.5, 0.3, 0.2])
    pB_true = np.array([0.3, 0.4, 0.3])
    terms_tr = [(0, 2), (1, 1), (2, 0)]
    terms_he = [(0, 0), (1, 1), (2, 2)]
    suts = ["exact", "addmult", "top1", "minmax"]

    def cat_vg(name, terms, PA, PB):
        vals = np.stack([PA[:, i] * PB[:, j] for (i, j) in terms], 1)
        B_ = PA.shape[0]
        gA = np.zeros((B_, 3)); gB = np.zeros((B_, 3))
        if name in ("exact", "addmult"):
            v = vals.sum(1)
            for (i, j) in terms:
                gA[:, i] += PB[:, j]; gB[:, j] += PA[:, i]
        elif name == "top1":
            o = np.argmax(vals, 1); v = vals[np.arange(B_), o]
            for t, (i, j) in enumerate(terms):
                m = o == t
                gA[m, i] += PB[m, j]; gB[m, j] += PA[m, i]
        elif name == "minmax":
            mins = np.stack([np.minimum(PA[:, i], PB[:, j]) for (i, j) in terms], 1)
            o = np.argmax(mins, 1); v = mins[np.arange(B_), o]
            for t, (i, j) in enumerate(terms):
                m = o == t
                aA = m & (PA[:, i] <= PB[:, j]); gA[aA, i] = 1.0
                aB = m & ~ (PA[:, i] <= PB[:, j]); gB[aB, j] = 1.0
        return v, gA, gB

    res = {n: dict(acc=[], cal=[], trans=[]) for n in suts}
    for seed in seeds:
        rng = np.random.default_rng(400 + seed)
        sigs = np.linalg.qr(rng.standard_normal((DIN, 3)))[0].T  # 3 orthonormal patterns
        def gen(M):
            zA = rng.choice(3, M, p=pA_true); zB = rng.choice(3, M, p=pB_true)
            XA = ALPHA * sigs[zA] + rng.standard_normal((M, DIN))
            XB = ALPHA * sigs[zB] + rng.standard_normal((M, DIN))
            def bayes(X, marg):
                lo = ALPHA * (X @ sigs.T) - ALPHA ** 2 / 2 + np.log(marg)
                e = np.exp(lo - lo.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
            return XA, XB, zA, zB, bayes(XA, pA_true), bayes(XB, pB_true)
        XA, XB, zA, zB, _, _ = gen(N)
        ytr = (zA + zB == 2).astype(float)
        XAte, XBte, zAte, zBte, bayA, bayB = gen(1500)
        yte = (zAte + zBte == 2).astype(float)
        for name in suts:
            net = MLP(DIN, 16, 3, np.random.default_rng(2000 + seed))
            steps = epochs * N // B
            for t in range(steps):
                ix = np.random.default_rng(t * 11 + seed).integers(0, N, B)
                la = net.forward(XA[ix]); ea = np.exp(la - la.max(1, keepdims=True))
                PA = np.clip(ea / ea.sum(1, keepdims=True), 1e-4, 1)
                gradsA_logits_stash = (net.X, net.h)
                lb = net.forward(XB[ix]); eb = np.exp(lb - lb.max(1, keepdims=True))
                PB = np.clip(eb / eb.sum(1, keepdims=True), 1e-4, 1)
                v, gA, gB = cat_vg(name, terms_tr, PA, PB)
                v = np.clip(v, 1e-4, 1 - 1e-4)
                d = ((v - ytr[ix]) / (v * (1 - v)) / B)[:, None]
                dlB = (gB * d - (np.sum(gB * d * PB, 1, keepdims=True))) * PB
                gB_params = net.backward(dlB)
                net.X, net.h = gradsA_logits_stash
                dlA = (gA * d - (np.sum(gA * d * PA, 1, keepdims=True))) * PA
                gA_params = net.backward(dlA)
                net.adam_step([a + b for a, b in zip(gA_params, gB_params)], lr=2e-3)
            def post(X):
                l = net.forward(X); e = np.exp(l - l.max(1, keepdims=True))
                return np.clip(e / e.sum(1, keepdims=True), 1e-4, 1)
            PAh, PBh = post(XAte), post(XBte)
            v, _, _ = cat_vg(name, terms_tr, PAh, PBh)
            res[name]["acc"].append(float(np.mean((v >= 0.5) == (yte >= 0.5))))
            res[name]["cal"].append(float(np.mean(np.abs(PAh - bayA)) / 2 +
                                          np.mean(np.abs(PBh - bayB)) / 2))
            vh, _, _ = cat_vg(name, terms_he, PAh, PBh)
            truth = (bayA * bayB).sum(1)
            res[name]["trans"].append(float(np.mean(np.abs(vh - truth))))
        print(f"  [control seed {seed}] done")
    RS["control"] = {n: {m: [float(np.mean(v)), float(np.std(v))]
                         for m, v in d.items()} for n, d in res.items()}


# --------------------------------------------------- E7: depth learning ----

def run_e7(seeds=(0, 1, 2), epochs=20, N=3000, B=64):
    """Chain of 6 cells; y = all open (single proof, depth 6). Convergent
    reasoner: v = prod p. Truncated unroll n=4: v == 0 identically (Remark 6),
    so the gradient is zero and the model cannot learn -- measured as AUC."""
    L = 6

    def auc(scores, y):
        o = np.argsort(scores); r = np.empty_like(o, float); r[o] = np.arange(len(scores))
        pos = y > 0.5
        if pos.all() or (~pos).all():
            return 0.5
        return float((r[pos].mean() - (pos.sum() - 1) / 2) / (~pos).sum())

    curves = {"convergent (fixed point)": [], "truncated (n=4 < depth 6)": []}
    for seed in seeds:
        rng = np.random.default_rng(500 + seed)
        sig = rng.standard_normal(DIN); sig /= np.linalg.norm(sig)
        Z = rng.random((N, L)) < 0.75
        X = Z[:, :, None] * ALPHA * sig[None, None, :] + rng.standard_normal((N, L, DIN))
        y = Z.all(1).astype(float)
        Xte_Z = rng.random((1200, L)) < 0.75
        Xte = Xte_Z[:, :, None] * ALPHA * sig[None, None, :] + rng.standard_normal((1200, L, DIN))
        yte = Xte_Z.all(1).astype(float)
        for mode in curves:
            net = MLP(DIN, 16, 1, np.random.default_rng(3000 + seed))
            hist = []
            for ep in range(epochs):
                for t in range(N // B):
                    ix = np.random.default_rng(ep * 97 + t).integers(0, N, B)
                    p = np.clip(sigmoid(net.forward(X[ix].reshape(-1, DIN))).reshape(B, L),
                                1e-4, 1 - 1e-4)
                    if mode.startswith("convergent"):
                        v = np.clip(p.prod(1), 1e-4, 1 - 1e-4)
                        dvdp = v[:, None] / p
                    else:
                        v = np.full(B, 1e-4)      # truncation: query unprovable at n=4
                        dvdp = np.zeros_like(p)   # identically zero gradient
                    dLdv = (v - y[ix]) / (v * (1 - v)) / B
                    dl = (dLdv[:, None] * dvdp * p * (1 - p)).reshape(-1, 1)
                    net.adam_step(net.backward(dl), lr=2e-3)
                pte = sigmoid(net.forward(Xte.reshape(-1, DIN))).reshape(-1, L)
                hist.append(auc(pte.prod(1), yte))
            curves[mode].append(hist)
    RS["e7"] = {m: [float(np.mean([h[-1] for h in cs])), float(np.std([h[-1] for h in cs]))]
                for m, cs in curves.items()}
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for mode, col in zip(curves, ("#0F6E56", "#D85A30")):
        H = np.array(curves[mode])
        ax.plot(range(1, H.shape[1] + 1), H.mean(0), "-o", ms=3, color=col, label=mode)
        ax.fill_between(range(1, H.shape[1] + 1), H.mean(0) - H.std(0),
                        H.mean(0) + H.std(0), color=col, alpha=0.15)
    ax.set_xlabel("epoch"); ax.set_ylabel("test AUC (chain query, depth 6)")
    ax.set_title("F8  Gradient starvation under truncation: the model never learns")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/F8_depth_learning.png", dpi=160); plt.close(fig)
    print(f"[E7] final AUC: {RS['e7']}")


# --------------------------------------- gradient liveness (radar axis) ----

def gradient_liveness():
    from nesyarena.provenances import ExactWMC, AddMult, TopK, MinMax
    provs = [("exact-wmc", ExactWMC()), ("add-mult(clamped)", AddMult()),
             ("top-1-proofs", TopK(1)), ("top-3-proofs", TopK(3)),
             ("min-max-prob", MinMax())]
    cover = {n: [] for n, _ in provs}
    for p in (0.3, 0.6, 0.9):
        for c in (0, 1, 2):
            for P in range(1, 7):
                if c + P * (3 - c) > 22:
                    continue
                proofs, probs = overlap_family(P, 3, c, p)
                _, ge = wmc_with_grad(proofs, probs)
                live_e = {f for f, g in ge.items() if abs(g) > 1e-12}
                for n, pv in provs:
                    g = pv.grad(proofs, probs)
                    live = sum(1 for f in live_e if abs(g.get(f, 0.0)) > 1e-12)
                    cover[n].append(live / max(1, len(live_e)))
    RS["gradient_liveness"] = {n: float(np.mean(v)) for n, v in cover.items()}
    print(f"[LIVENESS] {RS['gradient_liveness']}")


# ------------------------------------------------ stratified negation ------

def negation_demo():
    nodes = ["h", "a", "b", "c", "d", "t"]
    edges = [("h", "a"), ("a", "b"), ("c", "d"), ("d", "t"), ("a", "c"), ("s", "c")]
    nodes = sorted({u for u, v in edges} | {v for u, v in edges})
    # stratum 1: unsafe = reachable from hazard h (Boolean TC)
    f0 = {("edge", u, v): 1.0 for (u, v) in edges}
    hist = iterate(ground_tc(edges, nodes), f0, oplus=max, otimes=min,
                   zero=0.0, one=1.0, n_steps=12)
    unsafe = {"h"} | {w for w in nodes if hist[-1].get(("path", "h", w), 0.0) >= 1.0}
    # stratum 2: safe-path TC over edges into safe targets only
    edges2 = [(u, v) for (u, v) in edges if v not in unsafe and u not in unsafe]
    f02 = {("edge", u, v): 1.0 for (u, v) in edges2}
    hist2 = iterate(ground_tc(edges2, nodes), f02, oplus=max, otimes=min,
                    zero=0.0, one=1.0, n_steps=12)
    got = hist2[-1].get(("path", "s", "t"), 0.0)
    # reference: BFS on the filtered graph
    frontier, seen = {"s"}, {"s"}
    while frontier:
        nxt = {v for (u, v) in edges2 for f in [u] if u in frontier} - seen
        seen |= nxt; frontier = nxt
    ref = 1.0 if "t" in seen else 0.0
    # unstratified: win(x) <- move(x,y), not win(y) on a 2-cycle -> oscillation
    move = {("x", "y"), ("y", "x")}
    I = {("win", n): 0.0 for n in ("x", "y")}
    osc = []
    for _ in range(6):
        I = {("win", u): max((1.0 - I[("win", v)]) for (a, v) in move if a == u)
             for u in ("x", "y")}
        osc.append(I[("win", "x")])
    RS["negation"] = dict(stratified_ok=bool(abs(got - ref) < 1e-12),
                          unsafe=sorted(unsafe), oscillation=osc)
    print(f"[NEGATION] stratified per-stratum lfp == BFS reference: {RS['negation']['stratified_ok']}"
          f" | unstratified 2-cycle iterates of win(x): {osc} (period-2, no fixed point)")


# ------------------------------------------------------- gradient check ----

def gradient_check():
    rng = np.random.default_rng(9)
    sig = rng.standard_normal(DIN); sig /= np.linalg.norm(sig)
    proofs, _ = overlap_family(4, 3, 2, 0.6)
    facts = sorted(set().union(*proofs))
    S = BatchStructure(proofs, facts)
    X, Z, _ = gen_treatment(8, rng, sig)
    y = S.truth(Z).astype(float)
    net = MLP(DIN, 4, 1, rng)

    def loss_of(net_):
        p = np.clip(sigmoid(net_.forward(X.reshape(-1, DIN))).reshape(8, NCELL), 1e-4, 1 - 1e-4)
        v = np.clip(S.wmc_batch(p)[0], 1e-4, 1 - 1e-4)
        return float(np.mean(-(y * np.log(v) + (1 - y) * np.log(1 - v))))

    p = np.clip(sigmoid(net.forward(X.reshape(-1, DIN))).reshape(8, NCELL), 1e-4, 1 - 1e-4)
    v, dvdp = S.wmc_batch(p)
    v = np.clip(v, 1e-4, 1 - 1e-4)
    dLdv = (v - y) / (v * (1 - v)) / 8
    dl = (dLdv[:, None] * dvdp * p * (1 - p)).reshape(-1, 1)
    g = net.backward(dl)
    h = 1e-6
    for (pi, i, j) in [(0, 3, 2), (2, 1, 0)]:
        net.params[pi][i, j] += h; lp = loss_of(net)
        net.params[pi][i, j] -= 2 * h; lm = loss_of(net)
        net.params[pi][i, j] += h
        fd = (lp - lm) / (2 * h)
        an = g[pi][i, j]
        assert abs(fd - an) < 1e-4 * max(1, abs(fd)), (fd, an)
    print("[CHECK] analytic backprop through batched WMC == finite differences  OK")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    gradient_check()
    gradient_liveness()
    negation_demo()
    run_e7()
    print("[treatment] training 5 SUTs x 3 seeds with pixel perception ...")
    run_treatment()
    print("[control] training 4 SUTs x 3 seeds ...")
    run_control()
    with open(f"{OUT}/results_scale.json", "w") as fh:
        json.dump(RS, fh, indent=1)
    print(json.dumps(RS["treatment"], indent=1))
    print(json.dumps(RS["control"], indent=1))
    print("DONE.")
