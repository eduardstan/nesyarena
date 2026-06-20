"""NeSyArena MVP runner: gates, inference experiments (E1-E4), the H8
learning experiment (E6) with disjoint control vs overlap treatment,
scorecard, and RESULTS.md. Run:  python3 -m experiments.run_all"""

from __future__ import annotations
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nesyarena.oracle import wmc, wmc_with_grad
from nesyarena.provenances import ExactWMC, AddMult, TopK, MinMax, LSE, registry
from nesyarena.arena import (iterate, ground_tc, overlap_family, chain,
                             sweep_overlap, find_witness, problog_value)

OUT = os.environ.get("NESYARENA_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
R: dict = {}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def adam(grad_fn, theta0, steps=4000, lr=0.05):
    th = theta0.copy()
    m = np.zeros_like(th)
    v = np.zeros_like(th)
    for t in range(1, steps + 1):
        loss, g = grad_fn(th)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh, vh = m / (1 - 0.9 ** t), v / (1 - 0.999 ** t)
        th = th - lr * mh / (np.sqrt(vh) + 1e-8)
    return th


# ============================================================ GATE =========

def gate_problog(n_inst=30):
    rng = np.random.default_rng(1)
    diffs = []
    for i in range(n_inst):
        P = int(rng.integers(2, 5))
        L = int(rng.integers(2, 4))
        c = int(rng.integers(0, L))
        het = i >= n_inst - 10
        proofs, probs = overlap_family(P, L, c, p=float(rng.choice([0.3, 0.6, 0.9])),
                                       rng=rng, het=het)
        diffs.append(abs(wmc(proofs, probs) - problog_value(proofs, probs)))
    R["gate_problog_max_diff"] = float(max(diffs))
    print(f"[GATE] reference-WMC vs ProbLog on {n_inst} instances "
          f"(10 heterogeneous): max |diff| = {max(diffs):.2e}")


# ============================================================ E1 ===========

def e1_overlap():
    provs = [AddMult(), TopK(1), TopK(3), MinMax()]
    recs = sweep_overlap(provs, Ps=range(1, 7), cs=range(0, 3), ps=[0.3, 0.6, 0.9], L=3)
    R["e1_records"] = recs

    # F1: error surfaces at p=0.6
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.4), sharey=True)
    sel = [r for r in recs if r["p"] == 0.6]
    vmax = max(abs(r["err"]) for r in sel)
    for ax, pv in zip(axes, provs):
        M = np.full((3, 6), np.nan)
        for r in sel:
            if r["sut"] == pv.name:
                M[r["c"], r["P"] - 1] = r["err"]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
                       origin="lower", extent=[0.5, 6.5, -0.5, 2.5])
        for ci in range(3):
            for Pi in range(6):
                if not np.isnan(M[ci, Pi]):
                    ax.text(Pi + 1, ci, f"{M[ci, Pi]:+.2f}", ha="center",
                            va="center", fontsize=7)
        ax.set_title(pv.name, fontsize=10)
        ax.set_xlabel("proofs P")
    axes[0].set_ylabel("shared facts c (overlap)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="signed semantic error")
    fig.suptitle("F1  Error surfaces (L=3, p=0.6): red over-counts, blue under-counts", y=1.04)
    fig.savefig(f"{OUT}/F1_error_surfaces.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # F2: crossover (P=4, L=4)
    cs = range(0, 4)
    series = {pv.name: [] for pv in provs}
    for c in cs:
        proofs, probs = overlap_family(4, 4, c, 0.6)
        ex = wmc(proofs, probs)
        for pv in provs:
            series[pv.name].append(pv.value(proofs, probs) - ex)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(0, color="#0F6E56", lw=2, label="exact WMC (oracle)")
    style = {"add-mult(clamped)": ("o-", "#D85A30"), "top-1-proofs": ("^--", "#185FA5"),
             "top-3-proofs": ("s--", "#378ADD"), "min-max-prob": ("d:", "#7F77DD")}
    for name, vals in series.items():
        mk, col = style[name]
        ax.plot(list(cs), vals, mk, color=col, label=name)
    ax.set_xlabel("shared facts c (P=4, L=4, p=0.6)")
    ax.set_ylabel("signed semantic error")
    ax.set_title("F2  Opposite structural sensitivities (crossover)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/F2_crossover.png", dpi=160)
    plt.close(fig)

    sat = [r for r in recs if r["p"] == 0.9 and r["sut"] == "add-mult(clamped)" and r["P"] >= 3]
    R["e1_saturation_frac"] = float(np.mean([abs(r["err"] - (1.0 - r["exact"])) < 1e-12
                                             for r in sat]))
    print(f"[E1] {len(recs)} sweep records; saturation regime (p=0.9, P>=3): "
          f"{100*R['e1_saturation_frac']:.0f}% of add-mult cells pinned at value 1.0")


# ============================================================ E2 ===========

def e2_depth():
    om = dict(oplus=max, otimes=lambda a, b: a * b, zero=0.0, one=1.0)
    p = 0.9
    horizons = {}
    for n in (2, 4, 6, 8):
        h = None
        for L in range(1, 13):
            nodes, edges, probs = chain(L, p)
            rules = ground_tc(edges, nodes)
            f0 = {("edge", u, v): probs[(u, v)] for (u, v) in edges}
            hist = iterate(rules, f0, n_steps=n, **om)
            err = hist[n].get(("path", "v0", f"v{L}"), 0.0) - p ** L
            if abs(err) > 0.01 and h is None:
                h = L
        horizons[n] = h
    R["e2_horizons"] = horizons
    ok = all(horizons[n] == n + 1 for n in horizons)
    print(f"[E2] depth horizons h_delta per unroll-n: {horizons}  "
          f"(theory n+1: {'CONFIRMED' if ok else 'VIOLATED'})")

    # F3: value + finite-difference gradient at L=8
    L = 8
    nodes, edges, probs = chain(L, p)
    rules = ground_tc(edges, nodes)
    q = ("path", "v0", f"v{L}")

    def value_at(n, pm):
        f0 = {("edge", u, v): p for (u, v) in edges}
        f0[("edge", "v3", "v4")] = pm
        return iterate(rules, f0, n_steps=n, **om)[n].get(q, 0.0)

    ns = list(range(0, 13))
    vals = [value_at(n, p) for n in ns]
    h = 1e-5
    grads = [(value_at(n, p + h) - value_at(n, p - h)) / (2 * h) for n in ns]
    fig, ax1 = plt.subplots(figsize=(6.8, 4.0))
    ax1.plot(ns, vals, "o-", color="#0F6E56", label="value $I^{(n)}(q)$")
    ax2 = ax1.twinx()
    ax2.plot(ns, np.abs(grads), "s--", color="#D85A30", label="|gradient| (finite diff.)")
    ax1.axvline(L, color="gray", ls=":")
    ax1.set_xlabel("unrolling depth n (chain L=8, p=0.9)")
    ax1.set_ylabel("value", color="#0F6E56")
    ax2.set_ylabel("gradient", color="#D85A30")
    ax1.set_title("F3  Truncation: zero value and identically zero gradient below horizon")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/F3_depth_horizon.png", dpi=160)
    plt.close(fig)


# ============================================================ E3 ===========

def e3_surrogate():
    s = 0.5
    Ps = np.array([1, 2, 4, 8, 16, 32])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    for tau, col in zip((0.05, 0.1, 0.2), ("#185FA5", "#7F77DD", "#D85A30")):
        bias = []
        for P in Ps:
            proofs = [frozenset([f"f{j}"]) for j in range(P)]
            probs = {f"f{j}": s for j in range(P)}
            bias.append(LSE(tau).value(proofs, probs) - s)
        ax.plot(Ps, bias, "o-", color=col, label=f"measured, tau={tau}")
        ax.plot(Ps, tau * np.log(Ps), "k--", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of proofs P (equal scores 0.5)")
    ax.set_ylabel("bias LSE - max")
    ax.set_title("(a) Bias law: dashed = tau*ln(P)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    tg = np.geomspace(0.005, 0.4, 60)
    P, s1, sr = 5, 0.55, 0.5
    bias = [LSE(t).value([frozenset(["a"])] + [frozenset([f"b{j}"]) for j in range(P - 1)],
                         {"a": s1, **{f"b{j}": sr for j in range(P - 1)}}) - s1 for t in tg]
    nonmax = [1 - 1 / (1 + (P - 1) * np.exp(-(s1 - sr) / t)) for t in tg]
    ax.plot(tg, bias, "-", color="#D85A30", label="value bias (want low)")
    ax.plot(tg, nonmax, "-", color="#185FA5", label="gradient share to non-max (want high)")
    ax.set_xscale("log"); ax.set_xlabel("temperature tau")
    ax.set_title("(b) The surrogate dilemma"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/F4_surrogate.png", dpi=160)
    plt.close(fig)
    print("[E3] surrogate figure written (bias law + dilemma)")


# ============================================================ E4 ===========

def e4_witnesses(delta=0.05):
    rows = []
    for pv in [AddMult(), TopK(1), TopK(3), MinMax()]:
        w = find_witness(pv, delta)
        rows.append((pv.name, w))
        print(f"[E4] minimal witness for {pv.name}: {w}")
    R["e4_witnesses"] = [(n, w) for n, w in rows]


# ============================================================ E6 ===========

def _train_treatment(prov, q_true, proofs, train_q, targets, seed):
    facts = sorted(set().union(*proofs))
    rng = np.random.default_rng(seed)
    th0 = -1.0 + 0.1 * rng.standard_normal(len(facts))

    def grad_fn(th):
        p = np.clip(sigmoid(th), 1e-4, 1 - 1e-4)
        probs = dict(zip(facts, p))
        loss, g = 0.0, np.zeros_like(th)
        for idxs, y in zip(train_q, targets):
            sub = [proofs[i] for i in idxs]
            v = float(np.clip(prov.value(sub, probs), 1e-4, 1 - 1e-4))
            loss += -(y * np.log(v) + (1 - y) * np.log(1 - v))
            dLdv = (v - y) / (v * (1 - v))
            gd = prov.grad(sub, probs)
            for k, f in enumerate(facts):
                g[k] += dLdv * gd.get(f, 0.0) * p[k] * (1 - p[k])
        return loss / len(train_q), g / len(train_q)

    th = adam(grad_fn, th0)
    return dict(zip(facts, sigmoid(th)))


def e6_learning(n_seeds=5):
    suts = [ExactWMC(), AddMult(), TopK(1), TopK(3), MinMax()]
    # ---- treatment: G1 overlap (P=4, L=3, c=2), heterogeneous truth ----
    train_q = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 1, 2, 3)]
    held_q = [(0,), (1,), (2,), (3,), (0, 1, 2)]
    treat = {pv.name: dict(trans=[], facts_q=[], fact_mae=[]) for pv in suts}
    for seed in range(n_seeds):
        rng = np.random.default_rng(100 + seed)
        proofs, q_true = overlap_family(4, 3, 2, p=0.6, rng=rng, het=True)
        targets = [wmc([proofs[i] for i in idxs], q_true) for idxs in train_q]
        for pv in suts:
            learned = _train_treatment(pv, q_true, proofs, train_q, targets, seed)
            te, fq = [], []
            for idxs in held_q:
                sub = [proofs[i] for i in idxs]
                truth = wmc(sub, q_true)
                te.append(abs(pv.value(sub, learned) - truth))
                fq.append(abs(wmc(sub, learned) - truth))
            treat[pv.name]["trans"].append(float(np.mean(te)))
            treat[pv.name]["facts_q"].append(float(np.mean(fq)))
            treat[pv.name]["fact_mae"].append(
                float(np.mean([abs(learned[f] - q_true[f]) for f in q_true])))

    # ---- control: categorical sum (MNIST-sum_2 structure, |dom|=3) ----
    def cat_value_grad(sut_name, terms, pA, pB, k=None):
        vals = np.array([pA[i] * pB[j] for (i, j) in terms])
        gA, gB = np.zeros(3), np.zeros(3)
        if sut_name in ("exact-wmc", "add-mult(clamped)"):
            v = float(vals.sum())
            for (i, j), tv in zip(terms, vals):
                gA[i] += pB[j]; gB[j] += pA[i]
        elif sut_name.startswith("top-"):
            order = np.argsort(-vals)[:k]
            v = float(vals[order].sum())
            for o in order:
                i, j = terms[o]
                gA[i] += pB[j]; gB[j] += pA[i]
        elif sut_name == "min-max-prob":
            mins = [min(pA[i], pB[j]) for (i, j) in terms]
            o = int(np.argmax(mins)); i, j = terms[o]
            v = float(mins[o])
            if pA[i] <= pB[j]:
                gA[i] = 1.0
            else:
                gB[j] = 1.0
        return v, gA, gB

    ctrl = {pv.name: dict(trans=[]) for pv in suts}
    sums = {s: [(i, s - i) for i in range(3) if 0 <= s - i <= 2] for s in range(5)}
    diag = [(0, 0), (1, 1), (2, 2)]
    for seed in range(n_seeds):
        rng = np.random.default_rng(200 + seed)
        tA, tB = rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3))
        targets = {s: sum(tA[i] * tB[j] for (i, j) in sums[s]) for s in range(5)}
        for pv in suts:
            k = pv.k if isinstance(pv, TopK) else None

            def grad_fn(th):
                zA, zB = th[:3], th[3:]
                eA = np.exp(zA - zA.max()); pA = eA / eA.sum()
                eB = np.exp(zB - zB.max()); pB = eB / eB.sum()
                loss = 0.0
                gpA, gpB = np.zeros(3), np.zeros(3)
                for s in range(5):
                    v, gA, gB = cat_value_grad(pv.name, sums[s], pA, pB, k)
                    v = float(np.clip(v, 1e-4, 1 - 1e-4))
                    y = targets[s]
                    loss += -(y * np.log(v) + (1 - y) * np.log(1 - v))
                    d = (v - y) / (v * (1 - v))
                    gpA += d * gA; gpB += d * gB
                JA = np.diag(pA) - np.outer(pA, pA)
                JB = np.diag(pB) - np.outer(pB, pB)
                return loss / 5, np.concatenate([JA @ gpA, JB @ gpB]) / 5

            th = adam(grad_fn, 0.1 * np.random.default_rng(seed).standard_normal(6))
            eA = np.exp(th[:3] - th[:3].max()); pA = eA / eA.sum()
            eB = np.exp(th[3:] - th[3:].max()); pB = eB / eB.sum()
            v, _, _ = cat_value_grad(pv.name, diag, pA, pB, k)
            truth = sum(tA[i] * tB[i] for i in range(3))
            ctrl[pv.name]["trans"].append(abs(v - truth))

    R["e6_treatment"] = {n: dict(trans_mean=float(np.mean(d["trans"])),
                                 trans_std=float(np.std(d["trans"])),
                                 facts_q=float(np.mean(d["facts_q"])),
                                 fact_mae=float(np.mean(d["fact_mae"])))
                         for n, d in treat.items()}
    R["e6_control"] = {n: dict(trans_mean=float(np.mean(d["trans"])),
                               trans_std=float(np.std(d["trans"])))
                       for n, d in ctrl.items()}

    # F6 grouped bars
    names = [pv.name for pv in suts]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    x = np.arange(len(names)); w = 0.38
    cm = [R["e6_control"][n]["trans_mean"] for n in names]
    cs_ = [R["e6_control"][n]["trans_std"] for n in names]
    tm = [R["e6_treatment"][n]["trans_mean"] for n in names]
    ts = [R["e6_treatment"][n]["trans_std"] for n in names]
    ax.bar(x - w / 2, cm, w, yerr=cs_, color="#9FE1CB", label="control: disjoint task (MNIST-sum structure)")
    ax.bar(x + w / 2, tm, w, yerr=ts, color="#F0997B", label="treatment: overlap task (G1)")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("held-out query error after training (mean over 5 seeds)")
    ax.set_title("F6  Training through a misreasoner corrupts transferred knowledge\n"
                 "-- except on the disjoint task, where the default benchmark lives")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/F6_learning_transfer.png", dpi=160)
    plt.close(fig)
    print("[E6] treatment:", {n: f"{d['trans_mean']:.3f}+-{d['trans_std']:.3f}"
                              for n, d in R["e6_treatment"].items()})
    print("[E6] control:  ", {n: f"{d['trans_mean']:.3f}+-{d['trans_std']:.3f}"
                              for n, d in R["e6_control"].items()})


# ======================================================= SCORECARD =========

def scorecard():
    suts = ["exact-wmc", "add-mult(clamped)", "top-1-proofs", "top-3-proofs", "min-max-prob"]
    recs = R["e1_records"]

    def fid(name, sel):
        errs = [abs(r["err"]) for r in sel if r["sut"] == name]
        return max(0.0, 1.0 - float(np.mean(errs))) if errs else 1.0

    axes_def = {
        "multiplicity\n(c=0, p=0.6)": [r for r in recs if r["c"] == 0 and r["p"] == 0.6],
        "overlap\n(P=4, p=0.6)": [r for r in recs if r["P"] == 4 and r["p"] == 0.6],
        "saturation\n(p=0.9)": [r for r in recs if r["p"] == 0.9 and r["P"] >= 3],
    }
    prof = {s: [1.0 if s == "exact-wmc" else fid(s, sel) for sel in axes_def.values()]
            for s in suts}
    for s in suts:
        prof[s].append(max(0.0, 1.0 - R["e6_treatment"][s]["trans_mean"]))
    labels = list(axes_def.keys()) + ["learning\ntransfer (E6)"]
    R["scorecard"] = {s: dict(zip([l.replace("\n", " ") for l in labels], prof[s]))
                      for s in suts}

    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(6.4, 5.6), subplot_kw=dict(polar=True))
    cols = {"exact-wmc": "#0F6E56", "add-mult(clamped)": "#D85A30",
            "top-1-proofs": "#185FA5", "top-3-proofs": "#378ADD", "min-max-prob": "#7F77DD"}
    for s in suts:
        vals = prof[s] + prof[s][:1]
        ax.plot(ang + ang[:1], vals, "o-", lw=1.6, ms=3, color=cols[s], label=s)
        ax.fill(ang + ang[:1], vals, color=cols[s], alpha=0.06)
    ax.set_xticks(ang); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.0); ax.set_title("NeSyArena scorecard (measured fidelity profiles)", pad=18)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.1), fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/scorecard_radar.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[SCORECARD] written")


def write_results():
    lines = ["# NeSyArena MVP — measured results (auto-generated)\n"]
    lines.append(f"**Gate:** reference WMC vs ProbLog max diff = "
                 f"{R['gate_problog_max_diff']:.2e} on 30 instances (10 heterogeneous).\n")
    lines.append(f"**E2 depth horizons** (unroll n -> h): {R['e2_horizons']} — theory n+1.\n")
    lines.append("## E4 minimal witnesses (|error| > 0.05, shrunk)\n")
    lines.append("| SUT | P | L | c | p | facts | signed error |")
    lines.append("|---|---|---|---|---|---|---|")
    for n, w in R["e4_witnesses"]:
        lines.append(f"| {n} | {w['P']} | {w['L']} | {w['c']} | {w['p']} | {w['m']} | {w['err']:+.3f} |")
    lines.append("\n## E6 learning (H8): held-out query error after training, 5 seeds\n")
    lines.append("| SUT | control (disjoint) | treatment (overlap) | facts-quality (treat) | fact MAE (treat) |")
    lines.append("|---|---|---|---|---|")
    for n in R["e6_treatment"]:
        c, t = R["e6_control"][n], R["e6_treatment"][n]
        lines.append(f"| {n} | {c['trans_mean']:.3f} ± {c['trans_std']:.3f} | "
                     f"{t['trans_mean']:.3f} ± {t['trans_std']:.3f} | "
                     f"{t['facts_q']:.3f} | {t['fact_mae']:.3f} |")
    lines.append("\n## Scorecard (fidelity = 1 − mean |semantic error|)\n")
    axes_l = list(next(iter(R["scorecard"].values())).keys())
    lines.append("| SUT | " + " | ".join(axes_l) + " |")
    lines.append("|---|" + "---|" * len(axes_l))
    for s, d in R["scorecard"].items():
        lines.append(f"| {s} | " + " | ".join(f"{d[a]:.3f}" for a in axes_l) + " |")
    with open(f"{OUT}/RESULTS.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(f"{OUT}/results.json", "w") as fh:
        json.dump(R, fh, indent=1, default=str)
    print("[REPORT] RESULTS.md + results.json written")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    gate_problog()
    e1_overlap()
    e2_depth()
    e3_surrogate()
    e4_witnesses()
    e6_learning()
    scorecard()
    write_results()
    print("\nDONE.")
