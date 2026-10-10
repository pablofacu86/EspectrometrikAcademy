# -*- coding: utf-8 -*-
"""
Hyperparameter search spaces: library defaults, dataset-based recommendations (with the
equation behind each one) and parsing of user-edited ranges.

Every recommendation is a tuple (values, formula) where `formula` is a short human-readable
equation instantiated with the numbers of the current dataset, shown as a tooltip in the app.
"""
import ast
import math
import numpy as np


# ---------------------------------------------------------------------------
# Data summary used by the formulas
# ---------------------------------------------------------------------------
def resumen_datos(X, y=None, es_clf=True, prop_test=0.2, cv=5):
    """Numbers the recommendations depend on. X = matrix the models will see; y = classes or values."""
    X = np.asarray(X, dtype=float)
    n_tot, p = X.shape
    n_tr = int(round(n_tot * (1 - prop_test)))           # samples available to the search
    n_fit = int(round(n_tr * (cv - 1) / cv)) if isinstance(cv, int) else n_tr - 1   # samples in each CV fit
    d = {"n": n_tot, "p": p, "n_tr": max(n_tr, 2), "n_fit": max(n_fit, 2), "cv": cv, "es_clf": es_clf}
    Xc = X - np.nanmean(X, axis=0)
    d["var_x"] = float(np.nanvar(X))
    if y is not None:
        if es_clf:
            ys = np.asarray(y).astype(str)
            _, cnt = np.unique(ys, return_counts=True)
            d["k"], d["min_c"] = len(cnt), int(cnt.min())
            d["min_c_fit"] = max(1, int(cnt.min() * (1 - prop_test) * (cv - 1) / cv)) if isinstance(cv, int) else int(cnt.min())
        else:
            yv = np.asarray(y, dtype=float)
            ok = np.isfinite(yv)
            yv = yv[ok]
            d["y_mean"], d["y_sd"] = float(np.mean(yv)), float(np.std(yv, ddof=1)) if len(yv) > 1 else 1.0
            try:
                Xo = Xc[ok]
                d["alpha_max"] = float(np.max(np.abs(Xo.T @ (yv - yv.mean()))) / len(yv))
            except Exception:
                d["alpha_max"] = None
    return d


def _ent(x):
    return int(max(1, round(x)))


def _log_grid(centro, decadas, paso=1.0):
    """Values 10^(log10(centro) + k) for k in [-decadas[0], decadas[1]]."""
    base = math.log10(max(centro, 1e-12))
    ks = np.arange(-decadas[0], decadas[1] + 1e-9, paso)
    return [float(f"{10 ** (base + k):.3g}") for k in ks]


def _impares(a, b, tope=None):
    v = [k for k in range(max(1, a), int(b) + 1) if k % 2 == 1]
    if tope:
        v = [k for k in v if k <= tope]
    return v or [3]


def _dedup(vals):
    out = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# Recommendation per algorithm / parameter
#   each returns (values, "equation = value")
# ---------------------------------------------------------------------------
def _n_componentes(d):
    kmax = int(max(3, min(20, d["p"], d["n_fit"] // 5, d["n_fit"] - 2)))
    if not d["es_clf"] or True:
        pass
    cand = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    vals = [c for c in cand if c <= kmax]
    if kmax not in vals:
        vals.append(kmax)
    f = (f"k_max = min(20, p, n_fit/5) = min(20, {d['p']}, {d['n_fit']}/5) = {kmax}  "
         f"→ integers from 1 to {kmax}. (Rule: at least ~5 samples per latent variable; "
         "above the optimum the model fits noise.)")
    return vals, f


def _recomendar(nombre, par, d):
    n, p, nfit, ntr = d["n"], d["p"], d["n_fit"], d["n_tr"]
    es_clf = d["es_clf"]
    if par == "n_components":
        return _n_componentes(d)

    if nombre == "LDA" and par == "solver":
        if p > nfit:
            return ["lsqr", "svd"], f"p = {p} > n_fit = {nfit}: many variables per sample → both solvers are tried (use PCA scores or variable selection if LDA is unstable)."
        return ["svd", "lsqr"], f"p = {p} ≤ n_fit = {nfit}: 'svd' (no covariance inversion) and 'lsqr' are both fine."

    if nombre == "Logistic Regression" and par == "C":
        c0 = max(1e-3, min(1e3, nfit / max(p, 1)))
        return _log_grid(c0, (2, 2)), (f"C₀ = n_fit / p = {nfit} / {p} = {c0:.3g};  grid = C₀ × 10^k, k = −2…+2.  "
                                       "(C is the INVERSE of the regularisation: with many more variables than samples, use smaller C.)")

    if nombre == "Naive Bayes" and par == "var_smoothing":
        return [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-3], ("var_smoothing = fraction of the largest variance added to every variance for stability. "
                                                     "Grid 10⁻⁹…10⁻³ (increase it when some variables have almost no variance).")

    if par == "alpha" and nombre in ("Ridge",):
        c0 = max(p / max(nfit, 1), 1e-2)
        return _log_grid(c0, (3, 2)), (f"α₀ = p / n_fit = {p} / {nfit} = {c0:.3g};  grid = α₀ × 10^k, k = −3…+2.  "
                                       "(The more variables per sample, the more shrinkage is needed.)")

    if par == "alpha" and nombre in ("Lasso", "Elastic Net"):
        am = d.get("alpha_max")
        if am and np.isfinite(am) and am > 0:
            vals = [float(f"{am * f:.3g}") for f in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)]
            return vals, (f"α_max = max_j |x_jᵀ(y − ȳ)| / n = {am:.3g} (smallest α that sets every coefficient to 0);  "
                          "grid = α_max × {10⁻⁴, 10⁻³, 10⁻², 10⁻¹, 1}.")
        return [0.0001, 0.001, 0.01, 0.1, 1], "α_max could not be computed: logarithmic grid 10⁻⁴…1."

    if nombre == "Elastic Net" and par == "l1_ratio":
        return [0.1, 0.3, 0.5, 0.7, 0.9, 0.95], ("l1_ratio: 0 = Ridge, 1 = Lasso. Values close to 0 keep groups of correlated variables "
                                                  "(spectra); values close to 1 select few variables (tables).")

    if nombre in ("Random Forest",) and par == "n_estimators":
        return [100, 200, 300], "More trees never hurt accuracy (they only cost time): 100, 200, 300."

    if nombre == "Random Forest" and par == "max_depth":
        dmax = _ent(math.log2(max(ntr, 4)) * 2)
        return [None, max(3, dmax // 2), dmax, dmax * 2], (f"d_max ≈ 2·log₂(n_train) = 2·log₂({ntr}) ≈ {dmax}; grid = None (full), d_max/2, d_max, 2·d_max.")

    if nombre == "Random Forest" and par == "max_features":
        sq = math.sqrt(p) / p
        lg = math.log2(max(p, 2)) / p
        vals = _dedup(["sqrt", "log2", 0.1, 0.3] if p > 50 else ["sqrt", "log2", 0.3, 0.5])
        return vals, (f"sqrt(p) = {math.sqrt(p):.1f} variables ({sq:.1%} of p); log₂(p) = {math.log2(max(p, 2)):.1f} ({lg:.1%}); "
                      "with hundreds of correlated variables (spectra) small fractions work best, with few variables use 0.3–0.5.")

    if nombre == "Random Forest" and par == "min_samples_leaf":
        return [1, 2, 5], "Larger leaves smooth the model; useful with noisy y."

    if nombre == "Decision Tree" and par == "max_depth":
        dm = _ent(math.log2(max(ntr, 4)))
        return _dedup([None] + list(range(2, dm + 3))), f"d_max ≈ log₂(n_train) + 2 = log₂({ntr}) + 2 ≈ {dm + 2}: deeper trees memorise the samples."

    if nombre == "SVM" and par == "C" and es_clf:
        return [0.1, 1, 10, 100, 1000], "C: penalty for errors. Logarithmic grid 10⁻¹…10³ (data should be scaled)."

    if nombre in ("SVM", "SVM (SVR)") and par == "gamma":
        g0 = 1.0 / max(p * d["var_x"], 1e-12)
        vals = [float(f"{g0 * f:.3g}") for f in (0.1, 0.3, 1, 3, 10)]
        return vals, (f"γ₀ = 1 / (p·Var(X)) = 1 / ({p} × {d['var_x']:.3g}) = {g0:.3g}  (this is sklearn's 'scale');  grid = γ₀ × {{0.1, 0.3, 1, 3, 10}}.")

    if nombre == "SVM (SVR)" and par == "C":
        if d.get("y_sd") is not None:
            c0 = max(abs(d["y_mean"] + 3 * d["y_sd"]), abs(d["y_mean"] - 3 * d["y_sd"]), 1e-3)
            vals = [float(f"{c0 * f:.3g}") for f in (0.01, 0.1, 1, 10)]
            return vals, (f"C₀ = max(|ȳ + 3σ_y|, |ȳ − 3σ_y|) = {c0:.3g} (Cherkassky & Ma, 2004);  grid = C₀ × {{0.01, 0.1, 1, 10}}.")
        return [0.1, 1, 10, 100], "C: penalty; logarithmic grid."

    if nombre == "SVM (SVR)" and par == "epsilon":
        if d.get("y_sd") is not None:
            vals = [float(f"{d['y_sd'] * f:.3g}") for f in (0.01, 0.05, 0.1, 0.2)]
            return vals, (f"ε = fraction of σ_y: σ_y = {d['y_sd']:.3g};  grid = σ_y × {{0.01, 0.05, 0.1, 0.2}}.  (ε should be about the noise of the reference method.)")
        return [0.01, 0.1, 0.5], "ε: tolerance tube."

    if nombre == "KNN" and par == "n_neighbors":
        kmax_cls = d.get("min_c_fit", ntr) if es_clf else ntr
        k0 = _ent(math.sqrt(nfit))
        hi = min(2 * k0 + 1, max(3, kmax_cls), nfit - 1)
        vals = _impares(1 if k0 <= 3 else 3, hi)
        if len(vals) > 7:
            vals = vals[:: max(1, len(vals) // 6)]
        return vals, (f"k₀ ≈ √n_fit = √{nfit} ≈ {k0};  odd values up to 2·k₀+1 = {2 * k0 + 1}"
                      + (f", never above the smallest class size in training ({kmax_cls})" if es_clf else "") + ".")

    if nombre == "XGBoost" and par == "n_estimators":
        return [100, 200, 400], "More boosting rounds with a small learning rate = better but slower: 100, 200, 400."

    if nombre == "XGBoost" and par == "max_depth":
        dm = int(max(2, min(8, math.floor(math.log2(max(ntr / 10, 2))) + 1)))
        return list(range(2, dm + 1)), f"d_max = clip(⌊log₂(n_train/10)⌋ + 1, 2, 8) = {dm}  (few samples → shallow trees).  Grid 2…{dm}."

    if nombre == "XGBoost" and par == "learning_rate":
        return [0.03, 0.1, 0.3], "Small steps + many trees generalise better: 0.03, 0.1, 0.3."

    if nombre.startswith("Neural Network") and par == "hidden_layer_sizes":
        h = int(max(10, min(100, round(2 * math.sqrt(ntr)))))
        return [(h,), (2 * h,), (h, max(5, h // 2))], (f"h = clip(2·√n_train, 10, 100) = clip(2·√{ntr}, 10, 100) = {h};  grid = (h), (2h), (h, h/2).  "
                                                      "(Few samples → few neurons: a network with more weights than samples memorises.)")

    if nombre.startswith("Neural Network") and par == "alpha":
        a0 = max(1e-5, min(1.0, 1.0 / max(ntr, 1)))
        return _log_grid(a0, (1, 2)), f"α₀ = 1 / n_train = 1 / {ntr} = {a0:.3g};  grid = α₀ × 10^k, k = −1…+2  (L2 penalty: stronger with fewer samples)."

    return None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def espacio_recomendado(nombre, grilla_defecto, d):
    """For one algorithm: {param: (values, formula)} following the keys of the library default grid.
    Parameters without a rule keep their default values with an explanatory formula."""
    out = {}
    for par, vals_def in grilla_defecto.items():
        v, f = _recomendar(nombre, par, d)
        if v is None:
            out[par] = (list(vals_def), "Library default (no data-dependent rule for this parameter).")
        else:
            out[par] = (v, f)
    return out


def valores_a_texto(vals):
    return ", ".join(repr(v) for v in vals)


def texto_a_valores(texto):
    """Parses '2, 3, 5, None, "sqrt", (50,), 1e-3' → list. Raises ValueError with a friendly message."""
    texto = (texto or "").strip()
    if not texto:
        raise ValueError("write at least one value")
    try:
        v = ast.literal_eval("[" + texto + "]")
    except Exception:
        raise ValueError("could not read the list — separate values with commas; text values need quotes, e.g. 'sqrt', 'svd'; "
                         "tuples like (50,) or (50, 25)")
    if not v:
        raise ValueError("write at least one value")
    return v


def tamano_grilla(grilla):
    n = 1
    for v in grilla.values():
        n *= max(1, len(v))
    return n
