# -*- coding: utf-8 -*-
"""
Guidance layer of Metrika.edu.

* what_next(tab)       -> a "🧭 What's next?" popover with recommendations that depend on what the
                          user has done so far and on the actual results on screen.
* how_to_interpret(k)  -> a "📖 How to interpret?" popover that explains every metric / plot of a
                          section (optionally showing the user's own value and a verdict).
* pros_cons(kind)      -> a "⚖️ Pros & cons" popover comparing the options of a menu
                          (the selected one is highlighted).
* journey()            -> a progress tracker of the recommended learning path.

Everything here is plain text/logic, no heavy imports: it never slows the app down.
"""
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Learning journey
# ---------------------------------------------------------------------------
PASOS = [
    ("data", "1 · Load data", "Load spectra or a table, set IDs, classes and/or reference values."),
    ("prep", "2 · Preprocess", "Remove baseline / scatter / scale differences (or transform + scale a table)."),
    ("pca", "3 · Explore (PCA)", "Look at the structure of the data: groups, trends, anomalies."),
    ("outliers", "4 · Check outliers", "Find samples that do not behave like the others."),
    ("dendro", "5 · Cluster", "Do samples group the way you expect? (optional but very instructive)"),
    ("screening", "6 · Screen models", "Compare many preprocessing × algorithm combinations fairly."),
    ("refine", "7 · Refine a model", "Classification / Regression / SIMCA with tuning and validation."),
    ("pred", "8 · Predict new samples", "Use a saved model, with its domain of applicability."),
    ("report", "9 · Report", "Export a traceable report of everything you did."),
]


def _visitas():
    return st.session_state.setdefault("_guia_visitas", set())


def _marcar(tab):
    _visitas().add(tab)


def _tiene_datos(ss):
    return ss.get("X") is not None


def _estado_paso(ss):
    """Which steps of the journey look done, judging from the session state."""
    v = _visitas()
    hecho = {
        "data": _tiene_datos(ss),
        "prep": bool(ss.get("pasos_pretratamiento")) or ("prep" in v and _tiene_datos(ss) and ss.get("tab_prep") is not None),
        "pca": "pca" in v,
        "outliers": "outliers" in v,
        "dendro": ss.get("dendro_resultado") is not None,
        "screening": bool(ss.get("scr_filas")),
        "refine": any(ss.get(k) for k in ("clf_resultados", "reg_resultados", "simca_modelos")),
        "pred": ss.get("pred_ultimo") is not None,
        "report": ss.get("informe_final") is not None,
    }
    return hecho


def journey(compact=False):
    """Progress tracker. compact=True gives a single line (for the sidebar)."""
    ss = st.session_state
    h = _estado_paso(ss)
    hechos = sum(1 for k, _, _ in PASOS if h.get(k))
    if compact:
        st.progress(hechos / len(PASOS), text=f"Learning path: {hechos}/{len(PASOS)} steps")
        return
    st.progress(hechos / len(PASOS), text=f"Your learning path — {hechos}/{len(PASOS)} steps explored")
    cols = st.columns(3)
    for i, (k, titulo, texto) in enumerate(PASOS):
        with cols[i % 3]:
            marca = "✅" if h.get(k) else "⬜"
            st.markdown(f"**{marca} {titulo}**  \n<span style='font-size:0.85rem;color:#5A6A72'>{texto}</span>",
                        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Verdict helpers (traffic-light interpretation of a number)
# ---------------------------------------------------------------------------
def _sem(valor, malo, bueno, mayor_mejor=True):
    """Returns '🟢', '🟡' or '🔴' for a value given the 'poor' and 'good' thresholds."""
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return "⚪"
    if not np.isfinite(v):
        return "⚪"
    if mayor_mejor:
        return "🟢" if v >= bueno else ("🟡" if v >= malo else "🔴")
    return "🟢" if v <= bueno else ("🟡" if v <= malo else "🔴")


# ---------------------------------------------------------------------------
# HOW TO INTERPRET — one entry per metric / plot
#   (name, what it is, how to read it, rule of thumb, optional (key, poor, good, higher_is_better))
# ---------------------------------------------------------------------------
INTERPRETAR = {
    "resampling": {
        "title": "Reading the resampling results",
        "items": [
            ("Mean ± SD of each metric",
             "The model is refitted many times, each time with a different train/test partition; this is the average result and how much it moves from one partition to another.",
             "The mean is the performance you can honestly expect. The SD (spread) tells you how much it depends on WHICH samples happen to be in the test set.",
             "A small SD (say < 0.02 in accuracy or R²) means the result is stable. A large SD is typical of small or heterogeneous datasets — do not trust a single split."),
            ("95 % interval (bootstrap only)",
             "The range that contains the middle 95 % of the repetitions (2.5th to 97.5th percentile).",
             "It is an empirical interval for the metric: if you repeated the whole study with other samples from the same population, the result would most likely fall inside it.",
             "A wide interval means high uncertainty. If the intervals of two models overlap a lot, you cannot say that one is really better."),
            ("Box plot of the repetitions",
             "Each dot is one repetition; the box holds the middle 50 % of them and the line is the median.",
             "Compare boxes between models: separated boxes = a real difference; overlapping boxes = a difference that could be luck.",
             "Outlier dots far from the box show that some partitions were unlucky (e.g. a whole class missing from the test set)."),
            ("Resampling mean vs. the single test-set result",
             "Your first test result comes from one partition only.",
             "If the single result is far from the resampling mean (more than about 1 SD) it was a lucky or unlucky split — report the resampling mean ± SD instead.",
             None),
        ],
    },
    "pca": {
        "title": "Reading a PCA",
        "items": [
            ("Explained variance (per PC and cumulative)",
             "How much of the total variation in the data each principal component (PC) captures.",
             "PC1 holds the most variation, PC2 the most of what is left, and so on. The scree plot shows where "
             "adding PCs stops paying off (the 'elbow').",
             "Keep enough PCs to reach ~90–99 % cumulative variance for spectra (fewer for noisy tables). More PCs "
             "than needed just model noise."),
            ("Scores plot",
             "Each point is a sample, placed by its coordinates on the PCs.",
             "Samples that are close have similar spectra/profiles. Groups = classes or batches; a gradient = a trend "
             "(e.g. concentration); an isolated point = a candidate outlier.",
             "Colour by class: if classes separate in PC1–PC2, a classifier will probably work. Overlap does not "
             "mean failure — other PCs or a supervised model may still separate them."),
            ("Loadings",
             "The weight of each original variable (wavenumber, ppm, compound…) in a PC.",
             "Large positive/negative loadings mark the variables responsible for the separation seen in the scores. "
             "For spectra they look like spectra: peaks point to the chemistry.",
             "Compare the loading peaks with known bands (Glossary tab) to link the model to the chemistry."),
            ("Scores vs. batches / time",
             "If a PC separates measurement days, instruments or operators, that PC is capturing a nuisance effect.",
             "Colour by batch. If batches separate more than classes, improve preprocessing or randomise the "
             "measurement order next time.",
             "Always check for this before trusting a model."),
        ],
    },
    "outliers": {
        "title": "Reading the outlier plot (Hotelling's T² and Q)",
        "items": [
            ("Hotelling's T²",
             "Distance of a sample from the centre of the data INSIDE the PCA model (how extreme its scores are).",
             "High T² = the sample is unusual in the same directions as the others, but more extreme "
             "(e.g. a very high concentration).",
             "Above the limit (α = 0.05 by default): extreme but 'explained by the model'. Often still valid."),
            ("Q residual (SPE)",
             "How much of the sample's spectrum is NOT explained by the PCA model (distance to the model plane).",
             "High Q = something the model has never seen: contamination, a spike, a different matrix, instrument fault.",
             "Above the limit: look at the raw spectrum before keeping the sample."),
            ("Four quadrants",
             "Low/low = normal. High T² only = extreme but consistent. High Q only = new kind of variation. "
             "High both = very suspicious.",
             "Prioritise the top-right quadrant.",
             "With α = 0.05 about 5 % of good samples are flagged by chance — do not delete them blindly."),
            ("Before excluding",
             "Excluding is reversible here, and every exclusion is logged in the report.",
             "Ask: is there a physical reason (bad measurement, mislabeled, different product)? Then exclude. "
             "If not, keep it and mention it.",
             "Excluding more than ~5–10 % of the samples is a sign the preprocessing or the sampling should be revisited."),
        ],
    },
    "dendro": {
        "title": "Reading a dendrogram",
        "items": [
            ("Branch height",
             "The distance at which two groups merge. Low merges = very similar samples; tall merges = very different.",
             "Look for a few tall branches with many short ones below: that is natural group structure.",
             "Cut where there is a large jump in height to decide the number of clusters."),
            ("Leaf colours (classes)",
             "Labels coloured by the known class.",
             "If every cluster is one colour, the classes are naturally separable (pure clusters). "
             "Mixed colours = the classes overlap in this representation.",
             "The app reports a purity: the fraction of samples that fall in the majority class of their cluster "
             "(≥ 0.9 excellent, 0.7–0.9 acceptable, < 0.7 poor)."),
            ("Linkage method",
             "How the distance between two clusters is defined.",
             "Ward → compact, similar-sized clusters (best default for spectra). Average → compromise. "
             "Complete → tight clusters, sensitive to outliers. Single → chains, rarely good for spectra.",
             "If purity is poor, try another linkage and/or a different preprocessing before concluding anything."),
        ],
    },
    "clf": {
        "title": "Classification metrics",
        "items": [
            ("Accuracy",
             "Fraction of samples classified correctly.",
             "Easy to read but misleading when classes are unbalanced (90 % of samples in one class → a model that "
             "always answers that class gets 90 %).",
             "Use it only with balanced classes.", ("accuracy", 0.7, 0.9, True)),
            ("Balanced accuracy",
             "Average of the recall of each class (every class weighs the same).",
             "The honest version of accuracy for unbalanced data. 1 = perfect, 1/k = random guessing (k classes).",
             "≥ 0.9 good · 0.7–0.9 acceptable · < 0.7 weak.", ("balanced_accuracy", 0.7, 0.9, True)),
            ("Sensitivity / recall (per class)",
             "Of the samples that REALLY belong to a class, how many did the model find?",
             "Low sensitivity = samples of that class are being missed (false negatives).",
             "Critical when missing a case is costly (disease, adulteration).", ("sensibilidad_macro", 0.7, 0.9, True)),
            ("Specificity",
             "Of the samples that do NOT belong to a class, how many did the model correctly leave out?",
             "Low specificity = many false alarms (false positives).",
             "Critical when a false alarm is costly (rejecting good batches).", ("especificidad_macro", 0.7, 0.9, True)),
            ("Precision",
             "Of the samples the model assigned to a class, how many really belong to it?",
             "Low precision = the class label cannot be trusted when it appears.",
             "Look at precision and recall together (F1)."),
            ("F1-score",
             "Harmonic mean of precision and recall (0–1).",
             "Penalises models that are good on one and bad on the other.",
             "≥ 0.9 good.", ("f1_macro", 0.7, 0.9, True)),
            ("Cohen's kappa",
             "Agreement between prediction and truth beyond what chance would give (−1 to 1).",
             "0 = chance, 1 = perfect.",
             "> 0.8 almost perfect · 0.6–0.8 substantial · 0.4–0.6 moderate · < 0.4 weak.", ("kappa", 0.4, 0.8, True)),
            ("MCC (Matthews correlation)",
             "A single balanced score that uses all four cells of the confusion matrix (−1 to 1).",
             "Robust with unbalanced classes; 0 = chance.",
             "> 0.7 good.", ("mcc", 0.4, 0.7, True)),
            ("AUC (ROC)",
             "Probability that the model ranks a random sample of a class above a random sample of another (0.5–1).",
             "Measures how well classes are separated regardless of the decision threshold.",
             "> 0.9 excellent · 0.8–0.9 good · 0.7–0.8 fair · 0.5 = random.", ("auc", 0.7, 0.9, True)),
            ("Confusion matrix",
             "Rows = true class, columns = predicted class.",
             "The diagonal is correct. Off-diagonal cells tell you WHICH classes get confused — often neighbours "
             "chemically (e.g. two close origins).",
             "Use it to decide what to improve: more samples of the confused classes, other preprocessing, or merging classes."),
            ("Cross-validation (CV) vs. test",
             "CV is computed inside the training set (rotating which part is left out). The test set is never touched "
             "until the end.",
             "CV ≈ test: the estimate is reliable. CV much better than test (> 10–15 points): overfitting, or the test "
             "set differs from the training set. The test result is the one to report.",
             "With < 30 samples both are noisy: report them with caution and prefer repeated CV."),
        ],
    },
    "reg": {
        "title": "Regression metrics",
        "items": [
            ("RMSE (RMSEC / RMSECV / RMSEP)",
             "Typical size of the prediction error, in the units of the property you predict.",
             "C = calibration (training, optimistic), CV = cross-validation, P = prediction on the test set. "
             "Compare RMSECV and RMSEP: they should be similar.",
             "What matters is the error compared with the range of y and with the method's required precision."),
            ("R²",
             "Fraction of the variance of y explained by the model (1 = perfect, 0 = as good as predicting the mean).",
             "R² in calibration is always flattering. Look at R² of CV and test.",
             "> 0.9 excellent · 0.8–0.9 good · 0.6–0.8 screening only · < 0.6 weak.", ("r2", 0.6, 0.9, True)),
            ("RPD",
             "Standard deviation of y divided by the prediction error (SD/RMSE).",
             "How many times smaller the error is than the natural spread of the samples.",
             "> 3 good for quantification · 2–3 screening/rough · < 1.5 not usable. (Rule from NIR agricultural/food literature.)",
             ("rpd", 2.0, 3.0, True)),
            ("RPIQ",
             "Interquartile range of y divided by the error — like RPD but robust to skewed distributions.",
             "Preferred when y is not normally distributed (soils, biological data).",
             "> 4 good · 2.5–4 acceptable."),
            ("MAE",
             "Mean absolute error — like RMSE but without extra weight on large errors.",
             "If RMSE ≫ MAE a few predictions are very wrong; look at the residual plot.",
             "Report it together with RMSE."),
            ("Bias",
             "Average of (predicted − measured).",
             "Should be ≈ 0. A bias means the model systematically over/under-predicts (often a sign of a shift between "
             "training and new samples).",
             "Check the slope/intercept of predicted vs. measured too."),
            ("Predicted vs. measured plot",
             "Ideal = points on the 1:1 line.",
             "Curvature = non-linear relationship; fan shape = error grows with concentration; separate clouds = "
             "hidden groups.",
             "Colour by batch/class to find the cause of the pattern."),
            ("Residuals plot",
             "Measured − predicted for each sample.",
             "Should look like random noise around 0. Trends or isolated large residuals reveal model problems or outliers.",
             ""),
            ("Latent variables (PLS/PCR)",
             "Number of components used.",
             "RMSECV usually drops and then rises: choose the smallest number near the minimum (parsimony).",
             "Too many components fit the noise: good calibration, bad prediction."),
            ("CV vs. test",
             "Same logic as in classification.",
             "RMSEP much larger than RMSECV = overfitting or non-representative split.",
             "Report RMSEP (test set) as the final figure."),
        ],
    },
    "simca": {
        "title": "SIMCA — how to read the results",
        "items": [
            ("One PCA model per class",
             "SIMCA does not draw boundaries between classes; each class gets its own PCA model and a sample is accepted "
             "if it fits that model (T² and Q inside the limits).",
             "That means a sample can be accepted by ONE class, by SEVERAL (ambiguous) or by NONE (outlier / unknown class).",
             "Great for authentication ('is this really class A?') and for detecting samples of unknown classes."),
            ("Sensitivity of a class",
             "Percentage of the class's own test samples accepted by its model.",
             "Low = the model is too strict (raise α, use more components).",
             "Close to (1 − α) × 100 for a good model, e.g. ~95 %.", ("sens", 0.8, 0.9, True)),
            ("Specificity of a class",
             "Percentage of samples of OTHER classes rejected by this model.",
             "Low = the model is too permissive: classes overlap or too many components were kept.",
             "> 90 % good.", ("espec", 0.8, 0.9, True)),
            ("Accepted by several / by none",
             "Ambiguous vs. unrecognised samples.",
             "Many 'several' → classes are not distinguishable by this representation. Many 'none' → outliers, "
             "missing classes or a model that is too strict.",
             ""),
            ("Significance level α",
             "Probability that a good sample of the class is wrongly rejected.",
             "Lower α → wider limits → higher sensitivity, lower specificity (and vice versa).",
             "0.05 is the usual compromise."),
            ("Classical vs. DD-SIMCA",
             "Classical SIMCA uses separate T² and Q limits. DD-SIMCA combines them in one data-driven acceptance "
             "area estimated from the training data.",
             "DD-SIMCA is the recommended modern formulation and gives a more reliable false-alarm rate.",
             ""),
            ("Cooman's plot / T²–Q plot",
             "Distance of every sample to two class models at once, or T² vs. Q for one class.",
             "Samples in the corner of both models are ambiguous; those far from both are unknown.",
             ""),
        ],
    },
    "screening": {
        "title": "Reading the model screening",
        "items": [
            ("Fair comparison",
             "Every combination uses the same train/test split and the same validation.",
             "Differences between rows are therefore due to the method, not to luck in the split.",
             "Differences smaller than the CV noise (≈ ±2–3 points) are not meaningful."),
            ("Ranking metric",
             "Classification: AUC (or balanced accuracy). Regression: RMSE on the test set.",
             "The table also shows the CV values: a model that is good in CV but poor in test is overfitted.",
             "Prefer a simpler model (fewer variables, linear) when two are tied."),
            ("What works best on average",
             "Averages each preprocessing / algorithm over all combinations.",
             "Shows which preprocessing helps in general, rather than only the single winner.",
             "Take the top 2–3 combinations to the refinement step, not only the first."),
            ("Time per combination",
             "How long each one took.",
             "Useful to decide whether a marginal gain is worth a much slower method.",
             ""),
        ],
    },
    "pred": {
        "title": "Reading a prediction",
        "items": [
            ("Predicted class / value",
             "What the saved model answers for each new sample, after applying EXACTLY the same preprocessing.",
             "Always read it together with the confidence or interval and the domain check.",
             ""),
            ("Probability / margin",
             "How strongly the model prefers the predicted class.",
             "Probabilities near 1/k mean the model is guessing; many borderline samples = reconsider the model.",
             "Probabilities are model-dependent; calibrated sets (below) are more reliable."),
            ("Prediction interval (regression)",
             "A range that should contain the true value with the chosen confidence (e.g. 95 %).",
             "Wide intervals = low precision. Report value ± half-width.",
             "Valid only inside the calibrated range of y."),
            ("Prediction sets (classification, conformal)",
             "A set of classes that contains the true class with the chosen confidence.",
             "One class = confident; two or more = ambiguous; empty = nothing fits (unknown sample).",
             ""),
            ("Applicability domain (T², Q, DD)",
             "Does the new sample look like the data the model was trained on?",
             "Outside the domain → the prediction is an extrapolation and should NOT be trusted.",
             "Do not report predictions of samples outside the domain; investigate or add similar samples to the training set."),
        ],
    },
}


def how_to_interpret(kind, valores=None, key=None, etiqueta="📖 How to interpret?"):
    """Popover explaining each metric of `kind`. `valores` (dict, optional) maps the metric keys to the
    user's own numbers, which are shown next to the explanation with a traffic-light verdict."""
    info = INTERPRETAR.get(kind)
    if not info:
        return
    with st.popover(etiqueta, help="Explains every metric and plot shown in this section, in plain words.",
                    key=key or f"how_{kind}"):
        st.markdown(f"### {info['title']}")
        if valores:
            st.caption("🟢 good · 🟡 acceptable · 🔴 weak — rough rules of thumb, always judge them against your own application.")
        for it in info["items"]:
            nombre, que, como, regla = it[0], it[1], it[2], it[3]
            sem = ""
            if len(it) > 4 and valores:
                k, malo, bueno, mayor = it[4]
                if valores.get(k) is not None:
                    v = valores[k]
                    sem = f"  \n**Your value: {_sem(v, malo, bueno, mayor)} {float(v):.3g}**"
            st.markdown(f"**{nombre}**  \n{que}  \n*How to read it:* {como}"
                        + (f"  \n*Rule of thumb:* {regla}" if regla else "") + sem)
            st.divider()


# ---------------------------------------------------------------------------
# PROS & CONS — comparison of the options of a menu
# ---------------------------------------------------------------------------
PROS_CONTRAS = {
    "resampling": {
        "title": "Resampling methods",
        "items": {
            "Repeated hold-out (Monte Carlo)": (
                ["Easy to explain: you choose the test fraction and it is repeated many times.",
                 "The test set always has the size you decided and is stratified by class in classification.",
                 "Rotates the samples between train and test, so no single lucky/unlucky split decides."],
                ["Repetitions share many training samples, so the SD slightly underestimates the real uncertainty.",
                 "Some samples may be tested more often than others."],
                "Quick stability check of a result, or small/medium datasets where you want a fixed test size."),
            "Bootstrap (out-of-bag)": (
                ["Gives an empirical 95 % interval for each metric.",
                 "Uses the whole dataset size for training in each repetition (with repeated samples).",
                 "Test samples (out-of-bag) are never in that repetition's training set. Widely known and accepted."],
                ["Only ~63 % of the samples are distinct in training, so the estimate tends to be slightly pessimistic.",
                 "The out-of-bag size varies each time, and a class can be rare in it (such repetitions are discarded if training lacks a class).",
                 "Replicates of the same sample can leak between train and test if they are not grouped."],
                "You want confidence intervals, or the dataset is small and you want to use all of it for training."),
        },
    },
    "split_clf": {
        "title": "How to choose the test set (classification)",
        "items": {
            "Random": (
                ["Simplest and unbiased: every sample has the same chance.", "Needs no extra decisions."],
                ["With few samples a small class may end up (almost) absent from the test set.",
                 "The result depends on the seed — different seeds give different scores."],
                "Large datasets with balanced classes."),
            "Random (stratified)": (
                ["Simple and unbiased: every sample has the same chance.",
                 "Stratified: each class keeps its proportion in train and test.",
                 "Can be repeated with other seeds to see how stable the result is."],
                ["With few samples the test set may by chance be easier or harder than average.",
                 "Replicates of the same sample/subject can end up on both sides (optimistic result)."],
                "Default choice when you have ≥ 50 samples and no replicate structure."),
            "Kennard-Stone (representative)": (
                ["Training set covers the whole spectral space (including extremes).",
                 "Deterministic: the same split every time (reproducible).",
                 "Good when samples are scarce."],
                ["The test set becomes the 'typical' samples, so the performance estimate is slightly optimistic.",
                 "Does not guarantee class balance by itself (here it is applied class by class)."],
                "Few samples, and you want the model to be built on a representative training set."),
        },
    },
    "split_reg": {
        "title": "How to choose the test set (regression)",
        "items": {
            "Random": (
                ["Simplest and unbiased.", "Needs no extra decisions."],
                ["The test set can miss the extremes of y (low or high values).",
                 "With few samples the result depends strongly on the seed."],
                "Large datasets with a homogeneous distribution of y."),
            "Random (stratified by y range)": (
                ["Splits randomly INSIDE quantile bins of y, so train and test both cover low, middle and high values.",
                 "Still random, so it stays unbiased about the spectra.",
                 "Avoids the typical problem of a test set concentrated around the mean."],
                ["Ignores the spectra (X) completely.",
                 "With very few samples the bins can only be coarse."],
                "Recommended default for regression when y is skewed or the range matters."),
            "SPXY (extended Kennard-Stone)": (
                ["Training set spreads over BOTH the spectral space and the y range.",
                 "Deterministic and reproducible; widely used in NIR calibration."],
                ["The test set contains the 'typical' samples, so the error estimate tends to be optimistic.",
                 "Sensitive to outliers (they are selected first)."],
                "Calibration with few samples where every extreme is precious."),
        },
    },
    "seleccion": {
        "title": "Variable selection methods",
        "items": {
            "None (all variables)": (
                ["Fast and simple.", "No risk of selecting variables by chance.",
                 "Works well with PLS, which handles correlated variables naturally."],
                ["Noise variables stay in the model.", "Harder to interpret (hundreds of variables)."],
                "Always the first thing to try; keep it unless selection clearly improves the TEST result."),
            "Boruta": (
                ["Finds ALL relevant variables (not only the minimum set), based on Random Forest importance.",
                 "Compares each variable with random 'shadow' copies, so it has a statistical criterion.",
                 "Robust to correlated variables and non-linear relationships."],
                ["Slow with thousands of variables.", "Selected variables depend on the Random Forest (not on your final model).",
                 "Can keep many redundant variables."],
                "When you want to understand which variables matter, or reduce a table with many variables."),
            "Genetic Algorithm": (
                ["Searches subsets directly for the model that you will use (wrapper method).",
                 "Can find compact, high-performing subsets and interactions."],
                ["Slow and random (different runs → different subsets).",
                 "High risk of overfitting with few samples: the selection must be validated on an untouched test set.",
                 "Needs tuning (population, generations)."],
                "Many variables, enough samples, and a need for a small interpretable subset."),
        },
    },
    "optimizacion": {
        "title": "Hyperparameter search strategies",
        "items": {
            "Fast": (
                ["Quickest: 8 random combinations.", "Good first pass to see if tuning matters at all."],
                ["May miss the optimum.", "Results vary with the random draw."],
                "Exploration, or slow models."),
            "Balanced": (
                ["Successive halving: weak combinations are dropped early.",
                 "Good compromise between speed and quality."],
                ["Needs ≥ 40 samples to work as designed (otherwise it falls back to a random search).",
                 "Early rounds use few resources and can discard a late-bloomer."],
                "Default recommendation."),
            "Thorough": (
                ["Looks at every combination of the grid, but efficiently.",
                 "Reproducible: no random sampling of combinations."],
                ["Slowest of the grid-based options; cost grows multiplicatively with the grid.",
                 "Only as good as the grid you give it."],
                "Small grids, final tuning, enough computing time."),
            "Bayesian": (
                ["Learns from previous trials (Optuna/TPE): usually reaches a better result with fewer trials.",
                 "Excellent for expensive models."],
                ["Sequential, so it parallelises less.", "Treats each listed value as a category (does not interpolate between them)."],
                "Tuning a single model carefully."),
        },
    },
    "cv": {
        "title": "Cross-validation schemes",
        "items": {
            "k-fold (5 or 10)": (
                ["Good bias/variance balance.", "Fast enough for any model."],
                ["The result depends slightly on the random partition (repeat it to check)."],
                "Default."),
            "Leave-one-out (LOO)": (
                ["Uses almost all samples for training each time — good with very few samples.",
                 "Deterministic."],
                ["Slow with many samples.", "High variance of the estimate.",
                 "Very optimistic if there are replicates (the replicate of the left-out sample is still in training)."],
                "< 30 samples without replicates."),
        },
    },
    "linkage": {
        "title": "Linkage methods (dendrogram)",
        "items": {
            "ward": (["Compact, balanced clusters.", "Usually the best choice for spectra after PCA-like preprocessing."],
                     ["Assumes roughly spherical clusters of similar size.", "Sensitive to outliers."], "Default."),
            "average": (["Compromise between single and complete.", "Less sensitive to outliers than complete."],
                        ["Can merge clusters of different densities."], "When ward gives uneven clusters."),
            "complete": (["Tight, similar-diameter clusters."], ["Very sensitive to outliers."], "Clean data with compact groups."),
            "single": (["Finds elongated or chain-like structures."],
                       ["'Chaining' effect: one noisy sample can glue two groups. Rarely good for spectra."],
                       "Only to detect continuous trends or isolated samples."),
        },
    },
    "clf_algo": {
        "title": "Classification algorithms",
        "items": {
            "PLS-DA": (["The standard in chemometrics: copes with many correlated variables and few samples.",
                        "Interpretable scores and loadings (VIP)."],
                       ["Linear; the number of components must be tuned (risk of overfitting).",
                        "Not a real probabilistic model."], "First choice for spectra."),
            "LDA": (["Simple, fast, no hyperparameters.", "Excellent when classes are linearly separable."],
                    ["Needs more samples than variables (use PCA first or a selection).",
                     "Assumes similar covariance in all classes."], "Few, well-chosen variables (tables) or PCA scores."),
            "Logistic Regression": (["Probabilistic, interpretable coefficients.", "Regularisation handles many variables."],
                                    ["Linear boundaries.", "Needs scaling and tuning of C."], "Tables, interpretable models."),
            "Naive Bayes": (["Very fast; works with very few samples."],
                            ["Assumes independent variables (false in spectra).", "Probabilities are over-confident."],
                            "Quick baseline on tabular data."),
            "Random Forest": (["Captures non-linear effects and interactions.", "Robust, few assumptions, built-in variable importance."],
                              ["Slower; less accurate than PLS-DA with highly correlated smooth spectra.",
                               "Cannot extrapolate."], "Tables and non-linear problems."),
            "SVM": (["Very powerful with the RBF kernel; good with few samples and many variables."],
                    ["Needs tuning of C and gamma.", "Black box; slow with thousands of samples."], "Non-linear spectral problems."),
            "Decision Tree": (["Totally transparent rules."], ["Unstable and prone to overfitting."], "Teaching and simple rules."),
            "XGBoost": (["State of the art for tables; handles missing values."],
                        ["Many hyperparameters.", "Overfits easily with < 100 samples."], "Large tables."),
            "KNN": (["No training; intuitive."], ["Sensitive to scaling and to irrelevant variables.", "Slow to predict with many samples."],
                    "Small, clean datasets and as a sanity check."),
            "Neural Network (MLP)": (["Can model complex non-linearities."],
                                     ["Needs many samples.", "Black box, many hyperparameters, results vary with the seed."],
                                     "Large datasets only."),
        },
    },
    "reg_algo": {
        "title": "Regression algorithms",
        "items": {
            "PLS": (["The workhorse of multivariate calibration.", "Handles collinearity; few hyperparameters (components).",
                     "Interpretable (loadings, VIP, regression coefficients)."],
                    ["Linear.", "Choosing the number of latent variables needs care."], "First choice for spectra."),
            "PCR": (["Simple: PCA followed by regression."], ["Components are chosen without looking at y, so PLS is usually better."], "Teaching / comparison."),
            "Linear Regression (MLR)": (["Fully transparent."], ["Needs more samples than variables, fails with collinearity."], "Few, independent variables."),
            "Lasso": (["Selects variables automatically (zero coefficients)."], ["Picks one of several correlated variables arbitrarily."], "Tables with many variables."),
            "Ridge": (["Stable with correlated variables, keeps all of them."], ["No variable selection."], "Spectra, as an alternative to PLS."),
            "Elastic Net": (["Combines Lasso and Ridge: selects but keeps groups of correlated variables."], ["Two hyperparameters to tune."], "Correlated tables."),
            "Random Forest": (["Non-linear, robust."], ["Cannot extrapolate beyond the calibration range.", "Often worse than PLS for spectra."], "Non-linear tables."),
            "SVM (SVR)": (["Flexible, good with few samples."], ["Needs tuning (C, gamma, epsilon).", "Black box."], "Non-linear calibration."),
            "Decision Tree": (["Transparent."], ["Piecewise-constant predictions, unstable."], "Teaching."),
            "XGBoost": (["Excellent for tabular data."], ["Overfits with few samples; many hyperparameters."], "Large tables."),
            "KNN": (["Intuitive."], ["Sensitive to scaling; poor extrapolation."], "Sanity check."),
            "Neural Network (MLP)": (["Complex non-linear relations."], ["Needs many samples; unstable."], "Large datasets only."),
        },
    },
    "prep_spectral": {
        "title": "Spectral preprocessing — what each one fixes",
        "items": {
            "Savitzky–Golay smoothing": (["Reduces random noise preserving peak shape."],
                                         ["Too wide a window distorts peaks."], "Noisy spectra."),
            "1st / 2nd derivative (SG)": (["Removes baseline offset (1st) and slope (2nd).", "Resolves overlapping bands."],
                                          ["Amplifies noise (smooth first).", "Spectra become harder to read."], "NIR/MIR with baseline drift."),
            "SNV": (["Corrects scatter and multiplicative effects sample by sample.", "No reference spectrum needed."],
                    ["Can distort if there are strong real intensity differences between samples."], "NIR of solids/powders."),
            "MSC / EMSC": (["Corrects scatter using a reference spectrum.", "EMSC also models known interferents."],
                           ["Depends on the reference spectrum (the mean)."], "Scatter-dominated NIR."),
            "Baseline (ALS)": (["Removes curved baselines (fluorescence in Raman, chromatographic drift)."],
                               ["Parameters (λ, p) must be tuned; may remove broad real bands."], "Raman, chromatograms."),
            "Normalization (area / max / vector)": (["Removes concentration/amount differences."],
                                                    ["Destroys quantitative information when the amount is what you predict."], "Qualitative problems."),
            "PQN": (["Robust dilution correction for NMR / metabolomics."], ["Assumes most signals do not change."], "NMR of biofluids."),
            "Autoscaling": (["Gives every variable the same weight."], ["Amplifies noise in weak variables."], "Tabular data; mixed units."),
            "Mean centering": (["Always applied for PCA/PLS; removes the mean spectrum."], ["Alone does not equalise variables."], "Always."),
        },
    },
}


def pros_cons(kind, seleccionado=None, key=None, etiqueta="⚖️ Pros & cons"):
    """Popover comparing the options of a menu. The one currently selected is marked."""
    info = PROS_CONTRAS.get(kind)
    if not info:
        return
    with st.popover(etiqueta, help="Advantages, disadvantages and when to use each option.",
                    key=key or f"pc_{kind}"):
        st.markdown(f"### {info['title']}")
        for nombre, (pros, contras, cuando) in info["items"].items():
            sel = seleccionado is not None and str(seleccionado).startswith(nombre.split(" (")[0])
            st.markdown(("👉 " if sel else "") + f"**{nombre}**" + ("  _(selected)_" if sel else ""))
            st.markdown("✅ " + "  \n✅ ".join(pros))
            st.markdown("⚠️ " + "  \n⚠️ ".join(contras))
            st.markdown(f"💡 *Use it when:* {cuando}")
            st.divider()


# ---------------------------------------------------------------------------
# WHAT'S NEXT? — context-aware recommendations
# ---------------------------------------------------------------------------
def _rec(icono, titulo, texto, donde=None):
    return {"icono": icono, "titulo": titulo, "texto": texto, "donde": donde}


def _info_datos(ss):
    d = {"hay": _tiene_datos(ss)}
    if not d["hay"]:
        return d
    X = ss.get("X")
    d["n"], d["p"] = X.shape
    d["tab"] = bool(ss.get("_modo_tab", False)) and ss.get("nombres_var") is not None
    cl = ss.get("clases")
    d["clases"] = cl is not None
    if d["clases"]:
        import pandas as pd
        vc = pd.Series(np.asarray(cl).astype(str)).value_counts()
        d["k"], d["min_c"], d["max_c"] = len(vc), int(vc.min()), int(vc.max())
    y = ss.get("valores_y")
    d["y"] = y is not None and np.isfinite(np.asarray(y, dtype=float)).sum() >= 10
    d["pasos"] = len(ss.get("pasos_pretratamiento") or [])
    m = ss.get("mascara_excluidas")
    d["excl"] = int(np.sum(m)) if m is not None else 0
    return d


def _pureza_dendro(ss):
    """Purity of the clusters of the current dendrogram result (None if not computable)."""
    res = ss.get("dendro_resultado")
    if not res or res.get("clases") is None:
        return None
    try:
        from scipy.cluster import hierarchy
        import pandas as pd
        cl = np.asarray(res["clases"]).astype(str)
        k = len(np.unique(cl))
        if k < 2:
            return None
        lab = hierarchy.fcluster(res["Z"], t=k, criterion="maxclust")
        df = pd.DataFrame({"c": lab, "y": cl})
        return float(df.groupby("c")["y"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(df))
    except Exception:
        return None


def _mejor_clf(ss):
    res = ss.get("clf_resultados") or {}
    best = None
    for nombre, r in res.items():
        if not isinstance(r, dict) or "cv" not in r:
            continue
        cv = r["cv"].get("balanced_accuracy")
        te = r.get("test", {}).get("balanced_accuracy") if r.get("test") else None
        score = te if te is not None else cv
        if score is not None and (best is None or score > best["score"]):
            best = {"nombre": nombre, "cv": cv, "test": te, "score": score}
    return best


def _mejor_reg(ss):
    res = ss.get("reg_resultados") or {}
    best = None
    for nombre, r in res.items():
        if not isinstance(r, dict) or "cv" not in r:
            continue
        cv_r2, cv_rmse, cv_rpd = r["cv"].get("r2_cv"), r["cv"].get("rmse_cv"), r["cv"].get("rpd_cv")
        t = r.get("test") or {}
        score = t.get("r2", cv_r2)
        if score is not None and (best is None or score > best["score"]):
            best = {"nombre": nombre, "r2_cv": cv_r2, "rmse_cv": cv_rmse, "rpd_cv": cv_rpd,
                    "r2_te": t.get("r2"), "rmse_te": t.get("rmse"), "rpd_te": t.get("rpd"), "score": score}
    return best


def _simca_stats(ss):
    M = ss.get("simca_matriz_dentro")
    ytrue = ss.get("simca_clases_eval_true")
    if M is None or ytrue is None:
        return None
    try:
        ytrue = np.asarray(ytrue).astype(str)
        out = {}
        for c in M.columns:
            propias = ytrue == str(c)
            sens = float(M.loc[propias, c].mean()) if propias.any() else np.nan
            espec = float((~M.loc[~propias, c]).mean()) if (~propias).any() else np.nan
            out[str(c)] = (sens, espec)
        n_acc = M.sum(axis=1).to_numpy()
        return {"por_clase": out, "ninguna": float(np.mean(n_acc == 0)), "varias": float(np.mean(n_acc > 1))}
    except Exception:
        return None


def _recs_para(tab, ss, d):
    R = []
    sin_datos = not d["hay"]
    if tab != "home" and sin_datos and tab not in ("datasets", "contact", "learn", "hyper"):
        return [_rec("📂", "Load data first", "Use the left-hand panel: upload your file, or press one of the "
                     "**Try an example** buttons to start right away.", "Sidebar")]

    if tab == "home":
        if sin_datos:
            R += [_rec("🎯", "Start with an example", "In the left-hand panel choose **Try an example** (Wine, Iris, "
                       "Diabetes, olive-oil FT-MIR, soil NIR, simulated Raman). You will see the whole workflow in 10 minutes.", "Sidebar"),
                  _rec("📂", "Or load your own data", "Spectra (columns = wavenumbers) or tables (columns = named variables), "
                       "CSV or Excel. First column = sample ID.", "Sidebar"),
                  _rec("📚", "Need data to practise?", "The **Public datasets** tab lists where to download well-known datasets "
                       "(with the authors to cite).", "📚 Public datasets")]
        else:
            R.append(_rec("➡️", "Continue where you are", "Follow the learning path above; each tab has its own "
                          "**What's next?** button.", None))
        return R

    if tab == "data":
        if d["tab"]:
            R.append(_rec("🧾", "Tabular data", "Check that every column is numeric and that missing values are few. The next "
                          "step is the **transformation and scaling** (autoscaling is the usual choice when variables have "
                          "different units).", "🧪 Preprocessing"))
        else:
            R.append(_rec("📈", "Look at the raw spectra", "Do they look like what you measured? Noisy, saturated or flat spectra "
                          "should be fixed (or removed) before any model.", None))
        if not d["clases"] and not d["y"]:
            R.append(_rec("🏷️", "Define what you want to predict", "Without classes or reference values you can only explore "
                          "(PCA, dendrogram). To classify, load a class column; to predict a number, load the reference values.", "Sidebar"))
        if d["clases"] and d["min_c"] < 5:
            R.append(_rec("⚠️", "A class has fewer than 5 samples", f"The smallest class has {d['min_c']} samples. "
                          "Validation will be unreliable; consider merging classes or measuring more samples.", None))
        if d["n"] < 30:
            R.append(_rec("⚠️", "Few samples", f"With {d['n']} samples results are exploratory. Use cross-validation, "
                          "avoid complex models and do not over-interpret.", None))
        R += [_rec("➡️", "Next: preprocess", "Raw data → apply the corrections that fit your technique.", "🧪 Preprocessing"),
              _rec("🧭", "…or look first", "Running a PCA on the raw data is a good way to see whether preprocessing is needed.", "🧭 PCA")]
        return R

    if tab == "prep":
        if d["pasos"] == 0 and not d["tab"]:
            R.append(_rec("🧪", "Nothing applied yet", "Pick the signal type (NIR/MIR, Raman, chromatogram, NMR) and add steps. "
                          "Typical: NIR → SNV or derivative; Raman → baseline + normalization; chromatograms → baseline + alignment; "
                          "NMR → PQN.", None))
        else:
            R.append(_rec("✅", "Preprocessing defined", "Click **Apply** to send it to the rest of the app.", None))
        R += [_rec("🧭", "Check the effect with a PCA", "A good preprocessing makes classes separate better or batches overlap more.", "🧭 PCA"),
              _rec("🏁", "Not sure which one? Let the app compare", "Model screening tests many preprocessing × algorithm "
                   "combinations fairly.", "🏁 Model screening"),
              _rec("✂️", "Spectra: crop uninformative regions", "Removing noisy ends or regions without signal often improves models.", "✂️ Crop")]
        if d["pasos"] > 4:
            R.append(_rec("⚠️", "Many steps", "Long preprocessing chains are hard to justify and to reproduce. Keep only what "
                          "improves the validated result.", None))
        return R

    if tab == "crop":
        return [_rec("🧭", "Check with PCA", "Compare the scores with and without the crop.", "🧭 PCA"),
                _rec("🏁", "Test it objectively", "Model screening can compare the full vs. the cropped spectrum.", "🏁 Model screening"),
                _rec("💡", "Tip", "Crop regions with no chemical information (noise, water/CO₂ bands, the solvent region).", None)]

    if tab == "pca":
        if d["clases"]:
            R.append(_rec("🎨", "Colour by class", "Do the classes form groups? If yes, classification or SIMCA will probably work. "
                          "If they overlap, try other PCs, a different preprocessing, or SIMCA/PLS-DA which look for supervised directions.", None))
        elif d["y"]:
            R.append(_rec("🌈", "Colour by the reference value", "A smooth gradient across the scores means the signal carries "
                          "information about y: regression should work.", None))
        R += [_rec("🚩", "Look for outliers", "Isolated points in the scores or high T²/Q → check them in the outliers tab before modelling.", "🚩 Outliers"),
              _rec("🌳", "See the natural groups", "A dendrogram tells you whether samples cluster the way you expect.", "🌳 Dendrogram"),
              _rec("🛠️", "Understand the variables", "Loadings and variable ranking show WHICH variables drive the structure.", "🛠️ Other tools")]
        if d["tab"]:
            R.append(_rec("📏", "Tabular data: check the scaling", "PCA on unscaled tables is dominated by the variables with the largest "
                          "numbers. If one variable dominates PC1, use autoscaling.", "🧪 Preprocessing"))
        else:
            R.append(_rec("🧪", "Batches instead of classes?", "If a PC separates days or instruments, improve the preprocessing.", "🧪 Preprocessing"))
        R.append(_rec("🏁", "Ready to model?", "If the structure is reasonable, go to Model screening.", "🏁 Model screening"))
        return R

    if tab == "outliers":
        if d["excl"] > 0:
            pct = 100 * d["excl"] / d["n"]
            R.append(_rec("🧹", f"{d['excl']} sample(s) excluded ({pct:.0f} %)", "Re-run the PCA and dendrogram to see the effect. "
                          "They are also left out of every model." + (" That is a lot: revisit the preprocessing." if pct > 10 else ""), "🧭 PCA"))
        else:
            R.append(_rec("🔎", "Inspect before deleting", "Open the raw spectrum of every flagged sample. Remove it only if you can justify "
                          "it (bad measurement, wrong label, different product).", None))
        R += [_rec("🌳", "Next: dendrogram", "Check the group structure after cleaning.", "🌳 Dendrogram"),
              _rec("🏁", "Or go straight to the models", "Screening compares preprocessing and algorithms fairly.", "🏁 Model screening"),
              _rec("💡", "Remember", "About 5 % of perfectly good samples are flagged at α = 0.05; do not remove them just for that.", None)]
        return R

    if tab == "dendro":
        pur = _pureza_dendro(ss)
        res = ss.get("dendro_resultado")
        hist = ss.setdefault("_guia_dendro", {})
        if res is not None and pur is not None:
            hist[res.get("metodo")] = pur
        if res is None:
            R.append(_rec("▶️", "Compute it first", "Choose the linkage method (ward is a good start) and click Compute.", None))
        elif pur is None:
            R.append(_rec("🏷️", "Load classes to judge it", "With class labels the leaves are coloured and the app can report the "
                          "cluster purity.", "Sidebar"))
        else:
            tried = ", ".join(f"{m}: {v:.2f}" for m, v in hist.items())
            if pur >= 0.9:
                R.append(_rec("🟢", f"Excellent separation (purity {pur:.2f})", "Classes cluster naturally. A classifier should do very "
                              "well; SIMCA is also a good option.", "🏁 Model screening"))
            elif pur >= 0.7:
                otros = [m for m in ("ward", "average", "complete") if m not in hist]
                R.append(_rec("🟡", f"Moderate separation (purity {pur:.2f})", "Try " + (", ".join(otros) if otros else "another preprocessing")
                              + " linkage, or a different preprocessing/crop, and compare the purity. Supervised models can still separate "
                              "classes that clustering cannot.", None))
            else:
                otros = [m for m in ("ward", "average", "complete", "single") if m not in hist]
                R.append(_rec("🔴", f"Poor separation (purity {pur:.2f})", "Try " + (", ".join(otros) if otros else "other settings")
                              + ", change the preprocessing (SNV/derivative for NIR, baseline for Raman), crop uninformative regions, "
                              "or check outliers and labels. If nothing helps, the classes may not be distinguishable with this measurement.", "🧪 Preprocessing"))
            R.append(_rec("📋", "Linkages tried so far (purity)", tried or "–", None))
        R += [_rec("🚩", "Outliers can distort the tree", "A few extreme samples change the clusters a lot.", "🚩 Outliers"),
              _rec("🏁", "Next: models", "Move on to screening when you are happy with the structure.", "🏁 Model screening")]
        return R

    if tab == "otros":
        return [_rec("📈", "Loadings and variable ranking", "Which wavenumbers/variables explain the separation? Compare with known bands.", None),
                _rec("🗺️", "t-SNE / UMAP", "Non-linear maps are good for visualising; never use distances in them as quantitative.", None),
                _rec("🧬", "MCR-ALS", "If you expect a few pure components (e.g. a mixture), MCR gives their spectra and concentrations.", None),
                _rec("🏁", "Next: models", "Use what you learned to decide the preprocessing and the variables.", "🏁 Model screening")]

    if tab == "hyper":
        return [_rec("🎛️", "Choose how ranges are set", "Library defaults, dataset-based recommendation (recalculates automatically when the data "
                     "change) or your own edited ranges.", None),
                _rec("🏷️", "Next: use them", "The ranges are used whenever you tick **Optimize hyperparameters** in Classification/Regression "
                     "or in Model screening.", "🏁 Model screening"),
                _rec("💡", "Rule of thumb", "Start with the recommended ranges; widen them only if the best value lies at the edge of the range.", None)]

    if tab == "screening":
        filas = ss.get("scr_filas") or []
        if not filas:
            R.append(_rec("▶️", "Run a small first screening", "2–3 preprocessings × 3–4 algorithms is enough for a first impression "
                          "(PLS-DA / LDA / SVM for classes; PLS / Ridge / SVR for values).", None))
            return R
        try:
            import screening_en as sc
            tarea = (ss.get("scr_estado") or {}).get("tarea", "Classification")
            df = sc.tabla_ranking(filas, tarea)
            principal = sc.CLAVE_RANKING[tarea][0]
            top = df.iloc[0]
            txt = f"{top.get('Preprocessing', '?')} + {top.get('Algorithm', '?')} ({principal} = {top.get(principal, float('nan')):.3g})"
            R.append(_rec("🏆", "Best combination so far", txt, None))
        except Exception:
            pass
        R += [_rec("🤔", "Are you satisfied with the best results?", "If the top combinations are good and close to each other, take the best 1–3 "
                   "to the refinement step.", "🏷️ Classification / 📉 Regression / 🧬 SIMCA"),
              _rec("🔁", "Not yet? Widen the search", "Add preprocessings, variable selection (Boruta), more algorithms, or crop the spectrum. "
                   "Also check outliers and labels.", None),
              _rec("🎛️", "Tune the search spaces", "In the Hyperparameters tab you can adapt the ranges to your dataset.", "🎛️ Hyperparameters"),
              _rec("⚠️", "Beware of the winner's curse", "The top row is partly lucky. Prefer a combination that is also good in CV and in the neighbours of the ranking.", None)]
        return R

    if tab == "clf":
        b = _mejor_clf(ss)
        if b is None:
            R += [_rec("▶️", "Train your first models", "Start with PLS-DA, LDA and SVM (or Random Forest for tables). Keep 20–30 % for the test set.", None),
                  _rec("🏁", "Not sure which preprocessing?", "Screening can answer it objectively.", "🏁 Model screening")]
            return R
        ba = b["score"]
        gap = (b["cv"] - b["test"]) if (b["cv"] is not None and b["test"] is not None) else None
        if ba >= 0.9 and (gap is None or gap < 0.1):
            R.append(_rec("🟢", f"Metrics look good ({b['nombre']}: balanced accuracy {ba:.2f})",
                          "Do the metrics satisfy you? Then **save the model** and move to predicting new samples. Also look at the confusion matrix "
                          "to see which classes are the most confused.", "🔮 Prediction"))
        elif ba >= 0.7:
            R.append(_rec("🟡", f"Acceptable ({b['nombre']}: balanced accuracy {ba:.2f})",
                          "Try to improve: variable selection (Boruta), hyperparameter optimization, a different preprocessing, or removing outliers.", None))
        else:
            R.append(_rec("🔴", f"Weak ({b['nombre']}: balanced accuracy {ba:.2f})",
                          "Go back: check the PCA/dendrogram (are the classes separable?), the outliers, the labels, and the preprocessing. "
                          "More samples per class help more than a fancier algorithm.", "🧪 Preprocessing"))
        if gap is not None and gap > 0.15:
            R.append(_rec("⚠️", f"Overfitting suspected (CV − test = {gap:.2f})", "Use a simpler model (PLS-DA with fewer components, LDA), fewer variables, "
                          "or more data. Check that replicates are not split between train and test.", None))
        d_ = d
        if d_.get("clases") and d_["min_c"] > 0 and d_["max_c"] / d_["min_c"] > 2:
            R.append(_rec("⚖️", "Unbalanced classes", "Judge the model by balanced accuracy, kappa and MCC, not by accuracy.", None))
        R += [_rec("🧬", "Is it authentication you need?", "If the question is 'is this sample REALLY class A?' (or unknown classes may appear), SIMCA is the right tool.", "🧬 SIMCA"),
              _rec("💾", "Save and reuse", "Saved models keep their preprocessing, variables and traceability: use them any other day.", "🔮 Prediction"),
              _rec("📑", "Document it", "Add the model to the final report.", "📑 Final report")]
        return R

    if tab == "reg":
        b = _mejor_reg(ss)
        if b is None:
            R += [_rec("▶️", "Train your first models", "Start with PLS, Ridge and SVR. For PLS the key hyperparameter is the number of latent variables.", None),
                  _rec("🏁", "Compare preprocessings first", "Screening shows which one helps.", "🏁 Model screening")]
            return R
        rpd = b["rpd_te"] if b["rpd_te"] is not None else b["rpd_cv"]
        r2 = b["score"]
        if rpd is not None and np.isfinite(rpd) and rpd >= 3 and r2 >= 0.9:
            R.append(_rec("🟢", f"Metrics close ({b['nombre']}: R² {r2:.2f}, RPD {rpd:.1f})", "Good for quantification. Save the model and predict new samples.", "🔮 Prediction"))
        elif rpd is not None and np.isfinite(rpd) and rpd >= 2:
            R.append(_rec("🟡", f"Usable for screening ({b['nombre']}: R² {r2:.2f}, RPD {rpd:.1f})", "To reach quantitative level: better preprocessing, "
                          "variable selection, outlier removal, more samples covering the whole range, or a better reference method (its error limits you).", None))
        else:
            R.append(_rec("🔴", f"Weak ({b['nombre']}: R² {r2:.2f}" + (f", RPD {rpd:.1f}" if rpd is not None and np.isfinite(rpd) else "") + ")",
                          "Check the PCA coloured by y (is there a trend?), outliers, the reference values (errors in the lab data?), and the preprocessing.", "🧭 PCA"))
        if b["rmse_te"] is not None and b["rmse_cv"] and b["rmse_te"] > 1.3 * b["rmse_cv"]:
            R.append(_rec("⚠️", "Test error larger than CV error", f"RMSEP {b['rmse_te']:.3g} vs RMSECV {b['rmse_cv']:.3g}: overfitting or a test set unlike the training set. "
                          "Try fewer latent variables or the stratified/SPXY split.", None))
        R += [_rec("🔬", "Look at the plots", "Predicted vs. measured and residuals show non-linearity, heteroscedasticity and outliers that numbers hide.", None),
              _rec("💾", "Save and reuse", "The saved model keeps the preprocessing and its prediction interval.", "🔮 Prediction"),
              _rec("📑", "Document it", "Add it to the final report.", "📑 Final report")]
        return R

    if tab == "simca":
        st_ = _simca_stats(ss)
        if st_ is None:
            return [_rec("▶️", "Fit your first SIMCA", "Use 95 % variance and α = 0.05 as a start. Keep a test set so that sensitivity and specificity are honest.", None),
                    _rec("💡", "When SIMCA", "Authentication, detecting unknown classes, or when classes are not exhaustive.", None)]
        sens = [v[0] for v in st_["por_clase"].values() if np.isfinite(v[0])]
        espec = [v[1] for v in st_["por_clase"].values() if np.isfinite(v[1])]
        ms, me = (np.mean(sens) if sens else np.nan), (np.mean(espec) if espec else np.nan)
        R.append(_rec("📊", "Your result", f"Mean sensitivity {ms:.0%}, mean specificity {me:.0%}, accepted by none {st_['ninguna']:.0%}, ambiguous {st_['varias']:.0%}.", None))
        if np.isfinite(ms) and ms < 0.85:
            R.append(_rec("🔧", "Low sensitivity: models too strict", "Increase α a little, add components (explained variance 97–99 %), check class outliers in training.", None))
        if np.isfinite(me) and me < 0.85:
            R.append(_rec("🔧", "Low specificity: models too permissive", "Fewer components, lower α, better preprocessing or crop to the discriminating region; consider DD-SIMCA.", None))
        if st_["varias"] > 0.2:
            R.append(_rec("🔀", "Many ambiguous samples", "Classes are not well resolved by PCA models; compare with PLS-DA/SVM in Classification.", "🏷️ Classification"))
        if np.isfinite(ms) and np.isfinite(me) and ms >= 0.85 and me >= 0.85:
            R.append(_rec("🟢", "Metrics close", "Save the models and use them to test new samples (accepted / rejected / ambiguous).", "🔮 Prediction"))
        R.append(_rec("📑", "Document it", "Add it to the final report.", "📑 Final report"))
        return R

    if tab == "pred":
        if ss.get("pred_ultimo") is None:
            R += [_rec("💾", "Do you have a saved model?", "Train and save one in Classification, Regression or SIMCA (or load a .joblib from another day).", "🏷️ Classification / 📉 Regression"),
                  _rec("📂", "New samples", "Same kind of file as the training data. Spectra are interpolated to the model axis and preprocessed exactly as in training.", None)]
        else:
            R += [_rec("🛡️", "Check the applicability domain", "Predictions of samples outside the domain are extrapolations — do not report them as reliable.", None),
                  _rec("📏", "Read intervals / prediction sets", "Wide interval or several classes in the set = low confidence.", None),
                  _rec("🔁", "Many samples outside the domain?", "The model is not valid for them: measure similar samples and retrain.", "🏷️ Classification / 📉 Regression"),
                  _rec("📑", "Document it", "Add the predictions to the final report (with the model card).", "📑 Final report")]
        return R

    if tab == "report":
        return [_rec("📥", "Download and archive", "Keep the report together with the saved model (.joblib) and the original data.", None),
                _rec("🔁", "Not finished?", "You can return to any tab, change something and regenerate the report.", None),
                _rec("📚", "Cite your data", "If the data come from a public source, cite its authors (see Public datasets).", "📚 Public datasets"),
                _rec("✉️", "Questions?", "The author is available for questions and meetings — see the Contact tab.", "✉️ Contact")]
    return []


def what_next(tab, key=None, etiqueta="🧭 What's next?"):
    """Popover with context-aware suggestions for the given tab."""
    ss = st.session_state
    _marcar(tab)
    d = _info_datos(ss)
    recs = _recs_para(tab, ss, d)
    if not recs:
        return
    with st.popover(etiqueta, help="Recommendations on how to continue, based on what you have done and on your results.",
                    key=key or f"next_{tab}"):
        st.markdown("### What would I do now?")
        for r in recs:
            st.markdown(f"{r['icono']} **{r['titulo']}**  \n{r['texto']}"
                        + (f"  \n*→ {r['donde']}*" if r["donde"] else ""))
        st.caption("Suggestions only: chemometrics is iterative — going back a step is normal.")


def what_next_remuestreo(res, es_clf, boot, key=None, etiqueta="🧭 What's next?"):
    """Popover with suggestions based on the resampling results (dict model -> DataFrame of repetitions)."""
    if not res:
        return
    metrica = "Balanced accuracy" if es_clf else "R²"
    medias, sds = {}, {}
    for nm, df in res.items():
        if metrica in df.columns:
            medias[nm] = float(df[metrica].mean()); sds[nm] = float(df[metrica].std(ddof=1))
    recs = []
    if medias:
        mejor = max(medias, key=medias.get)
        sd_max = max(sds.values())
        recs.append(("🏆", "Best on average", f"**{mejor}** has the highest mean {metrica} ({medias[mejor]:.3f} ± {sds[mejor]:.3f})."))
        ord_ = sorted(medias, key=medias.get, reverse=True)
        if len(ord_) > 1:
            a, b2 = ord_[0], ord_[1]
            if medias[a] - medias[b2] < max(sds[a], sds[b2]):
                recs.append(("⚖️", "No clear winner", f"{a} and {b2} differ by less than their SD: you cannot say one is really better. Prefer the simpler/more interpretable one."))
        if sd_max > 0.05:
            recs.append(("🎲", "High variability", "The result depends strongly on the partition. Get more samples, use a simpler model or fewer variables, and always report mean ± SD rather than one split."))
        else:
            recs.append(("✅", "Stable result", "The SD is small, so one split is representative. You can go on and save the model."))
    recs.append(("📝", "Document it", "Write your conclusions in the observations box under the plot: they are printed in the Final report together with this table."))
    recs.append(("➡️", "Next step", "Save the chosen model in this tab and use it in 🔮 Prediction, or go to 📑 Final report."))
    with st.popover(etiqueta, help="Recommendations based on your resampling results.", key=key or "next_resampling"):
        st.markdown("### What would I do now?")
        for ic, t, tx in recs:
            st.markdown(f"{ic} **{t}**  \n{tx}")
        st.caption("Suggestions only: chemometrics is iterative — going back a step is normal.")


def barra_guia(tab, kind_interpretar=None, valores=None, pros=None):
    """Compact row of guidance buttons shown at the top of a tab."""
    cols = st.columns([1.5, 1.9, 1.6, 3])
    with cols[0]:
        what_next(tab)
    if kind_interpretar:
        with cols[1]:
            how_to_interpret(kind_interpretar, valores=valores, key=f"how_{tab}_{kind_interpretar}")
    if pros:
        with cols[2]:
            kind, sel = pros if isinstance(pros, tuple) else (pros, None)
            pros_cons(kind, sel, key=f"pc_{tab}_{kind}")


# ---------------------------------------------------------------------------
# Own values to show next to the explanations ("Your value: 🟢 0.93")
# ---------------------------------------------------------------------------
def valores_clf(ss=None):
    ss = ss or st.session_state
    res = ss.get("clf_resultados") or {}
    best, bs = None, -1
    for r in res.values():
        if not isinstance(r, dict) or "cv" not in r:
            continue
        src = r.get("test") or r["cv"]
        s = src.get("balanced_accuracy", -1)
        if s is not None and s > bs:
            best, bs = src, s
    if best is None:
        return None
    keys = ("accuracy", "balanced_accuracy", "sensibilidad_macro", "especificidad_macro", "f1_macro", "kappa", "mcc", "auc")
    return {k: best.get(k) for k in keys if best.get(k) is not None}


def valores_reg(ss=None):
    ss = ss or st.session_state
    b = _mejor_reg(ss or st.session_state)
    if not b:
        return None
    return {"r2": b["r2_te"] if b["r2_te"] is not None else b["r2_cv"],
            "rpd": b["rpd_te"] if b["rpd_te"] is not None else b["rpd_cv"]}


def valores_simca(ss=None):
    s = _simca_stats(ss or st.session_state)
    if not s:
        return None
    sens = [v[0] for v in s["por_clase"].values() if np.isfinite(v[0])]
    esp = [v[1] for v in s["por_clase"].values() if np.isfinite(v[1])]
    out = {}
    if sens:
        out["sens"] = float(np.mean(sens))
    if esp:
        out["espec"] = float(np.mean(esp))
    return out or None


# ---------------------------------------------------------------------------
# OBSERVATIONS under every plot (printed in the final report)
# ---------------------------------------------------------------------------
def _guardar_nota(plot_id, titulo, seccion):
    ss = st.session_state
    txt = ss.get(f"nota_{plot_id}", "")
    notas = ss.setdefault("notas", {})
    if txt.strip():
        notas[plot_id] = {"titulo": titulo, "texto": txt.strip(), "seccion": seccion}
    else:
        notas.pop(plot_id, None)


def nota(plot_id, titulo, seccion):
    """Text box for the user's own observations about the plot just drawn. `seccion` is the key of the section of the
    Final report where the note will be printed (pca, outliers, dendro, otros_loadings…, screening, clf, simca, reg, pred, dataset)."""
    ss = st.session_state
    guardada = (ss.get("notas") or {}).get(plot_id, {}).get("texto", "")
    try:
        cont = st.expander(("✏️ " if guardada else "📝 ") + f"Your observations on this plot — {titulo}", expanded=bool(guardada))
    except Exception:                      # already inside an expander (they cannot be nested)
        cont = st.container(border=True)
    with cont:
        st.text_area("Observations", value=guardada, key=f"nota_{plot_id}", height=90,
                     label_visibility="collapsed",
                     placeholder="What do you see? Groups, outliers, trends, decisions you take… "
                                 "These notes are printed next to the plot in the Final report.",
                     on_change=_guardar_nota, args=(plot_id, titulo, seccion),
                     help="Your own words. They are saved with the session/project and can be included in the Final report.")


def notas_por_seccion(ss):
    """{seccion: [(titulo, texto), ...]} in the order they were written."""
    out = {}
    for pid, n in (ss.get("notas") or {}).items():
        out.setdefault(n.get("seccion", "otros"), []).append((n["titulo"], n["texto"]))
    return out
