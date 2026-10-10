# -*- coding: utf-8 -*-
"""'Glossary & cheat sheet' tab: metric cheat sheet, glossary, decision guide and common pitfalls."""
import streamlit as st
import guia_utils as gu
from i18n import T

GLOSARIO = {
    "Absorbance": "−log₁₀ of the reflectance/transmittance. The usual unit for NIR/MIR so that signal is roughly proportional to concentration.",
    "Applicability domain": "The region of the data space where a model's predictions are trustworthy: samples that look like the training samples.",
    "Autoscaling": "Mean-centering plus dividing each variable by its standard deviation, so all variables have the same weight.",
    "Baseline": "Slowly varying background of a spectrum/chromatogram (scatter, fluorescence, drift) that is not chemical information of interest.",
    "Bias (model)": "Systematic difference between predictions and reference values (mean error). Should be close to zero.",
    "Calibration": "Building a model that relates the signal to a property (regression) — also the training part of a model.",
    "Class imbalance": "Classes with very different numbers of samples. Accuracy becomes misleading; use balanced accuracy, kappa or MCC.",
    "Confusion matrix": "Table of true class (rows) vs predicted class (columns).",
    "Cross-validation (CV)": "Repeatedly leaving part of the samples out, training on the rest and predicting the left-out part, to estimate performance on new samples.",
    "Data leakage": "Information from the test samples slipping into training (e.g. choosing preprocessing, variables or hyperparameters using all the data). It makes results look better than they are.",
    "Derivative (Savitzky–Golay)": "Smoothing + derivative in one step. Removes offsets (1st) and slopes (2nd) and sharpens overlapping bands.",
    "DD-SIMCA": "Data-driven SIMCA: one acceptance area built from the score and orthogonal distances, with a known false-alarm rate.",
    "Dendrogram": "Tree showing how samples are merged into clusters by similarity.",
    "EMSC / MSC": "(Extended) multiplicative scatter correction: removes scatter effects using a reference spectrum.",
    "F1-score": "Harmonic mean of precision and recall.",
    "Hotelling's T²": "Distance of a sample to the centre of the data inside the PCA model.",
    "Hyperparameter": "A setting of an algorithm chosen by the user (number of components, C, gamma, number of trees…) rather than learned from the data.",
    "Kennard–Stone / SPXY": "Algorithms that select a representative set of samples (SPXY also uses the reference values).",
    "Latent variable (LV)": "A component of PLS: a direction in the variable space that explains X and is related to y.",
    "Leverage": "How far a sample is from the centre of the model space; high-leverage samples have a large influence on the model.",
    "Loadings": "Weights of the original variables in a component.",
    "LOD / LOQ": "Limit of detection / quantification of an analytical method.",
    "Mean centering": "Subtracting the mean spectrum/variable mean from every sample.",
    "MCR-ALS": "Multivariate curve resolution: decomposes mixtures into pure-component spectra and concentrations.",
    "Overfitting": "A model that memorises the training samples (including noise) and fails on new ones. Symptom: very good calibration, much worse CV/test.",
    "PCA": "Principal component analysis: rotates the data to new axes (PCs) ordered by explained variance. Used to explore, compress and detect outliers.",
    "PLS / PLS-DA": "Partial least squares regression / discriminant analysis: finds components that explain X and correlate with y (values or classes).",
    "PQN": "Probabilistic quotient normalisation: corrects dilution effects (NMR, metabolomics).",
    "Q residual (SPE)": "Part of a sample not explained by the PCA model (distance to the model plane).",
    "Replicates": "Repeated measurements of the same sample. Keep them together in the same fold when validating.",
    "Residual": "Measured minus predicted value.",
    "RMSEC / RMSECV / RMSEP": "Root mean square error of calibration / cross-validation / prediction (test set).",
    "ROC curve / AUC": "Sensitivity vs 1−specificity for all thresholds; AUC is the area below (0.5 random, 1 perfect).",
    "RPD / RPIQ": "Ratio of the spread of the reference values (SD or interquartile range) to the prediction error.",
    "Scores": "Coordinates of the samples on the components.",
    "Sensitivity (recall)": "Fraction of the samples of a class that are correctly found.",
    "SIMCA": "Soft independent modelling of class analogy: one PCA model per class; a sample is accepted or rejected by each class model.",
    "SNV": "Standard normal variate: centres and scales each spectrum by its own mean and SD to remove scatter.",
    "Specificity": "Fraction of the samples NOT in a class that are correctly rejected.",
    "Test set": "Samples never used for any decision during modelling, kept to estimate the final performance honestly.",
    "VIP": "Variable importance in projection: importance of each variable in a PLS model (VIP > 1 usually 'important').",
    "Williams plot": "Standardised residuals vs leverage; shows outliers and high-influence samples in regression.",
}

DECISIONES = [
    ("I want to see how my samples are related", "PCA → Outliers → Dendrogram"),
    ("I want to know whether two or more groups can be distinguished", "PCA coloured by class → Model screening → Classification (PLS-DA / LDA / SVM)"),
    ("I want to confirm that a sample really belongs to one class (authentication)", "SIMCA / DD-SIMCA"),
    ("I want to predict a number (concentration, property)", "Preprocessing → Model screening → Regression (PLS first)"),
    ("I have a table of clinical / metabolomics variables", "Data (tabular mode) → PCA (autoscaled) → Classification / Regression / SIMCA"),
    ("I do not know which preprocessing to use", "Model screening with several preprocessing recipes"),
    ("My model is good in cross-validation but bad on new data", "Check outliers, replicates, domain (Prediction tab), preprocessing differences, and use a test set"),
    ("I have very few samples", "Use LOO or repeated CV, simple models (PLS, LDA), no variable selection, and report uncertainty"),
    ("I want to reuse my model another day", "Save it in Classification / Regression / SIMCA (a .joblib with its traceability) → Prediction tab"),
]

ERRORES = [
    ("Choosing the preprocessing or variables with ALL the data and then validating", "Information leaks into the validation. Keep a test set aside from the beginning."),
    ("Replicates split between training and validation", "The model 'recognises' the sample. Keep replicates in the same fold/set."),
    ("Trusting R² of calibration", "It always looks good. Report cross-validation and, above all, the test set."),
    ("Removing outliers until the model looks good", "Remove only what you can justify physically, and report it."),
    ("Too many latent variables or too complex a model", "Fits noise. Choose the simplest model within the noise of the best one."),
    ("Ignoring unbalanced classes", "Use balanced accuracy / kappa / MCC and the confusion matrix."),
    ("Predicting samples unlike the training set", "Always check the applicability domain."),
    ("Not recording how the data were measured", "Keep the instrument, date, operator and sample preparation: batches often explain more than classes."),
]


def render():
    st.subheader("Glossary & cheat sheet")
    t1, t2, t3, t4 = st.tabs(["📏 Metrics cheat sheet", "🔤 Glossary", "🧭 Which tool for what?", "⚠️ Common pitfalls"])
    with t1:
        st.caption("The same explanations are available through the **How to interpret?** buttons next to your results.")
        for k, titulo in (("clf", "Classification"), ("reg", "Regression"), ("simca", "SIMCA"), ("outliers", "Outliers (T² and Q)"),
                          ("pca", "PCA"), ("dendro", "Dendrogram"), ("pred", "Prediction")):
            info = gu.INTERPRETAR[k]
            with st.expander(f"{titulo}", expanded=(k == "clf")):
                for it in info["items"]:
                    st.markdown(f"**{it[0]}** — {it[1]}  \n*How to read it:* {it[2]}" + (f"  \n*Rule of thumb:* {it[3]}" if it[3] else ""))
    with t2:
        q = st.text_input("Search a term", key="glos_q", placeholder="e.g. overfitting, SNV, leverage…")
        n = 0
        for term, txt in sorted(GLOSARIO.items(), key=lambda x: x[0].lower()):
            if q and q.lower() not in (term + " " + txt + " " + T(term) + " " + T(txt)).lower():
                continue
            n += 1
            st.markdown(f"**{term}** — {txt}")
        if n == 0:
            st.info("No term matches your search.")
    with t3:
        for q_, a_ in DECISIONES:
            st.markdown(f"**{q_}**  \n→ {a_}")
    with t4:
        for e, c in ERRORES:
            st.markdown(f"❌ **{e}**  \n✅ {c}")
