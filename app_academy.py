"""
app_en.py — Espectrometrika
Streamlit application: spectra/chromatogram loading, preprocessing, PCA,
outlier detection, cluster analysis, classification, regression and
prediction on new samples — with a point-and-click interface (no coding
required). Includes model persistence, hyperparameter optimization, and a
downloadable traceability card (model card) for every trained model.

To run it:  streamlit run app_en.py
"""

import io
import os
import re
import sys
import time
import logging
import hashlib
import functools
import base64
import joblib
import smtplib
import datetime
from email.message import EmailMessage
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException
import matplotlib.pyplot as plt
import importlib.util
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA

import chemo_utils as cu
import pc_finder as pcf
import nl_finder as nlf
import ref_tab
import tabular_utils as tu
import asignaciones as asg
import confianza_utils as cf
import models_utils_en as mu
import guia_utils as gu
import i18n
from i18n import T
import hyperparam_utils as hu
import reporte_utils_en as ru
try:
    import screening_en as sc     # spectral crop + model screening
except ImportError:
    sc = None
try:
    import informe_final_en as inf   # combined 'Final report' tab
except ImportError:
    inf = None

st.set_page_config(
    page_title="Espectrometrika.edu",
    page_icon=os.path.join(os.path.dirname(__file__), "assets", "favicon.png"),
    layout="wide",
)

# Copyright footer (fixed at the bottom of every page, also on the Home screen)
COPYRIGHT_TEXTO = "Espectrometrika.edu · All rights reserved"
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "espectrometrika@gmail.com")   # shown in the Contact tab (mailto link)
EDICION = "Academy"
i18n.install()                                  # English / Castellano switch (per user session)
_top_cols = st.columns([2.6, 4.2, 2.4])
with _top_cols[1]:
    _niv = st.radio("Level", ["ini", "exp"], index=1, horizontal=True, key="nivel_sel", label_visibility="collapsed",
                    format_func=lambda k: T("🌱 First steps") if k == "ini" else T("🔬 I have experience"),
                    help=T("First steps keeps the essential workflow (load data, preprocessing, PCA, outliers, dendrogram, screening, classification, regression, prediction, report). The advanced tools (PC finder, non-linear finder, consensus, hyperparameters, SIMCA, other tools, resampling, variable selection and hyperparameter optimization) stay visible but greyed out. 'I have experience' unlocks everything."))
    st.session_state["_nivel"] = _niv
    st.caption(T("🌱 First steps: essentials only · 🔬 Experience: everything unlocked"))
with _top_cols[2]:
    i18n.language_switch()
_BASICO = (st.session_state.get("_nivel", "exp") == "ini")
import streamlit.components.v1 as _components
_components.html("""<script>
(function(){
  try{
    var d = window.parent.document;
    function lock(){
      d.querySelectorAll('[role="tab"]').forEach(function(t){
        if ((t.innerText||'').trim().indexOf('\\ud83d\\udd12') === 0){
          t.style.pointerEvents='none'; t.style.cursor='not-allowed'; t.setAttribute('aria-disabled','true'); t.tabIndex=-1;
        } else { t.style.pointerEvents=''; t.removeAttribute('aria-disabled'); }
      });
    }
    lock();
    if(!window.parent.__espLockObs){
      window.parent.__espLockObs = new MutationObserver(lock);
      window.parent.__espLockObs.observe(d.body,{childList:true,subtree:true,characterData:true});
    }
  }catch(e){}
})();
</script>""", height=0)
st.markdown(f"""
<style>
.block-container {{ padding-bottom: 3.5rem; }}
.app-footer {{
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 90;
    text-align: right; font-size: 14px; font-weight: 500; color: #0B3D54;
    padding: 6px 18px; border-top: 1px solid rgba(128,128,128,.25);
    background: rgba(255,255,255,.88); backdrop-filter: blur(4px);
}}
@media (prefers-color-scheme: dark) {{
    .app-footer {{ background: rgba(14,17,23,.88); color: #7FB8D0; }}
}}
</style>
<div class="app-footer">{COPYRIGHT_TEXTO} · Questions: <a href="mailto:{CONTACT_EMAIL}" style="color:inherit;">{CONTACT_EMAIL}</a></div>
""", unsafe_allow_html=True)

# Brand palette (kept as module-level constants so other parts of the app can reuse them)
PETROLEUM = "#0B3D54"
PETROLEUM_LIGHT = "#154D6B"
TEAL = "#14B8A6"
TEAL_LIGHT = "#5EEAD4"
TEAL_SOFT = "#E6FBF7"


@functools.lru_cache(maxsize=16)
@functools.lru_cache(maxsize=16)
def _img_b64(nombre_archivo, max_w=None, max_h=None, paleta=False):
    """Reads an image from assets/ and returns it as a base64 data URI for inline HTML.
    Done ONCE per process (cached) and downscaled to the size it is really displayed at:
    the original files are huge (up to ~850 KB) and were being re-encoded and re-sent to
    the browser on every interaction."""
    ruta = os.path.join(os.path.dirname(__file__), "assets", nombre_archivo)
    try:
        try:
            from PIL import Image
            im = Image.open(ruta)
            if max_w or max_h:
                im.thumbnail((max_w or im.width, max_h or im.height), Image.LANCZOS)
            buf = io.BytesIO()
            if paleta:
                im.convert("RGBA").quantize(256, method=Image.FASTOCTREE).save(buf, "PNG", optimize=True)
            else:
                im.save(buf, "PNG", optimize=True)
            datos = base64.b64encode(buf.getvalue()).decode()
        except Exception:                       # no PIL / odd file: send it as it is
            with open(ruta, "rb") as f:
                datos = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{datos}"
    except FileNotFoundError:
        return None


def _traza_espectro(eje, y, gl=False, **kw):
    """One spectrum as a Plotly trace, light on the wire: values as float32 and, when the
    spectral axis is evenly spaced, just (x0, dx) instead of repeating the whole axis in
    every trace. A 264 x 700 spectra plot goes from ~6.5 MB to ~1 MB sent to the browser."""
    eje = np.asarray(eje, dtype=float)
    y = np.asarray(y, dtype=np.float32)
    if len(eje) > 3000:
        # Very high resolution spectra (NMR with tens of thousands of points): a screen only has
        # ~1500 pixels across, so draw the min and max of each block of points. The shape and every
        # peak look identical, but the browser gets ~2000 points instead of up to 29000 per spectrum.
        # DISPLAY ONLY - the analyses always use the full-resolution data.
        _nb = 1000
        _lim = (len(y) // _nb) * _nb
        _b = np.nan_to_num(y[:_lim], nan=0.0).reshape(_nb, -1)
        _i0 = np.arange(_nb)[:, None] * _b.shape[1]
        _idx = np.sort(np.concatenate([(_i0[:, 0] + _b.argmin(axis=1)), (_i0[:, 0] + _b.argmax(axis=1))]))
        _idx = np.unique(np.concatenate([[0], _idx, [len(y) - 1]]))
        eje, y = eje[_idx], y[_idx]
    d = np.diff(eje)
    if len(eje) > 2 and np.allclose(d, d[0], rtol=1e-6, atol=0):
        extra = dict(x0=float(eje[0]), dx=float(d[0]))
    else:
        extra = dict(x=eje.astype(np.float32))
    return (go.Scattergl if gl else go.Scatter)(y=y, **extra, **kw)


CONTACTO_DESTINATARIO = "espectrometrika@gmail.com"


def _zona_horaria_usuario():
    """IANA timezone of the visitor's browser (e.g. 'America/Argentina/Buenos_Aires'),
    or None if it can't be read (older Streamlit, or no browser attached)."""
    try:
        return st.context.timezone
    except Exception:
        return None


def _fecha_hora_informe(con_segundos=False):
    """Current date and time in the USER's timezone (not the server's), always
    with the UTC offset so it can never be ambiguous."""
    return ru.ahora_str(_zona_horaria_usuario(), con_segundos=con_segundos)


def _datos_tabulares(X, nombres, clases):
    """Data tab for tabular variables: summary table, distributions and correlation map."""
    p = X.shape[1]
    nombres = np.array([str(n) for n in nombres])
    n_nan = int(np.isnan(X).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", X.shape[0])
    c2.metric("Variables", p)
    c3.metric("Missing values", f"{n_nan} ({100 * n_nan / X.size:.1f} %)")
    with np.errstate(all="ignore"):
        resumen = pd.DataFrame({
            "variable": nombres, "mean": np.nanmean(X, axis=0), "sd": np.nanstd(X, axis=0, ddof=1),
            "min": np.nanmin(X, axis=0), "median": np.nanmedian(X, axis=0), "max": np.nanmax(X, axis=0),
            "missing": np.isnan(X).sum(axis=0),
        })
        resumen["CV %"] = 100 * resumen["sd"] / resumen["mean"].abs().replace(0, np.nan)
    st.markdown("**Summary per variable**")
    st.dataframe(resumen.round(4), width='stretch', height=min(420, 38 + 35 * min(p, 10)))
    orden = np.argsort(-np.nan_to_num(np.nanstd(X, axis=0) / np.maximum(np.abs(np.nanmean(X, axis=0)), 1e-12)))
    k = st.slider("Variables shown in the plots", 5, int(min(60, p)), int(min(25, p)), key="tab_w_nvar", help="Only affects the plots (not the analysis). Fewer variables = faster and cleaner plots.") if p > 5 else p
    sel = np.sort(orden[:k]) if p > k else np.arange(p)
    Z = (X[:, sel] - np.nanmean(X[:, sel], axis=0)) / np.where(np.nanstd(X[:, sel], axis=0) > 0, np.nanstd(X[:, sel], axis=0), 1)
    figb = go.Figure()
    for j, jj in enumerate(sel):
        figb.add_trace(go.Box(y=Z[:, j], name=nombres[jj], boxpoints="outliers", marker_size=3, showlegend=False))
    figb.update_layout(height=420, title="Distribution of each variable (z-scores, for comparison)", yaxis_title="z-score")
    st.plotly_chart(figb, width='stretch')
    if len(sel) >= 3:
        C = pd.DataFrame(X[:, sel], columns=nombres[sel]).corr().to_numpy()
        figc = go.Figure(go.Heatmap(z=C, x=nombres[sel], y=nombres[sel], zmin=-1, zmax=1, colorscale="RdBu_r",
                                    colorbar=dict(title="r")))
        figc.update_layout(height=620, title="Correlation between variables (Pearson)", yaxis=dict(autorange="reversed"))
        st.plotly_chart(figc, width='stretch')
        st.caption("Strongly correlated variables carry redundant information; PCA/PLS handle that well, "
                   "but it matters for interpreting which variable 'drives' a result.")


@st.cache_data(show_spinner=False, max_entries=12, ttl=3600)
def _prep_tabular_cacheado(X_sub, imputacion, norm_filas, transformacion, escalado):
    P = tu.Preprocesador(imputacion, norm_filas, transformacion, escalado)
    return P.fit_transform(X_sub), P


def _pret_tabular():
    """Preprocessing tab for tabular variables (metabolomics / clinical / other)."""
    st.subheader("Preprocessing of tabular variables")
    gu.barra_guia("prep", pros=("prep_spectral", None))
    st.caption("Spectral corrections (derivatives, SNV, baseline...) make no sense for named variables. "
               "Here: clean the variables, then choose imputation → row normalisation → transformation → scaling. "
               "Scaling matters a lot: with raw values, the variable with the largest numbers dominates PCA/PLS.")
    X, nombres = st.session_state.X, np.array([str(n) for n in st.session_state.nombres_var])
    eje0 = np.asarray(st.session_state.numeros_onda, dtype=float)
    p = X.shape[1]
    with np.errstate(all="ignore"):
        falt = np.isnan(X).mean(axis=0)
        cte = (np.nanstd(X, axis=0) <= 1e-12) | np.isnan(np.nanstd(X, axis=0))
    c1, c2, c3 = st.columns(3)
    max_falt = c1.slider("Drop variables with more than … % missing", 0, 100, 50, key="pret_w_tab_falt", help="Variables with a larger share of empty cells than this are removed before analysis. Lower = stricter.")
    quitar_cte = c2.checkbox("Drop constant variables", value=True, key="pret_w_tab_cte", help="Variables with the same value in every sample carry no information (and break autoscaling), so they are removed.")
    excl = c3.multiselect("Also exclude these variables", list(nombres), key="pret_w_tab_excl", help="Variables you want to leave out of every analysis (e.g. IDs, duplicated measurements).") if p <= 3000 else []
    keep = (falt * 100 <= max_falt)
    if quitar_cte:
        keep &= ~cte
    if excl:
        keep &= ~np.isin(nombres, excl)
    st.caption(f"Variables kept: {int(keep.sum())}/{p}"
               + (f" — dropped: {int((falt * 100 > max_falt).sum())} for missing values, "
                  f"{int((quitar_cte & cte).sum())} constant." if keep.sum() < p else "."))
    if keep.sum() < 2:
        st.error("At least 2 variables are needed — relax the filters above.")
        return

    o1, o2, o3, o4 = st.columns(4)
    imp = o1.selectbox("1. Missing values", ["Median", "Mean", "Half of the minimum", "Zero"], key="pret_w_tab_imp",
                       help="Applied per variable. 'Half of the minimum' is common in metabolomics for values "
                            "below the detection limit.")
    nf = o2.selectbox("2. Row normalisation", ["None", "Sum", "Median", "PQN"], key="pret_w_tab_nf",
                      help="Corrects dilution / total-amount differences between samples (urine, plasma...). "
                           "Leave at None for clinical variables.")
    tr = o3.selectbox("3. Transformation", ["None", "log10", "ln", "log2", "Square root", "glog"], key="pret_w_tab_tr",
                      help="Reduces skewness and makes variance less dependent on concentration. "
                           "glog tolerates zeros and negatives.")
    es = o4.selectbox("4. Scaling", ["Autoscaling", "Pareto", "Mean centering", "Range", "Level", "Robust (median/MAD)", "None"],
                      key="pret_w_tab_es",
                      help="Autoscaling (mean 0, sd 1) gives every variable the same weight — the usual choice for "
                           "variables in different units. Pareto keeps part of the size information (typical in "
                           "metabolomics / NMR). Mean centering only keeps the original scale.")
    m_imp = {"Median": "median", "Mean": "mean", "Half of the minimum": "min2", "Zero": "zero"}[imp]
    m_nf = {"None": "none", "Sum": "sum", "Median": "median", "PQN": "pqn"}[nf]
    m_tr = {"None": "none", "log10": "log10", "ln": "ln", "log2": "log2", "Square root": "sqrt", "glog": "glog"}[tr]
    m_es = {"Autoscaling": "auto", "Pareto": "pareto", "Mean centering": "center", "Range": "range",
            "Level": "level", "Robust (median/MAD)": "robust", "None": "none"}[es]

    X_sub, eje_sub = X[:, keep], eje0[keep]
    try:
        X_pret, P = _prep_tabular_cacheado(X_sub, m_imp, m_nf, m_tr, m_es)
    except Exception as e:
        st.error(f"Could not apply this combination: {e}")
        return
    if not np.isfinite(X_pret).all():
        st.error("The result has non-finite values (e.g. log of a variable with all zeros). Change the transformation.")
        return
    desc = P.describir() or "none"
    st.caption("Combination in the preview: " + desc)
    propuesta = repr(("tab", m_imp, m_nf, m_tr, m_es, hash(keep.tobytes())))
    ya = (st.session_state.get("pret_propuesta_aplicada") == propuesta)

    k = min(15, X_sub.shape[1])
    sel = np.argsort(-np.nan_to_num(np.nanstd(X_sub, axis=0) / np.maximum(np.abs(np.nanmean(X_sub, axis=0)), 1e-12)))[:k]
    nm = nombres[keep][sel]
    cA, cB = st.columns(2)
    for col, M, tit in ((cA, X_sub[:, sel], "Before (raw values)"), (cB, X_pret[:, sel], "After")):
        f = go.Figure()
        for j in range(k):
            f.add_trace(go.Box(y=M[:, j], name=nm[j], boxpoints=False, showlegend=False))
        f.update_layout(title=tit + f" — {k} most variable", height=360)
        col.plotly_chart(f, width='stretch')

    if ya:
        st.success("✅ This combination is the one currently applied to the analysis.")
    else:
        st.warning("⚠️ This is only a **preview**. The rest of the app is using: "
                   f"**{st.session_state.get('pret_desc_aplicada', 'none')}**. Click **Apply** to use this one.")
    b1, b2, _ = st.columns([1.5, 1.5, 3])
    if b1.button("✅ Apply to the analysis", key="pret_btn_aplicar", type="secondary" if ya else "primary", disabled=ya):
        st.session_state.X_pret = X_pret
        st.session_state.numeros_onda_pret = eje_sub
        st.session_state.pasos_pretratamiento = []
        st.session_state.tab_prep = P
        st.session_state.pret_propuesta_aplicada = propuesta
        st.session_state.pret_desc_aplicada = desc
        _rerun_tab()
    if b2.button("↩ Reset (raw values)", key="pret_btn_reset"):
        resetear_prefijo("pret_w_")
        reiniciar_pretratamiento()
        _rerun_tab()
    nprev = min(40, X_pret.shape[1])
    st.caption(f"Preview of the 'After' data: first 10 samples × first {nprev} variables.")
    st.dataframe(pd.DataFrame(X_pret[:10, :nprev], index=st.session_state.ids[:10], columns=nombres[keep][:nprev]),
                 width='stretch')


def _generar_reporte(*args, **kwargs):
    """ru.generar_reporte, always stamping the report with date and time."""
    kwargs.setdefault("fecha_hora", _fecha_hora_informe())
    return ru.generar_reporte(*args, **kwargs)


def _generar_pdf_ficha(*args, **kwargs):
    """ru.generar_pdf_ficha_modelo, always stamping the PDF with date and time."""
    kwargs.setdefault("fecha_hora", _fecha_hora_informe())
    return ru.generar_pdf_ficha_modelo(*args, **kwargs)


def enviar_mensaje_contacto(mensaje, nombre_remitente, email_remitente):
    """Sends a feedback message to the app maintainer's inbox.

    Tries Formspree first (simplest to set up — no email credentials needed,
    just a free account at formspree.io), and falls back to sending via SMTP
    if Formspree isn't configured but SMTP secrets are.

    Expected structure in .streamlit/secrets.toml, EITHER of:

        [formspree]
        endpoint = "https://formspree.io/f/your-form-id"

    OR

        [smtp]
        server = "smtp.your-provider.com"
        port = 587
        user = "your-sending-address@example.com"
        password = "your-app-password"

    Returns (exito: bool, detalle: str).
    """
    cuerpo = mensaje.strip()
    if nombre_remitente.strip():
        cuerpo += f"\n\n— Sent by: {nombre_remitente.strip()}"
    if email_remitente.strip():
        cuerpo += f" ({email_remitente.strip()})"
    cuerpo += f"\n— Sent from Espectrometrika.edu on {datetime.datetime.now():%Y-%m-%d %H:%M}"

    # --- Option 1: Formspree (no email credentials needed) ---
    # Checked in two possible places, since different hosting platforms expose
    # configuration differently:
    #   - Posit Connect Cloud "Secret variables" become plain environment
    #     variables, so we check os.environ first.
    #   - Local development / Streamlit Community Cloud use a
    #     .streamlit/secrets.toml file, read through st.secrets.
    endpoint = os.environ.get("FORMSPREE_ENDPOINT")
    if not endpoint:
        try:
            endpoint = st.secrets["formspree"]["endpoint"]
        except Exception:
            endpoint = None
    if endpoint:
        # Defensive cleanup: strip whitespace and a common copy/paste mistake
        # (pasting a label like "URL: " or "Endpoint: " along with the value).
        endpoint = endpoint.strip()
        for _prefijo in ("URL:", "Url:", "url:", "Endpoint:", "endpoint:"):
            if endpoint.startswith(_prefijo):
                endpoint = endpoint[len(_prefijo):].strip()
        if not endpoint.startswith(("http://", "https://")):
            return False, (f"The configured Formspree endpoint doesn't look like a valid URL "
                            f"({endpoint!r}). Check the FORMSPREE_ENDPOINT value — it should be "
                            f"only the URL, e.g. https://formspree.io/f/xxxxxxxx, nothing else.")
        try:
            import requests
            respuesta = requests.post(
                endpoint,
                data={"message": cuerpo, "name": nombre_remitente.strip(),
                      "_replyto": email_remitente.strip() or "no-reply@espectrometrika.app"},
                headers={"Accept": "application/json"},
                timeout=15,
            )
            if respuesta.status_code in (200, 202):
                return True, "Message sent — thank you!"
            return False, f"Could not send the message right now (Formspree said: {respuesta.status_code})."
        except Exception as e:
            return False, f"Could not send the message right now ({e})."

    # --- Option 2: SMTP fallback ---
    # Same idea: plain env vars first (Posit Connect Cloud), then secrets.toml.
    if os.environ.get("SMTP_SERVER"):
        smtp_cfg = {
            "server": os.environ.get("SMTP_SERVER"),
            "port": os.environ.get("SMTP_PORT", 587),
            "user": os.environ.get("SMTP_USER"),
            "password": os.environ.get("SMTP_PASSWORD"),
        }
    else:
        try:
            smtp_cfg = st.secrets["smtp"]
        except Exception:
            return False, ("Feedback form is not configured yet (missing Formspree or SMTP "
                            "configuration). Ask the app maintainer to set it up.")

    msg = EmailMessage()
    msg["Subject"] = "Espectrometrika.edu — New feedback message"
    msg["From"] = smtp_cfg["user"]
    msg["To"] = CONTACTO_DESTINATARIO
    if email_remitente.strip():
        msg["Reply-To"] = email_remitente.strip()
    msg.set_content(cuerpo)

    try:
        with smtplib.SMTP(smtp_cfg["server"], int(smtp_cfg.get("port", 587))) as servidor:
            servidor.starttls()
            servidor.login(smtp_cfg["user"], smtp_cfg["password"])
            servidor.send_message(msg)
        return True, "Message sent — thank you!"
    except Exception as e:
        return False, f"Could not send the message right now ({e})."

# Custom CSS for a more polished, "scientific software" look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600;700&family=Nunito+Sans:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Nunito Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {padding-top: 3.5rem;}
    h1, h2, h3, h4 {color: #0B3D54; font-family: 'Montserrat', sans-serif; font-weight: 600;}
    div[data-testid="stMetricValue"] {color: #0B3D54;}
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        flex-wrap: wrap !important;
        row-gap: 2px !important;
        overflow-x: visible !important;
        white-space: normal !important;
        height: auto !important;
    }
    .stTabs [data-baseweb="tab-border"] {display: none !important;}
    .stTabs [data-baseweb="tab-highlight"] {display: none !important;}
    /* Hide the left/right scroll-arrow buttons Streamlit adds for overflowing
       tab bars — with wrapping forced above, they're not needed and were
       hiding tabs behind them instead of showing everything. */
    .stTabs button[data-testid="stTabsScrollButton"] {display: none !important;}
    .stTabs [data-baseweb="tab-list"] > div:first-child:not([role="tab"]) {display: none !important;}
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 6px 10px !important;
        font-weight: 500;
        font-size: 0.88rem !important;
    }
    .stTabs [aria-selected="true"] {
        color: #0B3D54 !important;
        border-bottom-color: #14B8A6 !important;
    }
    /* "Home" is always the first tab — give it its own accent color so it
       stands out from the rest, even when it isn't the selected tab. */
    .stTabs [data-baseweb="tab-list"] button:first-child {
        color: #D97706 !important;
    }
    .stTabs [data-baseweb="tab-list"] button:first-child[aria-selected="true"] {
        color: #D97706 !important;
        border-bottom-color: #D97706 !important;
    }
    .stTabs [data-baseweb="tab-list"] button:first-child p {
        color: #D97706 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stSidebar"] {
        border-right: 1px solid rgba(11, 61, 84, 0.12);
    }

    /* ---- tabs: Streamlit >= 1.6x renders them with role="tab" (not baseweb) ---- */
    [role="tablist"] {gap: 4px !important;}
    [role="tab"] {border-radius: 8px 8px 0 0; font-size: 0.9rem; padding-left: 12px !important; padding-right: 12px !important;}
    [role="tab"] p {font-weight: 600;}
    /* bigger, blue scroll arrows when a tab bar is wider than the window */
    button[data-testid="stTabsScrollRight"], button[data-testid="stTabsScrollLeft"] {
        width: 40px !important; height: 40px !important; min-width: 40px !important; border-radius: 50% !important;
        background: #14607F !important; color: #fff !important; opacity: 1 !important;
        box-shadow: 0 1px 6px rgba(11,61,84,.45) !important;
    }
    button[data-testid="stTabsScrollRight"] svg, button[data-testid="stTabsScrollLeft"] svg {
        width: 26px !important; height: 26px !important; fill: #fff !important; color: #fff !important;
    }
    /* ---- left panel: Espectrometrika blue with white text ---- */
    section[data-testid="stSidebar"] > div {background: linear-gradient(180deg, #0B3D54 0%, #0F5470 100%);}
    section[data-testid="stSidebar"] {border-right: none;}
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p, section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li, section[data-testid="stSidebar"] summary,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {color: #F1F7FA !important;}
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color: rgba(241,247,250,.82) !important;}
    section[data-testid="stSidebar"] a {color: #7DE3D6 !important;}
    section[data-testid="stSidebar"] [data-testid="stExpander"] {border: 1px solid rgba(255,255,255,.28); border-radius: 10px; background: rgba(255,255,255,.05);}
    section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg {fill: #CFE6EE !important; color: #CFE6EE !important;}
    section[data-testid="stSidebar"] [data-testid="stAlert"] p {color: inherit !important;}
    section[data-testid="stSidebar"] [data-baseweb="select"] *, section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea, section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {color: #1b1b1b !important;}
    section[data-testid="stSidebar"] [data-testid="stExpander"] details, section[data-testid="stSidebar"] [data-testid="stExpander"] summary {background: transparent !important;}
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {background: rgba(255,255,255,.08) !important;}
    section[data-testid="stSidebar"] [data-testid="stIconMaterial"] {color: #F1F7FA !important;}
    section[data-testid="stSidebar"] .stButton > button, section[data-testid="stSidebar"] .stDownloadButton > button {
        background: transparent !important; border: 1.5px solid rgba(255,255,255,.75) !important; color: #fff !important;}
    section[data-testid="stSidebar"] .stButton > button:hover, section[data-testid="stSidebar"] .stDownloadButton > button:hover {
        background: rgba(255,255,255,.14) !important; border-color: #7DE3D6 !important;}
    .sb-logo {background:#fff; border-radius:12px; padding:10px 14px 8px; margin:2px 0 0; border-bottom: 3px solid #fff;}
    .sb-sep {height:3px; background:#fff; border-radius:2px; margin:6px 0 10px; opacity:.9;}
    .stButton>button, .stDownloadButton>button {
        border-radius: 8px;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        border-color: #14B8A6;
        color: #0B3D54;
    }
    a {color: #14B8A6 !important;}

    /* ---------------- Mobile mode (phones / narrow windows) ---------------- */
    @media (max-width: 768px) {
        section.stMain .block-container, [data-testid="stMainBlockContainer"] {
            padding: 4rem 0.8rem 5rem 0.8rem !important;
        }
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.25rem !important; }
        h3 { font-size: 1.08rem !important; }
        /* columns stack one under the other */
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 0.5rem !important; }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            min-width: 100% !important; flex: 1 1 100% !important; width: 100% !important;
        }
        /* tabs: smaller text, swipe sideways */
        [role="tablist"] { overflow-x: auto !important; }
        [role="tab"] { padding: 0.4rem 0.55rem !important; font-size: 0.82rem !important; white-space: nowrap; }
        /* wide tables and plots scroll instead of overflowing the page */
        [data-testid="stDataFrame"], [data-testid="stTable"] { max-width: 100%; overflow-x: auto; }
        .js-plotly-plot, .plotly { max-width: 100% !important; }
        /* bigger touch targets */
        .stButton > button, .stDownloadButton > button { min-height: 2.8rem; width: 100%; }
        [data-baseweb="select"] > div, input, textarea { font-size: 16px !important; }
        /* hero logo and cards */
        img { max-width: 100% !important; height: auto !important; }
        [data-testid="stPopoverBody"], div[data-baseweb="popover"] > div { max-width: 94vw !important; }
    }
</style>
""", unsafe_allow_html=True)

# Well-distinguished color palette for coloring by class (avoids Plotly
# falling back to a continuous color scale with just a couple of similar
# shades when there are only 2-3 classes).
CLASS_PALETTE = px.colors.qualitative.Set1


def rgb_string_a_hex(color):
    """Converts a Plotly 'rgb(r,g,b)' color string to '#rrggbb' hex, so the
    same palette can also be used with matplotlib (which doesn't accept the
    'rgb(...)' CSS-style string format). Leaves already-hex colors untouched."""
    if isinstance(color, str) and color.startswith("rgb"):
        r, g, b = [int(v) for v in color[color.index("(") + 1: color.index(")")].split(",")]
        return f"#{r:02x}{g:02x}{b:02x}"
    return color

# Hyperparameter-search speed/quality presets. The tuple is (internal method
# name passed to mu.optimizar_hiperparametros, explanation shown to the user).
PRESETS_OPTIMIZACION = {
    "Fast": ("fast",
             "Tries a small number of random combinations (8). The quickest option — "
             "good for a first pass, may miss the exact optimum."),
    "Balanced": ("balanced",
                 "Successive-halving search from random combinations: quickly discards weak "
                 "options and only spends full effort on the promising ones. Good speed/quality "
                 "trade-off — recommended default."),
    "Thorough": ("thorough",
                 "Successive-halving search over the ENTIRE grid: still checks every "
                 "combination, just far more efficiently than a classic grid search."),
    "Grid search (most demanding)": ("grid",
             "Classic exhaustive grid search: trains EVERY combination of the ranges with the full "
             "cross-validation. It is the most demanding and the slowest option — best kept for a final "
             "model, with small ranges, when time does not matter."),
    "Bayesian": ("bayesian",
                 "Each combination it tries is chosen using what it learned from every "
                 "previous one (Optuna/TPE), so the search homes in on promising regions "
                 "instead of sampling blindly. Often finds a better result with fewer trials "
                 "than the other options — a good choice when training a single model."),
}


HP_MODOS = ["Library defaults", "Dataset-based (recommended)", "My edited ranges"]


def _grilla_activa(nombre, es_clf, X, y, prop_test, cv, modo=None):
    """Search space used when optimizing hyperparameters of `nombre`, according to the mode chosen in the
    Hyperparameters tab: library defaults, ranges computed from this dataset, or the ranges the user edited."""
    base = (mu.GRILLAS_CLASIFICACION if es_clf else mu.GRILLAS_REGRESION).get(nombre)
    if base is None:
        return None
    modo = modo or st.session_state.get("hp_modo_aplicado") or st.session_state.get("hp_modo", HP_MODOS[1])
    if modo == HP_MODOS[0]:
        return base
    if modo == HP_MODOS[2]:
        ed = (st.session_state.get("hp_edit_aplicado") or st.session_state.get("hp_edit") or {}).get("clf" if es_clf else "reg", {}).get(nombre)
        if ed:
            return ed
    try:
        d = hu.resumen_datos(X, y, es_clf, float(prop_test or 0.0), cv if isinstance(cv, int) else 5)
        rec = hu.espacio_recomendado(nombre, base, d)
        return {k: v[0] for k, v in rec.items()}
    except Exception:
        return base


def _ui_split(es_clf, prop, key, simca=False):
    """Shared selector of the test-set split method (+ random seed). Returns (label, metodo, semilla)."""
    if es_clf:
        opciones = ["Random", "Random (stratified)", "Kennard-Stone (representative)"]
        pc = "split_clf"
    else:
        opciones = ["Random", "Random (stratified by y range)", "SPXY (extended Kennard-Stone)"]
        pc = "split_reg"
    _ayuda_clf = ("How to choose which samples go into the independent test set. 'Random' picks them purely by chance. "
                  "'Random (stratified)' also uses chance, but keeps the proportion of each class equal in train and test. "
                  "'Kennard-Stone' instead picks the training set to cover the whole spectral space, class by class.")
    _ayuda_reg = ("How to choose which samples go into the independent test set. 'Random' picks them purely by chance. "
                  "'Random (stratified by y range)' also uses chance, but keeps the proportion of each level of y (quantile bins) equal in train and test. "
                  "'SPXY' (extended Kennard-Stone, Galvão 2005) picks the training set to cover both the spectral space and the y range.")
    sel = st.selectbox("Test set selection", opciones, index=1, disabled=(prop == 0), key=key,
                       help=_ayuda_clf if es_clf else _ayuda_reg)
    if sel.startswith("Kennard-Stone") or sel.startswith("SPXY"):
        metodo = "kennard_stone"
    elif "stratified" in sel:
        metodo = "random_strat"
    else:
        metodo = "random"
    semilla = st.number_input(
        "Random seed", min_value=0, max_value=999999, value=0, step=1, key=key + "_seed",
        disabled=(prop == 0 or metodo == "kennard_stone"),
        help="Think of the seed as the 'shuffle number' of the deck of samples. Seed 0, seed 1, seed 2... are just different "
             "shuffles: each one gives a different (but equally valid) random train/test split. The SAME seed always repeats "
             "exactly the same split and the same cross-validation folds, so your results are reproducible and other people can "
             "get your numbers. Changing the seed shows how much the result depends on the luck of the split — if the metric "
             "changes a lot between seed 0, 1 and 2, the dataset is small or heterogeneous and a single number cannot be trusted "
             "(use the resampling section below to repeat it many times). There is no 'best' seed: do not pick the one that "
             "looks better. Kennard-Stone/SPXY involve no randomness, so the seed is not used there.")
    if metodo != "kennard_stone" and prop > 0:
        st.caption("🎲 The seed is just the number of the 'shuffle': the same seed repeats the same split; another seed gives another split. Try 0, 1, 2 to see how much the result depends on the luck of the split.")
    gu.pros_cons(pc, sel, key="pc_" + key)
    if metodo == "kennard_stone":
        st.caption("ℹ️ Kennard-Stone is deterministic: it picks the TRAINING set to cover the spectral space within EACH class separately (including the more 'extreme' samples); the remaining, more 'typical' samples become the test set. No seed is needed."
                   if es_clf else
                   "ℹ️ SPXY is deterministic: it picks the TRAINING set to cover BOTH the spectral space and the y range (including the more 'extreme' samples); the remaining, more 'typical' samples become the test set. No seed is needed.")
    return sel, metodo, int(semilla)


def _diagnostico_curva(curva):
    """Plain-language reading of a learning curve (train vs. cross-validation)."""
    try:
        tr = np.asarray(curva["train_scores_mean"], dtype=float)
        va = np.asarray(curva["val_scores_mean"], dtype=float)
        brecha = float(tr[-1] - va[-1])
        sube = float(va[-1] - va[max(0, len(va) - 3)])
        if va[-1] < 0:
            return ("⚠️ The cross-validation score is below zero: with this few samples the model predicts worse than "
                    "simply using the mean. Typical of a model that is too flexible for the amount of data (overfitting) — "
                    "try a simpler model, fewer variables or stronger regularisation, and more samples if you can.")
        if brecha > 0.15 and sube > 0.01:
            return ("📈 Large gap between training and validation, and validation is still rising: the model is overfitting "
                    "and **more samples would likely help**.")
        if brecha > 0.15:
            return ("⚠️ Large, persistent gap: the model memorises the training set. Simplify it (fewer components / variables, "
                    "more regularisation).")
        if sube > 0.02:
            return "📈 Both curves are close but validation is still rising: more samples could still improve the model."
        return "✅ Both curves are flat and close together: the model has plateaued; more samples will change little."
    except Exception:
        return ""


def _ui_remuestreo(pref, modelos, X, y, es_clf, prop_default=0.25):
    """Optional resampling for ALL trained models: many random train/test splits → metrics ± SD."""
    st.markdown("#### 🔁 Resampling: metrics with their uncertainty (optional)")
    if _BASICO:
        st.caption(T("🔒 Advanced section — greyed out in First steps mode. Switch to 🔬 I have experience (top of the page) to use it."))
    st.caption("A single train/test split can be lucky or unlucky. Here the split is **repeated many times** (random"
               + (", stratified by class" if es_clf else "") + "); every trained model is refitted each time with the same "
               "hyperparameters and tested on the samples left out. The mean ± standard deviation tells you the "
               "result you can honestly expect and how much it depends on which samples ended up in the test set.")
    metodo_rs = st.radio("Resampling method", ["Repeated hold-out (Monte Carlo)", "Bootstrap (out-of-bag)"],
                         horizontal=True, key=f"{pref}_rs_met", disabled=_BASICO,
                         help="Repeated hold-out: each repetition holds out a fixed fraction of the samples as test, drawn at random without replacement. "
                              "Bootstrap: each repetition trains on N samples drawn WITH replacement (some repeated, ~63 % distinct) and tests on the samples that were never drawn (out-of-bag, ~37 %); the test samples are never in that training set. "
                              "Both rotate the samples between train and test, so no single lucky or unlucky test set decides the result.")
    _boot = metodo_rs.startswith("Bootstrap")
    c1, c2, c3 = st.columns([1, 1, 1])
    n_rep = c1.slider("Number of repetitions", 5, 200 if _boot else 100, 50 if _boot else 20, step=5, key=f"{pref}_rs_n{'b' if _boot else ''}", disabled=_BASICO)
    if _boot:
        prop = prop_default
        c2.caption("The test set of each repetition is the out-of-bag samples (about 37 % of the data, it varies each time).")
    else:
        prop = c2.slider("Test fraction (%)", 10, 40, int(prop_default * 100), step=5, key=f"{pref}_rs_prop", disabled=_BASICO) / 100

    def _correr():
        # runs BEFORE the page reruns, so the results are already there when it redraws
        try:
            st.session_state[f"{pref}_rs_res"] = {nm: mu.remuestreo_train_test(m, X, y, es_clf, int(n_rep), float(prop),
                                                                              metodo="bootstrap" if _boot else "holdout")
                                                  for nm, m in modelos.items()}
            st.session_state[f"{pref}_rs_boot"] = _boot
            st.session_state[f"{pref}_rs_cfg"] = {"n": int(n_rep), "prop": float(prop)}
            st.session_state[f"{pref}_rs_err"] = None
        except Exception as e:
            st.session_state[f"{pref}_rs_err"] = str(e)
    c3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    c3.button("▶ Run resampling", key=f"{pref}_rs_btn", on_click=_correr, type="primary", disabled=_BASICO,
              help="Repeats the split and refits every trained model each time (a few seconds to a few minutes, depending on the models and the data).")
    _g1, _g2, _g3, _ = st.columns([1.5, 1.9, 1.6, 3])
    with _g1:
        gu.what_next_remuestreo(st.session_state.get(f"{pref}_rs_res"), es_clf, st.session_state.get(f"{pref}_rs_boot"), key=f"{pref}_rs_next")
    with _g2:
        gu.how_to_interpret("resampling", key=f"{pref}_rs_how")
    with _g3:
        gu.pros_cons("resampling", metodo_rs, key=f"{pref}_rs_pc")
    if st.session_state.get(f"{pref}_rs_err"):
        st.error(f"Resampling failed: {st.session_state[f'{pref}_rs_err']}")
    res = st.session_state.get(f"{pref}_rs_res")
    if res:
        metricas = [c for c in next(iter(res.values())).columns if c != "split"]
        filas = []
        for nm, df in res.items():
            fila = {"Model": nm}
            for c in metricas:
                fila[c] = f"{df[c].mean():.3f} ± {df[c].std(ddof=1):.3f}"
                if st.session_state.get(f"{pref}_rs_boot"):
                    fila[c + " (95% CI)"] = f"{df[c].quantile(0.025):.3f} – {df[c].quantile(0.975):.3f}"
            filas.append(fila)
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
        principal = metricas[0]
        fig = go.Figure()
        for nm, df in res.items():
            fig.add_trace(go.Box(y=df[principal], name=nm, boxpoints="all", jitter=0.4, pointpos=0, marker=dict(size=4)))
        fig.update_layout(height=340, showlegend=False, yaxis_title=principal,
                          title=dict(text=f"{principal} over {len(next(iter(res.values())))} " + ("bootstrap repetitions" if st.session_state.get(f"{pref}_rs_boot") else "random splits"), x=0.01))
        st.plotly_chart(fig, width="stretch")
        st.caption("Report as **mean ± SD** over the repetitions. A large SD means the result depends strongly on which "
                   "samples are used for testing (typical of small datasets). If two models differ by less than their SD, "
                   "you cannot say that one is really better.")
        gu.nota(f"{pref}_resampling", "resampling plot", pref)
        st.divider()


def resetear_prefijo(prefijo):
    """Clears every session_state key starting with 'prefijo' — used by the
    per-tab 'Reset' buttons so leftover results from a previous run (a
    different set of models, an old learning curve, an old p-value table...)
    never linger on screen after starting a new analysis in that tab."""
    claves_a_borrar = [k for k in st.session_state.keys() if k.startswith(prefijo)]
    for k in claves_a_borrar:
        del st.session_state[k]


def df_a_excel_bytes(hojas):
    """Builds an in-memory .xlsx from a dict {sheet_name: DataFrame}."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for nombre_hoja, df_hoja in hojas.items():
            df_hoja.to_excel(writer, sheet_name=nombre_hoja[:31], index=False)
    return buffer.getvalue()


# =============================================================================
# CACHED, EXPENSIVE COMPUTATIONS
# -----------------------------------------------------------------------
# Streamlit reruns the ENTIRE script top-to-bottom on every single widget
# interaction, anywhere in the app — including tabs that have nothing to do
# with the widget that was actually touched. Without caching, that means
# toggling something as unrelated as "variable selection method" in the
# Classification tab would still silently recompute preprocessing, refit
# PCA, and recompute outlier statistics from scratch every single time,
# which is the main reason simple UI interactions could feel sluggish
# (especially on a shared/limited CPU). These cached wrappers make each of
# those only actually recompute when their real inputs change.
# =============================================================================

def _separador_csv(bytes_archivo):
    """Guess the CSV delimiter (comma, semicolon or tab) from the first line, and whether
    the numbers use a decimal comma (typical of semicolon-separated files from Excel in
    Spanish/other locales)."""
    texto = bytes_archivo[:20000].decode("utf-8-sig", errors="ignore")
    lineas = [l for l in texto.splitlines() if l.strip()][:15]
    primera = lineas[0] if lineas else ""
    sep = max([",", ";", "\t"], key=lambda c: primera.count(c)) if primera else ","
    if primera.count(sep) == 0:
        sep = ","
    coma_decimal = sep != "," and any(re.search(r"\d,\d", l) for l in lineas[1:])
    # Count fields over the first 15 COMPLETE lines (a header with tens of thousands of
    # variables is longer than any fixed byte window, so slicing by bytes would cut it short).
    _lineas15 = bytes_archivo.split(b"\n", 15)[:15]
    _sepb = sep.encode()
    n_campos = max((l.count(_sepb) for l in _lineas15), default=0) + 1
    return sep, coma_decimal, n_campos


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def _leer_crudo_cacheado(bytes_archivo, nombre_archivo):
    """Parses the file ONCE (no header assumed) and returns (sheet_names, {sheet: raw DataFrame},
    decimal_comma). Everything else (sheet list, raw preview, header row) is derived from this
    in memory, so changing the header row or a column choice never re-reads the file.
    Excel files use the much faster 'calamine' engine when it is installed."""
    if nombre_archivo.lower().endswith(".csv"):
        sep, coma_decimal, n_campos = _separador_csv(bytes_archivo)
        try:
            df = pd.read_csv(io.BytesIO(bytes_archivo), header=None, sep=sep, low_memory=False,
                             names=range(n_campos), encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(bytes_archivo), header=None, sep=sep, low_memory=False,
                             names=range(n_campos), encoding="latin-1")
        return ["(csv)"], {"(csv)": df}, coma_decimal
    buffer = io.BytesIO(bytes_archivo)
    try:
        hojas = pd.read_excel(buffer, header=None, sheet_name=None, engine="calamine")
    except Exception:
        buffer.seek(0)
        hojas = pd.read_excel(buffer, header=None, sheet_name=None)
    return list(hojas.keys()), hojas, False


def _aplicar_encabezado(raw, fila_encabezado, coma_decimal):
    """Turns a raw (header=None) table into a normal one using the chosen header row."""
    cabecera = raw.iloc[fila_encabezado - 1].tolist()
    datos = raw.iloc[fila_encabezado:].reset_index(drop=True)
    etiquetas, vistos = [], {}
    for i, h in enumerate(cabecera):
        if h is None or (isinstance(h, float) and np.isnan(h)):
            h = f"Unnamed: {i}"
        elif isinstance(h, float) and h.is_integer():
            h = int(h)
        h = str(h)
        if h in vistos:
            vistos[h] += 1
            h = f"{h}.{vistos[h]}"
        else:
            vistos[h] = 0
        etiquetas.append(h)
    # Fast path: convert the whole block to numbers column by column at the numpy level
    # (a pandas Series per column is very slow with 10^4+ variables). Columns that do not
    # convert (IDs, class labels, decimal-comma text...) fall through to the careful path below.
    _vals = datos.to_numpy(dtype=object)
    _bloque = np.full(_vals.shape, np.nan)
    _fallan = []
    for j in range(_vals.shape[1]):
        try:
            _bloque[:, j] = _vals[:, j].astype(float)
        except (ValueError, TypeError):
            _fallan.append(j)
    def _con_enteros(res):
        # leading columns that hold whole numbers (class codes, counts...) keep an integer type,
        # as they did when each column was parsed on its own
        for j in range(min(10, _bloque.shape[1])):
            c = _bloque[:, j]
            if j not in _fallan and len(c) and np.isfinite(c).all() and (c == np.floor(c)).all():
                res[etiquetas[j]] = c.astype("int64")
        return res

    if not _fallan:
        return _con_enteros(pd.DataFrame(_bloque, columns=etiquetas))
    if len(_fallan) < 0.5 * max(1, _vals.shape[1]):
        res = pd.DataFrame(_bloque, columns=etiquetas)
        _con_enteros(res)
        for j in _fallan:
            col = datos.iloc[:, j].reset_index(drop=True)
            if not pd.api.types.is_numeric_dtype(col):
                conv = pd.to_numeric(col, errors="coerce")
                if conv.notna().sum() == col.notna().sum():
                    col = conv
                elif coma_decimal:
                    conv = pd.to_numeric(col.astype(str).str.replace(",", ".", regex=False), errors="coerce")
                    if conv.notna().sum() == col.notna().sum():
                        col = conv
            res[etiquetas[j]] = col
        return res
    columnas = {}
    for j, etq in enumerate(etiquetas):
        col = datos.iloc[:, j]
        if not pd.api.types.is_numeric_dtype(col):
            conv = pd.to_numeric(col, errors="coerce")
            if conv.notna().sum() == col.notna().sum():
                col = conv
            elif coma_decimal:
                conv = pd.to_numeric(col.astype(str).str.replace(",", ".", regex=False), errors="coerce")
                if conv.notna().sum() == col.notna().sum():
                    col = conv
        columnas[etq] = col.reset_index(drop=True)
    return pd.DataFrame(columnas)


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def _leer_archivo_cacheado(bytes_archivo, nombre_archivo, hoja, fila_encabezado):
    """Header-applied table, derived from the single cached parse of the file."""
    _, hojas, coma_decimal = _leer_crudo_cacheado(bytes_archivo, nombre_archivo)
    raw = hojas[hoja] if hoja in hojas else next(iter(hojas.values()))
    return _aplicar_encabezado(raw, fila_encabezado, coma_decimal)


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def _leer_preview_cacheada(bytes_archivo, nombre_archivo, hoja):
    """First rows of the raw table (no header assumed), for the 'raw preview' expander."""
    _, hojas, _ = _leer_crudo_cacheado(bytes_archivo, nombre_archivo)
    raw = hojas[hoja] if hoja in hojas else next(iter(hojas.values()))
    # Only the first rows AND a handful of columns are needed to pick the header row; converting
    # tens of thousands of columns to text one by one (wide NMR/Raman files) took many seconds.
    n_col = raw.shape[1]
    cols = list(range(min(n_col, 14))) + (list(range(n_col - 3, n_col)) if n_col > 17 else [])
    vista = raw.iloc[:6, cols].to_numpy(dtype=object)
    out = pd.DataFrame([[("" if (isinstance(v, float) and np.isnan(v)) else str(v)) for v in fila] for fila in vista],
                       columns=[str(c) for c in cols])
    if n_col > 17:
        out.insert(14, "…", "…")
    return out


@st.cache_data(show_spinner=False, max_entries=6, ttl=3600)
def _alinear_cacheado(X, eje, n_int, max_d):
    return cu.alinear_intervalos(X, eje, n_int, max_d)


@st.cache_data(show_spinner=False, max_entries=12, ttl=3600)
def _aplicar_pretratamiento_cacheado(X_paso0, secuencia):
    """Cached preprocessing: only recomputes when the raw data or the chosen
    steps/parameters actually change, not on every unrelated rerun."""
    X_pret = X_paso0.copy()
    for paso_tup in secuencia:
        X_pret = aplicar_paso(X_pret, paso_tup)
    return X_pret


@st.cache_data(show_spinner=False, max_entries=12, ttl=3600)
def _calcular_linkage_cacheado(X_hca, metodo):
    """Cached hierarchical clustering: hierarchy.linkage does real (O(n^2)-O(n^3))
    work, and with no caching it was re-running on EVERY script rerun — even
    ones triggered by an unrelated widget in a completely different tab, since
    Streamlit re-executes the whole script top to bottom each time. Caching it
    means it only recomputes when the data or the linkage method actually change."""
    return hierarchy.linkage(X_hca, method=metodo, metric="euclidean")


def _bloque_cargas_biplot(scores, cargas, autovalores, var_exp, X, eje, ids, clases, color_vals, color_lab):
    """PCA loadings plot, biplot (correlation circle on top of the scores) and an angle/correlation helper."""
    tab = es_tabular()
    npc = cargas.shape[0]
    nombres = list(etiquetas_var(eje)) if tab else [f"{float(v):g}" for v in eje]
    p = len(nombres)
    opciones = list(range(1, min(15, npc) + 1))

    # ---------------------------------------------------------------- loadings
    st.markdown("### Loadings — which variables build each component")
    st.caption("The loading of a variable on a PC is its weight in that component: a large positive or negative loading means the variable "
               "contributes a lot to the direction of the PC. For spectra the loadings look like spectra (peaks = chemistry).")
    pcs_sel = st.multiselect("Components to plot", opciones, default=opciones[:2], key="pca_w_load_pcs",
                             help="One or several components; each is drawn in its own colour.")
    if pcs_sel:
        if not tab:
            fig_l = go.Figure()
            for k in pcs_sel:
                fig_l.add_trace(go.Scatter(x=np.asarray(eje, dtype=float), y=cargas[k - 1], mode="lines",
                                           name=f"PC{k} ({var_exp[k - 1]:.1f}%)"))
            fig_l.add_hline(y=0, line_color="lightgray")
            fig_l.update_layout(title="Loadings vs. position on the axis", xaxis_title="Wavenumber / wavelength / ppm / time",
                                yaxis_title="Loading", height=380)
        else:
            topn = st.slider("Variables shown (largest |loading|)", 3, min(40, p), min(15, p), key="pca_w_load_topn",
                             help="Shows the variables with the largest absolute loading on the first selected component.")
            k0 = pcs_sel[0]
            orden = np.argsort(-np.abs(cargas[k0 - 1]))[:topn]
            fig_l = go.Figure()
            for k in pcs_sel:
                fig_l.add_bar(x=[nombres[i] for i in orden], y=cargas[k - 1][orden], name=f"PC{k} ({var_exp[k - 1]:.1f}%)")
            fig_l.update_layout(title=f"Loadings of the {topn} variables with the largest weight on PC{k0}", barmode="group",
                                yaxis_title="Loading", height=400)
        st.plotly_chart(fig_l, width="stretch")
        gu.nota("pca_loadings", "loadings plot", "pca")

    # ---------------------------------------------------------------- biplot
    st.markdown("### Biplot — samples (scores) and variables (loadings) together")
    c1, c2, c3 = st.columns(3)
    px_ = c1.selectbox("X axis (PC)", opciones, format_func=lambda _k: f"PC{_k}", index=0, key="bip_x", help="Component on the horizontal axis.")
    py_ = c2.selectbox("Y axis (PC)", opciones, format_func=lambda _k: f"PC{_k}", index=min(1, len(opciones) - 1), key="bip_y", help="Component on the vertical axis.")
    nvar = c3.slider("Variables drawn as arrows", 3, min(40, p), min(12 if tab else 8, p), key="bip_n",
                     help="The variables with the longest arrows in this plane. Neighbouring points of a spectrum are almost "
                          "identical, so for spectra the app keeps variables that are far apart on the axis.")
    sd = np.nanstd(X, axis=0, ddof=1)
    sd[sd == 0] = np.nan
    # arrow = correlation of each variable with the two PCs (so it lies inside the unit circle)
    ax_x = cargas[px_ - 1] * np.sqrt(autovalores[px_ - 1]) / sd
    ax_y = cargas[py_ - 1] * np.sqrt(autovalores[py_ - 1]) / sd
    largo = np.sqrt(np.nan_to_num(ax_x) ** 2 + np.nan_to_num(ax_y) ** 2)
    elegidas = []
    sep = 0 if tab else 0.04 * (float(np.max(eje)) - float(np.min(eje)))
    for j in np.argsort(-largo):
        if len(elegidas) >= nvar:
            break
        if sep and any(abs(float(eje[j]) - float(eje[i])) < sep for i in elegidas):
            continue
        elegidas.append(int(j))
    sx, sy = scores[:, px_ - 1], scores[:, py_ - 1]
    esc = 0.9 * float(max(np.max(np.abs(sx)), np.max(np.abs(sy))))
    fig_b = go.Figure()
    if color_vals is not None:
        fig_b.add_trace(go.Scatter(x=sx, y=sy, mode="markers", text=list(ids), hoverinfo="text+x+y", name="samples",
                                   marker=dict(size=8, color=color_vals, colorscale="Viridis", showscale=True,
                                               colorbar=dict(title=color_lab))))
    elif clases is not None:
        for i, c in enumerate(pd.unique(clases)):
            m = np.asarray(clases) == c
            fig_b.add_trace(go.Scatter(x=sx[m], y=sy[m], mode="markers", text=list(np.asarray(ids)[m]), hoverinfo="text+x+y",
                                       name=str(c), marker=dict(size=8, color=CLASS_PALETTE[i % len(CLASS_PALETTE)])))
    else:
        fig_b.add_trace(go.Scatter(x=sx, y=sy, mode="markers", text=list(ids), hoverinfo="text+x+y", name="samples",
                                   marker=dict(size=8, color="#2E86AB")))
    th = np.linspace(0, 2 * np.pi, 120)
    fig_b.add_trace(go.Scatter(x=esc * np.cos(th), y=esc * np.sin(th), mode="lines", line=dict(color="lightgray", dash="dot"),
                               name="correlation = 1", hoverinfo="skip"))
    for j in elegidas:
        fig_b.add_annotation(x=esc * ax_x[j], y=esc * ax_y[j], ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                             showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.6, arrowcolor="#C0392B")
        fig_b.add_annotation(x=esc * ax_x[j], y=esc * ax_y[j], text=nombres[j], showarrow=False, font=dict(color="#C0392B", size=11),
                             yshift=10 if ax_y[j] >= 0 else -10)
    fig_b.add_hline(y=0, line_color="lightgray")
    fig_b.add_vline(x=0, line_color="lightgray")
    fig_b.update_layout(title=f"Biplot: PC{px_} ({var_exp[px_ - 1]:.1f}%) vs PC{py_} ({var_exp[py_ - 1]:.1f}%)", height=620,
                        xaxis_title=f"PC{px_}", yaxis_title=f"PC{py_}", yaxis=dict(scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig_b, width="stretch")
    plano = float(var_exp[px_ - 1] + var_exp[py_ - 1])
    st.caption(f"This plane explains **{plano:.1f} %** of the variance. " + (
        "The arrows can be read with confidence." if plano >= 70 else
        "It is a modest share: variables with SHORT arrows are poorly represented here, so do not interpret their direction."))
    st.session_state["pca_biplot_cfg"] = {"x": int(px_), "y": int(py_), "n": int(nvar)}
    gu.nota("pca_biplot", "biplot", "pca")

    with st.expander("🔍 How to read a biplot (correlations between variables, and the link between scores and loadings)", expanded=False):
        st.markdown(
            "**Each arrow is a variable.** Its coordinates are the correlation of the variable with the two components, so it always lies "
            "inside the dotted circle; the **closer the tip is to the circle, the better this plane represents the variable**.\n\n"
            "**Angles between arrows = correlation between variables**\n"
            "- Arrows pointing in the **same direction** (small angle) → variables **positively correlated** (when one rises, the other rises).\n"
            "- Arrows pointing in **opposite directions** (≈ 180°) → variables **negatively (inversely) correlated**.\n"
            "- Arrows at **≈ 90°** → variables **uncorrelated** (independent in this plane).\n"
            "- The cosine of the angle approximates the correlation: cos 0° = +1, cos 90° = 0, cos 180° = −1. It is an approximation: it is only "
            "reliable for long arrows in a plane that explains most of the variance.\n\n"
            "**How scores and loadings are related**\n"
            "- A score is a weighted sum of the variables: *score of sample i on PC k = Σ (value of variable j in sample i) × (loading of j on PC k)*. "
            "So the loadings are the *recipe* of each component and the scores are the *result* for each sample.\n"
            "- **A sample lying in the direction an arrow points has a high value of that variable**; in the opposite direction, a low one. "
            "Projecting a sample perpendicularly onto an arrow tells you how high that variable is for it.\n"
            "- A group of samples separated along PC1 is therefore explained by the variables whose arrows are aligned with PC1.\n"
            "- With **autoscaled** data (tables) every variable has the same weight, so the arrows are directly comparable; with unscaled "
            "spectra, variables with large signals dominate.")

    # ---------------------------------------------------------------- angle helper
    st.markdown("#### Compare two variables")
    st.caption("Pick two variables: the app measures their angle in this biplot and compares the correlation it suggests with the real correlation.")
    cand = [nombres[j] for j in elegidas]
    if len(cand) >= 2:
        a1, a2 = st.columns(2)
        va = a1.selectbox("Variable A", cand, index=0, key="bip_va", help="First variable.")
        vb = a2.selectbox("Variable B", cand, index=1, key="bip_vb", help="Second variable.")
        ja, jb = nombres.index(va), nombres.index(vb)
        if ja != jb:
            u = np.array([ax_x[ja], ax_y[ja]]); v = np.array([ax_x[jb], ax_y[jb]])
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu > 0 and nv > 0:
                cosang = float(np.clip(u @ v / (nu * nv), -1, 1))
                ang = float(np.degrees(np.arccos(cosang)))
                real = float(np.corrcoef(X[:, ja], X[:, jb])[0, 1])
                if cosang > 0.7:
                    lectura = "small angle → **positively correlated**"
                elif cosang < -0.7:
                    lectura = "angle near 180° → **inversely (negatively) correlated**"
                elif abs(cosang) < 0.3:
                    lectura = "angle near 90° → **practically uncorrelated** in this plane"
                else:
                    lectura = "intermediate angle → **weak / moderate correlation**"
                st.markdown(f"Angle in the plane: **{ang:.0f}°** → cos θ = **{cosang:+.2f}** ({lectura}).  \n"
                            f"Real Pearson correlation of the two variables: **{real:+.2f}**.")
                if min(nu, nv) < 0.6:
                    st.caption("At least one of the arrows is short (poorly represented in this plane): the angle can be misleading — trust the real correlation.")
        else:
            st.info("Choose two different variables.")


@st.cache_resource(show_spinner=False, max_entries=12, ttl=3600)
def _ajustar_pca_cacheado(X_pca_input, n_comp_max):
    """Cached PCA fit. Uses cache_resource (not cache_data) because it
    returns the fitted scikit-learn PCA object itself, which other tabs
    reuse directly (e.g. for its .transform() and .mean_) — cache_resource
    avoids deep-copying that object on every cache hit."""
    return PCA(n_components=n_comp_max).fit(X_pca_input)


@st.cache_data(show_spinner=False, max_entries=12, ttl=3600)
def _calcular_outliers_cacheado(X_pca_input, scores_completo, cargas_completo,
                                 autovalores, media_pca, n_comp, alpha):
    """Cached Hotelling's T² / residual Q computation for the Outliers tab."""
    T2 = cu.calcular_T2(scores_completo, autovalores, n_comp)
    T2_lim = cu.limite_T2(X_pca_input.shape[0], n_comp, alpha)
    Q = cu.calcular_Q(X_pca_input, scores_completo, cargas_completo, n_comp, media=media_pca)
    Q_lim = cu.limite_Q(autovalores, n_comp, alpha)
    Q_lim_confiable = cu.limite_Q_confiable(Q, Q_lim)
    if not Q_lim_confiable:
        Q_lim = float(np.percentile(Q, 100 * (1 - alpha)))
    return T2, T2_lim, Q, Q_lim, Q_lim_confiable


def es_tabular():
    return bool(st.session_state.get("_modo_tab", False)) and st.session_state.get("nombres_var") is not None


def etiquetas_var(eje):
    """Names for a (possibly cropped/subsetted) axis: real variable names in tabular mode,
    the axis values as text otherwise."""
    eje = np.asarray(eje, dtype=float)
    if es_tabular():
        nv = st.session_state.nombres_var
        idx = np.clip(np.rint(eje).astype(int) - 1, 0, len(nv) - 1)
        return np.array([str(nv[i]) for i in idx], dtype=object)
    return np.array([f"{v:g}" for v in eje], dtype=object)


def _con_nombres(df):
    """Display copy of a results table with variable NAMES instead of axis positions (tabular mode)."""
    if not es_tabular():
        return df
    df = df.copy()
    if "From" in df.columns and "To" in df.columns:
        f = etiquetas_var(df["From"].to_numpy(float)); t = etiquetas_var(df["To"].to_numpy(float))
        df.insert(0, "Variable(s)", [a if a == b else f"{a} … {b}" for a, b in zip(f, t)])
        df = df.drop(columns=["From", "To"])
    for c in ("variable", "Variable", "Position"):
        if c in df.columns:
            df[c] = etiquetas_var(df[c].to_numpy(float))
    return df


def _ticks_nombres(fig, eje):
    """Tabular mode: write the variable names on the x axis (when there are few enough)."""
    if es_tabular():
        eje = np.asarray(eje, dtype=float)
        if len(eje) <= 80:
            fig.update_xaxes(tickmode="array", tickvals=list(eje), ticktext=list(etiquetas_var(eje)), tickangle=-60)
    return fig


def _prediccion_confiable(bundle, X_final, ids_nuevo, nombre):
    """Optional reliability tools for the Prediction tab. Nothing is computed until a box is ticked."""
    tipo = bundle.get("tipo")
    if tipo not in ("classification", "regression"):
        return
    modelo = bundle["modelo"]
    cal, dom = bundle.get("calibracion"), bundle.get("dominio")
    eje_full = np.asarray(bundle["numeros_onda"], dtype=float)
    mask = bundle.get("mascara_variables")
    eje_sel = eje_full[mask] if mask is not None else eje_full
    if bundle.get("nombres_var") is not None:
        nv = bundle["nombres_var"]
        nombres_sel = [str(nv[int(round(v)) - 1]) for v in eje_sel]
    else:
        nombres_sel = [f"{v:g}" for v in eje_sel]
    ids_nuevo = np.asarray(ids_nuevo).astype(str)
    X_final = np.asarray(X_final, dtype=float)

    with st.expander("🔬 Reliability tools — conformal prediction, applicability domain, explanation (optional)"):
        st.caption("Three optional checks on how far to trust each prediction. **Nothing runs until you tick a box.**")

        # ---------------------------------------------------------------- 1. conformal
        st.markdown("**1 · Conformal prediction**")
        if st.checkbox("Add conformal intervals / prediction sets", key="conf_w_on",
                       help="Conformal prediction turns the model's cross-validation errors into intervals (regression) "
                            "or sets of classes (classification) with a guaranteed long-run coverage — e.g. 'the true "
                            "value is inside this interval in 90 % of new samples'. It only assumes new samples behave "
                            "like the training ones (check the applicability domain below)."):
            if cal is None:
                st.info("This model was saved before conformal prediction existed (or has no probabilities). "
                        "Retrain and save it again to enable it.")
            else:
                nivel = st.slider("Confidence level (%)", 70, 99, 90, key="conf_w_nivel",
                                  help="Higher = wider intervals / larger sets (safer, less informative). 90–95 % is typical.")
                alpha = 1 - nivel / 100
                if tipo == "regression":
                    por_rango = st.checkbox("Let the width depend on the predicted level", key="conf_w_rango",
                                            help="Uses a separate error quantile for the low / middle / high third of the "
                                                 "predicted values — better when the error grows with concentration. Needs "
                                                 "≥ 30 calibration samples.")
                    yhat = np.asarray(modelo.predict(X_final), dtype=float).ravel()
                    lo, hi, q = cf.intervalos_regresion(cal, yhat, alpha, por_rango)
                    dfc = pd.DataFrame({"id": ids_nuevo, "predicted_value": yhat, f"conformal_lower_{nivel}": lo,
                                        f"conformal_upper_{nivel}": hi, "half_width (±)": q})
                    st.dataframe(dfc, width='stretch')
                    st.caption(f"Based on {len(cal['y'])} out-of-fold errors. The interval keeps its "
                               f"{nivel} % coverage whatever the algorithm; unlike ±1.96×RMSE it makes no assumption "
                               "about the error distribution.")
                else:
                    if not hasattr(modelo, "predict_proba"):
                        st.info("This algorithm gives no probabilities, so prediction sets cannot be built.")
                        dfc = None
                    else:
                        proba = modelo.predict_proba(X_final)
                        sets, tam = cf.conjuntos_clasificacion(cal, proba, list(modelo.classes_), alpha)
                        dfc = pd.DataFrame({"id": ids_nuevo, "predicted_class": modelo.predict(X_final),
                                            f"prediction_set_{nivel}": sets, "set_size": tam})
                        st.dataframe(dfc, width='stretch')
                        st.caption("**set_size = 1**: one class is compatible with the data (confident). **> 1**: ambiguous "
                                   "— several classes are plausible, treat as undecided. **∅**: no class fits — the sample "
                                   "is unusual (new class, adulterant, out of domain). Thresholds are set per class so "
                                   "rare classes keep their coverage.")
                if dfc is not None:
                    st.download_button("⬇️ Download conformal results (Excel)", data=df_a_excel_bytes({"conformal": dfc}),
                                       file_name=f"conformal_{nombre}.xlsx", key="conf_dl",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # ---------------------------------------------------------------- 2. domain
        st.markdown("**2 · Applicability domain (DD-SIMCA-style distances)**")
        if st.checkbox("Check if each new sample is inside the model's domain", key="dom_w_on",
                       help="Builds a small PCA model of the training samples and measures how far each new sample is "
                            "(distance inside the PCA space = h, distance to it = q), with data-driven limits as in "
                            "DD-SIMCA. A sample outside the domain gets a prediction, but the model has never seen "
                            "anything like it — do not trust it."):
            if dom is None:
                st.info("This model was saved before the domain check existed. Retrain and save it again to enable it.")
            else:
                a_dom = st.slider("Significance level α (%)", 1, 20, 5, key="dom_w_alpha",
                                  help="Share of genuine training samples you accept to flag by mistake. Smaller α = a more "
                                       "tolerant domain (fewer false alarms, but more strange samples slip in).") / 100
                dd, crit = cf.distancias_dominio(dom, X_final, a_dom)
                dd.insert(0, "id", ids_nuevo)
                st.dataframe(dd.round(3), width='stretch')
                n_out = int((dd["domain"] != "inside").sum())
                (st.warning if n_out else st.success)(f"{n_out} of {len(dd)} new sample(s) fall outside the domain.")
                xm = float(max(np.max(dom["h_train"]), dd["h (score distance)"].max()) * 1.15)
                xs_, ys_ = cf.curva_aceptacion(dom, crit, xm)
                figd = go.Figure()
                figd.add_trace(go.Scatter(x=dom["h_train"], y=dom["q_train"], mode="markers", name="training",
                                          marker=dict(color="lightgray", size=6)))
                figd.add_trace(go.Scatter(x=dd["h (score distance)"], y=dd["q (residual distance)"], mode="markers",
                                          name="new samples", text=ids_nuevo,
                                          marker=dict(color=np.where(dd["domain"] == "inside", "#0F6E56", "#D62828"), size=9)))
                figd.add_trace(go.Scatter(x=xs_, y=ys_, mode="lines", name=f"acceptance limit (α={a_dom:.0%})",
                                          line=dict(color="black", dash="dash")))
                figd.update_layout(height=420, xaxis_title="h / h₀  (distance inside the model)",
                                   yaxis_title="q / q₀  (distance to the model)", title="Acceptance plot")
                figd.update_xaxes(range=[0, xm]); figd.update_yaxes(range=[0, max(float(dd["q (residual distance)"].max()), float(np.max(dom["q_train"]))) * 1.15])
                st.plotly_chart(figd, width='stretch')
                st.caption(f"PCA with {dom['k']} components ({dom['var_expl']:.0%} of the variance). Points above/right of "
                           "the dashed line are outside: right = unusual composition inside the model's space; up = "
                           "something the model cannot describe (new interferent, instrument problem).")

        # ---------------------------------------------------------------- 3. explanation
        st.markdown("**3 · Why this prediction? (explain one sample)**")
        if st.checkbox("Explain a single sample", key="expl_w_on",
                       help="Replaces, one region at a time, part of the sample by the average training sample and "
                            "watches how the prediction changes: regions whose removal changes it most are the ones "
                            "driving it. Model-agnostic and cheap (one prediction per region)."):
            if dom is None:
                st.info("This model was saved before explanations existed (it needs the training average). "
                        "Retrain and save it again to enable it.")
            else:
                c1, c2 = st.columns(2)
                id_sel = c1.selectbox("Sample", list(ids_nuevo), key="expl_w_id",
                                      help="Which of the new samples to explain.")
                es_tab = bundle.get("nombres_var") is not None
                if es_tab:
                    nw = len(eje_sel)
                    c2.caption("Tabular variables: each variable is its own region.")
                else:
                    nw = c2.slider("Number of regions", 10, 100, 40, key="expl_w_nw",
                                   help="The spectrum is cut into this many equal regions. Fewer = more robust; more = finer.")
                ix = int(np.where(ids_nuevo == id_sel)[0][0])
                clase = None
                if tipo == "classification":
                    pred = modelo.predict(X_final[ix:ix + 1])[0]
                    clase = c1.selectbox("Explain the probability of class", list(modelo.classes_),
                                         index=list(modelo.classes_).index(pred), key="expl_w_clase",
                                         help="By default the predicted class. Pick another to see what pushes towards it.")
                wins, efecto, base = cf.explicar_por_ventanas(modelo, X_final[ix], dom["mu"], tipo == "classification", nw, clase)
                et = "probability of " + str(clase) if clase is not None else "predicted value"
                _base_txt = ("%.0f %%" % (100 * base)) if clase is not None else ("%.3g" % base)
                st.caption(f"Reference prediction for this sample: **{_base_txt}** ({et}). Positive bars (green) = that region "
                           "pushes the prediction UP; negative (red) = pushes it DOWN.")
                col = np.where(efecto >= 0, "#0F6E56", "#D62828")
                if es_tab:
                    o = np.argsort(np.abs(efecto))[::-1][:25][::-1]
                    figx = go.Figure(go.Bar(x=efecto[o], y=[nombres_sel[w[0]] for w in [wins[i] for i in o]], orientation="h",
                                            marker_color=col[o]))
                    figx.update_layout(height=max(320, 22 * len(o)), xaxis_title=f"effect on {et}", yaxis_title="Variable")
                else:
                    cen = np.array([float(eje_sel[w].mean()) for w in wins])
                    anc = np.array([abs(float(eje_sel[w[-1]] - eje_sel[w[0]])) or 1.0 for w in wins])
                    figx = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.4, 0.6], vertical_spacing=0.04)
                    figx.add_trace(_traza_espectro(eje_sel, X_final[ix], mode="lines", line=dict(color="#5F5E5A", width=1)), row=1, col=1)
                    figx.add_trace(go.Bar(x=cen, y=efecto, width=anc, marker_color=col), row=2, col=1)
                    figx.update_layout(height=470, showlegend=False)
                    figx.update_yaxes(title_text="Signal", row=1, col=1); figx.update_yaxes(title_text=f"effect on {et}", row=2, col=1)
                    figx.update_xaxes(title_text="Variable", row=2, col=1)
                    if eje_sel[0] > eje_sel[-1]:
                        figx.update_xaxes(autorange="reversed")
                    top = np.argsort(np.abs(efecto))[::-1][:5]
                    regs = [(float(min(eje_sel[wins[k][0]], eje_sel[wins[k][-1]])), float(max(eje_sel[wins[k][0]], eje_sel[wins[k][-1]]))) for k in top]
                    _t_e, _m_e = _asig_pre(figx, eje_sel, regs, "pred")
                st.plotly_chart(figx, width='stretch')
                if not es_tab:
                    _asig_post(_t_e, _m_e, "pred")


def _extras_confianza(prefijo, cv):
    """Calibration (out-of-fold predictions) + applicability-domain model saved with a model, so the
    Prediction tab can offer conformal intervals/sets, domain checks and explanations."""
    es_clf = prefijo == "clf"
    try:
        cal = cf.calibracion_desde_cv(cv, es_clf)
    except Exception:
        cal = None
    ref = st.session_state.get(f"{prefijo}_dominio")
    return {"calibracion": cal, "dominio": ref}


def _extras_tabular():
    """What a saved model needs in order to score new TABULAR samples: variable names, the fitted
    preprocessor and the axis (positions) it was trained on."""
    if not es_tabular():
        return {}
    return {"nombres_var": list(st.session_state.nombres_var), "tab_prep": st.session_state.get("tab_prep"),
            "tab_eje": np.asarray(st.session_state.numeros_onda_pret, dtype=float)}


def _ui_aumento(prefijo, es_clf):
    """Augmentation is not part of this edition."""
    return None


def _asig_pre(fig, eje, regiones, clave):
    """Functional-group assignment, part 1 (BEFORE the plot): technique selector + labels written on top of
    the figure. Returns (technique, matches) or (None, None). Skipped for tabular variables."""
    if es_tabular() or not regiones:
        return None, None
    eje = np.asarray(eje, dtype=float)
    opciones = list(asg.TECNICAS.keys())
    sug = asg.sugerir_tecnica(eje, st.session_state.get("pret_w_tipo"))
    tec = st.selectbox(
        "Functional-group / proton assignment of the highlighted regions", ["Off"] + opciones,
        index=1 + opciones.index(sug), key=f"asg_w_tec_{clave}",
        help="Suggests which functional groups, overtones/combination bands or proton types typically absorb "
             "(or resonate) in each highlighted region, for the technique you choose. The default is guessed "
             "from your axis range (change it if wrong). Choose 'Off' to hide it. Indicative values only.")
    if tec == "Off":
        return None, None
    m = asg.buscar(tec, regiones)
    if not m.empty:
        niveles = 3
        vistos = set()
        for _, r in m.iterrows():
            if r["_k"] in vistos:
                continue
            vistos.add(r["_k"])
            a, b = regiones[int(r["_k"])]
            fig.add_annotation(x=(a + b) / 2, y=1.0 + 0.07 * (len(vistos) % niveles) + 0.01, yref="paper", yanchor="bottom",
                               text=asg.etiqueta_corta(r["Group / assignment"]), showarrow=False,
                               font=dict(size=10, color="#D62828"), bgcolor="rgba(255,255,255,0.7)")
        fig.update_layout(margin=dict(t=150))
        if fig.layout.title and fig.layout.title.text:
            # keep the figure title above the staggered labels so they never overlap
            fig.update_layout(title=dict(text=fig.layout.title.text, x=0.01, xanchor="left", y=0.985,
                                         yref="container", yanchor="top"))
    return tec, m


def _asig_post(tec, m, clave):
    """Functional-group assignment, part 2 (AFTER the plot): red warning, table and structure drawings."""
    if tec is None or m is None:
        return
    st.markdown(f":red[**⚠️ {asg.AVISO}**]")
    if m.empty:
        st.caption("No typical group of this technique falls inside the highlighted regions.")
        return
    with st.expander("🧪 Probable functional groups / protons for the highlighted regions (indicative)", expanded=True):
        st.dataframe(m.drop(columns=["_k", "_i", "_base"]), hide_index=True, width='stretch')
        st.caption("Structures: **red** = the atoms / bond that vibrate (IR, NIR, Raman) or the protons that "
                   "resonate (¹H NMR); everything else in black. R / R' = the rest of the molecule.")
        unicos = list(dict.fromkeys(m["_i"]))[:8]
        cols = st.columns(4)
        for n, i in enumerate(unicos):
            e = asg.entrada(tec, i)
            with cols[n % 4]:
                st.image(asg.svg_estructura(e["estructura"], e["rojos"], e["enlaces"]), width=170)
                st.caption(f"**{e['grupo']}**  \n{e['modo']}  \n{min(e['lo'], e['hi']):g}–{max(e['lo'], e['hi']):g}")


def df_variables_seleccionadas(eje, mascara):
    """Table with the detail of the selected variables (wavenumbers, or names in tabular mode)."""
    idx_sel = np.where(mascara)[0]
    if es_tabular():
        return pd.DataFrame({"position": idx_sel, "variable": etiquetas_var(eje[idx_sel])})
    return pd.DataFrame({
        "position": idx_sel,
        "wavenumber": eje[idx_sel],
    })


def como_texto(clases):
    """Forces classes to always be text/categorical (never numeric), so that
    Plotly always colors them as well-differentiated discrete categories,
    instead of interpreting them as a continuous numeric variable (which
    would give a single-hue scale, e.g. two similar shades of blue)."""
    if clases is None:
        return None
    return np.array([str(c) for c in clases])


def ejes_a_float(valores):
    """Converts a list/array of axis labels (wavenumbers, retention times,
    chemical shifts...) to float, auto-fixing the European decimal comma
    (e.g. "1500,5" -> "1500.5") when needed — some instruments/software
    export spectra with a comma instead of a dot as the decimal separator,
    and plain float() can't parse that on its own."""
    valores_texto = [str(v).strip() for v in valores]
    try:
        return np.array(valores_texto, dtype=float)
    except ValueError:
        pass

    def _arreglar(v):
        if "," in v and "." not in v:
            # Only a comma present: it's a decimal separator (e.g. "1500,5").
            return v.replace(",", ".")
        if "," in v and "." in v:
            # Both present: the comma is a thousands separator (e.g. "1,500.5").
            return v.replace(",", "")
        return v

    return np.array([_arreglar(v) for v in valores_texto], dtype=float)


# =============================================================================
# SESSION STATE
# =============================================================================

def init_state():
    defaults = {
        "df": None,
        "ids": None,
        "numeros_onda": None,
        "numeros_onda_pret": None,
        "tab_prep": None,            # fitted tabular preprocessor (for predicting new samples)
        "nombres_var": None,         # tabular mode: real variable names (axis is 1..p)
        "X": None,
        "clases": None,
        "mascara_excluidas": None,   # bool array: True = sample excluded as outlier
        "modelos_guardados": {},     # trained models saved for the Prediction tab
        "valores_y": None,           # continuous target (y) for Regression
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# =============================================================================
# KEEP SETTINGS WHEN YOU SWITCH TABS
# -----------------------------------------------------------------------
# Only the tab you are looking at runs (see the tabs section below) — that is
# what keeps the app fast. But Streamlit forgets the value of any widget that
# was not drawn during a run, so without this every slider/selector in a tab
# would silently go back to its default each time you visited another tab.
# Here every value widget (1) gets a stable key (derived from where it is in
# the code AND its parameters, so a changed default/range still resets it, just
# like before) and (2) has its value re-asserted at the start of each run.
# =============================================================================
_ES_APP = True   # marker: lets the wrapper recognise calls coming from THIS script

logging.getLogger("streamlit.elements.lib.policies").setLevel(logging.ERROR)

_WIDGETS_CON_VALOR = ("selectbox", "multiselect", "slider", "select_slider", "radio", "checkbox",
                      "toggle", "number_input", "text_input", "text_area", "segmented_control", "pills")


def _estable(v):
    """Printable form of a widget argument that doesn't change between runs."""
    if callable(v):
        return f"<fn:{getattr(v, '__qualname__', type(v).__name__)}>"
    return re.sub(r" at 0x[0-9a-fA-F]+", "", repr(v))


def _envolver_widget(original, es_metodo):
    @functools.wraps(original)
    def envuelto(*args, **kwargs):
        marco = sys._getframe(1)
        if not marco.f_globals.get("_ES_APP"):
            return original(*args, **kwargs)
        resto = args[1:] if es_metodo else args
        clave = kwargs.get("key")
        if clave is None:
            etiqueta = resto[0] if resto else kwargs.get("label", "")
            base = (f"{marco.f_lineno}|{etiqueta}|{[_estable(x) for x in resto[1:]]}|"
                    f"{sorted((k, _estable(v)) for k, v in kwargs.items())}")
            clave = "_w_" + hashlib.md5(base.encode("utf-8")).hexdigest()[:14]
            kwargs["key"] = clave
        st.session_state.setdefault("_reg", set()).add(str(clave))
        return original(*args, **kwargs)
    return envuelto


def _instalar_persistencia_widgets():
    if getattr(st, "_espectrometrika_patch", False):
        return                       # already installed in this process
    from streamlit.delta_generator import DeltaGenerator
    for nombre in _WIDGETS_CON_VALOR:
        if hasattr(DeltaGenerator, nombre):
            setattr(DeltaGenerator, nombre, _envolver_widget(getattr(DeltaGenerator, nombre), True))
        if hasattr(st, nombre):
            setattr(st, nombre, _envolver_widget(getattr(st, nombre), False))
    st._espectrometrika_patch = True


_instalar_persistencia_widgets()
for _k in list(st.session_state.get("_reg", ())):
    if _k in st.session_state:       # re-assert (before any widget is drawn in this run)
        st.session_state[_k] = st.session_state[_k]


def _rerun_tab():
    """Refresh only the tab (fragment) where a button was pressed, instead of the whole app
    (so the left panel and the other tabs are not redrawn). If for any reason we are not
    inside a fragment rerun, fall back to a normal full rerun — never an error."""
    try:
        st.rerun(scope="fragment")
    except StreamlitAPIException:
        st.rerun()


def hay_datos():
    return st.session_state.X is not None


def indice_activo():
    """Indices of the samples currently included (not excluded as outliers)."""
    n = st.session_state.X.shape[0]
    if st.session_state.mascara_excluidas is None:
        st.session_state.mascara_excluidas = np.zeros(n, dtype=bool)
    return ~st.session_state.mascara_excluidas


def datos_activos():
    """Returns (ids, X, clases) filtered by the exclusion mask.
    Classes are always returned as text (see como_texto)."""
    idx = indice_activo()
    ids = st.session_state.ids[idx]
    X = st.session_state.X[idx]
    clases = como_texto(st.session_state.clases[idx]) if st.session_state.clases is not None else None
    return ids, X, clases


# =============================================================================
# "COMPUTED WITH..." STATE FINGERPRINT
# -----------------------------------------------------------------------
# PCA, outlier detection, classification and regression are all fairly
# expensive, and every one of them depends on preprocessing + which samples
# are currently excluded as outliers. Recomputing them automatically every
# time ANYTHING changes elsewhere in the app (even in an unrelated tab) is
# wasteful and is the main reason the app can feel slow. Instead, each of
# these results is computed only on an explicit user action (first visit,
# or clicking an "Update" button), and is tagged with a lightweight
# "fingerprint" of the data/preprocessing/outlier state it was computed
# with. If that fingerprint no longer matches the CURRENT state, a small
# badge says so — but the old result stays on screen until the user
# actually asks for an update.
# =============================================================================

def firma_datos_activos():
    """A cheap fingerprint (shape + sum + sum-of-squares, NOT a full hash) of
    the currently active, preprocessed dataset — fast even for large arrays,
    and virtually impossible to collide on by chance. Returns None if there's
    no preprocessed data yet."""
    X_pret = st.session_state.get("X_pret")
    if X_pret is None:
        return None
    idx = indice_activo()
    X_activo = X_pret[idx]
    mascara = st.session_state.get("mascara_excluidas")
    pasos = st.session_state.get("pasos_pretratamiento")
    return (
        X_activo.shape,
        round(float(np.sum(X_activo)), 4),
        round(float(np.sum(X_activo ** 2)), 4),
        int(mascara.sum()) if mascara is not None else 0,
        str(pasos),
    )


def descripcion_firma(firma):
    """Human-readable one-liner for a firma_datos_activos() tuple, for the badge."""
    if firma is None:
        return "no data"
    n_muestras, n_vars = firma[0]
    n_excluidas = firma[3]
    pasos_txt = firma[4]
    resumen_pasos = "no preprocessing" if pasos_txt in ("None", "[]", "") else pasos_txt
    return f"{n_muestras} samples × {n_vars} variables, {n_excluidas} excluded as outliers, preprocessing: {resumen_pasos}"


def insignia_estado(clave_firma, etiqueta="This result", firma_actual=None):
    """Compares a stored fingerprint (session_state[clave_firma]) against the
    CURRENT one and renders a small badge. Returns True if the stored result
    is stale (current state has since changed) or doesn't exist yet."""
    firma_guardada = st.session_state.get(clave_firma)
    if firma_actual is None:
        firma_actual = firma_datos_activos()
    if firma_guardada is None:
        return True
    if firma_guardada == firma_actual:
        st.caption(f"✅ Up to date — computed with: {descripcion_firma(firma_guardada)}.")
        return False
    st.warning(
        f"⚠️ **{etiqueta} is out of date.** It was computed with: "
        f"{descripcion_firma(firma_guardada)}. The current data/preprocessing/outliers "
        f"are different now. The numbers below are still the last ones you asked for — "
        f"click **Update** below whenever you're ready to recompute them.",
        icon="⚠️",
    )
    return True


# =============================================================================
# COMPUTE / UPDATE / CLEAR CONTROLS (shared by every analysis tab)
# -----------------------------------------------------------------------
# Principle for the whole app: NOTHING heavy runs by itself. Each analysis is
# computed only when the user clicks Compute (or Update, once something it
# depends on has changed), and a result stays on screen — tagged with what it
# was computed from — until the user clears it.
# =============================================================================

CLAVES_PCA = ["pca_completo", "scores_completo", "cargas_completo", "autovalores",
              "var_explicada", "pca_X_input", "pca_ids", "pca_clases", "pca_eje",
              "pca_firma", "pca_origen", "pca_crop_desc", "scores", "cargas", "n_comp"]
CLAVES_OUTLIERS = ["outliers_resultado", "outliers_firma", "outliers_ids", "outliers_clases"]
PREFIJOS_DERIVADOS_PCA = ("otros_loadings", "otros_ranking", "otros_corr")


def limpiar_outliers():
    for k in CLAVES_OUTLIERS:
        st.session_state.pop(k, None)


def limpiar_pca():
    """Removes the PCA result and everything that was built on top of it
    (outlier detection, and the PCA-based tools in 'Other tools')."""
    for k in CLAVES_PCA:
        st.session_state.pop(k, None)
    limpiar_outliers()
    for p in PREFIJOS_DERIVADOS_PCA:
        resetear_prefijo(p)


def limpiar_resultados_exploratorios():
    """Clears every exploratory result (PCA, outliers, dendrogram, other tools)."""
    limpiar_pca()
    resetear_prefijo("dendro_")
    resetear_prefijo("otros_")
    st.session_state.pop("informe_final", None)


def reiniciar_pretratamiento():
    """Back to the raw spectra: no preprocessing applied to the analysis."""
    st.session_state.X_pret = st.session_state.X.copy()
    if es_tabular() and np.isnan(st.session_state.X_pret).any():
        # the analyses cannot run with gaps: fill them with the variable median (changeable in Preprocessing)
        st.session_state.X_pret = tu.imputar(st.session_state.X_pret, "median")
    st.session_state.numeros_onda_pret = st.session_state.numeros_onda
    st.session_state.pasos_pretratamiento = []
    if es_tabular():
        _p0 = tu.Preprocesador("median", "none", "none", "none")
        _p0.fit_transform(st.session_state.X)
        st.session_state.tab_prep = _p0
    else:
        st.session_state.tab_prep = None
    st.session_state.pret_desc_aplicada = ("none (raw values; gaps filled with the median)"
                                           if es_tabular() else "none (raw spectra)")
    st.session_state.pret_propuesta_aplicada = repr(((), None))   # same fingerprint as "no steps"
    st.session_state.pop("pret_descarga", None)


def firma_segun_origen(origen):
    """Fingerprint of what an analysis depends on: the data/preprocessing/outlier state
    PLUS (only if it used the cropped spectrum) the spectral crop."""
    return (firma_datos_activos() or ()) + (
        ("crop", st.session_state.get("crop_desc_aplicada") if origen == "cropped" else None, origen),)


def firma_modelos(mascara_crop):
    return firma_segun_origen("cropped" if mascara_crop is not None else "full")


def elegir_espectro_modelado(X, eje, prefijo):
    """Lets the user model on the full spectrum or on the cropped one (if a crop was
    applied in the Crop tab). Returns (X, eje, mascara_crop | None, eje_completo)."""
    crop = st.session_state.get("crop_aplicado")
    mascara = sc.mascara_recorte(eje, crop) if (sc is not None and crop) else None
    if mascara is None:
        return X, eje, None, eje
    opcion = st.radio(
        "Spectrum used for the model", ["Full spectrum (no crop)", "Cropped spectrum"], index=1,
        horizontal=True, key=f"{prefijo}_w_origen",
        help="The crop is applied AFTER the preprocessing (which is always computed on the full "
             "spectrum), so derivatives/smoothing are never distorted at the cut edges. Pick the "
             "full spectrum to ignore the crop for this model.")
    if opcion.startswith("Cropped"):
        st.caption(f"✂️ Using {int(mascara.sum())} of {len(mascara)} variables — {sc.describir_recorte(crop)}.")
        return X[:, mascara], eje[mascara], mascara, eje
    return X, eje, None, eje


def eje_y_mascara_para_guardar(prefijo, mascara_variables):
    """Axis + variable mask to store with a saved model. If the model was trained on a
    cropped spectrum, the FULL axis is stored together with a mask that combines the
    crop and the variable selection — so Prediction can interpolate and preprocess on
    the full axis first and only then keep the variables the model actually uses."""
    mcrop = st.session_state.get(f"{prefijo}_crop_mask")
    eje_comp = st.session_state.get(f"{prefijo}_eje_completo")
    if mcrop is None or eje_comp is None:
        return st.session_state[f"{prefijo}_eje_usado"], mascara_variables
    combinada = np.zeros(len(eje_comp), dtype=bool)
    idx = np.where(mcrop)[0]
    combinada[idx[mascara_variables] if mascara_variables is not None else idx] = True
    return eje_comp, combinada


def desc_con_recorte(desc, prefijo):
    if st.session_state.get(f"{prefijo}_crop_mask") is not None:
        return f"{desc}  |  crop: {st.session_state.get(f'{prefijo}_crop_desc', 'yes')}"
    return desc


def barra_control(prefijo, firma_actual, etiqueta, texto_compute="▶ Compute"):
    """The standard control bar of an analysis: [Compute / Update] [Clear] plus a
    badge saying whether the stored result is up to date.

    The result must live in st.session_state[f"{prefijo}_resultado"] and the
    fingerprint it was computed with in st.session_state[f"{prefijo}_firma"].
    Returns True ONLY on the run where the user clicked Compute/Update — the
    caller then does the (possibly heavy) work, stores both keys, and reruns.
    Clear removes the result without recomputing anything.
    """
    hay = st.session_state.get(f"{prefijo}_resultado") is not None
    desactualizado = hay and st.session_state.get(f"{prefijo}_firma") != firma_actual
    if hay and desactualizado:
        st.warning(f"⚠️ **{etiqueta} is out of date** — the data, preprocessing, outliers or "
                   "settings changed since it was computed. What you see below is still the "
                   "last result you asked for; click **Update** when you want a new one.")
    elif hay:
        st.caption("✅ Up to date with the current data and settings.")
    c1, c2, _ = st.columns([1.4, 1, 4])
    clic = c1.button("🔄 Update" if hay else texto_compute,
                     type="primary" if (desactualizado or not hay) else "secondary",
                     key=f"{prefijo}_btn_compute")
    if c2.button("🧹 Clear", key=f"{prefijo}_btn_clear", disabled=not hay,
                 help="Removes this result from the screen and frees its memory. "
                      "Nothing is recomputed."):
        st.session_state.pop(f"{prefijo}_resultado", None)
        st.session_state.pop(f"{prefijo}_firma", None)
        st.rerun()
    return clic


# Preprocessing functions that are not part of the original
# cu.aplicar_pretratamientos "catalog" (added for Raman/NMR/chromatograms).
# Shared between the Preprocessing tab and the Prediction tab, so the exact
# same steps can be reapplied to new samples.
FUNCIONES_EXTRA = {
    "pqn": cu.pqn,
    "linea_base_als": cu.linea_base_als,
    "eliminar_rayos_cosmicos": cu.eliminar_rayos_cosmicos,
    "emsc": cu.emsc,
}


def aplicar_paso(X_in, paso_tup):
    if paso_tup is None:
        return X_in
    nombre, params = paso_tup
    if nombre in FUNCIONES_EXTRA:
        return FUNCIONES_EXTRA[nombre](X_in, **params)
    return cu.aplicar_pretratamientos(X_in, [(nombre, params)])


# =============================================================================
# SIDEBAR: data and class loading (always visible)
# =============================================================================

class _ArchivoEjemplo:
    """Minimal stand-in for an uploaded file, used by the bundled example datasets."""
    def __init__(self, name, data):
        self.name, self._data = name, data
        self.size = len(data)
        self.file_id = "ej_" + name

    def getvalue(self):
        return self._data

    def read(self, *a):
        return self._data

    def seek(self, *a):
        return 0


with st.sidebar:
    _logo_b64 = _img_b64("logo_espectrometrika_edu.png", max_h=190, paleta=True)
    st.markdown(f"""
    <div class="sb-logo"><img src="{_logo_b64}" style="height:50px; display:block;"></div>
    <div class="sb-sep"></div>
    """, unsafe_allow_html=True)
    st.caption("The learning edition — explore, model, predict and understand every step")
    gu.journey(compact=True)

    with st.expander("📖 Theoretical guide (PDF)"):
        st.caption("A companion reference for this edition: a suggested learning path, preprocessing per technique, PCA, "
                   "outliers, clustering, SIMCA, variable selection, the classification/regression algorithms, validation, "
                   "every metric with its formula and how to read each diagnostic plot, plus a metric cheat sheet and "
                   "where to find public datasets. Read it alongside your analysis.")
        _ruta_guia = os.path.join(os.path.dirname(__file__), "assets", "theoretical_guide.pdf")
        try:
            with open(_ruta_guia, "rb") as _f:
                st.download_button(
                    "⬇️ Download theoretical guide (PDF)", data=_f.read(),
                    file_name="Introduction_to_Chemometrics_edu.pdf", mime="application/pdf",
                )
        except FileNotFoundError:
            st.caption("(Guide file not found — make sure assets/theoretical_guide.pdf is included.)")

    with st.expander("📤 Load a saved project"):
        st.caption("Restores data, preprocessing, and saved models from a project file downloaded "
                   "earlier — the fastest way to pick up exactly where you left off. This is "
                   "available whether or not you already have data loaded (it overwrites whatever "
                   "is currently loaded).")
        archivo_proyecto = st.file_uploader(
            "Project file (.joblib)", type=["joblib"], key="cargar_proyecto",
        help="A project file saved earlier from this app; it restores data, preprocessing and settings.")
        if archivo_proyecto is not None and st.button("📤 Load this project"):
            try:
                proyecto_cargado = joblib.load(archivo_proyecto)
                claves_esperadas_proyecto = {
                    "df", "ids", "numeros_onda", "X", "clases", "mascara_excluidas",
                    "X_pret", "numeros_onda_pret", "pasos_pretratamiento",
                    "valores_y", "modelos_guardados",
                }
                claves_faltantes = claves_esperadas_proyecto - set(proyecto_cargado.keys())
                if len(claves_faltantes) > 2:
                    st.error("This file doesn't look like a project saved by this app.")
                else:
                    for k, v in proyecto_cargado.items():
                        if k == "_modo_tab":
                            continue
                        st.session_state[k] = v
                    _tab_proy = bool(proyecto_cargado.get("_modo_tab", False))
                    st.session_state["_forzar_modo"] = ("Tabular variables (metabolomics, clinical, other)"
                                                        if _tab_proy else "Spectra / signals")
                    try:   # so the restored preprocessing isn't mistaken for "stale" data
                        _Xp = st.session_state.X
                        st.session_state["_X_firma"] = (_Xp.shape, round(float(np.nansum(_Xp)), 6), _tab_proy)
                    except Exception:
                        pass
                    limpiar_resultados_exploratorios()
                    _pasos_proy = st.session_state.get("pasos_pretratamiento") or []
                    st.session_state.pret_desc_aplicada = (
                        " → ".join(str(p[0]) for p in _pasos_proy) if _pasos_proy else "none (raw spectra)")
                    st.session_state.pret_propuesta_aplicada = None
                    st.success("Project loaded successfully.")
                    st.rerun()
            except Exception as e:
                st.error(f"Could not load the project file: {e}")

    with st.expander("✉️ Questions about chemometrics? Write to me", expanded=False):
        st.caption("Doubts about your analysis, a second opinion on a model, or a meeting to discuss your data? "
                   f"Write to **{CONTACT_EMAIL}** or use the form below (more options in the **✉️ Contact** tab).")
        _mensaje_contacto = st.text_area(
            "Your message", key="contacto_mensaje", label_visibility="collapsed",
            placeholder="Write your message here...", height=100,
        help="Free text sent to the author.")
        _col_nombre, _col_email = st.columns(2)
        with _col_nombre:
            _nombre_contacto = st.text_input("Your name (optional)", key="contacto_nombre", help="Optional; so the author can reply.")
        with _col_email:
            _email_contacto = st.text_input(
                "Your email (optional)", key="contacto_email",
                help="Only if you'd like a reply.",
            )
        if st.button("📨 Send message", use_container_width=True):
            if not _mensaje_contacto.strip():
                st.warning("Please write a message before sending.")
            else:
                _ok, _detalle = enviar_mensaje_contacto(
                    _mensaje_contacto, _nombre_contacto, _email_contacto,
                )
                if _ok:
                    st.success(_detalle)
                else:
                    st.error(_detalle)

    with st.expander("🎓 Need data to practise?", expanded=False):
        st.caption("Open the **📚 Public datasets** tab: it lists free tables and spectra (wine, soil, olive oil, "
                   "Raman, metabolomics…) with a direct link and the citation of each one. Download a file there and load it here.")

    st.header("1. Load data")
    if "_forzar_modo" in st.session_state:      # set by "Load a saved project"
        st.session_state["modo_datos"] = st.session_state.pop("_forzar_modo")
    _modo_datos = st.radio(
        "Type of data",
        ["Spectra / signals", "Tabular variables (metabolomics, clinical, other)"],
        key="modo_datos",
        help="Spectra / signals: each column is a point of a spectrum or chromatogram (the header is the "
             "wavenumber, ppm, time...). Tabular variables: each column is a named variable "
             "(a compound concentration, a clinical analysis, any measured quantity). In this mode the "
             "spectral preprocessing is replaced by transformation / scaling suited to tabular data, "
             "and variable NAMES are shown in the results (the axis is simply 1, 2, 3...).",
    )
    MODO_TAB = _modo_datos.startswith("Tabular")
    st.session_state["_modo_tab"] = MODO_TAB
    archivo = st.file_uploader(
        "Data file" if MODO_TAB else "Spectra file",
        type=["csv", "xlsx", "xls"],
        help="You can choose afterwards which row and which columns correspond to what.",
    key="ac_ld_file")

    if archivo is not None:
        st.session_state.pop("_ejemplo", None)          # a file the user uploads always wins over an example
    elif st.session_state.get("_ejemplo"):
        _e = st.session_state["_ejemplo"]
        archivo = _ArchivoEjemplo(_e["name"], _e["bytes"])

    clases_desde_archivo = None
    valores_y_desde_archivo = None

    if archivo is not None:
        try:
            id_archivo = getattr(archivo, "file_id", None) or f"{archivo.name}_{archivo.size}"
            es_excel = not archivo.name.lower().endswith(".csv")
            bytes_archivo = archivo.getvalue()

            hoja_elegida = None
            if es_excel:
                nombres_hojas = _leer_crudo_cacheado(bytes_archivo, archivo.name)[0]
                if len(nombres_hojas) > 1:
                    hoja_elegida = st.selectbox(
                        "Sheet", nombres_hojas, key=f"hoja_{id_archivo}",
                        help="This Excel file has more than one sheet — pick which one to load.",
                    )
                else:
                    hoja_elegida = nombres_hojas[0]

            with st.expander("👁️ Raw preview (to choose the header row)"):
                df_crudo = _leer_preview_cacheada(bytes_archivo, archivo.name, hoja_elegida)
                st.dataframe(df_crudo, width='stretch')

            fila_encabezado = st.number_input(
                "Which row contains the wavenumbers / variable names?",
                min_value=1, max_value=10, value=1, step=1,
                help="1 = first row of the file. Increase it if your file has metadata above the real header.",
                key=f"fila_encabezado_{id_archivo}",
            )

            df_completo = _leer_archivo_cacheado(bytes_archivo, archivo.name, hoja_elegida, fila_encabezado)
            # Normalize column labels to text right away. Excel keeps numeric
            # header cells (e.g. wavenumbers) as actual int/float column labels,
            # while CSV headers are always text — without this, selecting a
            # column by its displayed name later would fail for Excel files
            # with numeric headers (KeyError: label not found).
            df_completo.columns = [str(c) for c in df_completo.columns]
            columnas = list(df_completo.columns)

            # Smart defaults: a column whose HEADER is not a number cannot be a spectral variable.
            # The first text column is the sample ID; a numeric column with a text header is
            # offered as the reference value, a text column as the class. (All can be changed.)
            def _es_numero(txt):
                try:
                    float(str(txt).replace(",", "."))
                    return True
                except ValueError:
                    return False
            _id_def = columnas[0] if (columnas and not _es_numero(columnas[0])
                                      and not pd.api.types.is_numeric_dtype(df_completo[columnas[0]])) else None
            _extra = [c for c in columnas if c != _id_def and not _es_numero(c) and not c.startswith("Unnamed")]
            _y_def = (None if MODO_TAB else
                      next((c for c in _extra if pd.api.types.is_numeric_dtype(df_completo[c])), None))
            _cl_def = next((c for c in _extra if not pd.api.types.is_numeric_dtype(df_completo[c])), None)
            col1, col2 = st.columns(2)
            with col1:
                _id_por_defecto = 1 + columnas.index(_id_def) if _id_def is not None else 0
                opcion_id = st.selectbox(
                    "Sample ID column",
                    ["(none — auto-generate)"] + columnas,
                    index=_id_por_defecto,
                    key=f"opcion_id_{id_archivo}",
                    help="Which column holds each sample's unique name/ID. Pick 'none' to auto-generate "
                         "Sample_1, Sample_2... if your file doesn't have one.",
                )
            with col2:
                opcion_clase = st.selectbox(
                    "Class column (for classification / SIMCA, if already in the file)",
                    ["(none)"] + columnas,
                    index=1 + columnas.index(_cl_def) if _cl_def is not None else 0,
                    key=f"opcion_clase_{id_archivo}",
                    help="Which column holds the class/group label for each sample. Only use this for "
                         "a CATEGORICAL label (e.g. origin, variety). Leave as 'none' if you don't need "
                         "classification, or if your target is a continuous number — use the reference "
                         "value column below for that instead.",
                )
            opcion_valor_y = st.selectbox(
                "Reference value column (for regression, if already in the file)",
                ["(none)"] + columnas,
                index=1 + columnas.index(_y_def) if _y_def is not None else 0,
                key=f"opcion_valory_{id_archivo}",
                help="Which column holds the continuous numeric value you want to predict with "
                     "regression (e.g. a lab-measured concentration). This is different from the class "
                     "column above — use this one for numbers, not categories.",
            )

            columnas_restantes = columnas.copy()

            if opcion_id == "(none — auto-generate)":
                ids = np.array([f"Sample_{i+1}" for i in range(len(df_completo))])
            else:
                ids = df_completo[opcion_id].astype(str).to_numpy()
                columnas_restantes.remove(opcion_id)
                n_duplicados = len(ids) - len(set(ids))
                if n_duplicados > 0:
                    st.warning(f"⚠️ The ID column has {n_duplicados} repeated value(s) — sample IDs "
                               "should normally be unique (otherwise things like exported predictions "
                               "become hard to trace back to a specific sample). Making them unique by "
                               "appending a running number.")
                    contador = {}
                    ids_unicos = []
                    for i in ids:
                        contador[i] = contador.get(i, 0) + 1
                        ids_unicos.append(i if contador[i] == 1 else f"{i}_{contador[i]}")
                    ids = np.array(ids_unicos)

            if opcion_clase != "(none)":
                clases_desde_archivo = df_completo[opcion_clase].astype(str).to_numpy()
                if opcion_clase in columnas_restantes:
                    columnas_restantes.remove(opcion_clase)

            if opcion_valor_y != "(none)":
                try:
                    valores_y_desde_archivo = pd.to_numeric(df_completo[opcion_valor_y]).to_numpy(dtype=float)
                except (ValueError, TypeError):
                    st.error(f"Column '{opcion_valor_y}' has non-numeric values — it can't be used as "
                             "a regression reference value.")
                    valores_y_desde_archivo = None
                if opcion_valor_y in columnas_restantes:
                    columnas_restantes.remove(opcion_valor_y)

            if MODO_TAB:
                # tabular mode: keep numeric columns only; the axis is the position 1..p and the
                # real names travel separately (nombres_var)
                _num = [c for c in columnas_restantes if pd.api.types.is_numeric_dtype(df_completo[c])]
                _no_num = [c for c in columnas_restantes if c not in _num]
                if _no_num:
                    st.warning(f"{len(_no_num)} non-numeric column(s) were left out of the variables: "
                               + ", ".join(_no_num[:8]) + ("..." if len(_no_num) > 8 else "")
                               + ". If one is a class, assign it above.")
                if len(_num) < 2:
                    raise ValueError("Tabular mode needs at least 2 numeric variable columns.")
                columnas_restantes = _num
                numeros_onda = np.arange(1, len(_num) + 1, dtype=float)
                st.session_state.nombres_var = np.array(_num, dtype=object)
            else:
                numeros_onda = ejes_a_float(columnas_restantes)
                st.session_state.nombres_var = None
            X = df_completo[columnas_restantes].to_numpy(dtype=float)

            cambio_tamano = (st.session_state.X is None) or (st.session_state.X.shape[0] != X.shape[0])
            archivo_realmente_nuevo = (
                st.session_state.get("archivo_cargado_id") is not None
                and st.session_state.get("archivo_cargado_id") != id_archivo
            )
            df_final = pd.DataFrame(X, index=ids, columns=(st.session_state.nombres_var if MODO_TAB else numeros_onda))
            df_final.index.name = "id"

            st.session_state.df = df_final
            st.session_state.ids = ids
            st.session_state.numeros_onda = numeros_onda
            st.session_state.X = X
            # Different data than last time (new file, other columns/header row...)?
            # Then whatever was preprocessed/computed before no longer applies:
            # go back to raw spectra and drop the old exploratory results.
            _firma_X = (X.shape, round(float(np.nansum(X)), 6), MODO_TAB)
            if st.session_state.get("_X_firma") != _firma_X:
                st.session_state["_X_firma"] = _firma_X
                resetear_prefijo("pret_w_")
                resetear_prefijo("crop_")
                reiniciar_pretratamiento()
                limpiar_resultados_exploratorios()
            if cambio_tamano:
                st.session_state.mascara_excluidas = np.zeros(X.shape[0], dtype=bool)
            if archivo_realmente_nuevo or cambio_tamano:
                # A genuinely different file was loaded (or the sample count changed) — clear
                # classes/reference values from the PREVIOUS dataset unless the new file itself
                # already supplies them, so stale labels from an unrelated dataset never linger.
                if clases_desde_archivo is None:
                    st.session_state.clases = None
                if valores_y_desde_archivo is None:
                    st.session_state.valores_y = None
            if clases_desde_archivo is not None:
                st.session_state.clases = clases_desde_archivo
            if valores_y_desde_archivo is not None:
                st.session_state.valores_y = valores_y_desde_archivo
            if archivo_realmente_nuevo:
                # A genuinely different file was loaded (not just a re-parse of the same one) —
                # clear any trained-model results from the PREVIOUS dataset so stale
                # tables/plots referencing old samples can't linger on screen. Saved models
                # (in "Predict" tab) are intentionally kept, since re-using an already-trained
                # model on a new file is a legitimate workflow.
                resetear_prefijo("clf_")
                resetear_prefijo("reg_")
                resetear_prefijo("simca_")
                resetear_prefijo("mcr_")
            st.session_state["archivo_cargado_id"] = id_archivo
            st.session_state["archivo_cargado_nombre"] = archivo.name

            st.success(f"{X.shape[0]} samples × {X.shape[1]} variables"
                       + (" · reference values loaded" if valores_y_desde_archivo is not None else ""))
            if archivo_realmente_nuevo:
                st.caption("ℹ️ New file detected — previous Classification/Regression/SIMCA results "
                           "were cleared. Saved models (for the Predict tab) were kept.")
        except Exception as e:
            st.error(f"Could not read '{archivo.name}' with that configuration: {e}  \n"
                     "Tip: every column that is not a spectral variable (sample ID, class, "
                     "reference value such as concentration) must be assigned in the three "
                     "selectors above; all the remaining columns are read as spectra.")
            if st.session_state.get("archivo_cargado_nombre") not in (None, archivo.name):
                st.warning(f"⚠️ Still showing the previous file "
                           f"('{st.session_state['archivo_cargado_nombre']}') because this one could not be read. "
                           "Check the header row and the chosen columns above.")

    if hay_datos():
        st.header("2. Classes / groups (optional)")

        if clases_desde_archivo is not None:
            st.success(f"Classes loaded from the file: {', '.join(map(str, np.unique(clases_desde_archivo)))}")
            usar_otra_fuente = st.checkbox(
                "Define classes another way instead",
                help="Override the class column from the file and use one of the methods below "
                     "instead (extract from ID, type manually, or upload a separate file).",
            key="ac_ld_cls_other")
        else:
            usar_otra_fuente = True

        if usar_otra_fuente:
            modo_clases = st.radio(
                "How do you want to define the classes?",
                ["None", "Extract from ID (separator)", "Extract from ID (character position)",
                 "Enter manually", "Upload file (id, class)"],
                index=0 if clases_desde_archivo is None else 0,
                help="How to assign each sample to a class/group. Extracting from the ID works "
                     "when your sample names already encode the class (e.g. 'A_01', or 'AB01' by "
                     "character position). Otherwise type them in or upload a lookup file.",
            key="ac_ld_cls_mode")

            if modo_clases == "None":
                pass  # no-op: keep whatever classes are already loaded (from a project, a
                      # previous file, etc.) — classes are only cleared when a genuinely new
                      # file is loaded (see 'archivo_realmente_nuevo' below), never just
                      # because this radio's default option happens to be showing.

            elif modo_clases == "Extract from ID (separator)":
                sep = st.text_input("Separator in the ID", value="_", help="Character that separates the parts of the ID (e.g. '_' in 'Greece_01').", key="ac_ld_sep")
                posicion = st.number_input("Fragment position (0 = first)", min_value=0, value=0, step=1, help="Which piece of the ID, counting from 0, is the class.", key="ac_ld_pos")
                try:
                    clases = np.array([str(i).split(sep)[posicion] for i in st.session_state.ids])
                    st.session_state.clases = clases
                    st.caption(f"Classes detected: {', '.join(map(str, np.unique(clases)))}")
                except Exception:
                    st.warning("Could not extract the class with that separator/position for every ID.")

            elif modo_clases == "Extract from ID (character position)":
                st.caption("For example, in the ID \"AB01\", positions 1 and 2 (\"AB\") could be the class.")
                c1, c2 = st.columns(2)
                pos_inicio = c1.number_input("Start position (1 = first character)", min_value=1, value=1, step=1, help="First character of the ID (1 = first) that holds the class.", key="ac_ld_pos_ini")
                pos_fin = c2.number_input("End position", min_value=1, value=2, step=1, help="Last character of the ID that holds the class.", key="ac_ld_pos_fin")
                try:
                    clases = np.array([str(i)[pos_inicio - 1: pos_fin] for i in st.session_state.ids])
                    st.session_state.clases = clases
                    st.caption(f"Classes detected: {', '.join(map(str, np.unique(clases)))}")
                except Exception:
                    st.warning("Could not extract the class with those positions for every ID.")

            elif modo_clases == "Enter manually":
                st.caption("One class per sample, comma-separated, in the same order as the IDs:")
                st.code(", ".join(st.session_state.ids[:8]) + (", ..." if len(st.session_state.ids) > 8 else ""))
                texto = st.text_area("Classes (comma-separated)", help="Names of the classes, separated by commas, in the same order you will assign them.", key="ac_ld_cls_text")
                if texto.strip():
                    lista = [c.strip() for c in texto.split(",")]
                    if len(lista) != len(st.session_state.ids):
                        st.error(f"You entered {len(lista)} classes but there are {len(st.session_state.ids)} samples.")
                    else:
                        st.session_state.clases = np.array(lista)

            elif modo_clases == "Upload file (id, class)":
                archivo_clases = st.file_uploader("CSV with columns: id, class", type=["csv"], key="clases_csv", help="Table with the sample ID and its class; matched to the spectra by ID.")
                if archivo_clases is not None:
                    try:
                        df_clases = pd.read_csv(archivo_clases, dtype=str)
                        df_clases = df_clases.set_index(df_clases.columns[0])
                        mapa = df_clases.iloc[:, 0].to_dict()
                        clases = np.array([mapa.get(i, "no_class") for i in st.session_state.ids])
                        st.session_state.clases = clases
                        if "no_class" in clases:
                            st.warning("Some IDs were not found in the class file (marked 'no_class').")
                    except Exception as e:
                        st.error(f"Error reading the class file: {e}")

        n_excluidas = int(st.session_state.mascara_excluidas.sum()) if st.session_state.mascara_excluidas is not None else 0
        if n_excluidas > 0:
            st.info(f"🚫 {n_excluidas} sample(s) excluded as outliers ('Outliers' tab).")

        with st.expander("💾 Save full project"):
            st.caption("Bundles your data, preprocessing settings, and every saved model into a "
                       "single file, so you can close the app and pick up exactly where you left "
                       "off — instead of re-uploading the spectra and retraining everything.")
            CLAVES_PROYECTO = [
                "df", "ids", "numeros_onda", "X", "clases", "mascara_excluidas",
                "X_pret", "numeros_onda_pret", "pasos_pretratamiento",
                "valores_y", "modelos_guardados", "nombres_var", "tab_prep", "_modo_tab", "notas",
            ]
            if st.button("💾 Prepare project file", help="Packages everything listed above into one downloadable file."):
                proyecto = {k: st.session_state[k] for k in CLAVES_PROYECTO if k in st.session_state}
                buffer_proyecto = io.BytesIO()
                joblib.dump(proyecto, buffer_proyecto)
                st.session_state["_proyecto_bytes"] = buffer_proyecto.getvalue()
            if "_proyecto_bytes" in st.session_state:
                st.download_button(
                    "⬇️ Download project file", data=st.session_state["_proyecto_bytes"],
                    file_name="chemometrics_project.joblib", mime="application/octet-stream",
                )


# =============================================================================
# MAIN BODY: tabs
# =============================================================================

# Safety net: whenever there's data, the analysis must have a preprocessed version
# to work from (raw spectra until the user applies something in the Preprocessing tab).
if hay_datos() and (st.session_state.get("X_pret") is None
                    or st.session_state.X_pret.shape[0] != st.session_state.X.shape[0]):
    reiniciar_pretratamiento()

_logo_top_b64 = _img_b64("logo_espectrometrika_edu.png", max_h=190, paleta=True)
with _top_cols[0]:
    st.markdown(f"""
<div style="margin-bottom:10px; margin-top:4px;">
    <img src="{_logo_top_b64}" style="height:42px; opacity:0.9; display:block;">
</div>
""", unsafe_allow_html=True)

_NOMBRES_TABS = {
    0: "🏠 Home", 1: "📈 Data", 2: "🧪 Preprocessing", 3: "✂️ Crop",
    4: "🧭 PCA", 5: "🔎 PC finder", 6: "🌲 Non-linear finder", 7: "🤝 Consensus", 32: "📡 Reference spectra", 8: "🚩 Outliers", 9: "🌳 Dendrogram", 10: "🛠️ Other tools",
    11: "🏁 Model screening", 12: "🏷️ Classification", 13: "🧬 SIMCA",
    14: "📉 Regression", 15: "🔮 Prediction", 16: "📑 Final report",
    27: "📚 Public datasets", 28: "✉️ Contact", 29: "📖 Glossary & cheat sheet",
    30: "🎛️ Hyperparameters", 31: "🧭 Learning path",
}
# Two-level navigation. Each group is a top tab; its tools are inner tabs.
_GRUPOS = [
    ("🏠 Home", [0]),
    ("🧭 How do we start?", [31, 27, 29]),
    ("📂 Data & preprocessing", [1, 2, 3]),
    ("🔍 Exploratory", [4, 5, 6, 7, 8, 9, 10, 32]),
    ("🎯 Models & prediction", [30, 11, 12, 13, 14, 15]),
    ("📑 Final report", [16]),
    ("✉️ Contact", [28]),
]


class _TabAnidada:
    """A tool tab living inside a group tab. It is 'open' only when its group
    AND itself are the selected ones, so unseen tools never run."""
    def __init__(self, grupo, interna):
        self.g, self.i = grupo, interna

    @property
    def open(self):
        try:
            og = self.g.open
        except Exception:
            og = None
        if og is False:
            return False
        if self.i is None:
            return og
        try:
            oi = self.i.open
        except Exception:
            oi = None
        return False if oi is False else (True if (og is None and oi is None) else True)

    def __enter__(self):
        return (self.i if self.i is not None else self.g).__enter__()

    def __exit__(self, *a):
        return (self.i if self.i is not None else self.g).__exit__(*a)


_AVANZADAS = {5, 6, 7, 10, 13, 30}      # locked (greyed out) in "First steps" mode


class _TabBloqueada:
    """Placeholder for a tool locked in 'First steps' mode: never open."""
    open = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _etq_tab(k):
    n = _NOMBRES_TABS[k]
    return f":gray[🔒 {T(n)}]" if (_BASICO and k in _AVANZADAS) else n


def _crear_tabs():
    try:
        grupos = st.tabs([g[0] for g in _GRUPOS], on_change="rerun", key="nav_principal")
        lazy = True
    except TypeError:                  # older Streamlit without lazy tabs
        grupos = st.tabs([g[0] for g in _GRUPOS])
        lazy = False
    out = {}
    for gi, ((_, idx), gt) in enumerate(zip(_GRUPOS, grupos)):
        if len(idx) == 1:
            out[idx[0]] = _TabAnidada(gt, None)
            continue
        with gt:
            nombres = [_etq_tab(k) for k in idx]
            try:
                if lazy:
                    inner = st.tabs(nombres, on_change="rerun", key=f"nav_g{gi}{'b' if _BASICO else 'x'}")
                else:
                    inner = st.tabs(nombres)
            except TypeError:
                inner = st.tabs(nombres)
        for k, it in zip(idx, inner):
            if _BASICO and k in _AVANZADAS:
                with it:
                    st.info(T("🔒 This is an advanced tool. Switch to **🔬 I have experience** at the top of the page to unlock it."))
                out[k] = _TabBloqueada()
            else:
                out[k] = _TabAnidada(gt, it)
    return out


tabs = _crear_tabs()


def _abierta(tab):
    """True if this tab's code should run now: it is the selected tab, or tabs
    aren't lazy in this Streamlit version (then .open is None)."""
    try:
        return tab.open is not False
    except Exception:
        return True


# -----------------------------------------------------------------------
# TAB: HOME
# -----------------------------------------------------------------------
if _abierta(tabs[0]):
    with tabs[0]:
        _logo_hero_b64 = _img_b64("logo_espectrometrika_edu.png", max_h=240, paleta=True)
        st.markdown(f"""
    <div style="margin-bottom:6px;">
        <img src="{_logo_hero_b64}" style="height:120px; display:block;">
    </div>
    <p style="font-family:'Nunito Sans',sans-serif; font-size:1.08rem; color:#4A5A63; max-width:860px;
              margin-top:6px; margin-bottom:18px; line-height:1.55;">
        <b>Espectrometrika.edu</b> is the edition of the app designed to <b>start in chemometrics</b>: make your first exploratory
        analyses, train and validate your first models, understand every metric. Models can be <b>saved and reused
        another day</b> to predict new samples, with full traceability.
    </p>
    """, unsafe_allow_html=True)

        with st.expander("💡 Why chemometrics?", expanded=False):
            st.markdown(
                "**Chemometrics** is the discipline that applies mathematical and statistical methods to extract chemical "
                "information from measurements — usually spectra, chromatograms or tables of chemical variables — and to design "
                "experiments and validate the results. It makes it possible, for example, to classify samples, to predict a "
                "property (a concentration, a quality parameter) from a spectrum, or to detect anomalous samples.")
            st.markdown(
                "**Chemometric models do not replace traditional chemical analysis: they complement and strengthen it.** "
                "Combining spectroscopy (or any instrumental data) with multivariate modelling gives results that are fast, "
                "usually non-destructive and with little or no use of reagents. They can reveal patterns that would otherwise "
                "go unnoticed, and they allow many more samples to be analysed — which means more confidence in the conclusions "
                "about a whole batch, not only about the few samples a laboratory can measure.")
            st.markdown(
                "They are especially useful when large volumes of data have to be analysed, and they are widely used, for example, in:\n"
                "- **Food control**: genuineness, geographic or botanical origin, adulteration.\n"
                "- **Metabolomics**: finding the metabolites of interest.\n"
                "- **Instrument calibration**: multivariate calibrations of measurement instruments.\n"
                "- **Pharmaceutical quality control**: identification, quantification and process monitoring.\n"
                "- **Research projects**: a general tool to explore and model data.")
            st.markdown(
                "For all these reasons it is worth knowing these alternative tools — and knowing just as well their scope and "
                "limitations: a model is only as good as the data and the validation behind it, and it must be checked and "
                "maintained over time. That is the idea of this app: to learn to use them well, with a critical mind.")

        with st.expander("🎓 What is Espectrometrika.edu?", expanded=False):
            st.markdown(
                "**Espectrometrika.edu is a tool to bring chemometrics closer to analytical chemists, students, researchers "
                "and anyone who works with instrumental data.** It is designed above all for **learning**: you load your own data "
                "(spectra or tables) and go through the whole workflow — exploring, preprocessing, comparing models, validating, "
                "predicting — seeing what each step does and why.")
            st.markdown(
                "It is built so that you can **guide yourself** through the learning:\n"
                "- A suggested **learning path** (*How do we start?* tab) that shows your progress.\n"
                "- On every tab, a **What's next?** button that suggests what to do depending on your results, a **How to interpret?** "
                "button that explains each plot and metric in plain words, and a **Pros & cons** button for each algorithm.\n"
                "- A **glossary and metrics cheat sheet**, a **theoretical guide** (PDF) and a list of **public datasets** to practise.\n"
                "- **Observation boxes** where you write what you see; they are printed in the final report together with the settings you used.")
            st.markdown(
                "Besides learning, the models you build can be **saved and reused** to predict new samples, with full traceability. "
                "The app is free for teaching and self-study, and works in English and Castellano.")

        def _tarjeta(titulo, texto, bg, borde):
            st.markdown(f"""
        <div style="background:{bg}; border-left:4px solid {borde}; border-radius:10px;
                    padding:18px 20px; min-height:200px;">
            <div style="font-family:'Montserrat',sans-serif; font-weight:600; font-size:1.02rem;
                        color:#0B3D54; margin-bottom:8px;">{titulo}</div>
            <div style="font-family:'Nunito Sans',sans-serif; font-size:0.92rem; color:#3D4A52;
                        line-height:1.5;">{texto}</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            _tarjeta("🔍 1 · Explore",
                     "Load <b>spectra</b> (NIR, MIR, Raman, NMR, chromatograms) or <b>tables</b> (clinical, metabolomics, chemical analyses). "
                     "Preprocess, run PCA, detect outliers and cluster the samples — each step explained.",
                     "#E6F0F6", PETROLEUM)
        with col2:
            _tarjeta("🎯 2 · Model",
                     "Compare many preprocessing × algorithm combinations fairly (<b>screening</b>), then refine <b>classification</b>, "
                     "<b>regression</b> and <b>SIMCA</b> models. Tune hyperparameters with ranges adapted to your data.",
                     "#FFF4DC", "#F5A623")
        with col3:
            _tarjeta("🔮 3 · Predict & report",
                     "Save your model with its preprocessing and a model card, load it another day, predict new samples with their "
                     "domain of applicability, and export a traceable <b>report</b>.",
                     "#FDECEA", "#F23A2F")

        st.markdown("""
        <div style="margin-top:22px; padding:12px 16px; border-radius:8px; background:#F4F8F9;
                    font-size:0.92rem; color:#3D4A52; line-height:1.55;">
        <b>How it works:</b> nothing heavy runs by itself. Every analysis tab has a
        <b>Compute</b> button; when something it depends on changes (the preprocessing, the
        excluded outliers...), the result is flagged <b>out of date</b> and keeps showing the
        last version you asked for until you click <b>Update</b>. Preprocessing only reaches
        the rest of the app when you click <b>Apply</b>.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:12px; padding:12px 16px; border-radius:8px; background:#FFF4DC; border-left:4px solid #F5A623;
                    font-size:0.92rem; color:#3D4A52; line-height:1.55;">
        <b>🌱 First steps / 🔬 I have experience</b> (switch at the top of the page): in <b>First steps</b> mode you get the essential
        workflow — load data, preprocessing, PCA, outliers, dendrogram, model screening, classification, regression, prediction and report —
        and the advanced tools (PC finder, non-linear finder, consensus, hyperparameters, SIMCA, other tools, resampling, variable selection,
        hyperparameter optimization) stay visible but <b>greyed out</b>, so you can see what is coming without getting lost.
        Choose <b>I have experience</b> to unlock everything. You can switch at any time without losing your work.
        </div>
        """, unsafe_allow_html=True)

        if hay_datos():
            st.success("✅ You have a dataset loaded — continue with the tabs above, or press **🧭 What's next?**")
            gu.what_next("home", key="next_home_datos")
            if st.button("🗑️ Clear data and start over", key="btn_reset_home"):
                st.session_state.clear()
                st.rerun()
        else:
            st.info("👈 To get started, upload your own file in the left-hand panel (spectra or a table, CSV or Excel), "
                    "or find free data to practise in the **Public datasets** tab.")
            gu.what_next("home", key="next_home_vacio")
            c_a, c_b = st.columns(2)
            with c_a:
                st.markdown("**New to chemometrics?** Open the **🧭 How do we start?** tab and read the *Theoretical guide* (sidebar).")
            with c_b:
                st.markdown("**Questions or want to talk about your data?** See the **✉️ Contact** tab — messages are welcome.")

        _hero_img_b64 = _img_b64("hero_spectra.png", max_w=1300, paleta=True)
        if _hero_img_b64:
            st.markdown(f"""
        <div style="margin-top:30px; text-align:center;">
            <img src="{_hero_img_b64}" style="max-width:78%; opacity:0.92;">
        </div>
        """, unsafe_allow_html=True)


def _leer_espectros(archivo, tabular=False):
    """Reads an uploaded file in the app's usual format: first column = sample ID, the rest = variables.
    Returns (ids, axis_or_None, names_or_None, X)."""
    if archivo.name.lower().endswith(".csv"):
        df = pd.read_csv(archivo, index_col=0)
    else:
        df = pd.read_excel(archivo, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    ids = df.index.astype(str).to_numpy()
    if tabular:
        return ids, np.arange(1, df.shape[1] + 1, dtype=float), np.array([str(c) for c in df.columns], dtype=object), df.to_numpy(dtype=float)
    try:
        eje = ejes_a_float(df.columns.tolist())
    except Exception:
        raise ValueError("the column headers are not numbers (e.g. '%s'), so this does not look like a spectrum — "
                         "if it holds named variables, choose 'Tabular variables' above." % str(df.columns[0]))
    return ids, eje, None, df.to_numpy(dtype=float)





def _bundle_a_X(bundle, X, eje):
    """Applies what a saved (classification / regression) model needs to RAW spectra: interpolation to its axis,
    the same pre-treatment steps and the same variable selection. Returns the matrix the model expects."""
    Xi, _fuera = cu.interpolar_a_eje(X, eje, bundle["numeros_onda"])
    for paso in bundle["pasos_pretratamiento"]:
        Xi = aplicar_paso(Xi, paso)
    if bundle["mascara_variables"] is not None:
        Xi = Xi[:, bundle["mascara_variables"]]
    return Xi


# -----------------------------------------------------------------------
# TAB: QC — control charts over time (nothing runs until a button is clicked)
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# TAB: INSTRUMENT TRANSFER LAB — several instruments, nothing runs until a button is clicked
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# TAB: HYPERSPECTRAL — own data, light (reduced cube, PCA maps, k-means map)
# -----------------------------------------------------------------------



# -----------------------------------------------------------------------
# TAB: HOW DO WE START? (learning path + progress)
# -----------------------------------------------------------------------
if _abierta(tabs[31]):
    with tabs[31]:
        st.subheader("How do we start?")
        st.markdown("New to chemometrics? Follow these nine steps in order — each one has its own tab. The bar fills in as you "
                    "explore them. If you do not have data yet, find free datasets to practise in the **Public datasets** tab.")
        gu.journey()
        st.caption("Every tab has a **🧭 What's next?** button that recommends what to do depending on your results, and a "
                   "**📖 How to interpret?** button that explains each metric in plain words.")
        st.markdown("**Tip:** the *Theoretical guide* (sidebar) explains every method used here, from preprocessing to validation.")


# -----------------------------------------------------------------------
# TABS THAT NEED NO DATA: public datasets, contact, glossary
# -----------------------------------------------------------------------
if _abierta(tabs[27]):
    with tabs[27]:
        import datasets_tab as _dst
        _dst.render()
if _abierta(tabs[28]):
    with tabs[28]:
        import contact_tab as _cnt
        _cnt.render(enviar_mensaje_contacto, CONTACT_EMAIL)
if _abierta(tabs[29]):
    with tabs[29]:
        import glosario_tab as _glo
        _glo.render()

if not hay_datos():
    # All the data-dependent tabs need a loaded dataset to make sense — show a friendly
    # placeholder in each of them (so they're not just blank if clicked) and stop here,
    # before their real logic below (which assumes data exists).
    for _k_vacia in sorted(k for k in tabs if 1 <= k <= 16 or k == 30):
        with tabs[_k_vacia]:
            st.info("📁 Load your data in the left-hand panel — or find free data in the **Public datasets** tab — to get started.")
            gu.what_next("home", key=f"next_vacio_{_k_vacia}")
    st.stop()

# -----------------------------------------------------------------------
# TAB: DATA
# -----------------------------------------------------------------------
if _abierta(tabs[1]):
    # Each tab re-runs ON ITS OWN when one of its widgets changes (fragment), instead of
    # re-executing the whole app. Buttons that must refresh other parts call st.rerun().
    @st.fragment
    def _frag_tab_1():
        st.subheader("Data preview")
        gu.barra_guia("data")
        _df_prev = st.session_state.df.head(10)
        if _df_prev.shape[1] > 20:
            # wide spectra (thousands of variables): a table with every column is slow to build and
            # draw and unreadable anyway - show the leading columns and the last few
            st.dataframe(pd.concat([_df_prev.iloc[:, :14], _df_prev.iloc[:, -3:]], axis=1), width='stretch')
            st.caption(f"Showing 17 of {_df_prev.shape[1]} columns (the first 14 and the last 3). "
                       "All variables are used in the analyses.")
        else:
            st.dataframe(_df_prev, width='stretch')

        ids, X, clases = datos_activos()

        if es_tabular():
            _datos_tabulares(X, st.session_state.nombres_var, clases)
            return
        st.subheader("Spectra")
        fig = go.Figure()
        colores_clase = None
        if clases is not None:
            paleta = CLASS_PALETTE
            clases_unicas = np.unique(clases)
            colores_clase = {c: paleta[i % len(paleta)] for i, c in enumerate(clases_unicas)}

        _vistos = set()
        for i in range(X.shape[0]):
            _g = str(clases[i]) if clases is not None else "samples"
            fig.add_trace(_traza_espectro(
                st.session_state.numeros_onda, X[i], gl=(X.shape[0] > 500), mode="lines",
                line=dict(width=1, color=colores_clase[clases[i]] if colores_clase is not None else "steelblue"),
                opacity=0.6, name=_g, legendgroup=_g, showlegend=(clases is not None and _g not in _vistos),
                hovertext=str(ids[i]), hoverinfo="text+x+y"))
            _vistos.add(_g)
        fig.update_layout(height=450, xaxis_title="Wavenumber / wavelength", yaxis_title="Signal")
        if st.session_state.numeros_onda[0] > st.session_state.numeros_onda[-1]:
            fig.update_xaxes(autorange="reversed")
        st.plotly_chart(fig, width='stretch')
        gu.nota("data_spectra", "raw spectra plot", "dataset")
        st.caption(f"Showing {X.shape[0]} of {st.session_state.X.shape[0]} samples "
                   f"({int(st.session_state.mascara_excluidas.sum())} excluded as outliers).")
    with tabs[1]:
        _frag_tab_1()


# -----------------------------------------------------------------------
# TAB: PREPROCESSING
# -----------------------------------------------------------------------
if _abierta(tabs[2]):
    # Each tab re-runs ON ITS OWN when one of its widgets changes (fragment), instead of
    # re-executing the whole app. Buttons that must refresh other parts call st.rerun().
    @st.fragment
    def _frag_tab_2():
        if es_tabular():
            _pret_tabular()
            return
        st.subheader("Spectral preprocessing")
        gu.barra_guia("prep", pros=("prep_spectral", None))

        # Important: preprocessing is always computed on ALL samples (not just the
        # active ones), because further down (PCA, Outliers, Dendrogram, etc.) this
        # same result is filtered with the exclusion mask. If it came pre-filtered
        # here, that second filtering would break as soon as an outlier is excluded
        # (mismatched sizes).
        ids, X, clases = st.session_state.ids, st.session_state.X, st.session_state.clases
        numeros_onda_crudo = st.session_state.numeros_onda

        tipo_senal = st.radio(
            "Signal type",
            ["NIR / MIR", "Raman", "Chromatogram", "NMR"],
            horizontal=True, key="pret_w_tipo",
            help="Changes which preprocessing options are available below.",
        )

        # --- Step 0: NMR only (bucketing). Also changes the spectral axis. ---
        X_paso0 = X
        numeros_onda_paso0 = numeros_onda_crudo
        ancho_bucket_usado = None
        alin_firma = None
        if tipo_senal in ("NMR", "Chromatogram"):
            _alinear = st.checkbox(
                "Align peaks (correct band / peak shifts)", value=False, key="pret_w_align",
                help="Peaks that move between samples (NMR: pH, ionic strength, temperature; chromatograms: "
                     "retention-time drift) look like different variables to PCA/PLS. Each interval of the "
                     "spectrum is shifted, sample by sample, to match a reference (the median spectrum) "
                     "(icoshift-style). Centering does NOT fix this — alignment does. It is applied before "
                     "bucketing and does not change the axis.")
            if _alinear:
                _rg = abs(float(numeros_onda_crudo[-1]) - float(numeros_onda_crudo[0]))
                ca1, ca2 = st.columns(2)
                _n_int = ca1.slider("Number of intervals", 5, 200, 40, key="pret_w_align_n",
                                    help="More intervals = more local correction (and more risk of distorting "
                                         "wide peaks). 30–80 is a good start.")
                _max_d = ca2.slider("Maximum shift (axis units, e.g. ppm or min)",
                                    min_value=float(_rg / 2000), max_value=float(_rg / 20),
                                    value=float(_rg / 200), format="%.5f", key="pret_w_align_d",
                                    help="Largest displacement allowed in each interval. Keep it smaller than the "
                                         "distance between neighbouring peaks.")
                try:
                    X_paso0, _desp = _alinear_cacheado(X, np.asarray(numeros_onda_crudo, dtype=float), int(_n_int), round(float(_max_d), 8))
                    alin_firma = ("align", int(_n_int), round(float(_max_d), 8))
                    st.caption(f"Alignment applied: mean |shift| = {np.abs(_desp).mean():.2f} points, "
                               f"max = {np.abs(_desp).max()} points (the search is capped at ±80 points).")
                except Exception as e:
                    st.error(f"Alignment failed: {e}")
        if tipo_senal == "NMR":
            usar_bucketing = st.checkbox(
                "Apply bucketing (binning)", value=True, key="pret_w_bucketing",
                help="Groups the high-resolution NMR spectrum into fixed-width bins, summing the signal "
                     "within each — drastically reduces the number of variables and makes the data more "
                     "robust to small peak shifts between samples.",
            )
            if usar_bucketing:
                rango_eje = abs(float(numeros_onda_crudo[-1]) - float(numeros_onda_crudo[0]))
                ancho_default = max(rango_eje / 300, 1e-6)
                ancho_min = max(rango_eje / 3000, 1e-6)
                ancho_max = max(rango_eje / 10, ancho_default * 2)
                ancho_bucket = st.slider(
                    "Bucket width (same units as your spectral axis, e.g. ppm or Hz)",
                    min_value=float(ancho_min), max_value=float(ancho_max),
                    value=float(ancho_default), format="%.5f", key="pret_w_ancho",
                    help="Wider buckets = fewer variables and more tolerance to peak shifts, but less "
                         "resolution (nearby peaks can merge). Narrower buckets keep more detail but "
                         "are more sensitive to small shifts between samples.",
                )
                try:
                    X_paso0, numeros_onda_paso0 = cu.bucketing(X_paso0, numeros_onda_crudo, ancho_bucket)
                    ancho_bucket_usado = round(float(ancho_bucket), 8)
                    st.caption(f"Bucketing applied: {X.shape[1]} variables → {X_paso0.shape[1]} buckets.")
                    if X_paso0.shape[1] > 3000:
                        st.warning(
                            f"Bucketing left {X_paso0.shape[1]} variables — that's a lot for "
                            "a small dataset. Try increasing the bucket width."
                        )
                except Exception as e:
                    st.error(f"Bucketing failed: {e}")

        st.caption("Build a combination in 2 steps: first a scatter/scale/baseline "
                   "correction, then a Savitzky-Golay filter (when it applies to the "
                   "chosen signal type).")

        opciones_a_por_tipo = {
            "NIR / MIR": ["None", "Normalization (area)", "Normalization (vector)",
                          "Standardization", "SNV", "MSC", "EMSC"],
            "Raman": ["None", "Remove cosmic rays", "Baseline correction (ALS)",
                      "SNV", "Standardization"],
            "Chromatogram": ["None", "Baseline correction (ALS)",
                             "Normalization (area)", "Standardization"],
            "NMR": ["None", "Normalization (PQN)", "Standardization", "SNV"],
        }
        opciones_b_por_tipo = {
            "NIR / MIR": ["None", "Smoothing (order 0)", "1st derivative", "2nd derivative"],
            "Raman": ["None", "Smoothing (order 0)", "1st derivative", "2nd derivative"],
            "Chromatogram": ["None"],
            "NMR": ["None"],
        }

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            paso_a = st.selectbox("Step A", opciones_a_por_tipo[tipo_senal], key=f"pret_w_a_{tipo_senal}", help="First pre-treatment. Baseline/scatter correction is usually applied first; 'None' skips it.")
        with col2:
            paso_b = st.selectbox("Step B (smoothing / derivative)", opciones_b_por_tipo[tipo_senal], key=f"pret_w_b_{tipo_senal}", help="Savitzky-Golay smoothing or derivative. A derivative removes baseline and sharpens overlapped bands but amplifies noise.")
        with col3:
            orden = st.radio("Application order", ["A → B (recommended)", "B → A"], key="pret_w_orden", help="Order in which the two steps are applied. The order changes the result.")

        if paso_a == "Baseline correction (ALS)":
            c1, c2 = st.columns(2)
            lam_als = c1.select_slider("Baseline smoothness (lambda)",
                                        options=[1e3, 1e4, 1e5, 1e6, 1e7, 1e8], value=1e5, key="pret_w_lam", help="Higher lambda = smoother (stiffer) baseline. Use higher values for broad baselines.")
            p_als = c2.slider("Asymmetry (p)", min_value=0.001, max_value=0.1, value=0.01, step=0.001, key="pret_w_p", help="Low p (0.001–0.05) keeps the baseline under the peaks; higher p lets it follow the signal.")
        if paso_a == "Remove cosmic rays":
            c1, c2 = st.columns(2)
            ventana_rc = c1.slider("Median filter window (odd)", min_value=3, max_value=15, value=5, step=2, key="pret_w_vrc", help="Window of the median filter used to remove spikes. Larger windows also flatten narrow real peaks.")
            umbral_rc = c2.slider("Threshold (modified z-score)", min_value=3, max_value=15, value=7, key="pret_w_urc", help="Points deviating more than this many robust z-units are treated as spikes. Lower = removes more.")
        if paso_a == "EMSC":
            orden_emsc = st.slider(
                "Polynomial order (wavelength-dependent effects)", min_value=0, max_value=4, value=2, key="pret_w_emsc",
                help="How many polynomial terms of the wavelength axis to model and remove, on top of "
                     "the usual offset + scaling that plain MSC already corrects for. Order 0 behaves "
                     "like plain MSC; higher orders can correct sloped or curved baseline effects that "
                     "vary smoothly across the spectrum, at the cost of possibly removing some real "
                     "chemical signal if set too high.",
            )

        if paso_b != "None":
            c1, c2 = st.columns(2)
            ventana = c1.slider("SG window", min_value=5, max_value=51, value=11, step=2, key="pret_w_ventana", help="Number of points in the smoothing window (odd). Bigger = smoother but wider peaks are distorted; keep it narrower than your narrowest peak.")
            orden_poly = c2.slider("Polynomial order", min_value=1, max_value=5, value=2, key="pret_w_opoly", help="Degree of the local polynomial. Higher keeps peak shape but smooths less; must be lower than the window.")
        else:
            ventana, orden_poly = 11, 2

        mapa_a = {
            "Normalization (area)": ("normalizar", {"modo": "area"}),
            "Normalization (vector)": ("normalizar", {"modo": "vector"}),
            "Normalization (PQN)": ("pqn", {}),
            "Standardization": ("estandarizar", {}),
            "SNV": ("snv", {}),
            "MSC": ("msc", {}),
            "EMSC": ("emsc", {"eje": numeros_onda_paso0, "orden_polinomio": orden_emsc} if paso_a == "EMSC" else {}),
            "Baseline correction (ALS)": (
                "linea_base_als", {"lam": lam_als, "p": p_als} if paso_a == "Baseline correction (ALS)" else {}
            ),
            "Remove cosmic rays": (
                "eliminar_rayos_cosmicos", {"ventana": ventana_rc, "umbral": umbral_rc}
                if paso_a == "Remove cosmic rays" else {}
            ),
        }
        mapa_b = {
            "Smoothing (order 0)": ("suavizado_sg", {"ventana": ventana, "orden_polinomio": orden_poly}),
            "1st derivative": ("derivada_sg", {"orden": 1, "ventana": ventana, "orden_polinomio": orden_poly}),
            "2nd derivative": ("derivada_sg", {"orden": 2, "ventana": ventana, "orden_polinomio": orden_poly}),
        }

        # (aplicar_paso and FUNCIONES_EXTRA are defined above, at module level)

        paso_a_tup = mapa_a.get(paso_a)
        paso_b_tup = mapa_b.get(paso_b)
        secuencia = [paso_a_tup, paso_b_tup] if orden.startswith("A") else [paso_b_tup, paso_a_tup]

        # PREVIEW only: what these settings would produce. It is NOT sent to the rest
        # of the app until the user clicks "Apply" below.
        error_pretratamiento = None
        try:
            X_pret = _aplicar_pretratamiento_cacheado(X_paso0, tuple(secuencia))
        except Exception as e:
            error_pretratamiento = str(e)
            X_pret = X_paso0.copy()

        if error_pretratamiento:
            st.error(
                "Could not apply the preprocessing (check the chosen parameters, "
                "e.g. that the SG window is odd and larger than the polynomial order). "
                f"Details: {error_pretratamiento}"
            )

        def _sin_arrays(paso):
            if paso is None:
                return None
            nombre, params = paso
            return (nombre, tuple(sorted(
                (k, f"array{v.shape}" if hasattr(v, "shape") else v) for k, v in params.items())))

        _propuesta = repr((tuple(_sin_arrays(p) for p in secuencia if p is not None),
                           ancho_bucket_usado if alin_firma is None else (ancho_bucket_usado, alin_firma)))
        _ya_aplicada = (st.session_state.get("pret_propuesta_aplicada") == _propuesta)

        st.markdown("**Before / after comparison**")
        col_izq, col_der = st.columns(2)
        with col_izq:
            fig1 = go.Figure()
            for i in range(min(X_paso0.shape[0], 60)):
                fig1.add_trace(_traza_espectro(numeros_onda_paso0, X_paso0[i],
                                           mode="lines", line=dict(width=0.8), opacity=0.5,
                                           showlegend=False))
            fig1.update_layout(title="Before", height=380)
            if numeros_onda_paso0[0] > numeros_onda_paso0[-1]:
                fig1.update_xaxes(autorange="reversed")
            st.plotly_chart(fig1, width='stretch')
            gu.nota("prep_before", "spectra before preprocessing", "dataset")
        with col_der:
            fig2 = go.Figure()
            for i in range(min(X_pret.shape[0], 60)):
                fig2.add_trace(_traza_espectro(numeros_onda_paso0, X_pret[i],
                                           mode="lines", line=dict(width=0.8), opacity=0.5,
                                           showlegend=False))
            fig2.update_layout(title="After", height=380)
            if numeros_onda_paso0[0] > numeros_onda_paso0[-1]:
                fig2.update_xaxes(autorange="reversed")
            st.plotly_chart(fig2, width='stretch')
            gu.nota("prep_after", "spectra after preprocessing", "dataset")

        nombres_pasos_es = [p[0] for p in secuencia if p is not None]
        nombres_mostrar = []
        if orden.startswith("A"):
            if paso_a != "None": nombres_mostrar.append(paso_a)
            if paso_b != "None": nombres_mostrar.append(paso_b)
        else:
            if paso_b != "None": nombres_mostrar.append(paso_b)
            if paso_a != "None": nombres_mostrar.append(paso_a)
        if alin_firma is not None:
            nombres_mostrar.insert(0, f"Peak alignment ({alin_firma[1]} intervals)")
        if nombres_mostrar:
            st.caption("Combination in the preview: " + " → ".join(nombres_mostrar))
        else:
            st.caption("No preprocessing in the preview (raw spectra).")

        if _ya_aplicada:
            st.success("✅ This combination is the one currently applied to the analysis "
                       "(PCA, Outliers, Dendrogram, Classification, Regression...).")
        else:
            st.warning("⚠️ The combination above is only a **preview**. The rest of the app is still "
                       f"using: **{st.session_state.get('pret_desc_aplicada', 'none (raw spectra)')}**. "
                       "Click **Apply** to use this one instead.")
        col_ap1, col_ap2, _ = st.columns([1.5, 1.5, 3])
        if col_ap1.button("✅ Apply to the analysis", key="pret_btn_aplicar",
                          type="secondary" if _ya_aplicada else "primary",
                          disabled=_ya_aplicada or error_pretratamiento is not None,
                          help="Sends this preprocessing to every other tab. Results already computed "
                               "there will be flagged as out of date — nothing is recomputed until you ask."):
            st.session_state.X_pret = X_pret
            st.session_state.numeros_onda_pret = numeros_onda_paso0
            st.session_state.pasos_pretratamiento = [p for p in secuencia if p is not None]
            st.session_state.pret_propuesta_aplicada = _propuesta
            st.session_state["pret_tipo_aplicado"] = tipo_senal
            st.session_state.pret_desc_aplicada = " → ".join(nombres_mostrar) if nombres_mostrar else "none (raw spectra)"
            _rerun_tab()
        _es_crudo = (st.session_state.get("pret_propuesta_aplicada") == repr(((), None)))
        if col_ap2.button("↩ Reset (raw spectra)", key="pret_btn_reset",
                          disabled=_es_crudo and _ya_aplicada,
                          help="Goes back to the raw spectra and puts these settings back to 'None'."):
            resetear_prefijo("pret_w_")
            reiniciar_pretratamiento()
            _rerun_tab()

        # Light preview only (building and sending the whole table on every interaction
        # is slow for big datasets); the full files are generated on request below.
        n_prev_filas, n_prev_cols = min(10, X_pret.shape[0]), min(40, X_pret.shape[1])
        st.caption(f"Preview of the 'After' data: first {n_prev_filas} samples × first {n_prev_cols} "
                   f"variables (of {X_pret.shape[0]} × {X_pret.shape[1]}).")
        st.dataframe(pd.DataFrame(X_pret[:n_prev_filas, :n_prev_cols], index=ids[:n_prev_filas],
                                  columns=[f"{v:.6g}" if i == 0 or f"{v:.6g}" != f"{numeros_onda_paso0[i-1]:.6g}" else f"{v:.8g}" for i, v in enumerate(numeros_onda_paso0[:n_prev_cols])]),
                     width='stretch', height=220)

        if st.button("📦 Prepare download files (CSV / Excel)", key="pret_btn_preparar",
                     help="Builds the complete 'After' dataset as files. Done only on request to keep the tab fast."):
            with st.spinner("Preparing files..."):
                df_exportar = cu.armar_dataframe_exportable(ids, numeros_onda_paso0, X_pret)
                st.session_state["pret_descarga"] = {
                    "propuesta": _propuesta,
                    "csv": df_exportar.to_csv().encode("utf-8"),
                    "xlsx": df_a_excel_bytes({"preprocessed_spectra": df_exportar.reset_index(names="id")}),
                }
        _desc = st.session_state.get("pret_descarga")
        if _desc is not None and _desc["propuesta"] == _propuesta:
            col_dl1, col_dl2 = st.columns(2)
            col_dl1.download_button(
                "⬇️ Download preprocessed dataset (CSV)", data=_desc["csv"],
                file_name="preprocessed_spectra.csv", mime="text/csv", key="pret_dl_csv",
            )
            col_dl2.download_button(
                "⬇️ Download preprocessed dataset (Excel)", data=_desc["xlsx"],
                file_name="preprocessed_spectra.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="pret_dl_xlsx",
            )
    with tabs[2]:
        _frag_tab_2()


# -----------------------------------------------------------------------
# TAB: PCA
# -----------------------------------------------------------------------
if _abierta(tabs[4]):
    # Each tab re-runs ON ITS OWN when one of its widgets changes (fragment), instead of
    # re-executing the whole app. Buttons that must refresh other parts call st.rerun().
    @st.fragment
    def _frag_tab_4():
        st.subheader("Principal Component Analysis (PCA)")
        gu.barra_guia("pca", "pca")

        ids_actuales, _, clases_actuales = datos_activos()
        X_pca_input = st.session_state.X_pret[indice_activo()]
        X_pca_input, _eje_pca_in, _crop_mask_pca, _ = elegir_espectro_modelado(
            X_pca_input, np.array(st.session_state.numeros_onda_pret, dtype=float), "pca")
        n_muestras, n_variables = X_pca_input.shape
        n_comp_max = min(n_muestras - 1, n_variables)

        if n_comp_max < 2:
            st.warning("At least 3 samples are needed to compute PCA.")
        else:
            esta_desactualizado_pca = insignia_estado("pca_firma", "The PCA result", firma_actual=firma_modelos(_crop_mask_pca))
            col_btn_pca, col_clr_pca, _ = st.columns([1.4, 1, 4])
            if col_clr_pca.button("🧹 Clear", key="btn_limpiar_pca",
                                  disabled=st.session_state.get("pca_completo") is None,
                                  help="Removes the PCA result — and the Outliers result and the PCA-based "
                                       "tools in 'Other tools', which are built on it. Nothing is recomputed."):
                limpiar_pca()
                _rerun_tab()
            with col_btn_pca:
                _texto_boton_pca = "▶ Compute PCA" if st.session_state.get("pca_completo") is None else "🔄 Update PCA"
                if st.button(_texto_boton_pca, type="primary" if esta_desactualizado_pca else "secondary",
                             key="btn_computar_pca"):
                    with st.spinner("Fitting PCA..."):
                        _pca_completo = _ajustar_pca_cacheado(X_pca_input, n_comp_max)
                        st.session_state.pca_completo = _pca_completo
                        st.session_state.scores_completo = _pca_completo.transform(X_pca_input)
                        st.session_state.cargas_completo = _pca_completo.components_
                        st.session_state.autovalores = _pca_completo.explained_variance_
                        st.session_state.var_explicada = _pca_completo.explained_variance_ratio_ * 100
                        # IMPORTANT: freeze EVERYTHING this PCA result is consistent with —
                        # if outlier exclusions change afterwards (without clicking Update),
                        # using live data/ids anywhere downstream would silently mismatch
                        # this frozen result in length. Outliers (next tab) reuses all of this.
                        st.session_state.pca_X_input = X_pca_input
                        st.session_state.pca_eje = np.array(_eje_pca_in)
                        st.session_state.pca_origen = "cropped" if _crop_mask_pca is not None else "full"
                        st.session_state.pca_crop_desc = st.session_state.get("crop_desc_aplicada") if _crop_mask_pca is not None else None
                        st.session_state.pca_ids = ids_actuales
                        st.session_state.pca_clases = clases_actuales
                        st.session_state.pca_firma = firma_modelos(_crop_mask_pca)
                    _rerun_tab()

            if st.session_state.get("pca_completo") is None:
                st.info("Click **Compute PCA** above to get started.")
            else:
                pca_completo = st.session_state.pca_completo
                scores_completo = st.session_state.scores_completo
                cargas_completo = st.session_state.cargas_completo
                autovalores = st.session_state.autovalores
                ids = st.session_state.pca_ids
                clases = st.session_state.pca_clases
                var_explicada = st.session_state.var_explicada
                var_acumulada = np.cumsum(var_explicada)
                # Use the STORED result's own dimensions from here on, not the live
                # dataset's — if outliers were excluded/restored after this PCA was
                # computed (without clicking Update), the live data could have a
                # different number of samples/components than this stale result.
                n_comp_max_guardado = len(var_explicada)

                col1, col2 = st.columns([2, 1])
                with col1:
                    fig = go.Figure()
                    fig.add_bar(x=list(range(1, len(var_explicada) + 1)), y=var_explicada, name="% individual")
                    fig.add_trace(go.Scatter(x=list(range(1, len(var_acumulada) + 1)), y=var_acumulada,
                                              mode="lines+markers", name="% cumulative", yaxis="y2"))
                    fig.update_layout(
                        title="Explained variance (scree plot)",
                        xaxis_title="Principal component",
                        yaxis=dict(title="% individual"),
                        yaxis2=dict(title="% cumulative", overlaying="y", side="right"),
                        xaxis=dict(range=[0.5, min(15, len(var_explicada)) + 0.5]),
                        height=380,
                    )
                    st.plotly_chart(fig, width='stretch')
                    gu.nota("pca_scree", "scree plot", "pca")
                with col2:
                    n_comp = st.slider("Components to retain (n_comp)", min_value=2,
                                        max_value=min(10, n_comp_max_guardado), value=min(3, n_comp_max_guardado),
                                        help="How many principal components to keep for scores/loadings plots "
                                             "and for outlier detection (T²/Q). Check the scree plot on the "
                                             "left: pick enough to capture most of the variance, without "
                                             "including components that just look like noise.")
                    st.metric("Cumulative explained variance", f"{var_acumulada[n_comp - 1]:.1f}%")
                    st.session_state.n_comp = n_comp
                if es_tabular() and float(var_explicada[0]) > 85 and st.session_state.get("tab_prep") is None:
                    st.warning(f"⚠️ **PC1 explains {var_explicada[0]:.1f} % of the variance** — with unscaled tables this almost always means one "
                               "variable with very large numbers (e.g. proline vs. a pH) dominates the PCA. Go to **🧪 Preprocessing**, "
                               "choose *autoscaling* and apply it, then recompute the PCA: every variable will weigh the same.")

                scores = scores_completo[:, :n_comp]
                cargas = cargas_completo[:n_comp, :]
                st.session_state.scores = scores
                st.session_state.cargas = cargas

                # ---- colouring: by class, or by a continuous value (colour gradient)
                _color_vals, _color_lab = None, None
                _valores_pca = None
                if st.session_state.get("valores_y") is not None:
                    _mapa_y = dict(zip(map(str, st.session_state.ids), st.session_state.valores_y))
                    _valores_pca = np.array([_mapa_y.get(str(i), np.nan) for i in ids], dtype=float)
                _opc_color = ["Class" if clases is not None else "None"]
                if _valores_pca is not None:
                    _opc_color.append("Reference value (gradient)")
                _opc_color.append("Variable (gradient)" if es_tabular() else "Spectral variable (gradient)")
                _modo_color = st.radio("Color samples by", _opc_color, horizontal=True, key="pca_w_color",
                                       help="'Reference value' paints each sample on a colour scale according to its "
                                            "analyte concentration / reference value. 'Spectral variable' uses the "
                                            "(preprocessed) signal at one wavelength / variable instead.")
                if _modo_color.startswith("Reference"):
                    _color_vals, _color_lab = _valores_pca, "Reference value"
                    if np.isnan(_valores_pca).any():
                        st.caption(f"{int(np.isnan(_valores_pca).sum())} sample(s) have no reference value and are drawn in grey.")
                elif _modo_color.startswith(("Spectral", "Variable")):
                    _eje_col = np.asarray(st.session_state.pca_eje, dtype=float)
                    _X_col = st.session_state.pca_X_input
                    if es_tabular():
                        _nm_col = list(etiquetas_var(_eje_col))
                        _j = st.selectbox("Variable", range(len(_nm_col)), format_func=lambda i: _nm_col[i], key="pca_w_color_varT", help="Variable to inspect.")
                        _color_vals, _color_lab = _X_col[:, _j], _nm_col[_j]
                    else:
                        _v_txt = st.select_slider("Variable (wavelength / wavenumber)",
                                                  options=[float(v) for v in _eje_col],
                                                  value=float(_eje_col[len(_eje_col) // 2]), key="pca_w_color_var",
                                                  format_func=lambda v: f"{v:g}", help="Position of the variable on the spectral axis.")
                        _j = int(np.argmin(np.abs(_eje_col - _v_txt)))
                        _color_vals, _color_lab = _X_col[:, _j], f"Signal @ {_eje_col[_j]:g}"

                def grafico_scores(pc_x, pc_y):
                    _kw = dict(color=clases if clases is not None else None,
                               color_discrete_sequence=CLASS_PALETTE)
                    if _color_vals is not None:
                        _kw = dict(color=_color_vals, color_continuous_scale="Viridis")
                    fig = px.scatter(
                        x=scores_completo[:, pc_x - 1], y=scores_completo[:, pc_y - 1],
                        hover_name=ids,
                        labels={"x": f"PC{pc_x} ({var_explicada[pc_x-1]:.1f}%)",
                                "y": f"PC{pc_y} ({var_explicada[pc_y-1]:.1f}%)",
                                "color": _color_lab or "Class"},
                        title=f"Scores: PC{pc_x} vs PC{pc_y}", **_kw,
                    )
                    if _color_vals is not None:
                        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="rgba(60,60,60,.5)")))
                        fig.update_layout(coloraxis_colorbar=dict(title=_color_lab))
                    fig.add_hline(y=0, line_color="lightgray")
                    fig.add_vline(x=0, line_color="lightgray")
                    fig.update_layout(height=420)
                    return fig

                n_comp_disponibles = min(15, n_comp_max_guardado)
                opciones_pc = list(range(1, n_comp_disponibles + 1))
                st.markdown("**2D scores plots** — pick any pair of components to compare "
                             f"(up to PC{n_comp_disponibles}).")
                c1, c2 = st.columns(2)
                with c1:
                    cc1, cc2 = st.columns(2)
                    pcx_1 = cc1.selectbox("X axis (PC)", opciones_pc, format_func=lambda _k: f"PC{_k}", index=0, key="pcx_1",
                                           help="Which principal component to plot on the horizontal axis.")
                    pcy_1 = cc2.selectbox("Y axis (PC)", opciones_pc, format_func=lambda _k: f"PC{_k}", index=min(1, n_comp_disponibles - 1), key="pcy_1",
                                           help="Which principal component to plot on the vertical axis.")
                    st.plotly_chart(grafico_scores(pcx_1, pcy_1), width='stretch')
                    gu.nota("pca_scores_a", "scores plot (left)", "pca")
                with c2:
                    cc3, cc4 = st.columns(2)
                    idx_default_x2 = min(1, n_comp_disponibles - 1)
                    idx_default_y2 = min(2, n_comp_disponibles - 1)
                    pcx_2 = cc3.selectbox("X axis (PC)", opciones_pc, format_func=lambda _k: f"PC{_k}", index=idx_default_x2, key="pcx_2", help="Component shown on the horizontal axis.")
                    pcy_2 = cc4.selectbox("Y axis (PC)", opciones_pc, format_func=lambda _k: f"PC{_k}", index=idx_default_y2, key="pcy_2", help="Component shown on the vertical axis.")
                    st.plotly_chart(grafico_scores(pcx_2, pcy_2), width='stretch')
                    gu.nota("pca_scores_b", "scores plot (right)", "pca")

                st.markdown("**Interactive 3D plot (drag to rotate)**")
                if n_comp_max_guardado >= 3:
                    cc5, cc6, cc7 = st.columns(3)
                    pcx_3d = cc5.selectbox("X axis (PC)", opciones_pc, format_func=lambda _k: f"PC{_k}", index=0, key="pcx_3d", help="Component shown on the X axis.")
                    pcy_3d = cc6.selectbox("Y axis (PC)", opciones_pc, format_func=lambda _k: f"PC{_k}", index=min(1, n_comp_disponibles - 1), key="pcy_3d", help="Component shown on the Y axis.")
                    pcz_3d = cc7.selectbox("Z axis (PC)", opciones_pc, format_func=lambda _k: f"PC{_k}", index=min(2, n_comp_disponibles - 1), key="pcz_3d", help="Component shown on the Z axis.")
                    col_x, col_y, col_z = f"PC{pcx_3d}", f"PC{pcy_3d}", f"PC{pcz_3d}"
                    df3d = pd.DataFrame({
                        col_x: scores_completo[:, pcx_3d - 1],
                        col_y: scores_completo[:, pcy_3d - 1],
                        col_z: scores_completo[:, pcz_3d - 1],
                    })
                    df3d["ID"] = ids
                    if _color_vals is not None:
                        df3d[_color_lab] = _color_vals
                        fig3d = px.scatter_3d(df3d, x=col_x, y=col_y, z=col_z, color=_color_lab, hover_name="ID",
                                               color_continuous_scale="Viridis")
                    else:
                        df3d["Class"] = clases if clases is not None else "All samples"
                        fig3d = px.scatter_3d(df3d, x=col_x, y=col_y, z=col_z, color="Class", hover_name="ID",
                                               color_discrete_sequence=CLASS_PALETTE)
                    fig3d.update_traces(marker=dict(size=5))
                    fig3d.update_layout(height=550)
                    st.plotly_chart(fig3d, width='stretch')
                else:
                    st.caption("At least 3 components are needed for the 3D plot.")
                gu.nota("pca_3d", "3D scores plot", "pca")

                st.divider()
                _bloque_cargas_biplot(scores_completo, cargas_completo, autovalores, var_explicada,
                                      st.session_state.pca_X_input, np.asarray(st.session_state.pca_eje, dtype=float),
                                      ids, clases, _color_vals, _color_lab)
    with tabs[4]:
        _frag_tab_4()


# -----------------------------------------------------------------------
# TAB: OUTLIERS
# -----------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=6, ttl=3600)
def _pca_pcf_cacheado(X, k):
    return pcf.ajustar_pca(X, k)


if _abierta(tabs[5]):
    @st.fragment
    def _frag_tab_14():
        st.subheader("PC finder — which PCs carry the difference, and which variables explain it")
        st.caption(
            "A quick exploratory shortcut. **Classification:** finds the PCs (alone, in pairs or in triples) where the "
            "classes are most separated. **Regression:** finds the PCs where the reference values (e.g. concentration) "
            "change along a clear direction. The informative PCs are often NOT the first ones — they can be partly "
            "hidden behind larger sources of variation. The chosen combination is then traced back to the original "
            "variables, so you can see which part of the spectrum / chromatogram (ppm, wavenumber, retention time) "
            "separates the groups or follows the concentration: candidate functional groups, compounds or markers.")

        ids_a, _, clases_a = datos_activos()
        X_in = st.session_state.X_pret[indice_activo()]
        X_in, eje_in, _m_pcf, _ = elegir_espectro_modelado(
            X_in, np.array(st.session_state.numeros_onda_pret, dtype=float), "pcf")
        eje_in = np.asarray(eje_in, dtype=float)

        valores_a = None
        if st.session_state.get("valores_y") is not None:
            _mapa = dict(zip(map(str, st.session_state.ids), st.session_state.valores_y))
            valores_a = np.array([_mapa.get(str(i), np.nan) for i in ids_a], dtype=float)

        tareas = []
        if clases_a is not None:
            _cnt = pd.Series(clases_a).value_counts()
            if len(_cnt) >= 2 and _cnt.min() >= 2:
                tareas.append("Classification")
        if valores_a is not None and np.isfinite(valores_a).sum() >= 8:
            tareas.append("Regression")
        if X_in.shape[0] < 6 or not tareas:
            st.info("Load classes (≥ 2 classes with ≥ 2 samples each) or reference values (≥ 8 samples) in the "
                    "left-hand panel to use this tab.")
            return

        tarea = st.radio("What do you want to find?", tareas, horizontal=True, key="pcf_w_tarea", help="Pick the kind of structure to search for; each option explains what it looks for once selected.")
        if tarea == "Regression":
            _ok = np.isfinite(valores_a)
            X_f, y_f, ids_f, cl_f = X_in[_ok], valores_a[_ok], ids_a[_ok], None
        else:
            X_f, y_f, ids_f, cl_f = X_in, None, ids_a, np.asarray(clases_a)

        c1, c2 = st.columns(2)
        k_tope = int(max(3, min(15, X_f.shape[0] - 1)))
        k_max = c1.slider("PCs to explore", 3, k_tope, min(8, k_tope), key="pcf_w_k",
                          help="Only the first N principal components are searched.")
        tam = c2.radio("Combination size", [1, 2, 3], index=1, horizontal=True, key="pcf_w_tam",
                       format_func=lambda v: {1: "1 PC", 2: "2 PCs", 3: "3 PCs"}[v],
                       help="2 PCs gives a plot you can read directly; 3 PCs finds more but is harder to see.")

        _sig_pcf = (tarea, X_f.shape, float(np.nansum(X_f[:, ::97])), k_max, tam)
        if st.button("▶ Run PC search", type="primary", key="pcf_btn"):
            st.session_state["pcf_sig"] = _sig_pcf
        if st.session_state.get("pcf_sig") != _sig_pcf:
            st.info("Choose the settings above and click **Run PC search** (nothing is computed until you do; "
                    "changing a setting asks you to run it again).")
            return

        with st.spinner("Searching..."):
            scores, cargas, var_exp = _pca_pcf_cacheado(X_f, k_tope)
            if tarea == "Classification":
                rank = pcf.buscar_clasificacion(scores, cl_f, k_max, tam)
                col_crit, col_cv = "Separation (1-Wilks)", "CV balanced accuracy"
            else:
                rank = pcf.buscar_regresion(scores, y_f, k_max, tam)
                col_crit, col_cv = "Adjusted R²", "Q² (CV)"

        st.markdown(f"**Best combinations of {tam} PC{'s' if tam > 1 else ''}**")
        tabla = rank.head(10)[["PCs", col_crit, col_cv]].copy()
        tabla.insert(1, "Explained variance (PCA)", [
            f"{sum(var_exp[j] for j in c):.1f}%" for c in rank.head(10)["cols"]])
        st.dataframe(tabla.round(3), width="stretch", hide_index=True)
        if tarea == "Classification":
            st.caption("**Separation** = share of the variance in those PCs that lies between classes (0–1). The "
                       "cross-validated accuracy confirms the top candidates with a simple LDA on those PCs only. "
                       "Always compare combinations of the same size.")
        else:
            st.caption("**Adjusted R²** = how well those PCs explain the reference values (linear fit); **Q²** is the "
                       "cross-validated version (closer to R² = more trustworthy).")

        opciones = [f"{r.PCs}  ·  {r[col_crit]:.3f}" for _, r in rank.head(10).iterrows()]
        elegido = st.selectbox("Combination to inspect", opciones, index=0, key="pcf_w_sel", help="Which of the best-ranked combinations to display below.")
        fila = rank.iloc[opciones.index(elegido)]
        cols = list(fila["cols"])
        etiq = [f"PC{j + 1} ({var_exp[j]:.1f}%)" for j in cols]
        S = scores[:, cols]

        # ---- scores plot of the chosen combination
        if tarea == "Classification":
            if tam == 1:
                fig = px.box(x=cl_f, y=S[:, 0], points="all", color=cl_f, color_discrete_sequence=CLASS_PALETTE,
                             hover_name=ids_f, labels={"x": "Class", "y": etiq[0], "color": "Class"})
            elif tam == 2:
                fig = px.scatter(x=S[:, 0], y=S[:, 1], color=cl_f, color_discrete_sequence=CLASS_PALETTE,
                                 hover_name=ids_f, labels={"x": etiq[0], "y": etiq[1], "color": "Class"})
                fig.add_hline(y=0, line_color="lightgray"); fig.add_vline(x=0, line_color="lightgray")
            else:
                fig = px.scatter_3d(x=S[:, 0], y=S[:, 1], z=S[:, 2], color=cl_f, color_discrete_sequence=CLASS_PALETTE,
                                    hover_name=ids_f, labels={"x": etiq[0], "y": etiq[1], "z": etiq[2], "color": "Class"})
        else:
            if tam == 1:
                fig = px.scatter(x=S[:, 0], y=y_f, color=y_f, color_continuous_scale="Viridis", hover_name=ids_f,
                                 labels={"x": etiq[0], "y": "Reference value", "color": "Reference value"})
            elif tam == 2:
                fig = px.scatter(x=S[:, 0], y=S[:, 1], color=y_f, color_continuous_scale="Viridis", hover_name=ids_f,
                                 labels={"x": etiq[0], "y": etiq[1], "color": "Reference value"})
                fig.add_hline(y=0, line_color="lightgray"); fig.add_vline(x=0, line_color="lightgray")
                _A = np.column_stack([np.ones(len(y_f)), S])
                _b = np.linalg.lstsq(_A, y_f, rcond=None)[0][1:]
                _u = _b / (np.linalg.norm(_b) or 1.0)
                _L = 0.45 * min(np.ptp(S[:, 0]), np.ptp(S[:, 1]))
                _c = S.mean(axis=0)
                fig.add_annotation(x=_c[0] + _u[0] * _L, y=_c[1] + _u[1] * _L, ax=_c[0] - _u[0] * _L, ay=_c[1] - _u[1] * _L,
                                   xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3,
                                   arrowwidth=3, arrowcolor="crimson")
                st.caption("The red arrow marks the direction in which the reference value increases.")
            else:
                fig = px.scatter_3d(x=S[:, 0], y=S[:, 1], z=S[:, 2], color=y_f, color_continuous_scale="Viridis",
                                    hover_name=ids_f, labels={"x": etiq[0], "y": etiq[1], "z": etiq[2], "color": "Reference value"})
        fig.update_layout(height=480, title="Samples in the chosen PCs")
        st.plotly_chart(fig, width="stretch")

        # ---- back to the variables
        st.markdown("### Which variables explain it?")
        imp, signed = pcf.direccion_variables(scores, cargas, cols, clases=cl_f, y=y_f)
        cc1, cc2 = st.columns(2)
        pct = cc1.slider("Highlight the top % of variables", 1, 20, 5, key="pcf_w_pct",
                         help="Contiguous variables in the top % of importance are grouped into regions.")
        modo_v = cc2.radio("Curve", ["Importance"] + (["Signed direction"] if signed is not None else []),
                           horizontal=True, key="pcf_w_curva",
                           help="Importance (0–1) shows WHERE the difference is. The signed direction also shows "
                                "which way: positive = higher in the second class / higher with concentration.")
        curva = imp if modo_v == "Importance" else signed
        regiones, umbral = pcf.regiones_importantes(eje_in, imp, percentil=100 - pct)
        picos = pcf.picos_importantes(eje_in, imp, signed)

        from plotly.subplots import make_subplots
        figv = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.4, 0.6], vertical_spacing=0.04)
        figv.add_trace(_traza_espectro(eje_in, X_f.mean(axis=0).round(6), mode="lines", name="Mean spectrum",
                                       line=dict(color="#5F5E5A", width=1)), row=1, col=1)
        figv.add_trace(_traza_espectro(eje_in, curva, mode="lines", name=modo_v,
                                       line=dict(color="#0F6E56", width=1.3)), row=2, col=1)
        for _, r in regiones.iterrows():
            figv.add_vrect(x0=min(r["From"], r["To"]), x1=max(r["From"], r["To"]), fillcolor="crimson",
                           opacity=0.18, line_width=0)
        figv.update_layout(height=520, showlegend=False,
                           title=f"Mean spectrum (top) and variable {modo_v.lower()} (bottom) for {etiq and ' + '.join(f'PC{j+1}' for j in cols)}")
        figv.update_yaxes(title_text="Signal", row=1, col=1)
        figv.update_yaxes(title_text=modo_v, row=2, col=1)
        figv.update_xaxes(title_text=("Variable" if es_tabular() else "Variable (wavenumber / ppm / time)"), row=2, col=1)
        if eje_in[0] > eje_in[-1]:
            figv.update_xaxes(autorange="reversed")
        _ticks_nombres(figv, eje_in)
        _t_p, _m_p = _asig_pre(figv, eje_in, [(min(r["From"], r["To"]), max(r["From"], r["To"])) for _, r in regiones.head(15).iterrows()], "pcf")
        st.plotly_chart(figv, width="stretch")
        _asig_post(_t_p, _m_p, "pcf")
        st.caption("Red bands = regions in the top "
                   f"{pct}% of importance. Check them against your reference table (chemical shifts, functional-group "
                   "bands, retention times) to name the compounds.")

        t1, t2 = st.columns(2)
        t1.markdown("**Regions**")
        t1.dataframe(_con_nombres(regiones).round(4), width="stretch", hide_index=True)
        t2.markdown("**Strongest single peaks**")
        t2.dataframe(_con_nombres(picos).round(4), width="stretch", hide_index=True)

        with st.expander("📈 Loadings of the chosen PCs"):
            figl = go.Figure()
            for j in cols:
                figl.add_trace(_traza_espectro(eje_in, cargas[j], mode="lines", name=f"PC{j + 1}", line=dict(width=1.2)))
            figl.update_layout(height=380, xaxis_title="Variable", yaxis_title="Loading")
            if eje_in[0] > eje_in[-1]:
                figl.update_xaxes(autorange="reversed")
            _ticks_nombres(figl, eje_in)
            st.plotly_chart(figl, width="stretch")

        _vars_df = pd.DataFrame({"variable": (etiquetas_var(eje_in) if es_tabular() else eje_in), "importance": imp})
        if signed is not None:
            _vars_df["signed_direction"] = signed
        st.download_button(
            "⬇️ Download results (Excel)",
            data=df_a_excel_bytes({"ranking": rank.drop(columns=["cols"]), "regions": regiones,
                                   "peaks": picos, "variable_importance": _vars_df}),
            file_name="pc_finder_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="pcf_dl")
    with tabs[5]:
        _frag_tab_14()


def _preparar_finder(prefijo):
    """Common input for the finder tabs: active samples, preprocessed spectra (full / cropped),
    task (classification / regression) and the matching targets. Returns None if not possible."""
    ids_a, _, clases_a = datos_activos()
    X_in = st.session_state.X_pret[indice_activo()]
    X_in, eje_in, _m, _ = elegir_espectro_modelado(
        X_in, np.array(st.session_state.numeros_onda_pret, dtype=float), prefijo)
    eje_in = np.asarray(eje_in, dtype=float)
    valores_a = None
    if st.session_state.get("valores_y") is not None:
        _mapa = dict(zip(map(str, st.session_state.ids), st.session_state.valores_y))
        valores_a = np.array([_mapa.get(str(i), np.nan) for i in ids_a], dtype=float)
    tareas = []
    if clases_a is not None:
        _cnt = pd.Series(clases_a).value_counts()
        if len(_cnt) >= 2 and _cnt.min() >= 3:
            tareas.append("Classification")
    if valores_a is not None and np.isfinite(valores_a).sum() >= 10:
        tareas.append("Regression")
    if X_in.shape[0] < 8 or not tareas:
        st.info("Load classes (≥ 2 classes with ≥ 3 samples each) or reference values (≥ 10 samples) in the "
                "left-hand panel to use this tab.")
        return None
    tarea = st.radio("What do you want to find?", tareas, horizontal=True, key=f"{prefijo}_w_tarea", help="Pick the kind of structure to search for; each option explains what it looks for once selected.")
    if tarea == "Regression":
        ok = np.isfinite(valores_a)
        return dict(tarea=tarea, es_clf=False, X=X_in[ok], y=valores_a[ok], ids=ids_a[ok], eje=eje_in)
    return dict(tarea=tarea, es_clf=True, X=X_in, y=np.asarray(clases_a), ids=ids_a, eje=eje_in)


@st.cache_data(show_spinner=False, max_entries=6, ttl=3600)
def _nl_cacheado(X, y, es_clf, n_vent):
    wins = nlf.ventanas(X.shape[1], n_vent)
    imp_w, imp_v, err = nlf.permutacion_ventanas(X, y, es_clf, wins, n_est=150, n_rep=2, n_jobs=mu.N_JOBS)
    mi = nlf.mi_ventanas(X, y, es_clf, wins)
    return imp_w, imp_v, err, mi


@st.cache_data(show_spinner=False, max_entries=6, ttl=3600)
def _modelos_lineal_vs_nl(X, y, es_clf):
    from sklearn.base import clone
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
    from sklearn.metrics import balanced_accuracy_score, r2_score
    cat = mu.crear_clasificadores() if es_clf else mu.crear_regresores()
    nombres = [("PLS-DA" if es_clf else "PLS", "linear"), ("Random Forest", "non-linear"),
               ("SVM" if es_clf else "SVM (SVR)", "non-linear")]
    if es_clf:
        cv = StratifiedKFold(n_splits=int(min(5, pd.Series(y).value_counts().min())), shuffle=True, random_state=0)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
    filas = []
    for nom, tipo in nombres:
        try:
            pred = cross_val_predict(clone(cat[nom]), X, y, cv=cv, n_jobs=mu.N_JOBS)
            m = balanced_accuracy_score(y, pred) if es_clf else r2_score(y, pred)
        except Exception:
            m = np.nan
        filas.append({"Model": nom, "Type": tipo, "CV balanced accuracy" if es_clf else "CV Q²": round(float(m), 3)})
    return pd.DataFrame(filas)


def _cv_con_variables(X, y, es_clf, mascara):
    from sklearn.base import clone
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
    from sklearn.metrics import balanced_accuracy_score, r2_score
    cat = mu.crear_clasificadores() if es_clf else mu.crear_regresores()
    est = clone(cat["PLS-DA" if es_clf else "PLS"])
    if es_clf:
        cv = StratifiedKFold(n_splits=int(min(5, pd.Series(y).value_counts().min())), shuffle=True, random_state=0)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
    Xs = X[:, mascara]
    if hasattr(est, "n_components"):
        est.set_params(n_components=int(max(1, min(5, Xs.shape[1], Xs.shape[0] - 2))))
    pred = cross_val_predict(est, Xs, y, cv=cv)
    return float(balanced_accuracy_score(y, pred) if es_clf else r2_score(y, pred))


if _abierta(tabs[6]):
    @st.fragment
    def _frag_tab_15():
        st.subheader("Non-linear finder — which regions matter when the relationship is not a straight line")
        st.caption(
            "The PC finder looks for straight-line (linear) structure in PCs. This tab works directly on the "
            "variables with a Random Forest, so it can catch curved relationships, thresholds and interactions "
            "between regions. Importance is measured per **window** of neighbouring variables (they are strongly "
            "correlated, so single variables are unreliable): a window matters if scrambling it, on samples the "
            "model has NOT seen, makes the predictions clearly worse.")
        d = _preparar_finder("nlf")
        if d is None:
            return
        c1, c2 = st.columns(2)
        if es_tabular():
            n_vent = int(d["X"].shape[1])
            c1.caption(f"Tabular variables: each variable is its own 'window' ({n_vent})."
                       + (" ⚠️ With many variables this can be slow." if n_vent > 300 else ""))
        else:
            n_vent = c1.slider("Number of windows", 20, 150, 60, step=10, key="nlf_w_vent",
                               help="The axis is cut into this many equal windows. More windows = finer resolution but "
                                    "noisier and slower.")
        top_n = c2.slider("Windows to highlight", 1, 15, 5, key="nlf_w_top", help="How many top-ranked windows of the spectrum are highlighted.")
        sig = (d["tarea"], d["X"].shape, float(np.nansum(d["X"][:, ::97])), n_vent)
        if st.button("▶ Run non-linear search", type="primary", key="nlf_btn"):
            with st.spinner("Training a Random Forest and permuting windows on held-out samples..."):
                _nl_cacheado(d["X"], d["y"], d["es_clf"], n_vent)
                _modelos_lineal_vs_nl(d["X"], d["y"], d["es_clf"])
            st.session_state["nlf_sig"] = sig
        if st.session_state.get("nlf_sig") != sig:
            st.info("Click **Run non-linear search** (takes a few seconds to about half a minute).")
            return
        imp_w, imp_v, err, mi = _nl_cacheado(d["X"], d["y"], d["es_clf"], n_vent)
        wins = nlf.ventanas(d["X"].shape[1], n_vent)
        eje = d["eje"]

        st.markdown("**Does non-linearity help for this dataset?**")
        comp = _modelos_lineal_vs_nl(d["X"], d["y"], d["es_clf"])
        st.dataframe(comp, width="stretch", hide_index=True)
        st.caption("Cross-validated performance on all variables. If the non-linear models are clearly better than "
                   "the linear one, curved relationships or interactions are likely present; if not, the linear "
                   "PC finder already tells most of the story.")

        orden = np.argsort(imp_w)[::-1][:top_n]
        figv = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.3, 0.35, 0.35], vertical_spacing=0.04)
        figv.add_trace(_traza_espectro(eje, d["X"].mean(axis=0).round(6), mode="lines", line=dict(color="#5F5E5A", width=1)), row=1, col=1)
        figv.add_trace(_traza_espectro(eje, imp_v, mode="lines", line=dict(color="#7B3FA0", width=1.1)), row=2, col=1)
        centros = [float(eje[w].mean()) for w in wins]
        anchos = [abs(float(eje[w[-1]] - eje[w[0]])) or 1.0 for w in wins]
        figv.add_trace(go.Bar(x=centros, y=imp_w, width=anchos, marker_color="#0F6E56", name="Permutation importance"), row=3, col=1)
        figv.add_trace(go.Scatter(x=centros, y=mi, mode="lines+markers", line=dict(color="#C07A00", width=1),
                                  marker=dict(size=4), name="Mutual information"), row=3, col=1)
        for k in orden:
            w = wins[k]
            figv.add_vrect(x0=float(min(eje[w[0]], eje[w[-1]])), x1=float(max(eje[w[0]], eje[w[-1]])),
                           fillcolor="crimson", opacity=0.18, line_width=0)
        figv.update_layout(height=640, showlegend=True, legend=dict(orientation="h", y=-0.12),
                           title="Mean spectrum · Random Forest importance per variable · importance per window")
        figv.update_yaxes(title_text="Signal", row=1, col=1)
        figv.update_yaxes(title_text="RF impurity", row=2, col=1)
        figv.update_yaxes(title_text="Per window (0–1)", row=3, col=1)
        figv.update_xaxes(title_text=("Variable" if es_tabular() else "Variable (wavenumber / ppm / time)"), row=3, col=1)
        if eje[0] > eje[-1]:
            figv.update_xaxes(autorange="reversed")
        _ticks_nombres(figv, eje)
        _t_n, _m_n = _asig_pre(figv, eje, [(float(min(eje[wins[k][0]], eje[wins[k][-1]])), float(max(eje[wins[k][0]], eje[wins[k][-1]]))) for k in orden], "nlf")
        st.plotly_chart(figv, width="stretch")
        _asig_post(_t_n, _m_n, "nlf")
        st.caption("Green bars: how much the held-out predictions get worse when that window is scrambled "
                   "(non-linear, multivariate). Orange: mutual information of the window's mean signal with the "
                   "class / value (non-linear, one window at a time). Purple: classic Random Forest importance of "
                   "each single variable (fine detail, but biased towards correlated variables).")

        tabla = pd.DataFrame({"From": [float(eje[wins[k][0]]) for k in orden], "To": [float(eje[wins[k][-1]]) for k in orden],
                              "Permutation importance": imp_w[orden], "Mutual information": mi[orden]}).round(3)
        st.markdown("**Top windows**")
        st.dataframe(_con_nombres(tabla), width="stretch", hide_index=True)
        todo = _con_nombres(pd.DataFrame({"From": [float(eje[w[0]]) for w in wins], "To": [float(eje[w[-1]]) for w in wins],
                             "permutation_importance": imp_w, "mutual_information": mi}))
        st.download_button("⬇️ Download results (Excel)",
                           data=df_a_excel_bytes({"windows": todo, "model_comparison": comp,
                                                  "variable_importance": pd.DataFrame({"variable": (etiquetas_var(eje) if es_tabular() else eje), "rf_importance": imp_v})}),
                           file_name="nonlinear_finder_results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="nlf_dl")
    with tabs[6]:
        _frag_tab_15()


if _abierta(tabs[7]):
    @st.fragment
    def _frag_tab_16():
        st.subheader("Consensus — do two independent approaches agree?")
        st.caption(
            "Runs the **linear** approach (best PC combination from the PC finder, traced back to the variables) and "
            "the **non-linear** one (Random Forest, grouped permutation) completely independently, then compares "
            "them window by window. Regions found by BOTH are the most trustworthy candidates. Regions found by only "
            "one are informative too: linear-only often means a smooth/collinear effect, non-linear-only suggests "
            "curved relationships or interactions.")
        d = _preparar_finder("cons")
        if d is None:
            return
        c1, c2, c3 = st.columns(3)
        if es_tabular():
            n_vent = int(d["X"].shape[1])
            c1.caption(f"Tabular variables: each variable is its own 'window' ({n_vent})."
                       + (" ⚠️ With many variables this can be slow." if n_vent > 300 else ""))
        else:
            n_vent = c1.slider("Number of windows", 20, 150, 60, step=10, key="cons_w_vent", help="The spectrum is divided into this many windows; each is evaluated separately.")
        tam = c2.radio("PCs per combination (linear side)", [1, 2, 3], index=1, horizontal=True, key="cons_w_tam", help="How many principal components are used by the linear side of each combination. More = more detail but higher risk of noise.")
        frac = c3.slider("'Top' = best X % of windows", 5, 40, 20, key="cons_w_frac", help="Windows within the best X % of the ranking are highlighted.") / 100
        sig = (d["tarea"], d["X"].shape, float(np.nansum(d["X"][:, ::97])), n_vent, tam, frac)
        if st.button("▶ Run both approaches", type="primary", key="cons_btn"):
            with st.spinner("Running the linear and the non-linear search..."):
                _nl_cacheado(d["X"], d["y"], d["es_clf"], n_vent)
            st.session_state["cons_sig"] = sig
        if st.session_state.get("cons_sig") != sig:
            st.info("Click **Run both approaches**.")
            return
        X, y, eje = d["X"], d["y"], d["eje"]
        wins = nlf.ventanas(X.shape[1], n_vent)
        # linear side
        k_tope = int(max(3, min(12, X.shape[0] - 1)))
        scores, cargas, var_exp = _pca_pcf_cacheado(X, k_tope)
        if d["es_clf"]:
            rank = pcf.buscar_clasificacion(scores, y, k_tope, tam)
            imp_lin, _ = pcf.direccion_variables(scores, cargas, rank.loc[0, "cols"], clases=y)
        else:
            rank = pcf.buscar_regresion(scores, y, k_tope, tam)
            imp_lin, _ = pcf.direccion_variables(scores, cargas, rank.loc[0, "cols"], y=y)
        lin_w = nlf.a_ventanas(imp_lin, wins)
        # non-linear side
        imp_nl_w, imp_nl_v, _, _mi = _nl_cacheado(X, y, d["es_clf"], n_vent)
        tabla, res = nlf.comparar_ventanas(eje, wins, lin_w, imp_nl_w, top_frac=frac)

        m1, m2, m3 = st.columns(3)
        m1.metric("Best PCs (linear side)", rank.loc[0, "PCs"])
        m2.metric("Rank agreement (Spearman)", f"{res['spearman']:.2f}" if res["spearman"] == res["spearman"] else "n/a",
                  help="Correlation of the window rankings of the two methods (1 = identical).")
        m3.metric(f"Overlap of top {res['k']} windows", f"{res['overlap'] * 100:.0f}%",
                  help="Share of windows that are in the top list of BOTH methods.")

        centros = [float(eje[w].mean()) for w in wins]
        anchos = [abs(float(eje[w[-1]] - eje[w[0]])) or 1.0 for w in wins]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.3, 0.7], vertical_spacing=0.04)
        fig.add_trace(_traza_espectro(eje, X.mean(axis=0).round(6), mode="lines", line=dict(color="#5F5E5A", width=1)), row=1, col=1)
        fig.add_trace(go.Bar(x=centros, y=lin_w / (lin_w.max() or 1), width=anchos, name="Linear (PC direction)",
                             marker_color="rgba(31,119,180,0.65)"), row=2, col=1)
        fig.add_trace(go.Bar(x=centros, y=imp_nl_w, width=anchos, name="Non-linear (Random Forest)",
                             marker_color="rgba(214,95,0,0.65)"), row=2, col=1)
        colores = {"Both": "seagreen", "Linear only": "royalblue", "Non-linear only": "darkorange"}
        for _, r in tabla[tabla["Found by"] != ""].iterrows():
            fig.add_vrect(x0=min(r["From"], r["To"]), x1=max(r["From"], r["To"]), fillcolor=colores[r["Found by"]],
                          opacity=0.16, line_width=0)
        fig.update_layout(height=560, barmode="overlay", legend=dict(orientation="h", y=-0.12),
                          title="Importance per window — green band = found by both approaches")
        fig.update_yaxes(title_text="Signal", row=1, col=1); fig.update_yaxes(title_text="Importance (0–1)", row=2, col=1)
        fig.update_xaxes(title_text=("Variable" if es_tabular() else "Variable (wavenumber / ppm / time)"), row=2, col=1)
        if eje[0] > eje[-1]:
            fig.update_xaxes(autorange="reversed")
        _ticks_nombres(fig, eje)
        _tc = tabla[tabla["Found by"] != ""].copy()
        _tc["_o"] = (_tc["Found by"] != "Both").astype(int)
        _tc = _tc.sort_values(["_o", "Consensus (rank mean)"], ascending=[True, False]).head(12)
        _t_c, _m_c = _asig_pre(fig, eje, [(min(r["From"], r["To"]), max(r["From"], r["To"])) for _, r in _tc.iterrows()], "cons")
        st.plotly_chart(fig, width="stretch")
        _asig_post(_t_c, _m_c, "cons")

        vistas = tabla[tabla["Found by"] != ""].sort_values("Consensus (rank mean)", ascending=False)
        st.markdown("**Windows in the top list of at least one method**")
        st.dataframe(_con_nombres(vistas.drop(columns=["_i"])).round(3), width="stretch", hide_index=True)
        n_both = int((tabla["Found by"] == "Both").sum())
        if n_both:
            st.success(f"{n_both} window(s) found by BOTH independent approaches — the most reliable regions.")
        else:
            st.warning("No window is in the top list of both methods. Try more windows / a larger 'top' share, a "
                       "different PC combination size, or check that the classes / values really are structured.")

        # independent confirmation: does a model using only those regions keep its performance?
        st.markdown("**Confirmation: how well do the regions predict on their own?** (cross-validated PLS)")
        etq = "balanced accuracy" if d["es_clf"] else "Q²"
        filas = []
        todos = np.ones(X.shape[1], dtype=bool)
        filas.append({"Variables used": "All variables", "n variables": int(todos.sum()), etq: round(_cv_con_variables(X, y, d["es_clf"], todos), 3)})
        for etiqueta, cat in [("Found by both", "Both"), ("Linear side top windows", None), ("Non-linear side top windows", "nl")]:
            if cat == "Both":
                idx = [w for i, w in enumerate(wins) if tabla.loc[i, "Found by"] == "Both"]
            elif cat is None:
                idx = [wins[i] for i in np.argsort(lin_w)[::-1][:res["k"]]]
            else:
                idx = [wins[i] for i in np.argsort(imp_nl_w)[::-1][:res["k"]]]
            if not idx:
                continue
            m = np.zeros(X.shape[1], dtype=bool); m[np.concatenate(idx)] = True
            filas.append({"Variables used": etiqueta, "n variables": int(m.sum()), etq: round(_cv_con_variables(X, y, d["es_clf"], m), 3)})
        rng = np.random.RandomState(0)
        aleat = []
        for _ in range(5):
            sel = rng.choice(len(wins), size=res["k"], replace=False)
            m = np.zeros(X.shape[1], dtype=bool); m[np.concatenate([wins[i] for i in sel])] = True
            aleat.append(_cv_con_variables(X, y, d["es_clf"], m))
        filas.append({"Variables used": f"Random {res['k']} windows (mean of 5)", "n variables": "–", etq: round(float(np.mean(aleat)), 3)})
        st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)
        st.caption("If the regions found by the two approaches predict about as well as all the variables and "
                   "clearly better than random windows, they carry the relevant information.")
        st.download_button("⬇️ Download consensus table (Excel)",
                           data=df_a_excel_bytes({"windows": _con_nombres(tabla.drop(columns=["_i"])), "confirmation": pd.DataFrame(filas)}),
                           file_name="consensus_results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="cons_dl")
    with tabs[7]:
        _frag_tab_16()


# -----------------------------------------------------------------------
# TAB: OUTLIERS
# -----------------------------------------------------------------------

if _abierta(tabs[32]):
    with tabs[32]:
        ref_tab.render()


if _abierta(tabs[8]):
    # Each tab re-runs ON ITS OWN when one of its widgets changes (fragment), instead of
    # re-executing the whole app. Buttons that must refresh other parts call st.rerun().
    @st.fragment
    def _frag_tab_5():
        st.subheader("Outlier detection: Hotelling's T² and Q residual")
        gu.barra_guia("outliers", "outliers")

        _msg_toast = st.session_state.pop("_toast_exclusion", None)
        if _msg_toast:
            st.toast(_msg_toast, icon="🚫")
        _mask_excl = st.session_state.get("mascara_excluidas")
        if _mask_excl is not None and _mask_excl.any():
            _ids_excl = st.session_state.ids[_mask_excl]
            _n_ex = int(_mask_excl.sum())
            _pf, _of = st.session_state.get("pca_firma"), st.session_state.get("outliers_firma")
            _pca_al_dia = _pf is not None and len(_pf) > 3 and _pf[3] == _n_ex
            _out_al_dia = _of is not None and len(_of) > 3 and _of[3] == _n_ex
            _lista = ", ".join(map(str, _ids_excl[:15])) + (" …" if len(_ids_excl) > 15 else "")
            if st.session_state.get("pca_completo") is None or (_pca_al_dia and _out_al_dia):
                st.info(f"🚫 **{_n_ex} sample(s) excluded** from all analyses: {_lista}. "
                        "The results shown below were computed without them.")
            elif not _pca_al_dia:
                st.warning(f"🚫 **{_n_ex} sample(s) excluded** from all analyses: {_lista}. "
                           "The PCA and outlier plots below were computed **before** this change — click "
                           "**Update PCA** (PCA tab), then **Update outliers** here.", icon="⚠️")
            else:
                st.warning(f"🚫 **{_n_ex} sample(s) excluded** from all analyses: {_lista}. "
                           "PCA is already updated; click **Update outliers** below to recompute without them.",
                           icon="⚠️")
        else:
            st.caption("No samples are excluded right now: all samples are used in every analysis.")

        if st.session_state.get("pca_completo") is None:
            st.info("Compute PCA first, in the **PCA** tab (needs at least 2 components).")
        else:
            # Use the SAME frozen snapshot the PCA result itself was computed from —
            # not live data — so they always stay consistent in length with each other,
            # regardless of outlier exclusions made afterwards without clicking Update.
            ids = st.session_state.pca_ids
            clases = st.session_state.pca_clases
            X_pca_input = st.session_state.pca_X_input
            scores_completo = st.session_state.scores_completo
            cargas_completo = st.session_state.cargas_completo
            autovalores = st.session_state.autovalores
            n_comp = min(st.session_state.get("n_comp", 3), len(st.session_state.var_explicada))
            n_muestras = X_pca_input.shape[0]

            st.caption("Outlier detection is built on the PCA result, so it uses the same spectrum as the PCA: "
                       + (f"✂️ cropped ({st.session_state.get('pca_crop_desc')})." if st.session_state.get("pca_origen") == "cropped"
                          else "the full spectrum. (Change it in the PCA tab and click Update there.)"))
            col1, col2, col3 = st.columns(3)
            alpha = col1.slider("Significance level (alpha)", 0.01, 0.10, 0.05, step=0.01, help="Probability of a false alarm. Smaller = fewer samples flagged (only the most extreme).", key="ac_out_alpha")
            criterio = col2.selectbox("Criterion to flag outliers",
                                       ["T² and Q (both, conservative)", "T² or Q (either, aggressive)",
                                        "T² only", "Q only"], help="Which statistic decides that a sample is an outlier (T², Q or both).", key="ac_out_crit")

            _firma_outliers_actual = firma_segun_origen(st.session_state.get("pca_origen", "full")) + (n_comp, alpha)
            _resultado_outliers_previo = st.session_state.get("outliers_resultado")
            esta_desactualizado_out = (
                _resultado_outliers_previo is None
                or st.session_state.get("outliers_firma") != _firma_outliers_actual
            )
            if _resultado_outliers_previo is not None:
                if esta_desactualizado_out:
                    st.warning("⚠️ **This outlier result is out of date** — PCA, preprocessing, excluded "
                               "samples, alpha, or n_comp changed since it was last computed. The plot below still shows the "
                               "last computed result. Click **Update** to recompute it.", icon="⚠️")
                else:
                    st.caption("✅ Up to date with the current PCA result, preprocessing, and settings.")
            _texto_boton_out = "▶ Compute outliers" if _resultado_outliers_previo is None else "🔄 Update outliers"
            _col_b_out, _col_c_out, _ = st.columns([1.4, 1, 4])
            if _col_c_out.button("🧹 Clear", key="btn_limpiar_outliers",
                                 disabled=_resultado_outliers_previo is None,
                                 help="Removes the outlier result from the screen. Samples you already "
                                      "excluded stay excluded (use 'Restore all samples' to undo that)."):
                limpiar_outliers()
                _rerun_tab()
            if _col_b_out.button(_texto_boton_out, type="primary" if esta_desactualizado_out else "secondary",
                                 key="btn_computar_outliers"):
                with st.spinner("Computing T² / Q..."):
                    st.session_state.outliers_resultado = _calcular_outliers_cacheado(
                        X_pca_input, scores_completo, cargas_completo, autovalores,
                        st.session_state.pca_completo.mean_, n_comp, alpha,
                    )
                    # Freeze the ids/classes THIS result goes with. PCA can be updated
                    # independently afterwards (different sample count) — if that
                    # happens before Outliers is also updated, using PCA's "current"
                    # ids here instead of this frozen snapshot would mismatch T2/Q below.
                    st.session_state.outliers_ids = ids
                    st.session_state.outliers_clases = clases
                    st.session_state.outliers_firma = _firma_outliers_actual
                _rerun_tab()

            if st.session_state.get("outliers_resultado") is None:
                st.info("Click **Compute outliers** above to get started.")
            else:
                ids = st.session_state.outliers_ids
                clases = st.session_state.outliers_clases
                T2, T2_lim, Q, Q_lim, Q_lim_confiable = st.session_state.outliers_resultado
                if not Q_lim_confiable:
                    st.warning(
                        "⚠ The theoretical Q limit (Jackson-Mudholkar) is not reliable with this "
                        "'n_comp' (happens when almost all the variance is already explained). "
                        "An EMPIRICAL limit (95th percentile of your own Q values) is used instead."
                    )

                col3.metric("T² limit", f"{T2_lim:.2f}")
                col3.metric("Q limit", f"{Q_lim:.4g}" + (" (empirical)" if not Q_lim_confiable else ""))

                T2_lo, T2_hi = cu.rango_con_margen(T2, T2_lim)
                Q_lo, Q_hi = cu.rango_con_margen(Q, Q_lim)

                # ---- colouring: by class, or by a continuous value (colour gradient)
                _col_vals_o, _col_lab_o = None, None
                _vals_o = None
                if st.session_state.get("valores_y") is not None:
                    _mapa_o = dict(zip(map(str, st.session_state.ids), st.session_state.valores_y))
                    _vals_o = np.array([_mapa_o.get(str(i), np.nan) for i in ids], dtype=float)
                _opc_o = ["Class" if clases is not None else "None"]
                if _vals_o is not None:
                    _opc_o.append("Reference value (gradient)")
                _eje_o = st.session_state.get("pca_eje")
                _X_o = st.session_state.get("pca_X_input")
                _spec_ok_o = _eje_o is not None and _X_o is not None and len(_X_o) == len(ids)
                if _spec_ok_o:
                    _opc_o.append("Variable (gradient)" if es_tabular() else "Spectral variable (gradient)")
                _modo_o = st.radio("Color samples by", _opc_o, horizontal=True, key="out_w_color",
                                   help="Paint each sample by its class, by its analyte concentration / reference "
                                        "value (colour gradient), or by the signal at one wavelength / variable — "
                                        "useful to see at a glance whether the outliers share a concentration or class.")
                if _modo_o.startswith("Reference"):
                    _col_vals_o, _col_lab_o = _vals_o, "Reference value"
                elif _modo_o.startswith(("Spectral", "Variable")):
                    _eje_oo = np.asarray(_eje_o, dtype=float)
                    if es_tabular():
                        _nm_o = list(etiquetas_var(_eje_oo))
                        _jo = st.selectbox("Variable", range(len(_nm_o)), format_func=lambda i: _nm_o[i], key="out_w_color_varT", help="Variable to inspect.")
                        _col_vals_o, _col_lab_o = np.asarray(_X_o)[:, _jo], _nm_o[_jo]
                    else:
                        _v_o = st.select_slider("Variable (wavelength / wavenumber)",
                                                options=[float(v) for v in _eje_oo],
                                                value=float(_eje_oo[len(_eje_oo) // 2]), key="out_w_color_var",
                                                format_func=lambda v: f"{v:g}", help="Position of the variable on the spectral axis.")
                        _jo = int(np.argmin(np.abs(_eje_oo - _v_o)))
                        _col_vals_o, _col_lab_o = np.asarray(_X_o)[:, _jo], f"Signal @ {_eje_oo[_jo]:g}"

                if _col_vals_o is not None:
                    fig = px.scatter(x=T2, y=Q, hover_name=ids, color=_col_vals_o, color_continuous_scale="Viridis",
                                     labels={"x": "Hotelling's T²", "y": "Q residual", "color": _col_lab_o})
                    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="rgba(60,60,60,.5)")))
                    fig.update_layout(coloraxis_colorbar=dict(title=_col_lab_o))
                else:
                    fig = px.scatter(x=T2, y=Q, hover_name=ids,
                                     color=clases if clases is not None else None,
                                     color_discrete_sequence=CLASS_PALETTE,
                                     labels={"x": "Hotelling's T²", "y": "Q residual"})
                _ids_ya_excl = set(map(str, st.session_state.ids[st.session_state.mascara_excluidas]))
                _m_ya = np.array([str(i) in _ids_ya_excl for i in ids])
                if _m_ya.any():
                    fig.add_trace(go.Scatter(
                        x=T2[_m_ya], y=Q[_m_ya], mode="markers", name="already excluded",
                        hovertext=[str(i) for i in ids[_m_ya]],
                        marker=dict(symbol="x", size=12, color="black", line=dict(width=2))))
                fig.add_shape(type="rect", x0=T2_lo, x1=T2_lim, y0=Q_lo, y1=Q_lim,
                              fillcolor="green", opacity=0.08, line_width=0, layer="below")
                fig.add_shape(type="rect", x0=T2_lim, x1=T2_hi, y0=Q_lo, y1=Q_lim,
                              fillcolor="orange", opacity=0.08, line_width=0, layer="below")
                fig.add_shape(type="rect", x0=T2_lo, x1=T2_lim, y0=Q_lim, y1=Q_hi,
                              fillcolor="orange", opacity=0.08, line_width=0, layer="below")
                fig.add_shape(type="rect", x0=T2_lim, x1=T2_hi, y0=Q_lim, y1=Q_hi,
                              fillcolor="red", opacity=0.12, line_width=0, layer="below")
                fig.add_vline(x=T2_lim, line_dash="dash", line_color="red")
                fig.add_hline(y=Q_lim, line_dash="dash", line_color="red")
                fig.add_annotation(x=(T2_lo + T2_lim) / 2, y=(Q_lo + Q_lim) / 2, text="OK",
                                    showarrow=False, font=dict(color="green"))
                fig.add_annotation(x=(T2_lim + T2_hi) / 2, y=(Q_lo + Q_lim) / 2, text="Outlier T²",
                                    showarrow=False, font=dict(color="darkorange"))
                fig.add_annotation(x=(T2_lo + T2_lim) / 2, y=(Q_lim + Q_hi) / 2, text="Outlier Q",
                                    showarrow=False, font=dict(color="darkorange"))
                fig.add_annotation(x=(T2_lim + T2_hi) / 2, y=(Q_lim + Q_hi) / 2, text="Outlier T² & Q",
                                    showarrow=False, font=dict(color="darkred"))
                fig.update_xaxes(range=[T2_lo, T2_hi])
                fig.update_yaxes(range=[Q_lo, Q_hi])
                fig.update_layout(height=520, title="Influence plot")
                st.plotly_chart(fig, width='stretch')
                gu.nota("outliers", "influence plot (T² vs Q)", "outliers")
                st.caption("Axes zoom automatically to your data (T² and Q are rarely exactly 0).")

                if criterio.startswith("T² and Q"):
                    es_outlier = (T2 > T2_lim) & (Q > Q_lim)
                elif criterio.startswith("T² or Q"):
                    es_outlier = (T2 > T2_lim) | (Q > Q_lim)
                elif criterio == "T² only":
                    es_outlier = T2 > T2_lim
                else:
                    es_outlier = Q > Q_lim

                _ids_ya_excl = set(map(str, st.session_state.ids[st.session_state.mascara_excluidas]))
                _ya = np.array([str(i) in _ids_ya_excl for i in ids])
                candidatos = ids[es_outlier & ~_ya]          # new ones, not yet excluded
                candidatos_ya = ids[es_outlier & _ya]        # flagged but already excluded
                st.markdown(f"**New outlier candidates with this criterion ({len(candidatos)}):** "
                            + (", ".join(map(str, candidatos)) if len(candidatos) else "none"))
                if len(candidatos_ya):
                    st.caption(f"Already excluded and still flagged in this (older) result: "
                               f"{', '.join(map(str, candidatos_ya))}. They are drawn as ✖ in the plot.")

                if clases is not None and len(candidatos) > 0:
                    conteo_candidatos_por_clase = pd.Series(clases[es_outlier]).value_counts()
                    conteo_total_por_clase = pd.Series(clases).value_counts()
                    proporciones = (conteo_candidatos_por_clase / conteo_total_por_clase).dropna()
                    clases_con_muchos = proporciones[proporciones > 0.3].index.tolist()
                    if clases_con_muchos:
                        st.info(f"ℹ️ A large share of class(es) **{', '.join(clases_con_muchos)}** show up here. "
                                "This outlier detection compares every sample to a SINGLE overall model — if a "
                                "whole class is chemically quite different from the rest, much of it can look "
                                "like an 'outlier' even though it's really just a distinct, valid population. "
                                "If that looks like what's happening, consider the **SIMCA** tab instead: it "
                                "builds a separate model per class, so a class won't be penalized just for "
                                "being different from the others.")

                st.caption("Nothing is removed until you click **Exclude**. Excluded samples are kept in your "
                           "file and can be brought back with **Restore**.")
                col_a, col_b = st.columns(2)
                if col_a.button(f"🚫 Exclude {len(candidatos)} new candidate(s) from the analysis",
                                disabled=(len(candidatos) == 0)):
                    idx_global = np.array([np.where(st.session_state.ids == i)[0][0] for i in candidatos])
                    st.session_state.mascara_excluidas[idx_global] = True
                    st.session_state["_toast_exclusion"] = (
                        f"{len(candidatos)} sample(s) excluded. Update PCA and outliers to recompute.")
                    st.rerun()
                if col_b.button("♻️ Restore all samples (undo exclusions)",
                                disabled=not st.session_state.mascara_excluidas.any()):
                    st.session_state.mascara_excluidas[:] = False
                    st.session_state["_toast_exclusion"] = "All samples restored. Update PCA and outliers to recompute."
                    st.rerun()

                st.divider()
                st.markdown("**📄 Exploratory analysis report**")
                st.caption("Includes: spectra, PCA (explained variance and scores), and the T²/Q "
                           "outlier plot with the critical limits and the candidate samples.")
                if st.button("🖨️ Generate report (PDF)", key="generar_reporte_exp"):
                    var_explicada = st.session_state.var_explicada
                    secciones_rep_exp = [
                        {"tipo": "titulo", "texto": "1. Dataset"},
                        {"tipo": "clave_valor", "pares": [
                            ("Samples used", int(X_pca_input.shape[0])),
                            ("Variables", int(X_pca_input.shape[1])),
                            ("Preprocessing", " -> ".join(p[0] for p in st.session_state.pasos_pretratamiento) or "none"),
                        ]},
                        {"tipo": "imagen", "fig": ru.fig_espectros(
                            st.session_state.numeros_onda_pret, X_pca_input,
                            clases=como_texto(clases) if clases is not None else None, titulo="Preprocessed spectra")},
                        {"tipo": "titulo", "texto": "2. PCA"},
                        {"tipo": "clave_valor", "pares": [
                            ("Components retained", n_comp),
                            ("Cumulative explained variance", f"{np.cumsum(var_explicada)[n_comp-1]:.1f}%"),
                        ]},
                        {"tipo": "imagen", "fig": ru.fig_scree(var_explicada, n_comp_marcado=n_comp)},
                        {"tipo": "imagen", "fig": ru.fig_scores(
                            scores_completo, var_explicada, 1, 2,
                            clases=como_texto(clases) if clases is not None else None, ids=ids)},
                        {"tipo": "salto_pagina"},
                        {"tipo": "titulo", "texto": "3. Outlier detection (Hotelling's T² and Q residual)"},
                        {"tipo": "clave_valor", "pares": [
                            ("Significance level (alpha)", alpha),
                            ("T² limit", f"{T2_lim:.2f}"),
                            ("Q limit", f"{Q_lim:.4g}" + (" (empirical)" if not Q_lim_confiable else " (theoretical)")),
                            ("Criterion used", criterio),
                            ("Outlier candidate samples", ", ".join(candidatos) if len(candidatos) else "none"),
                            ("Currently excluded samples", ", ".join(st.session_state.ids[st.session_state.mascara_excluidas])
                             if st.session_state.mascara_excluidas.any() else "none"),
                        ]},
                        {"tipo": "imagen", "fig": ru.fig_outliers(
                            T2, Q, T2_lim, Q_lim, T2_lo, T2_hi, Q_lo, Q_hi,
                            clases=como_texto(clases) if clases is not None else None)},
                        {"tipo": "salto_pagina"},
                        {"tipo": "titulo", "texto": "4. Hierarchical Cluster Analysis (HCA)"},
                        {"tipo": "clave_valor", "pares": [("Linkage method", "ward")]},
                    ]
                    if X_pca_input.shape[0] >= 3:
                        Z_reporte = hierarchy.linkage(X_pca_input, method="ward")
                        secciones_rep_exp.append({"tipo": "imagen", "fig": ru.fig_dendrograma(
                            Z_reporte, list(ids), clases=como_texto(clases) if clases is not None else None)})
                    pdf_reporte_exp = _generar_reporte(
                        "Exploratory Analysis Report",
                        f"{X_pca_input.shape[0]} samples - {n_comp} principal components", secciones_rep_exp,
                    )
                    st.download_button(
                        "⬇️ Download full report (PDF)",
                        data=bytes(pdf_reporte_exp.output()),
                        file_name="exploratory_analysis_report.pdf", mime="application/pdf",
                        key="descargar_reporte_exp",
                    )
    with tabs[8]:
        _frag_tab_5()


# -----------------------------------------------------------------------
# TAB: DENDROGRAM
# -----------------------------------------------------------------------
if _abierta(tabs[9]):
    # Each tab re-runs ON ITS OWN when one of its widgets changes (fragment), instead of
    # re-executing the whole app. Buttons that must refresh other parts call st.rerun().
    @st.fragment
    def _frag_tab_6():
        st.subheader("Hierarchical Cluster Analysis (HCA)")
        gu.barra_guia("dendro", "dendro", pros=("linkage", st.session_state.get("dendro_w_metodo")))

        X_hca_vivo = st.session_state.X_pret[indice_activo()]
        X_hca_vivo, _eje_den, _crop_mask_den, _ = elegir_espectro_modelado(
            X_hca_vivo, np.array(st.session_state.numeros_onda_pret, dtype=float), "dendro")

        if X_hca_vivo.shape[0] < 3:
            st.warning("At least 3 samples are needed.")
        else:
            col1, col2 = st.columns(2)
            metodo = col1.selectbox(
                "Linkage method", ["ward", "average", "complete", "single"], key="dendro_w_metodo",
                help="How the distance between two CLUSTERS (not individual samples) is defined when "
                     "deciding what to merge next:\n"
                     "• ward: merges the pair that increases within-cluster variance the least — usually "
                     "gives the most balanced, compact clusters (good default).\n"
                     "• average: distance = average distance between all pairs across the two clusters.\n"
                     "• complete: distance = the FARTHEST pair between the two clusters — tends to give "
                     "tight, evenly-sized clusters, sensitive to outliers.\n"
                     "• single: distance = the CLOSEST pair between the two clusters — can chain together "
                     "long, straggly clusters (sensitive to noise, but can find elongated shapes).",
            )
            vista = col2.radio("Dendrogram type", ["Linear (interactive)", "Circular"], horizontal=True,
                               key="dendro_w_vista",
                               help="Only changes how the SAME result is drawn — it doesn't recompute anything.")

            # --- Compute / Update / Clear -------------------------------------------------
            if barra_control("dendro", (firma_modelos(_crop_mask_den), metodo), "The dendrogram"):
                ids_snap, _, clases_snap = datos_activos()
                with st.spinner("Clustering..."):
                    Z_nuevo = _calcular_linkage_cacheado(X_hca_vivo, metodo)
                # Frozen snapshot: the result keeps its OWN samples/classes, so excluding
                # outliers afterwards can never leave the drawing out of sync with them.
                st.session_state["dendro_resultado"] = {
                    "Z": Z_nuevo, "X": X_hca_vivo, "ids": ids_snap, "clases": clases_snap,
                    "metodo": metodo, "figuras": {},
                "crop_desc": st.session_state.get("crop_desc_aplicada") if _crop_mask_den is not None else None,
                }
                st.session_state["dendro_firma"] = (firma_modelos(_crop_mask_den), metodo)
                st.session_state["dendro_origen"] = "cropped" if _crop_mask_den is not None else "full"
                _rerun_tab()

            res = st.session_state.get("dendro_resultado")
            if res is None:
                st.info("Choose the linkage method above and click **Compute**. "
                        "Nothing is calculated until you do.")
            else:
                Z, X_hca, ids, clases = res["Z"], res["X"], res["ids"], res["clases"]
                st.caption(f"Showing the result computed with the **{res['metodo']}** linkage on "
                           f"{X_hca.shape[0]} samples"
                           + (f", on the ✂️ cropped spectrum ({res['crop_desc']})." if res.get("crop_desc") else "."))
                _dd1, _dd2, _dd3 = st.columns([1.5, 1.9, 3])
                with _dd1:
                    gu.how_to_interpret("dendro", key="how_res_dendro")
                with _dd2:
                    gu.what_next("dendro", key="next_res_dendro")

                mapa_color_clase = None
                if clases is not None:
                    clases_unicas = list(pd.unique(como_texto(clases)))
                    mapa_color_clase = {c: rgb_string_a_hex(CLASS_PALETTE[i % len(CLASS_PALETTE)])
                                         for i, c in enumerate(clases_unicas)}
                    st.caption("Leaf labels are colored by class, so you can see at a glance whether "
                               "same-class samples cluster together (cluster purity).")
                    leyenda = "  ".join(f"<span style='color:{color}'>■</span> {c}"
                                         for c, color in mapa_color_clase.items())
                    st.markdown(leyenda, unsafe_allow_html=True)

                # ---------------- where to cut the tree
                _Zh = np.asarray(Z[:, 2], dtype=float)
                _hmax, _ns = float(_Zh.max()), int(X_hca.shape[0])
                _k_def = int(min(max(2, len(set(como_texto(clases))) if clases is not None else 3), max(2, _ns - 1), 15))
                _salto = int(np.argmax(np.diff(_Zh))) if len(_Zh) > 2 else 0
                _k_sug = int(min(max(2, _ns - (_salto + 1)), max(2, _ns - 1)))
                st.markdown("**✂️ Where to cut the tree?**")
                _modo_corte = st.radio("Cut by", ["Number of clusters", "Distance (height)"], horizontal=True, key="dendro_w_modo_corte",
                                       help="The horizontal cut line decides the clusters: every branch that the line crosses is one cluster. "
                                            "You can set the number of clusters you want, or the height of the line.")
                def _h_para_k(kk):
                    kk = int(min(max(kk, 2), _ns - 1))
                    return float((_Zh[_ns - kk - 1] + _Zh[_ns - kk]) / 2) if _ns - kk < len(_Zh) else _hmax
                if _modo_corte.startswith("Number"):
                    _kc = st.slider("Number of clusters", 2, int(max(2, min(15, _ns - 1))), _k_def, key="dendro_w_k",
                                    help="How many groups you want. The cut height is placed between the last two merges that separate them.")
                    _hc = _h_para_k(_kc)
                else:
                    _hc = st.slider("Cut height (distance)", 0.0, _hmax, float(_h_para_k(_k_def)), step=_hmax / 200 if _hmax > 0 else 0.01,
                                    key="dendro_w_h", help="Height of the horizontal line. Low line = many tight clusters; high line = few broad clusters.")
                _lab_c = hierarchy.fcluster(Z, t=_hc, criterion="distance")
                _kc_real = int(len(np.unique(_lab_c)))
                _pur_c = None
                if clases is not None:
                    _df_c = pd.DataFrame({"c": _lab_c, "y": como_texto(clases)})
                    _pur_c = float(_df_c.groupby("c")["y"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(_df_c))
                _m1, _m2, _m3, _m4 = st.columns(4)
                _m1.metric("Clusters", _kc_real)
                _m2.metric("Cut height", f"{_hc:.4g}", f"{100 * _hc / _hmax:.0f} % of the maximum" if _hmax > 0 else None, delta_color="off")
                _m3.metric("Smallest cluster", int(pd.Series(_lab_c).value_counts().min()))
                _m4.metric("Purity vs. classes", f"{_pur_c:.2f}" if _pur_c is not None else "–",
                           help="Fraction of samples that belong to the majority class of their cluster (1 = every cluster contains a single class).")
                st.caption(f"💡 The largest jump in the merge heights suggests **{_k_sug} clusters** (a long vertical branch means the groups "
                           "below it are very different from one another).")
                st.session_state["dendro_corte"] = {"k": _kc_real, "h": float(_hc), "pureza": _pur_c, "metodo": res["metodo"], "sug": _k_sug}
                with st.expander("📘 What does the cut height mean?", expanded=False):
                    st.markdown(
                        "- The **vertical axis is the distance (dissimilarity)** at which two groups join: the lower the height of a junction, "
                        "the **more similar** the samples (or groups) that merge there.\n"
                        "- The **cut line** says: *'everything that joins below this height is similar enough to be one cluster'*. Samples that only "
                        "meet above the line are considered different groups.\n"
                        "- **Lower the line** → more and tighter clusters (down to one sample each). **Raise it** → fewer, broader clusters "
                        "(up to a single cluster).\n"
                        "- The **number of clusters is the number of branches the line crosses.** A good cut sits in a *long vertical stretch* "
                        "without junctions: those groups are well separated.\n"
                        "- The scale depends on the linkage (Ward heights are larger than average linkage ones), so compare heights only "
                        "within the same tree.")

                # Each drawing is built once and kept with the result (not rebuilt on every rerun).
                if vista.startswith("Linear"):
                    if "lineal" not in res["figuras"]:
                        # distfun is skipped on purpose: Z is already computed, and plotly would
                        # otherwise recompute all pairwise distances just to throw them away.
                        fig = ff.create_dendrogram(X_hca, labels=list(ids), distfun=lambda x: None,
                                                   linkagefun=lambda x: Z)
                        fig.update_layout(height=550 if mapa_color_clase is None else 610,
                                          title="Linear dendrogram")
                        if mapa_color_clase is not None:
                            orden_hojas = list(fig.layout.xaxis.ticktext)
                            pos_id = {str(i): k for k, i in enumerate(ids)}
                            colores_hojas = [mapa_color_clase.get(str(clases[pos_id[str(hoja)]]), "black")
                                             for hoja in orden_hojas]
                            altura_max = float(Z[:, 2].max())
                            y_marcador = -altura_max * 0.05
                            fig.add_trace(go.Scatter(
                                x=list(fig.layout.xaxis.tickvals), y=[y_marcador] * len(orden_hojas),
                                mode="markers", marker=dict(size=9, color=colores_hojas, symbol="square"),
                                hoverinfo="skip", showlegend=False,
                            ))
                            fig.update_yaxes(range=[y_marcador * 1.8, altura_max * 1.05])
                        res["figuras"]["lineal"] = fig
                    _figc = go.Figure(res["figuras"]["lineal"])
                    _orden_h = list(_figc.layout.xaxis.ticktext)
                    _pos = {str(i): k for k, i in enumerate(ids)}
                    _paleta_c = px.colors.qualitative.Safe
                    _col_cl = [_paleta_c[(int(_lab_c[_pos[str(h_)]]) - 1) % len(_paleta_c)] for h_ in _orden_h]
                    _figc.add_trace(go.Scatter(x=list(_figc.layout.xaxis.tickvals), y=[-_hmax * 0.12] * len(_orden_h), mode="markers",
                                               marker=dict(size=9, color=_col_cl, symbol="diamond"), hoverinfo="skip", showlegend=False))
                    _figc.add_hline(y=_hc, line_dash="dash", line_color="#C0392B", line_width=2,
                                    annotation_text=f"cut = {_hc:.3g} → {_kc_real} clusters", annotation_position="top left",
                                    annotation_font_color="#C0392B")
                    _figc.update_yaxes(range=[-_hmax * 0.2, _hmax * 1.05])
                    st.plotly_chart(_figc, width='stretch')
                    st.caption("Lower strip: ◼ colour of the class (if any) · ◆ colour of the cluster obtained with the cut.")
                    gu.nota("dendro", "dendrogram", "dendro")
                else:
                    if "circular" not in res["figuras"]:
                        id_a_clase = None
                        if clases is not None:
                            id_a_clase = {str(i): str(c) for i, c in zip(ids, clases)}
                        with st.spinner("Drawing..."):
                            fig_mpl = cu.dendrograma_circular_fig(Z, list(ids), id_a_clase=id_a_clase,
                                                                  mapa_color_clase=mapa_color_clase)
                            _buf = io.BytesIO()
                            fig_mpl.savefig(_buf, format="png", dpi=130, bbox_inches="tight")
                            plt.close(fig_mpl)
                        res["figuras"]["circular"] = _buf.getvalue()
                    st.image(res["figuras"]["circular"])
                    st.caption(f"Cut at {_hc:.3g}: {_kc_real} clusters (the cut line is shown in the linear view).")
                    gu.nota("dendro", "dendrogram", "dendro")
    with tabs[9]:
        _frag_tab_6()


# -----------------------------------------------------------------------
# TAB: OTHER TOOLS
# -----------------------------------------------------------------------
if _abierta(tabs[10]):
    # Each tab re-runs ON ITS OWN when one of its widgets changes (fragment), instead of
    # re-executing the whole app. Buttons that must refresh other parts call st.rerun().
    @st.fragment
    def _frag_tab_7():
        st.subheader("Other exploratory analysis tools")
        gu.barra_guia("otros")
        st.caption("Pick a tool, adjust its settings, and click **Compute**. Nothing runs until you do, "
                   "and each tool keeps its own result until you clear it.")

        ids_vivo, _, clases_vivo = datos_activos()
        eje_vivo = np.array(st.session_state.numeros_onda_pret)
        X_vivo = st.session_state.X_pret[indice_activo()]

        herramienta = st.selectbox(
            "Choose a tool",
            ["Loadings vs wavenumber", "Ranking of important variables",
             "Loadings correlation plot", "Mean spectrum by class",
             "Clustermap (heatmap + dendrogram)", "t-SNE", "UMAP", "MCR-ALS (mixture resolution)"],
            key="otros_w_herramienta",
            help="Pick which exploratory technique to run on the preprocessed data.",
        )

        tiene_pca = st.session_state.get("pca_completo") is not None
        _crop_mask_otros = None
        if herramienta not in ("Loadings vs wavenumber", "Ranking of important variables", "Loadings correlation plot"):
            X_vivo, eje_vivo, _crop_mask_otros, _ = elegir_espectro_modelado(X_vivo, eje_vivo, "otros")
        n_vivo = X_vivo.shape[0]

        if herramienta in ["Loadings vs wavenumber", "Ranking of important variables",
                            "Loadings correlation plot"] and not tiene_pca:
            st.info("Compute PCA first, in the **PCA** tab — these views are built from its loadings.")

        # ------------------------------------------------------------------ PCA-based views
        elif herramienta == "Loadings vs wavenumber":
            _firma = ("pca", st.session_state.get("pca_firma"))
            if barra_control("otros_loadings", _firma, "This view"):
                st.session_state["otros_loadings_resultado"] = {
                    "cargas": np.array(st.session_state.cargas_completo),
                    "eje": np.array(st.session_state.pca_eje)}
                st.session_state["otros_loadings_firma"] = _firma
                _rerun_tab()
            r = st.session_state.get("otros_loadings_resultado")
            if r is None:
                st.info("Click **Compute** to draw the loadings of the PCA you computed.")
            else:
                cargas, eje = r["cargas"], r["eje"]
                n_pcs = st.slider("How many PCs to show", 1, min(5, cargas.shape[0]), min(3, cargas.shape[0]),
                                   key="otros_w_loadings_npcs",
                                   help="Number of components to overlay on the loading plot, so you can "
                                        "compare which wavenumbers drive each one.")
                fig = go.Figure()
                for i in range(n_pcs):
                    if es_tabular():
                        fig.add_trace(go.Scatter(x=eje, y=cargas[i], mode="lines+markers", name=f"PC{i+1}",
                                                 text=etiquetas_var(eje), hovertemplate="%{text}<br>loading %{y:.3f}<extra>PC" + str(i + 1) + "</extra>"))
                    else:
                        fig.add_trace(go.Scatter(x=eje, y=cargas[i], mode="lines", name=f"PC{i+1}"))
                fig.update_layout(height=450, xaxis_title=("Variable (index; hover for the name)" if es_tabular() else "Wavenumber"),
                                  yaxis_title="Loading")
                if eje[0] > eje[-1]:
                    fig.update_xaxes(autorange="reversed")
                st.plotly_chart(fig, width='stretch')
                gu.nota("otros_loadings", "loadings plot", "otros_loadings")

        elif herramienta == "Ranking of important variables":
            _firma = ("pca", st.session_state.get("pca_firma"))
            if barra_control("otros_ranking", _firma, "This ranking"):
                st.session_state["otros_ranking_resultado"] = {
                    "cargas": np.array(st.session_state.cargas_completo),
                    "eje": np.array(st.session_state.pca_eje)}
                st.session_state["otros_ranking_firma"] = _firma
                _rerun_tab()
            r = st.session_state.get("otros_ranking_resultado")
            if r is None:
                st.info("Click **Compute** to rank the variables by their weight in a principal component.")
            else:
                cargas, eje = r["cargas"], r["eje"]
                col1, col2 = st.columns(2)
                pc_elegido = col1.number_input("PC to analyze", min_value=1, max_value=cargas.shape[0], value=1,
                                                key="otros_w_ranking_pc", help="Which principal component's loadings are analysed.")
                top_n = col2.slider("How many variables to show", 5, 40, 15, key="otros_w_ranking_top", help="Number of variables with the highest loadings to list/mark.")
                idx_pc = pc_elegido - 1
                orden = np.argsort(np.abs(cargas[idx_pc]))[::-1][:top_n]
                orden = orden[np.argsort(cargas[idx_pc, orden])]
                colores = ["crimson" if v < 0 else "steelblue" for v in cargas[idx_pc, orden]]
                fig = go.Figure(go.Bar(
                    x=cargas[idx_pc, orden],
                    y=(list(etiquetas_var(eje[orden])) if es_tabular() else [f"{eje[i]:.0f}" for i in orden]),
                    orientation="h", marker_color=colores,
                ))
                fig.update_layout(height=max(350, 22 * top_n), xaxis_title=f"Loading on PC{pc_elegido}",
                                   yaxis_title=("Variable" if es_tabular() else "Wavenumber"))
                st.plotly_chart(fig, width='stretch')
                gu.nota("otros_ranking", "ranking of variables", "otros_ranking")

        elif herramienta == "Loadings correlation plot":
            _firma = ("pca", st.session_state.get("pca_firma"))
            if barra_control("otros_corr", _firma, "This plot"):
                st.session_state["otros_corr_resultado"] = {
                    "cargas": np.array(st.session_state.cargas_completo),
                    "autovalores": np.array(st.session_state.autovalores),
                    "eje": np.array(st.session_state.pca_eje)}
                st.session_state["otros_corr_firma"] = _firma
                _rerun_tab()
            r = st.session_state.get("otros_corr_resultado")
            if r is None:
                st.info("Click **Compute** to draw the loadings correlation plot of the PCA you computed.")
            else:
                cargas, autovalores, eje = r["cargas"], r["autovalores"], r["eje"]
                col1, col2, col3 = st.columns(3)
                pc_x = col1.number_input("PC on X axis", min_value=1, max_value=cargas.shape[0], value=1,
                                          key="otros_w_corr_x", help="Component on the horizontal axis.")
                pc_y = col2.number_input("PC on Y axis", min_value=1, max_value=cargas.shape[0],
                                          value=min(2, cargas.shape[0]), key="otros_w_corr_y", help="Component on the vertical axis.")
                top_n = col3.slider("Variables to highlight", 5, 40, 15, key="otros_w_corr_top", help="How many variables with the strongest influence are marked.")

                load_x = cargas[pc_x - 1] * np.sqrt(autovalores[pc_x - 1])
                load_y = cargas[pc_y - 1] * np.sqrt(autovalores[pc_y - 1])
                escala = 1 / max(np.max(np.abs(load_x)), np.max(np.abs(load_y)))
                load_x_n, load_y_n = load_x * escala, load_y * escala
                peso = load_x_n ** 2 + load_y_n ** 2
                top_idx = np.argsort(peso)[-top_n:]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=load_x_n, y=load_y_n, mode="markers",
                                          marker=dict(size=6, color="steelblue", opacity=0.35),
                                          name="All variables"))
                fig.add_trace(go.Scatter(x=load_x_n[top_idx], y=load_y_n[top_idx], mode="markers+text",
                                          marker=dict(size=9, color="darkred"),
                                          text=(list(etiquetas_var(eje[top_idx])) if es_tabular() else [f"{eje[i]:.0f}" for i in top_idx]),
                                          textposition="top center", name=f"Top {top_n}"))
                theta = np.linspace(0, 2 * np.pi, 100)
                for rad in (0.5, 1.0):
                    fig.add_trace(go.Scatter(x=rad * np.cos(theta), y=rad * np.sin(theta), mode="lines",
                                              line=dict(dash="dash", color="gray"), showlegend=False))
                fig.update_layout(height=600, xaxis_title=f"PC{pc_x}", yaxis_title=f"PC{pc_y}",
                                   xaxis=dict(range=[-1.2, 1.2]), yaxis=dict(range=[-1.2, 1.2], scaleanchor="x"))
                st.plotly_chart(fig, width='stretch')
                gu.nota("otros_corr", "loadings correlation plot", "otros_corr")

        # ------------------------------------------------------------------ Mean spectrum
        elif herramienta == "Mean spectrum by class":
            if clases_vivo is None:
                st.warning("Define classes in the left-hand panel first.")
            else:
                _firma = (firma_modelos(_crop_mask_otros),)
                if barra_control("otros_media", _firma, "This plot"):
                    por_clase = {}
                    for cl in np.unique(clases_vivo):
                        mask = clases_vivo == cl
                        por_clase[str(cl)] = (X_vivo[mask].mean(axis=0), X_vivo[mask].std(axis=0))
                    st.session_state["otros_media_resultado"] = {"por_clase": por_clase, "eje": eje_vivo.copy()}
                    st.session_state["otros_media_firma"] = _firma
                    st.session_state["otros_media_origen"] = "cropped" if _crop_mask_otros is not None else "full"
                    _rerun_tab()
                r = st.session_state.get("otros_media_resultado")
                if r is None:
                    st.info("Click **Compute** to get the mean spectrum (± standard deviation) of each class.")
                else:
                    eje = r["eje"]
                    fig = go.Figure()
                    for nombre, (promedio, desvio) in r["por_clase"].items():
                        fig.add_trace(go.Scatter(x=eje, y=promedio, mode="lines", name=nombre))
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([eje, eje[::-1]]),
                            y=np.concatenate([promedio + desvio, (promedio - desvio)[::-1]]),
                            fill="toself", opacity=0.15, line=dict(width=0), showlegend=False,
                        ))
                    fig.update_layout(height=450, xaxis_title="Wavenumber", yaxis_title="Signal")
                    if eje[0] > eje[-1]:
                        fig.update_xaxes(autorange="reversed")
                    st.plotly_chart(fig, width='stretch')
                    gu.nota("otros_media", "mean spectrum by class", "otros_media")

        # ------------------------------------------------------------------ Clustermap
        elif herramienta == "Clustermap (heatmap + dendrogram)":
            metodo_cm = st.selectbox("Linkage method", ["ward", "average", "complete", "single"],
                                      key="otros_w_cm_metodo", help="How clusters are merged. Ward gives compact clusters; single can chain; average is intermediate.")
            _firma = (firma_modelos(_crop_mask_otros), metodo_cm)
            if barra_control("otros_cm", _firma, "This clustermap"):
                with st.spinner("Clustering and drawing the heatmap — this can take a while with many samples..."):
                    df_heat = pd.DataFrame(X_vivo, index=ids_vivo, columns=np.round(eje_vivo, 0))
                    import seaborn as sns
                    fig_cm = sns.clustermap(df_heat, method=metodo_cm, metric="euclidean",
                                             col_cluster=False, cmap="viridis", figsize=(10, 7),
                                             xticklabels=False)
                    _buf = io.BytesIO()
                    fig_cm.fig.savefig(_buf, format="png", dpi=130, bbox_inches="tight")
                    plt.close(fig_cm.fig)
                st.session_state["otros_cm_resultado"] = {"png": _buf.getvalue(), "metodo": metodo_cm}
                st.session_state["otros_cm_firma"] = _firma
                st.session_state["otros_cm_origen"] = "cropped" if _crop_mask_otros is not None else "full"
                _rerun_tab()
            r = st.session_state.get("otros_cm_resultado")
            if r is None:
                st.info("Choose the linkage method and click **Compute**. This is one of the heavier "
                        "tools, so it only runs on demand.")
            else:
                st.caption(f"Clustermap computed with the **{r['metodo']}** linkage.")
                st.image(r["png"])
                gu.nota("otros_cm", "clustermap", "otros_cm")

        # ------------------------------------------------------------------ t-SNE
        elif herramienta == "t-SNE":
            perplejidad = st.slider("Perplexity", 5, min(50, max(6, n_vivo - 1)), min(30, max(6, n_vivo - 1)),
                                     key="otros_w_tsne_perp", help="Roughly the number of neighbours each point considers. Low = local structure, high = global structure.")
            _firma = (firma_modelos(_crop_mask_otros), perplejidad)
            if barra_control("otros_tsne", _firma, "This t-SNE map"):
                from sklearn.manifold import TSNE
                with st.spinner("Computing t-SNE..."):
                    emb = TSNE(n_components=2, perplexity=perplejidad, init="pca",
                               random_state=0).fit_transform(X_vivo)
                st.session_state["otros_tsne_resultado"] = {"emb": emb, "ids": ids_vivo, "clases": clases_vivo}
                st.session_state["otros_tsne_firma"] = _firma
                st.session_state["otros_tsne_origen"] = "cropped" if _crop_mask_otros is not None else "full"
                _rerun_tab()
            r = st.session_state.get("otros_tsne_resultado")
            if r is None:
                st.info("Set the perplexity and click **Compute**.")
            else:
                fig = px.scatter(x=r["emb"][:, 0], y=r["emb"][:, 1], hover_name=r["ids"],
                                  color=r["clases"] if r["clases"] is not None else None,
                                  color_discrete_sequence=CLASS_PALETTE,
                                  labels={"x": "t-SNE 1", "y": "t-SNE 2"})
                fig.update_layout(height=500)
                st.plotly_chart(fig, width='stretch')
                gu.nota("otros_tsne", "t-SNE map", "otros_tsne")

        # ------------------------------------------------------------------ UMAP
        elif herramienta == "UMAP":
            if importlib.util.find_spec("umap") is None:     # check only; importing umap takes ~25 s
                st.error("The `umap-learn` package is missing (pip install umap-learn) for this option.")
            else:
                st.caption("⏱ The first UMAP run after the app starts takes longer (30–60 s: the library "
                           "compiles itself once). Later runs are fast.")
                vecinos = st.slider("n_neighbors", 2, min(50, max(3, n_vivo - 1)), min(15, max(3, n_vivo - 1)),
                                     key="otros_w_umap_vecinos", help="Neighbourhood size. Small = local detail, large = global structure.")
                _firma = (firma_modelos(_crop_mask_otros), vecinos)
                if barra_control("otros_umap", _firma, "This UMAP map"):
                    with st.spinner("Computing UMAP... (first time: up to a minute)"):
                        import umap
                        emb = umap.UMAP(n_components=2, n_neighbors=vecinos, random_state=0).fit_transform(X_vivo)
                    st.session_state["otros_umap_resultado"] = {"emb": emb, "ids": ids_vivo, "clases": clases_vivo}
                    st.session_state["otros_umap_firma"] = _firma
                    st.session_state["otros_umap_origen"] = "cropped" if _crop_mask_otros is not None else "full"
                    _rerun_tab()
                r = st.session_state.get("otros_umap_resultado")
                if r is None:
                    st.info("Set n_neighbors and click **Compute**.")
                else:
                    fig = px.scatter(x=r["emb"][:, 0], y=r["emb"][:, 1], hover_name=r["ids"],
                                      color=r["clases"] if r["clases"] is not None else None,
                                      color_discrete_sequence=CLASS_PALETTE,
                                      labels={"x": "UMAP 1", "y": "UMAP 2"})
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, width='stretch')
                    gu.nota("otros_umap", "UMAP map", "otros_umap")

        # ------------------------------------------------------------------ MCR-ALS
        elif herramienta == "MCR-ALS (mixture resolution)":
            st.caption("Multivariate Curve Resolution — Alternating Least Squares: decomposes your "
                       "spectra into a set of 'pure component' spectra and their concentration profile "
                       "across samples, without needing to know the pure spectra beforehand. Useful when "
                       "your samples are mixtures and you want to recover what the individual "
                       "constituents look like and how much of each is in every sample. Assumes "
                       "non-negative concentrations and spectra (the usual physical case).")
            n_componentes_mcr = st.slider(
                "Number of components to resolve", 2, max(2, min(8, n_vivo - 1)), 2, key="otros_w_mcr_n",
                help="How many pure/underlying components MCR-ALS should try to recover. Too few won't "
                     "explain the mixtures well; too many risk splitting real signal into noise-fitting "
                     "components. Try comparing the lack-of-fit for a couple of values.",
            )
            _firma = (firma_modelos(_crop_mask_otros), n_componentes_mcr)
            if barra_control("otros_mcr", _firma, "This MCR-ALS result", texto_compute="▶ Run MCR-ALS"):
                with st.spinner("Running alternating least squares..."):
                    resultado_mcr = cu.mcr_als(X_vivo, n_componentes=n_componentes_mcr, max_iter=200)
                st.session_state["otros_mcr_resultado"] = {
                    "res": resultado_mcr, "ids": ids_vivo, "eje": eje_vivo.copy(), "n": n_componentes_mcr}
                st.session_state["otros_mcr_firma"] = _firma
                st.session_state["otros_mcr_origen"] = "cropped" if _crop_mask_otros is not None else "full"
                _rerun_tab()
            r = st.session_state.get("otros_mcr_resultado")
            if r is None:
                st.info("Choose the number of components and click **Run MCR-ALS**.")
            else:
                resultado_mcr, ids_mcr, eje_mcr = r["res"], r["ids"], r["eje"]
                st.metric("Lack of fit", f"{resultado_mcr['lof_pct']:.2f}%",
                          help="Percentage of the data's variance NOT explained by the resolved "
                               "components — lower is better. Under ~5% is generally considered a good fit.")
                st.caption(f"{r['n']} components · converged in {resultado_mcr['n_iter']} iterations.")

                fig_spectra_mcr = go.Figure()
                for k in range(resultado_mcr["S"].shape[0]):
                    fig_spectra_mcr.add_trace(go.Scatter(x=eje_mcr, y=resultado_mcr["S"][k],
                                                           mode="lines", name=f"Component {k+1}"))
                fig_spectra_mcr.update_layout(height=400, title="Resolved pure-component spectra",
                                               xaxis_title="Wavenumber", yaxis_title="Signal (a.u.)")
                if eje_mcr[0] > eje_mcr[-1]:
                    fig_spectra_mcr.update_xaxes(autorange="reversed")
                st.plotly_chart(fig_spectra_mcr, width='stretch')

                df_conc_mcr = pd.DataFrame(
                    resultado_mcr["C"], index=ids_mcr,
                    columns=[f"Component {k+1}" for k in range(resultado_mcr["C"].shape[1])],
                )
                fig_conc_mcr = go.Figure()
                for col in df_conc_mcr.columns:
                    fig_conc_mcr.add_trace(go.Bar(x=df_conc_mcr.index.astype(str), y=df_conc_mcr[col], name=col))
                fig_conc_mcr.update_layout(height=400, barmode="stack", title="Relative concentration profile per sample",
                                            xaxis_title="Sample", yaxis_title="Relative concentration")
                st.plotly_chart(fig_conc_mcr, width='stretch')
                gu.nota("otros_mcr", "MCR-ALS results", "otros_mcr")

                st.download_button(
                    "⬇️ Download concentration profiles (Excel)",
                    data=df_a_excel_bytes({"mcr_concentrations": df_conc_mcr.reset_index(names="id")}),
                    file_name="mcr_als_concentrations.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="descargar_mcr_excel",
                )
    with tabs[10]:
        _frag_tab_7()


# -----------------------------------------------------------------------
# TAB: CLASSIFICATION
# -----------------------------------------------------------------------
if _abierta(tabs[12]):
    # Each tab re-runs ON ITS OWN when one of its widgets changes (fragment), instead of
    # re-executing the whole app. Buttons that must refresh other parts call st.rerun().
    @st.fragment
    def _frag_tab_9():
        st.subheader("Supervised classification")
        st.caption("**Model refinement.** Use this tab to refine the candidates that came out of the 🏁 Model screening: "
                   "train them with the settings you prefer (test set, cross-validation, variable selection, hyperparameter "
                   "optimization), inspect the details, and **save the model you will use to predict new samples** in the 🔮 Prediction tab.")
        gu.barra_guia("clf", "clf", valores=gu.valores_clf(), pros=("clf_algo", None))
        if st.button("🔄 Reset this tab", key="reset_clf",
                     help="Clears all trained models, metrics, and plots from this tab, so you can "
                          "start a completely fresh run without any leftover results from before."):
            resetear_prefijo("clf_")
            _rerun_tab()

        ids_activos, X_activo_clf, clases_activas = datos_activos()
        X_modelado = st.session_state.X_pret[indice_activo()]
        eje_modelado = st.session_state.numeros_onda_pret
        X_modelado, eje_modelado, _crop_mask_clf, _eje_completo_clf = elegir_espectro_modelado(X_modelado, eje_modelado, "clf")

        if clases_activas is None:
            st.warning("Define classes in the left-hand panel (Data tab) to run a classification.")
        else:
            conteos = pd.Series(clases_activas).value_counts()
            if conteos.min() < 2:
                st.error(f"Some class has fewer than 2 samples ({conteos.to_dict()}). "
                         "At least 2 samples per class are needed to validate.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    modelos_elegidos = st.multiselect(
                        "Models to compare",
                        list(mu.crear_clasificadores().keys()),
                        default=["LDA", "PLS-DA", "Random Forest"],
                        help="Pick one or more algorithms to train and compare side by side in the same "
                             "run — the results table below will show every metric for each of them.",
                    key="ac_clf_models")
                    metodo_seleccion = st.selectbox(
                        "Variable selection",
                        ["None", "Boruta", "Genetic Algorithm"],
                        help="Reduce the spectrum to just the most informative wavenumbers before "
                             "training. Boruta is conservative and keeps anything that helps at all; "
                             "the Genetic Algorithm explores combinations more freely but is slower and "
                             "a bit more random. 'None' uses the full spectrum.",
                    key="ac_clf_varsel", disabled=_BASICO)
                    if _BASICO:
                        metodo_seleccion = "None"
                    gu.pros_cons("seleccion", metodo_seleccion, key="pc_sel_clf")
                with col2:
                    cv_folds_sel = st.selectbox(
                        "Cross-validation folds (internal)",
                        [3, 5, 10, "LOO (leave 1 sample out at a time)"], index=1,
                        help="How many folds to use for cross-validation. More folds = less biased but "
                             "slower and more variance between folds. LOO (Leave-One-Out) trains one "
                             "model per sample — most thorough, slowest, and typical for small datasets.",
                    key="ac_clf_cv")
                    cv_folds = "LOO" if cv_folds_sel == "LOO (leave 1 sample out at a time)" else cv_folds_sel
                    gu.pros_cons("cv", "Leave-one-out (LOO)" if cv_folds == "LOO" else "k-fold (5 or 10)", key="pc_cv_clf")
                    if cv_folds == "LOO" and X_modelado.shape[0] > 150:
                        st.caption("⚠ LOO trains one model per sample — with many samples this can take a while.")
                    prop_test = st.slider(
                        "Proportion for independent test set (0 = use everything for CV)",
                        0.0, 0.4, 0.2, step=0.05,
                        help="Fraction of samples set aside BEFORE training, never used to choose "
                             "anything — only to get one final, honest check of how the model performs "
                             "on data it has never influenced. 0 means skip this and use every sample "
                             "for cross-validation instead.",
                    key="ac_clf_ptest")
                    metodo_split_sel, metodo_split, semilla_clf = _ui_split(True, prop_test, "ac_clf_split")
                    optimizar = st.checkbox(
                        "Optimize hyperparameters (slower)",
                        help="Automatically search for better settings for each model (e.g. how many "
                             "trees in a Random Forest) instead of using the defaults. Usually improves "
                             "results a bit, at the cost of extra computation time.",
                    key="ac_clf_optim", disabled=_BASICO)
                    if _BASICO:
                        optimizar = False
                    if optimizar:
                        st.caption("Search ranges in use: **" + st.session_state.get("hp_modo_aplicado", HP_MODOS[1])
                                   + "** — change them in the 🎛️ Hyperparameters tab.")
                    preset_opt = st.selectbox(
                        "Optimization budget", list(PRESETS_OPTIMIZACION.keys()), index=1, disabled=not optimizar,
                        help="How much effort to spend searching for good hyperparameters — see the "
                             "explanation below once you pick one.",
                    key="ac_clf_preset")
                    metodo_opt = PRESETS_OPTIMIZACION[preset_opt][0]
                    gu.pros_cons("optimizacion", preset_opt, key="pc_opt_clf")
                    st.caption("The ranges explored are shown (and editable) in the **🎛️ Hyperparameters** tab.")
                    cfg_aug = _ui_aumento("clf", True)

                if optimizar:
                    st.caption(f"ℹ️ **{preset_opt}**: {PRESETS_OPTIMIZACION[preset_opt][1]}")

                if optimizar and any(m in mu.ALGORITMOS_LENTOS_AL_OPTIMIZAR for m in modelos_elegidos):
                    lentos_elegidos = [m for m in modelos_elegidos if m in mu.ALGORITMOS_LENTOS_AL_OPTIMIZAR]
                    extra = " With LOO this can take several minutes." if cv_folds == "LOO" else ""
                    st.caption(f"⚠ Optimizing hyperparameters for {', '.join(lentos_elegidos)} can still take a while "
                               f"on a shared/limited CPU. If it's too slow, try 'Fast', or fewer folds.{extra}")

                if optimizar and len(modelos_elegidos) > 3:
                    st.warning(f"⚠️ You're optimizing hyperparameters for {len(modelos_elegidos)} models at once. "
                               "Each one runs its own search in parallel across CPU cores, so doing several "
                               "together can exhaust the machine's memory and cause the app to crash or freeze. "
                               "We recommend optimizing **up to 3 models at a time** — train the rest with "
                               "default settings first, then optimize the best candidates separately.")

                if metodo_seleccion == "Boruta":
                    c1, c2 = st.columns(2)
                    boruta_max_iter = c1.slider(
                        "Boruta iterations", 10, 500, 40, step=10, key="boruta_iter_clf",
                        help="How many comparison rounds to run against the random 'shadow' variables. "
                             "More iterations give a more stable decision but take longer.",
                    )
                    boruta_alpha = c2.select_slider(
                        "Boruta p-value (alpha)", options=[0.01, 0.02, 0.05, 0.1], value=0.05, key="boruta_alpha_clf",
                        help="Significance level for keeping a variable: smaller (e.g. 0.01) is stricter "
                             "and keeps fewer, more confidently relevant variables; larger (e.g. 0.1) is "
                             "more permissive.",
                    )
                if metodo_seleccion == "Genetic Algorithm":
                    c1, c2 = st.columns(2)
                    ga_poblacion = c1.slider("Population size", 10, 60, 20, key="ga_pob_clf", help="Number of candidate solutions per generation. Larger = better search but slower.")
                    ga_generaciones = c2.slider("Generations", 5, 50, 15, key="ga_gen_clf", help="How many generations the search runs. More = better but slower.")

                if st.session_state.get("clf_resultados") is not None:
                    insignia_estado("clf_firma", "This trained classification model", firma_actual=firma_modelos(_crop_mask_clf))

                if metodo_seleccion != "None" or optimizar:
                    st.caption("ℹ️ Variable selection and hyperparameter optimization use ONLY the training samples "
                               "(the test set is set aside first and never touched). Cross-validation figures can still be "
                               "slightly optimistic when selection is used; the independent test set is the honest estimate."
                               + (" ⚠ With no test set (0%), selection uses all the samples, so the CV figures are optimistic." if prop_test == 0 else ""))
                _run_clf = st.session_state.get("clf_run")
                _incompleto_clf = bool(_run_clf) and not _run_clf.get("completo", True)
                _valida_clf = _incompleto_clf and _run_clf.get("firma") == firma_modelos(_crop_mask_clf)
                if _incompleto_clf:
                    _hechos_clf = len([n_ for n_, r_ in _run_clf["resultados"].items() if "error" not in r_])
                    st.warning(f"⏸ Incomplete run: {_hechos_clf} of {len(_run_clf['modelos'])} models finished (the run was interrupted). "
                               + ("Click **Resume** to continue with the same settings, skipping the finished models."
                                  if _valida_clf else "The data, preprocessing or crop changed since then, so it can't be resumed — train again."))
                _ct1, _ct2, _ = st.columns([1.7, 1.6, 3])
                _ent_clf = _ct1.button("🚀 Train and evaluate (Classification)", disabled=len(modelos_elegidos) == 0)
                _rean_clf = _ct2.button("⏩ Resume interrupted run", key="clf_btn_resume", disabled=not _valida_clf,
                                        help="Continues the interrupted run with the SAME settings it started with, skipping "
                                             "the models that already finished (and not repeating the variable selection).")
                if _ent_clf or _rean_clf:
                    if _rean_clf:
                        _r = _run_clf
                        modelos_elegidos = list(_r["modelos"]); cv_folds = _r["cv_folds"]; prop_test = _r["prop_test"]
                        metodo_split = _r["metodo_split"]; metodo_split_sel = _r["metodo_split_sel"]; semilla_clf = _r.get("semilla", 0)
                        optimizar = _r["optimizar"]; metodo_opt = _r["metodo_opt"]; metodo_seleccion = _r["metodo_seleccion"]
                        boruta_max_iter, boruta_alpha = _r["boruta_iter"], _r["boruta_alpha"]
                        ga_poblacion, ga_generaciones = _r["ga_pob"], _r["ga_gen"]
                        _idx_split_clf = _r["idx_split"]
                    else:
                        # Clear secondary results tied to the PREVIOUS set of trained models (a
                        # statistical comparison or learning curve computed for models A/B/C would
                        # otherwise linger on screen after retraining with a different D/E/F).
                        for _clave in ["clf_pvalores", "clf_puntajes_cv", "clf_comparacion_metodo",
                                        "clf_curva_aprendizaje", "clf_curva_modelo", "clf_ultima_ficha", "clf_rs_res"]:
                            st.session_state.pop(_clave, None)
                        # The train/test split is decided FIRST: variable selection and hyperparameter optimization
                        # then use ONLY the training samples, so nothing about the test set leaks into the model.
                        _idx_split_clf = None
                        if prop_test > 0:
                            try:
                                _idx_split_clf = mu.dividir_train_test(X_modelado, clases_activas, ids_activos, prop_test, True, semilla_clf, metodo_split)
                            except Exception:
                                _idx_split_clf = None
                        # Everything needed to RESUME is kept in this record and updated after every model.
                        st.session_state["clf_run"] = {
                            "completo": False, "firma": firma_modelos(_crop_mask_clf), "modelos": list(modelos_elegidos),
                            "cv_folds": cv_folds, "prop_test": prop_test, "metodo_split": metodo_split, "semilla": semilla_clf,
                            "metodo_split_sel": metodo_split_sel, "optimizar": optimizar, "metodo_opt": metodo_opt,
                            "metodo_seleccion": metodo_seleccion,
                            "boruta_iter": boruta_max_iter if metodo_seleccion == "Boruta" else None,
                            "boruta_alpha": boruta_alpha if metodo_seleccion == "Boruta" else None,
                            "ga_pob": ga_poblacion if metodo_seleccion == "Genetic Algorithm" else None,
                            "ga_gen": ga_generaciones if metodo_seleccion == "Genetic Algorithm" else None,
                            "idx_split": _idx_split_clf, "mascara": None, "seleccion_hecha": False,
                            "resultados": {}, "hp": {}, "desc": {},
                        }
                    _r = st.session_state["clf_run"]
                    resultados, hiperparametros_optimos, descripcion_opt_usada = _r["resultados"], _r["hp"], _r["desc"]
                    _idx_tr_clf = _idx_split_clf[0] if _idx_split_clf is not None else np.arange(len(clases_activas))

                    def _guardar_clf():
                        st.session_state["clf_resultados"] = resultados
                        st.session_state["clf_mascara_variables"] = mascara_variables
                        st.session_state["clf_descripcion_opt"] = descripcion_opt_usada
                        st.session_state["clf_ids_usados"] = ids_activos
                        st.session_state["clf_eje_usado"] = eje_modelado
                        st.session_state["clf_pasos_pretratamiento"] = st.session_state.pasos_pretratamiento
                        st.session_state["clf_hiperparametros"] = hiperparametros_optimos
                        st.session_state["clf_espectro_promedio"] = X_modelado.mean(axis=0)
                        st.session_state["clf_cv_folds"] = cv_folds
                        st.session_state["clf_prop_test"] = prop_test
                        st.session_state["clf_metodo_split"] = metodo_split_sel
                        st.session_state["clf_metodo_seleccion"] = metodo_seleccion
                        st.session_state["clf_n_muestras"] = X_modelado.shape[0]
                        st.session_state["clf_firma"] = firma_modelos(_crop_mask_clf)
                        st.session_state["clf_eje_completo"] = _eje_completo_clf
                        st.session_state["clf_crop_mask"] = _crop_mask_clf
                        st.session_state["clf_origen"] = "cropped" if _crop_mask_clf is not None else "full"
                        st.session_state["clf_crop_desc"] = st.session_state.get("crop_desc_aplicada")

                    mascara_variables = _r["mascara"]
                    _msg_espera = "This may take a few minutes..." if (optimizar or metodo_seleccion in ("Boruta", "Genetic Algorithm")) else "Training..."
                    with st.spinner(f"Selecting variables ({_msg_espera})" if (metodo_seleccion != "None" and not _r["seleccion_hecha"]) else _msg_espera):
                        if not _r["seleccion_hecha"]:
                            if metodo_seleccion == "Boruta":
                                try:
                                    mascara_variables = mu.seleccionar_variables_boruta(
                                        X_modelado[_idx_tr_clf], clases_activas[_idx_tr_clf], es_clasificacion=True,
                                        max_iter=boruta_max_iter, alpha=boruta_alpha,
                                    )
                                    if mascara_variables.sum() == 0:
                                        st.warning("Boruta did not select any variable; using all of them.")
                                        mascara_variables = None
                                except Exception as e:
                                    st.error(f"Boruta failed ({e}); using all variables.")
                            elif metodo_seleccion == "Genetic Algorithm":
                                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                                cv_ga = 3 if cv_folds == "LOO" else min(3, cv_folds)
                                ga = mu.SeleccionGenetica(
                                    LinearDiscriminantAnalysis(), es_clasificacion=True,
                                    tam_poblacion=ga_poblacion, n_generaciones=ga_generaciones,
                                    cv=cv_ga, random_state=0,
                                )
                                ga.fit(X_modelado[_idx_tr_clf], clases_activas[_idx_tr_clf])
                                mascara_variables = ga.mejor_mascara_
                            _r["mascara"], _r["seleccion_hecha"] = mascara_variables, True

                        X_sel = X_modelado[:, mascara_variables] if mascara_variables is not None else X_modelado
                        try:
                            st.session_state["clf_dominio"] = cf.ajustar_dominio(X_sel[_idx_tr_clf])
                        except Exception:
                            st.session_state["clf_dominio"] = None
                        catalogo = mu.crear_clasificadores()
                        for nombre in modelos_elegidos:
                            if nombre in resultados and "error" not in resultados[nombre]:
                                continue                      # finished in the interrupted run: skip it
                            modelo = catalogo[nombre]
                            try:
                                if optimizar and nombre in mu.GRILLAS_CLASIFICACION:
                                    modelo, mejores_params, _, desc_opt = mu.optimizar_hiperparametros(
                                        modelo, _grilla_activa(nombre, True, X_sel[_idx_tr_clf], clases_activas[_idx_tr_clf], 0.0, cv_folds),
                                        X_sel[_idx_tr_clf], clases_activas[_idx_tr_clf],
                                        es_clasificacion=True, cv=cv_folds, metodo=metodo_opt,
                                    )
                                    hiperparametros_optimos[nombre] = mejores_params
                                    descripcion_opt_usada[nombre] = desc_opt
                                st.session_state["clf_aug_desc"] = None
                                resultados[nombre] = mu.entrenar_evaluar_clasificacion(
                                    modelo, X_sel, clases_activas, ids=ids_activos,
                                    cv=cv_folds, proporcion_test=prop_test, metodo_split=metodo_split, random_state=semilla_clf,
                                    indices_split=_idx_split_clf,
                                )
                            except Exception as e:
                                resultados[nombre] = {"error": str(e)}
                            _guardar_clf()                    # saved right away: an interruption keeps what is done
                    _r["completo"] = True
                    _guardar_clf()
                    if _incompleto_clf:      # redraw: the 'incomplete run' banner above is now outdated
                        _rerun_tab()

                if "clf_resultados" in st.session_state:
                    resultados = st.session_state["clf_resultados"]
                    mascara_variables = st.session_state["clf_mascara_variables"]
                    hiperparametros_optimos = st.session_state.get("clf_hiperparametros", {})
                    descripcion_opt_usada = st.session_state.get("clf_descripcion_opt", {})

                    if mascara_variables is not None:
                        st.caption(f"Selected variables: {int(mascara_variables.sum())} of {len(mascara_variables)}.")
                        with st.expander("📍 View selected variables on the mean spectrum"):
                            fig_vars = go.Figure()
                            eje_r = st.session_state["clf_eje_usado"]
                            fig_vars.add_trace(go.Scatter(x=eje_r, y=st.session_state["clf_espectro_promedio"],
                                                           mode="lines", line=dict(color="#5F5E5A"), name="Mean spectrum"))
                            idx_sel = np.where(mascara_variables)[0]
                            bloques = np.split(idx_sel, np.where(np.diff(idx_sel) != 1)[0] + 1) if len(idx_sel) else []
                            for j, bloque in enumerate(bloques):
                                x0, x1 = eje_r[bloque[0]], eje_r[bloque[-1]]
                                fig_vars.add_vrect(x0=min(x0, x1), x1=max(x0, x1), fillcolor="#0F6E56",
                                                    opacity=0.25, line_width=0)
                            fig_vars.update_layout(height=380, xaxis_title="Wavenumber", yaxis_title="Signal")
                            if eje_r[0] > eje_r[-1]:
                                fig_vars.update_xaxes(autorange="reversed")
                            if not es_tabular():
                                _t_a, _m_a = _asig_pre(fig_vars, eje_r, asg.regiones_desde_mascara(eje_r, mascara_variables), "clf")
                                st.plotly_chart(fig_vars, width='stretch')
                                _asig_post(_t_a, _m_a, "clf")

                            st.markdown("**List of selected variables**")
                            df_vars_sel = df_variables_seleccionadas(eje_r, mascara_variables)
                            st.dataframe(df_vars_sel, width='stretch', height=200)
                            st.download_button(
                                "⬇️ Download selected variables (Excel)",
                                data=df_a_excel_bytes({"selected_variables": df_vars_sel}),
                                file_name="selected_variables_classification.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="descargar_vars_clf",
                            )

                    _cm1, _cm2, _cm3 = st.columns([1.5, 1.9, 3])
                    with _cm1:
                        gu.how_to_interpret("clf", valores=gu.valores_clf(), key="how_res_clf")
                    with _cm2:
                        gu.what_next("clf", key="next_res_clf")
                    filas = []
                    for nombre, res in resultados.items():
                        if "error" in res:
                            filas.append({"Model": nombre, "Error": res["error"]})
                            continue
                        fila = {
                            "Model": nombre,
                            "Accuracy (CV)": round(res["cv"]["accuracy"], 3),
                            "Bal. Accuracy (CV)": round(res["cv"]["balanced_accuracy"], 3),
                            "Sensitivity (CV)": round(res["cv"]["sensibilidad_macro"], 3),
                            "Specificity (CV)": round(res["cv"]["especificidad_macro"], 3),
                            "F1 macro (CV)": round(res["cv"]["f1_macro"], 3),
                            "Kappa (CV)": round(res["cv"]["kappa"], 3),
                            "MCC (CV)": round(res["cv"]["mcc"], 3),
                            "AUC (CV)": round(res["cv"]["auc"], 3) if res["cv"]["auc"] is not None else "n/a",
                        }
                        if "test" in res:
                            fila["Accuracy (test)"] = round(res["test"]["accuracy"], 3)
                            fila["Sensitivity (test)"] = round(res["test"]["sensibilidad_macro"], 3)
                            fila["Specificity (test)"] = round(res["test"]["especificidad_macro"], 3)
                            fila["Kappa (test)"] = round(res["test"]["kappa"], 3)
                            fila["MCC (test)"] = round(res["test"]["mcc"], 3)
                            fila["AUC (test)"] = round(res["test"]["auc"], 3) if res["test"]["auc"] is not None else "n/a"
                        filas.append(fila)
                    df_metricas_clf = pd.DataFrame(filas)
                    st.dataframe(df_metricas_clf, width='stretch')
                    st.download_button(
                        "⬇️ Download metrics table (Excel)",
                        data=df_a_excel_bytes({"classification_metrics": df_metricas_clf}),
                        file_name="classification_metrics.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="descargar_metricas_clf",
                    )

                    modelos_exitosos = [n for n in resultados if "error" not in resultados[n]]
                    if len(modelos_exitosos) >= 2:
                        hay_test_clf = "test" in resultados[modelos_exitosos[0]]
                        if _BASICO:
                            st.caption(T("🔒 Compare models statistically — advanced, greyed out in First steps mode."))  # LOCKED
                        else:
                            with st.expander("📊 Compare models statistically"):
                                if hay_test_clf:
                                    st.caption("Compares models on the held-out TEST set predictions, sample by "
                                               "sample (McNemar's test) — this is what can actually reveal "
                                               "overfitting, since two models can look equally good in "
                                               "cross-validation yet behave very differently on genuinely unseen "
                                               "data. A p-value below 0.05 means the difference is unlikely to be "
                                               "just chance.")
                                else:
                                    st.caption("No independent test set was configured, so this compares models "
                                               "using a paired t-test on their per-fold cross-validation scores "
                                               "instead (same folds for every model). Note this can't detect "
                                               "overfitting the way a held-out test set would — consider setting "
                                               "aside a test proportion above 0% for a more trustworthy comparison.")
                                if st.button("Run comparison", key="comparar_modelos_clf"):
                                    if hay_test_clf:
                                        pvalores, puntajes = mu.comparar_modelos_en_test(
                                            {n: resultados[n] for n in modelos_exitosos}, es_clasificacion=True,
                                        )
                                    else:
                                        mascara_cmp = st.session_state["clf_mascara_variables"]
                                        # Use the exact same train-only subset the main CV metrics used, so the
                                        # held-out test set (if any) never leaks into model comparison.
                                        ids_train_ref = resultados[modelos_exitosos[0]]["cv"]["ids"]
                                        mask_train_cmp = np.isin(ids_activos, ids_train_ref)
                                        X_cmp = X_modelado[mask_train_cmp]
                                        y_cmp = clases_activas[mask_train_cmp]
                                        if mascara_cmp is not None:
                                            X_cmp = X_cmp[:, mascara_cmp]
                                        modelos_para_comparar = {
                                            n: mu.clone(resultados[n]["modelo_final"]) for n in modelos_exitosos
                                        }
                                        with st.spinner("Running cross-validation for every model on the same folds..."):
                                            pvalores, puntajes = mu.comparar_modelos_estadisticamente(
                                                modelos_para_comparar, X_cmp, y_cmp, es_clasificacion=True,
                                                cv=st.session_state["clf_cv_folds"],
                                            )
                                    st.session_state["clf_pvalores"] = pvalores
                                    st.session_state["clf_puntajes_cv"] = puntajes
                                    st.session_state["clf_comparacion_metodo"] = (
                                        "McNemar's test on the held-out test set" if hay_test_clf
                                        else "Paired t-test on cross-validation folds (no test set configured)"
                                    )
                                if "clf_pvalores" in st.session_state:
                                    interpretacion = mu.interpretar_comparacion_modelos(
                                        st.session_state["clf_pvalores"], st.session_state["clf_puntajes_cv"],
                                    )
                                    if interpretacion["hay_diferencias"]:
                                        for msg in interpretacion["mensajes"]:
                                            st.success(msg)
                                    else:
                                        st.info(interpretacion["mensajes"][0])
                                    st.markdown("**p-values:**")
                                    st.dataframe(st.session_state["clf_pvalores"].style.format("{:.4f}"), width='stretch')

                    _m_rs = st.session_state.get("clf_mascara_variables")
                    _ui_remuestreo("clf", {n_: mu.clone(r_["modelo_final"]) for n_, r_ in resultados.items() if "error" not in r_},
                                   X_modelado[:, _m_rs] if _m_rs is not None else X_modelado, clases_activas, True)

                    nombre_detalle = st.selectbox(
                        "View details for", [n for n in resultados if "error" not in resultados[n]],
                        help="Choose which of the trained models to inspect below: confusion matrix, "
                             "hyperparameters, learning curve, and the report/save options.",
                    )
                    if nombre_detalle:
                        res = resultados[nombre_detalle]

                        if hiperparametros_optimos.get(nombre_detalle):
                            with st.container(border=True):
                                st.markdown(f"**⚙️ Optimized hyperparameters — {nombre_detalle}**")
                                cols_hp = st.columns(len(hiperparametros_optimos[nombre_detalle]))
                                for col_hp, (k, v) in zip(cols_hp, hiperparametros_optimos[nombre_detalle].items()):
                                    col_hp.metric(k, str(v))
                                st.caption(f"🔧 Method: {descripcion_opt_usada.get(nombre_detalle, 'n/a')}")

                        fuente = st.radio("Confusion matrix on:", ["CV", "Test"] if "test" in res else ["CV"],
                                           horizontal=True, key="fuente_matriz_clf",
                                           help="'CV' shows the cross-validation predictions; 'Test' shows "
                                                "the independent held-out test set, if one was configured.")
                        datos_matriz = res["cv"] if fuente == "CV" else res["test"]
                        clases_orden = res["cv"]["clases"]
                        if fuente == "Test":
                            y_t, y_p = res["test"]["y_true"], res["test"]["y_pred"]
                            matriz = mu.confusion_matrix(y_t, y_p, labels=clases_orden)
                        else:
                            matriz = res["cv"]["matriz_confusion"]

                        fig_cm = px.imshow(matriz, text_auto=True, x=list(clases_orden), y=list(clases_orden),
                                            labels=dict(x="Predicted", y="Actual", color="Count"),
                                            color_continuous_scale="Blues")
                        fig_cm.update_layout(height=420, title=f"Confusion matrix — {nombre_detalle} ({fuente})")
                        st.plotly_chart(fig_cm, width='stretch')
                        gu.nota("clf_cm", "confusion matrix", "clf")

                        ids_fuente = res["cv"]["ids"] if fuente == "CV" else res["test"]["ids"]
                        y_true_fuente = res["cv"]["y_true"] if fuente == "CV" else res["test"]["y_true"]
                        y_pred_fuente = res["cv"]["y_pred"] if fuente == "CV" else res["test"]["y_pred"]
                        df_exp = mu.exportar_predicciones_clasificacion(ids_fuente, y_true_fuente, y_pred_fuente)
                        st.download_button(
                            f"⬇️ Download predictions ({nombre_detalle}, {fuente})",
                            data=df_exp.to_csv(index=False).encode("utf-8"),
                            file_name=f"predictions_{nombre_detalle}_{fuente}.csv", mime="text/csv",
                        )

                        if _BASICO:
                            st.caption(T("🔒 Learning curve — advanced, greyed out in First steps mode."))  # LOCKED
                        else:
                            with st.expander(f"📈 Learning curve — {nombre_detalle}"):
                                st.caption("Shows performance vs. how many training samples were used. If both "
                                           "curves are still rising and far apart, more samples would likely help. "
                                           "If they're flat and close together, the model has plateaued.")
                                if st.button("Compute learning curve", key="curva_aprendizaje_clf"):
                                    mascara_lc = st.session_state["clf_mascara_variables"]
                                    X_lc = X_modelado[:, mascara_lc] if mascara_lc is not None else X_modelado
                                    with st.spinner("Training with increasing sample sizes (this may take a few minutes)..."):
                                        curva = mu.calcular_curva_aprendizaje(
                                            mu.clone(res["modelo_final"]), X_lc, clases_activas, es_clasificacion=True,
                                            cv=min(5, st.session_state["clf_cv_folds"]) if st.session_state["clf_cv_folds"] != "LOO" else 5,
                                        )
                                    st.session_state["clf_curva_aprendizaje"] = curva
                                    st.session_state["clf_curva_modelo"] = nombre_detalle
                                if st.session_state.get("clf_curva_modelo") == nombre_detalle and "clf_curva_aprendizaje" in st.session_state:
                                    curva = st.session_state["clf_curva_aprendizaje"]
                                    fig_lc = go.Figure()
                                    fig_lc.add_trace(go.Scatter(x=curva["train_sizes"], y=curva["train_scores_mean"],
                                                                 mode="lines+markers", name="Training score",
                                                                 line=dict(color="#185FA5")))
                                    fig_lc.add_trace(go.Scatter(x=curva["train_sizes"], y=curva["val_scores_mean"],
                                                                 mode="lines+markers", name="Cross-validation score",
                                                                 line=dict(color="#B91C1C")))
                                    fig_lc.update_layout(height=380, xaxis_title="Training set size (samples)",
                                                          yaxis_title=curva["scoring"])
                                    st.plotly_chart(fig_lc, width='stretch')
                                    st.info(_diagnostico_curva(curva))


                        st.divider()
                        st.markdown("**📄 Report for this analysis**")
                        st.caption("Includes: dataset used, spectra, preprocessing, selected variables, "
                                   "a comparison table of all trained models, and the detail "
                                   f"({', '.join([nombre_detalle])}) with confusion matrix and hyperparameters.")
                        if st.button("🖨️ Generate report (PDF)", key="generar_reporte_clf"):
                            secciones_rep = [
                                {"tipo": "titulo", "texto": "1. Dataset"},
                                {"tipo": "clave_valor", "pares": [
                                    ("Samples used", int(X_modelado.shape[0])),
                                    ("Original variables", int(X_modelado.shape[1])),
                                    ("Class distribution", ", ".join(
                                        f"{c}: {n}" for c, n in pd.Series(clases_activas).value_counts().items())),
                                    ("Preprocessing", " -> ".join(
                                        p[0] for p in st.session_state["clf_pasos_pretratamiento"]) or "none"),
                                ]},
                                {"tipo": "imagen", "fig": ru.fig_espectros(
                                    eje_modelado, X_modelado, clases=como_texto(clases_activas), titulo="Preprocessed spectra")},
                                {"tipo": "titulo", "texto": "2. Variable selection"},
                                {"tipo": "clave_valor", "pares": [
                                    ("Method", st.session_state["clf_metodo_seleccion"]),
                                    ("Selected variables", int(mascara_variables.sum()) if mascara_variables is not None
                                     else f"all ({X_modelado.shape[1]})"),
                                ]},
                            ]
                            if mascara_variables is not None:
                                secciones_rep.append({"tipo": "imagen", "fig": ru.fig_variables_seleccionadas(
                                    eje_modelado, X_modelado.mean(axis=0), mascara_variables)})
                                idx_sel_rep = np.where(mascara_variables)[0]
                                secciones_rep.append({"tipo": "tabla",
                                    "encabezados": ["Position", "Wavenumber"],
                                    "filas": [[str(i), f"{eje_modelado[i]:.2f}"] for i in idx_sel_rep[:60]],
                                    "anchos": [30, 70]})
                                if len(idx_sel_rep) > 60:
                                    secciones_rep.append({"tipo": "parrafo",
                                        "texto": f"(showing the first 60 of {len(idx_sel_rep)} variables — "
                                                 "the full list is in the Excel file downloadable from the app)"})
                            secciones_rep += [
                                {"tipo": "salto_pagina"},
                                {"tipo": "titulo", "texto": "3. Model comparison"},
                                {"tipo": "clave_valor", "pares": [
                                    ("Cross-validation", f"{res['cv']['cv_folds_usados']}-fold" if cv_folds != "LOO"
                                     else f"Leave-One-Out ({res['cv']['cv_folds_usados']} repetitions)"),
                                    ("Test proportion", f"{int(prop_test * 100)}% ({metodo_split_sel}, seed {semilla_clf})" if metodo_split != "kennard_stone" else f"{int(prop_test * 100)}% ({metodo_split_sel})"),
                                ]},
                                {"tipo": "tabla",
                                 "encabezados": list(df_metricas_clf.columns),
                                 "filas": df_metricas_clf.astype(str).values.tolist()},
                                {"tipo": "titulo", "texto": f"4. Detail: {nombre_detalle}"},
                                {"tipo": "imagen", "fig": ru.fig_matriz_confusion(matriz, list(clases_orden))},
                            ]
                            if hiperparametros_optimos.get(nombre_detalle):
                                secciones_rep.append({"tipo": "subtitulo", "texto": "Optimal hyperparameters"})
                                secciones_rep.append({"tipo": "clave_valor",
                                    "pares": [("Optimization method", descripcion_opt_usada.get(nombre_detalle, "n/a"))]
                                    + list(hiperparametros_optimos[nombre_detalle].items())})
                            if st.session_state.get("clf_curva_modelo") == nombre_detalle and "clf_curva_aprendizaje" in st.session_state:
                                secciones_rep.append({"tipo": "subtitulo", "texto": f"Learning curve — {nombre_detalle}"})
                                secciones_rep.append({"tipo": "imagen",
                                    "fig": ru.fig_curva_aprendizaje(st.session_state["clf_curva_aprendizaje"], nombre_detalle)})
                            if "clf_pvalores" in st.session_state:
                                secciones_rep.append({"tipo": "salto_pagina"})
                                secciones_rep.append({"tipo": "titulo", "texto": "5. Statistical comparison between models"})
                                secciones_rep.append({"tipo": "parrafo",
                                    "texto": f"Method: {st.session_state.get('clf_comparacion_metodo', 'n/a')}. "
                                             "A p-value below 0.05 suggests the performance difference between "
                                             "that pair of models is unlikely to be due to chance alone."})
                                interpretacion_rep = mu.interpretar_comparacion_modelos(
                                    st.session_state["clf_pvalores"], st.session_state["clf_puntajes_cv"],
                                )
                                for msg in interpretacion_rep["mensajes"]:
                                    secciones_rep.append({"tipo": "parrafo", "texto": msg.replace("**", "")})
                                secciones_rep.append({"tipo": "imagen",
                                    "fig": ru.fig_comparacion_pvalores(st.session_state["clf_pvalores"])})

                            try:
                                secciones_rep += inf._b_remuestreo(st.session_state, "clf")
                            except Exception:
                                pass
                            pdf_reporte_clf = _generar_reporte(
                                "Classification Report", f"Models: {', '.join(modelos_elegidos)}", secciones_rep,
                            )
                            st.download_button(
                                "⬇️ Download full report (PDF)",
                                data=bytes(pdf_reporte_clf.output()),
                                file_name="classification_report.pdf", mime="application/pdf",
                                key="descargar_reporte_clf",
                            )
                        st.divider()

                        nombre_guardado = st.text_input("Name to save this model as", value=nombre_detalle, help="Name under which the model is kept in this session (and shown in Prediction).")
                        if st.button("💾 Save this model for the Prediction tab"):
                            id_trazabilidad = mu.generar_id_trazabilidad()
                            if clases_activas is not None:
                                desc_y = ", ".join(f"{c}: {n}" for c, n in pd.Series(clases_activas).value_counts().items())
                            else:
                                desc_y = "n/a"
                            cv_desc = (f"Leave-One-Out ({res['cv']['cv_folds_usados']} repetitions)"
                                       if cv_folds == "LOO" else f"{res['cv']['cv_folds_usados']}-fold")
                            metricas_ficha = {
                                "Accuracy (CV)": round(res["cv"]["accuracy"], 3),
                                "Balanced Accuracy (CV)": round(res["cv"]["balanced_accuracy"], 3),
                                "Sensitivity (CV)": round(res["cv"]["sensibilidad_macro"], 3),
                                "Specificity (CV)": round(res["cv"]["especificidad_macro"], 3),
                                "F1 macro (CV)": round(res["cv"]["f1_macro"], 3),
                                "Kappa (CV)": round(res["cv"]["kappa"], 3),
                                "MCC (CV)": round(res["cv"]["mcc"], 3),
                                "AUC (CV)": round(res["cv"]["auc"], 3) if res["cv"]["auc"] is not None else "n/a",
                            }
                            if "test" in res:
                                metricas_ficha.update({
                                    "Accuracy (test)": round(res["test"]["accuracy"], 3),
                                    "Sensitivity (test)": round(res["test"]["sensibilidad_macro"], 3),
                                    "Specificity (test)": round(res["test"]["especificidad_macro"], 3),
                                    "Kappa (test)": round(res["test"]["kappa"], 3),
                                    "MCC (test)": round(res["test"]["mcc"], 3),
                                    "AUC (test)": round(res["test"]["auc"], 3) if res["test"]["auc"] is not None else "n/a",
                                })
                            ficha = {
                                "id_trazabilidad": id_trazabilidad,
                                "fecha_creacion": _fecha_hora_informe(con_segundos=True),
                                "nombre_modelo": nombre_guardado,
                                "tipo": "classification",
                                "algoritmo": nombre_detalle,
                                "n_muestras": int(X_modelado.shape[0]),
                                "n_variables_totales": int(X_modelado.shape[1]),
                                "n_variables_usadas": int(mascara_variables.sum()) if mascara_variables is not None else int(X_modelado.shape[1]),
                                "descripcion_y": desc_y,
                                "pretratamiento_desc": desc_con_recorte(" -> ".join(p[0] for p in st.session_state["clf_pasos_pretratamiento"]) or "none", "clf"),
                                "seleccion_variables_desc": st.session_state["clf_metodo_seleccion"],
                                "hiperparametros": hiperparametros_optimos.get(nombre_detalle, {}),
                                "metodo_optimizacion": descripcion_opt_usada.get(nombre_detalle, "none"),
                                "aumento_datos": st.session_state.get("clf_aug_desc") or "none",
                                "cv_desc": cv_desc,
                                "prop_test_desc": f"{int(st.session_state['clf_prop_test'] * 100)}% ({st.session_state.get('clf_metodo_split', 'Random')})",
                                "metricas": metricas_ficha,
                                "entorno_software": mu.info_entorno_software(),
                            }
                            st.session_state.modelos_guardados[nombre_guardado] = {
                                "tipo": "classification",
                                "modelo": res["modelo_final"],
                                "mascara_variables": eje_y_mascara_para_guardar("clf", mascara_variables)[1],
                                "numeros_onda": eje_y_mascara_para_guardar("clf", mascara_variables)[0],
                                "pasos_pretratamiento": st.session_state["clf_pasos_pretratamiento"], "tipo_senal": st.session_state.get("pret_tipo_aplicado", "NIR / MIR"),
                                **_extras_tabular(),
                                **_extras_confianza("clf", res["cv"]),
                                "ficha": ficha,
                            }
                            st.session_state["clf_ultima_ficha"] = ficha
                            st.success(f"Model '{nombre_guardado}' saved (traceability ID: {id_trazabilidad}). "
                                       "It is now available in the Prediction tab.")

                        if st.session_state.get("clf_ultima_ficha", {}).get("nombre_modelo") == nombre_guardado:
                            pdf_ficha = _generar_pdf_ficha(
                                st.session_state["clf_ultima_ficha"],
                                fig_extra=ru.fig_matriz_confusion(matriz, list(clases_orden)),
                                titulo_fig_extra="Confusion matrix",
                            )
                            st.download_button(
                                "📄 Download model card (PDF)",
                                data=bytes(pdf_ficha.output()),
                                file_name=f"model_card_{nombre_guardado}.pdf", mime="application/pdf",
                            )
    with tabs[12]:
        _frag_tab_9()


# TAB: SIMCA
# -----------------------------------------------------------------------
if _abierta(tabs[13]):
    # Each tab re-runs ON ITS OWN when one of its widgets changes (fragment), instead of
    # re-executing the whole app. Buttons that must refresh other parts call st.rerun().
    @st.fragment
    def _frag_tab_10():
        st.subheader("SIMCA — Soft Independent Modeling of Class Analogies")
        gu.barra_guia("simca", "simca", valores=gu.valores_simca())
        if st.button("🔄 Reset this tab", key="reset_simca",
                     help="Clears all trained SIMCA models and results from this tab, so you can start "
                          "a completely fresh run without any leftover results from before."):
            resetear_prefijo("simca_")
            _rerun_tab()
        st.caption("Unlike LDA/PLS-DA/Random Forest (which always force a pick among the trained "
                   "classes), SIMCA builds one PCA model PER CLASS and asks, independently for each "
                   "one, 'does this sample look like a member of this class?'. A sample can end up "
                   "accepted by exactly one class (a clean call), by several (an ambiguous / "
                   "overlapping region), or by none (a genuine outlier — possibly a class the model "
                   "has never seen). This makes it a natural fit for authenticity / quality-control "
                   "questions such as 'is this really wine type B, or something else entirely?'.")

        ids_simca, _, clases_simca = datos_activos()
        X_modelado_simca = st.session_state.X_pret[indice_activo()]
        eje_modelado_simca = st.session_state.numeros_onda_pret
        X_modelado_simca, eje_modelado_simca, _crop_mask_simca, _eje_completo_simca = elegir_espectro_modelado(X_modelado_simca, eje_modelado_simca, "simca")

        if clases_simca is None:
            st.warning("Define classes in the left-hand panel (Data tab) to build SIMCA models.")
        else:
            conteos_simca = pd.Series(clases_simca).value_counts()
            if conteos_simca.min() < 4:
                st.error(f"Some class has fewer than 4 samples ({conteos_simca.to_dict()}). SIMCA needs "
                         "a handful of samples per class to fit a meaningful class-specific PCA model.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    varianza_objetivo_simca = st.slider(
                        "Target variance per class model", 0.80, 0.99, 0.90, step=0.01,
                        help="Each class gets its own PCA with just enough components to reach this "
                             "cumulative explained variance — capped at roughly a third of that class's "
                             "calibration samples, so the model doesn't run out of residual degrees of "
                             "freedom (which would make it fit calibration noise instead of real "
                             "structure, and fail to generalize to new samples). To DETECT MORE ADULTERATED "
                             "samples use a higher value (0.95–0.99): a stricter model, but more false alarms.",
                    key="ac_simca_var")
                    regla_simca = st.radio(
                        "Acceptance rule", ["Classical (T² and Q limits)", "Data-driven (DD-SIMCA)"], key="regla_simca",
                        horizontal=True,
                        help="Classical: a sample is accepted if BOTH its T² and its Q are below their own limits (a box). "
                             "DD-SIMCA (the modern standard for authenticity / adulteration): the two distances are "
                             "combined into one, and its limit is estimated from the data, giving a diagonal boundary "
                             "and a better-controlled false-alarm rate. Try it when classical SIMCA over- or under-rejects.")
                    alpha_simca = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, step=0.01,
                                             key="alpha_simca",
                                             help="Controls how strict each class boundary is. Smaller "
                                                  "alpha (e.g. 0.01) = a wider, more permissive boundary; "
                                                  "larger alpha (e.g. 0.10) = a tighter one that rejects "
                                                  "more borderline samples. 0.05 is the usual default. To DETECT MORE "
                                                  "ADULTERATED samples raise it (0.10+): fewer adulterated samples "
                                                  "slip through, but more genuine ones are rejected (false alarms).")
                with col2:
                    prop_test_simca = st.slider(
                        "Proportion for independent test set (0 = validate on calibration only)",
                        0.0, 0.4, 0.3, step=0.05, key="test_simca",
                        help="Fraction of each class's samples set aside to check whether the class "
                             "model actually accepts genuinely new samples of that class — leaving this "
                             "at 0 only checks how well the model fits the same data it was built from, "
                             "which is optimistic.",
                    )
                    metodo_split_sel_simca, metodo_split_simca, semilla_simca = _ui_split(True, prop_test_simca, "split_simca")

                _run_simca = st.session_state.get("simca_run")
                _incompleto_simca = bool(_run_simca) and not _run_simca.get("completo", True)
                _valida_simca = _incompleto_simca and _run_simca.get("firma") == firma_modelos(_crop_mask_simca)
                if _incompleto_simca:
                    st.warning(f"⏸ Incomplete run: {len(_run_simca['modelos']) + len(_run_simca['errores'])} of {len(_run_simca['clases'])} "
                               "class models finished (the run was interrupted). "
                               + ("Click **Resume** to continue with the same settings, skipping the finished classes."
                                  if _valida_simca else "The data, preprocessing or crop changed since then, so it can't be resumed — train again."))
                _cs1, _cs2, _ = st.columns([1.9, 1.6, 3])
                _ent_simca = _cs1.button("🚀 Train SIMCA models (one per class)")
                _rean_simca = _cs2.button("⏩ Resume interrupted run", key="simca_btn_resume", disabled=not _valida_simca,
                                          help="Continues the interrupted run with the SAME settings, skipping the classes already fitted.")
                if _ent_simca or _rean_simca:
                    if _rean_simca:
                        _rs = _run_simca
                        prop_test_simca = _rs["prop_test"]; metodo_split_sel_simca = _rs["metodo_split_sel"]; semilla_simca = _rs.get("semilla", 0)
                        varianza_objetivo_simca = _rs["varianza"]; alpha_simca = _rs["alpha"]
                        regla_simca = _rs.get("regla", "Classical (T² and Q limits)")
                        idx_train_s, idx_test_s = _rs["idx_train"], _rs["idx_test"]
                        clases_unicas_simca = np.array(_rs["clases"])
                    else:
                        clases_unicas_simca = np.unique(clases_simca)
                        if prop_test_simca > 0:
                            idx_train_s, idx_test_s = mu.dividir_train_test(
                                X_modelado_simca, clases_simca, ids_simca, prop_test_simca,
                                es_clasificacion=True, random_state=semilla_simca, metodo_split=metodo_split_simca,
                            )
                        else:
                            idx_train_s, idx_test_s = np.arange(len(clases_simca)), np.arange(len(clases_simca))
                        st.session_state["simca_run"] = {
                            "completo": False, "firma": firma_modelos(_crop_mask_simca), "clases": list(clases_unicas_simca),
                            "idx_train": idx_train_s, "idx_test": idx_test_s, "prop_test": prop_test_simca,
                            "metodo_split_sel": metodo_split_sel_simca, "semilla": semilla_simca, "varianza": varianza_objetivo_simca,
                            "alpha": alpha_simca, "regla": regla_simca, "modelos": {}, "errores": {},
                        }
                    _rs = st.session_state["simca_run"]
                    modelos_simca, errores_simca = _rs["modelos"], _rs["errores"]
                    with st.spinner("Fitting one PCA model per class (this may take a few minutes)..."):
                        for c in clases_unicas_simca:
                            if c in modelos_simca or c in errores_simca:
                                continue                      # finished in the interrupted run: skip it
                            mask_c_train = (clases_simca[idx_train_s] == c)
                            X_c = X_modelado_simca[idx_train_s][mask_c_train]
                            try:
                                modelos_simca[c] = cu.entrenar_modelo_simca(
                                    X_c, varianza_objetivo=varianza_objetivo_simca, alpha=alpha_simca,
                                    regla="dd" if str(regla_simca).startswith("Data") else "clasica",
                                )
                            except Exception as e:
                                errores_simca[c] = str(e)

                    # Evaluate every TEST sample (regardless of its true class) against every class model
                    X_eval = X_modelado_simca[idx_test_s]
                    clases_eval_true = clases_simca[idx_test_s]
                    ids_eval = ids_simca[idx_test_s]
                    resultados_por_clase = {}
                    matriz_dentro = pd.DataFrame(index=ids_eval, columns=list(modelos_simca.keys()), dtype=bool)
                    for c, modelo_c in modelos_simca.items():
                        T2_c, Q_c, dentro_c = cu.evaluar_muestras_simca(X_eval, modelo_c)
                        resultados_por_clase[c] = {"T2": T2_c, "Q": Q_c, "dentro": dentro_c}
                        matriz_dentro[c] = dentro_c

                    n_clases_aceptado = matriz_dentro.sum(axis=1)
                    st.session_state["simca_modelos"] = modelos_simca
                    st.session_state["simca_errores"] = errores_simca
                    st.session_state["simca_resultados_por_clase"] = resultados_por_clase
                    st.session_state["simca_matriz_dentro"] = matriz_dentro
                    st.session_state["simca_clases_eval_true"] = clases_eval_true
                    st.session_state["simca_ids_eval"] = ids_eval
                    st.session_state["simca_n_clases_aceptado"] = n_clases_aceptado
                    st.session_state["simca_eje_usado"] = eje_modelado_simca
                    st.session_state["simca_pasos_pretratamiento"] = st.session_state.pasos_pretratamiento
                    st.session_state["simca_prop_test"] = prop_test_simca
                    st.session_state["simca_metodo_split"] = metodo_split_sel_simca
                    st.session_state["simca_alpha"] = alpha_simca
                    st.session_state["simca_varianza_objetivo"] = varianza_objetivo_simca
                    st.session_state["simca_n_muestras"] = int(X_modelado_simca.shape[0])
                    st.session_state["simca_validacion_es_calibracion"] = (prop_test_simca == 0)
                    st.session_state["simca_eje_completo"] = _eje_completo_simca
                    st.session_state["simca_crop_mask"] = _crop_mask_simca
                    st.session_state["simca_origen"] = "cropped" if _crop_mask_simca is not None else "full"
                    st.session_state["simca_crop_desc"] = st.session_state.get("crop_desc_aplicada")
                    _rs["completo"] = True
                    if _incompleto_simca:      # redraw: the 'incomplete run' banner above is now outdated
                        _rerun_tab()

                if "simca_modelos" in st.session_state:
                    modelos_simca = st.session_state["simca_modelos"]
                    errores_simca = st.session_state["simca_errores"]
                    resultados_por_clase = st.session_state["simca_resultados_por_clase"]
                    matriz_dentro = st.session_state["simca_matriz_dentro"]
                    clases_eval_true = st.session_state["simca_clases_eval_true"]
                    n_clases_aceptado = st.session_state["simca_n_clases_aceptado"]

                    if errores_simca:
                        for c, err in errores_simca.items():
                            st.error(f"Class '{c}': {err}")

                    clases_limitadas = [c for c, m in modelos_simca.items() if m.get("limitado_por_muestras")]
                    if clases_limitadas:
                        st.info(f"ℹ️ For class(es) {', '.join(clases_limitadas)}, the target variance could not "
                                "be reached without using too many components for the available calibration "
                                "samples — the number of components was capped for stability. Consider adding "
                                "more calibration samples for these classes, or lowering the target variance.")

                    if st.session_state["simca_validacion_es_calibracion"]:
                        st.caption("⚠ No independent test set was used — the stats below are computed on "
                                   "the same samples used to fit each model (optimistic; only a rough check).")

                    _sm1, _sm2, _sm3 = st.columns([1.5, 1.9, 3])
                    with _sm1:
                        gu.how_to_interpret("simca", valores=gu.valores_simca(), key="how_res_simca")
                    with _sm2:
                        gu.what_next("simca", key="next_res_simca")
                    st.markdown("**Per-class model summary**")
                    filas_simca = []
                    for c, modelo_c in modelos_simca.items():
                        es_miembro = (clases_eval_true == c)
                        dentro_c = matriz_dentro[c].to_numpy()
                        sensibilidad = dentro_c[es_miembro].mean() if es_miembro.any() else np.nan
                        especificidad = (~dentro_c[~es_miembro]).mean() if (~es_miembro).any() else np.nan
                        filas_simca.append({
                            "Class": c,
                            "Components": modelo_c["n_comp"],
                            "Variance explained": f"{modelo_c['varianza_explicada_pct']:.1f}%",
                            "Calibration samples": modelo_c["n_muestras_calibracion"],
                            "Genuine samples accepted": round(sensibilidad, 3) if pd.notna(sensibilidad) else "n/a",
                            "Adulterated / other-class detected": round(especificidad, 3) if pd.notna(especificidad) else "n/a",
                            "False alarms (genuine rejected)": round(1 - sensibilidad, 3) if pd.notna(sensibilidad) else "n/a",
                            "Missed (adulterated accepted)": round(1 - especificidad, 3) if pd.notna(especificidad) else "n/a",
                        })
                    df_simca_resumen = pd.DataFrame(filas_simca)
                    st.dataframe(df_simca_resumen, width='stretch')
                    st.caption("Read per class, taking that class as the GENUINE product and everything else as "
                               "adulterated / other: **Adulterated / other-class detected** is the detection rate "
                               "(you want it high); **False alarms** are genuine samples wrongly rejected; "
                               "**Missed** are adulterated samples wrongly accepted as genuine. "
                               "(Genuine accepted = classical SIMCA sensitivity; adulterated detected = specificity.)")
                    with st.expander("🎯 How to tune for detecting adulteration"):
                        st.markdown(
                            "- **To catch more adulterated samples (lower 'Missed')**: raise the significance level "
                            "(alpha 0.10 or more) and/or raise the target variance (0.95–0.99). The class boundary "
                            "gets tighter, so fewer foreign samples slip in — at the price of more false alarms.\n"
                            "- **To reduce false alarms on genuine samples**: lower alpha (0.01–0.02) and/or lower the "
                            "target variance (0.85–0.90). Missed adulterated samples will go up.\n"
                            "- Beyond these two knobs, the preprocessing (e.g. SNV + derivative) and cropping the "
                            "spectrum to the informative region usually matter more. Remove outliers from the "
                            "genuine class first, or the boundary widens and lets adulterated samples through.\n"
                            "- Always judge on an **independent test set** that includes adulterated samples, "
                            "changing one knob at a time.")
                    st.download_button(
                        "⬇️ Download SIMCA summary (Excel)",
                        data=df_a_excel_bytes({"simca_summary": df_simca_resumen}),
                        file_name="simca_summary.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="descargar_simca_excel",
                    )

                    st.markdown("**Sample assignment overview**")
                    n_ambiguo = int((n_clases_aceptado > 1).sum())
                    n_afuera = int((n_clases_aceptado == 0).sum())
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Accepted by exactly 1 class", int((n_clases_aceptado == 1).sum()))
                    c2.metric("Accepted by 2+ classes (ambiguous)", n_ambiguo)
                    c3.metric("Rejected by all classes (outlier)", n_afuera)

                    clase_detalle_simca = st.selectbox(
                        "View T² / Q plot for class", list(modelos_simca.keys()),
                        help="Shows the influence plot for that class's model — samples inside the "
                             "dashed box are accepted as members of this class.",
                    )
                    if clase_detalle_simca:
                        modelo_c = modelos_simca[clase_detalle_simca]
                        res_c = resultados_por_clase[clase_detalle_simca]
                        T2_lo, T2_hi = cu.rango_con_margen(res_c["T2"], modelo_c["T2_lim"])
                        Q_lo, Q_hi = cu.rango_con_margen(res_c["Q"], modelo_c["Q_lim"])
                        fig_simca = px.scatter(
                            x=res_c["T2"], y=res_c["Q"], hover_name=st.session_state["simca_ids_eval"],
                            color=como_texto(clases_eval_true), color_discrete_sequence=CLASS_PALETTE,
                            labels={"x": "T² (distance within class model)", "y": "Q (distance to class model)",
                                    "color": "True class"},
                        )
                        if modelo_c.get("dd") is not None:
                            _dd = modelo_c["dd"]
                            _xx = np.linspace(0, _dd["crit"] * _dd["h0"] / _dd["Nh"], 100)
                            _yy = (_dd["crit"] - _dd["Nh"] * _xx / _dd["h0"]) * _dd["q0"] / _dd["Nq"]
                            fig_simca.add_trace(go.Scatter(x=_xx, y=_yy, mode="lines", name="DD-SIMCA limit",
                                                           line=dict(color="red", dash="dash")))
                        else:
                            fig_simca.add_vline(x=modelo_c["T2_lim"], line_dash="dash", line_color="red")
                            fig_simca.add_hline(y=modelo_c["Q_lim"], line_dash="dash", line_color="red")
                        fig_simca.update_xaxes(range=[T2_lo, T2_hi])
                        fig_simca.update_yaxes(range=[Q_lo, Q_hi])
                        fig_simca.update_layout(height=480, title=f"SIMCA model — class '{clase_detalle_simca}' "
                                                                   + ("(samples below the dashed line are accepted)" if modelo_c.get("dd") is not None
                                                                   else "(samples inside the dashed box are accepted)"))
                        st.plotly_chart(fig_simca, width='stretch')
                        gu.nota("simca_plot", "SIMCA class model plot", "simca")

                    st.divider()
                    st.markdown("**📄 Report for this SIMCA analysis**")
                    if st.button("🖨️ Generate report (PDF)", key="generar_reporte_simca"):
                        secciones_simca = [
                            {"tipo": "titulo", "texto": "1. Dataset"},
                            {"tipo": "clave_valor", "pares": [
                                ("Samples used", st.session_state["simca_n_muestras"]),
                                ("Classes modeled", ", ".join(modelos_simca.keys())),
                                ("Preprocessing", " -> ".join(
                                    p[0] for p in st.session_state["simca_pasos_pretratamiento"]) or "none"),
                                ("Target variance per class", f"{st.session_state['simca_varianza_objetivo']*100:.0f}%"),
                                ("Significance level (alpha)", st.session_state["simca_alpha"]),
                                ("Test proportion", f"{int(st.session_state['simca_prop_test']*100)}% "
                                 f"({st.session_state['simca_metodo_split']})"),
                            ]},
                            {"tipo": "titulo", "texto": "2. Per-class model summary"},
                            {"tipo": "tabla", "encabezados": list(df_simca_resumen.columns),
                             "filas": df_simca_resumen.astype(str).values.tolist()},
                            {"tipo": "clave_valor", "pares": [
                                ("Accepted by exactly 1 class", int((n_clases_aceptado == 1).sum())),
                                ("Accepted by 2+ classes (ambiguous)", n_ambiguo),
                                ("Rejected by all classes (outlier)", n_afuera),
                            ]},
                        ]
                        for c, modelo_c in modelos_simca.items():
                            res_c = resultados_por_clase[c]
                            T2_lo, T2_hi = cu.rango_con_margen(res_c["T2"], modelo_c["T2_lim"])
                            Q_lo, Q_hi = cu.rango_con_margen(res_c["Q"], modelo_c["Q_lim"])
                            secciones_simca.append({"tipo": "salto_pagina"})
                            secciones_simca.append({"tipo": "subtitulo", "texto": f"Class '{c}'"})
                            secciones_simca.append({"tipo": "imagen", "fig": ru.fig_outliers(
                                res_c["T2"], res_c["Q"], modelo_c["T2_lim"], modelo_c["Q_lim"],
                                T2_lo, T2_hi, Q_lo, Q_hi, clases=como_texto(clases_eval_true))})
                        pdf_simca = _generar_reporte(
                            "SIMCA Report", f"Classes: {', '.join(modelos_simca.keys())}", secciones_simca,
                        )
                        st.download_button(
                            "⬇️ Download full report (PDF)",
                            data=bytes(pdf_simca.output()),
                            file_name="simca_report.pdf", mime="application/pdf",
                            key="descargar_reporte_simca",
                        )
                    st.divider()

                    nombre_guardado_simca = st.text_input("Name to save this SIMCA model set as", value="SIMCA", help="Name under which this set of class models is kept in the session.")
                    if st.button("💾 Save these SIMCA models for the Prediction tab"):
                        id_trazabilidad_simca = mu.generar_id_trazabilidad()
                        ficha_simca = {
                            "id_trazabilidad": id_trazabilidad_simca,
                            "fecha_creacion": _fecha_hora_informe(con_segundos=True),
                            "nombre_modelo": nombre_guardado_simca,
                            "tipo": "simca",
                            "algoritmo": "SIMCA",
                            "n_muestras": st.session_state["simca_n_muestras"],
                            "n_variables_totales": int(X_modelado_simca.shape[1]),
                            "n_variables_usadas": int(X_modelado_simca.shape[1]),
                            "descripcion_y": ", ".join(f"{c}: {int(n)}" for c, n in conteos_simca.items()),
                            "pretratamiento_desc": " -> ".join(
                                p[0] for p in st.session_state["simca_pasos_pretratamiento"]) or "none",
                            "seleccion_variables_desc": "none (SIMCA uses full spectrum per class)",
                            "hiperparametros": {"target_variance": st.session_state["simca_varianza_objetivo"],
                                                "alpha": st.session_state["simca_alpha"]},
                            "cv_desc": "n/a (class-modeling, not cross-validated the same way)",
                            "prop_test_desc": f"{int(st.session_state['simca_prop_test']*100)}% "
                                              f"({st.session_state['simca_metodo_split']})",
                            "metricas": {f"Genuine samples accepted ({row['Class']})": row["Genuine samples accepted"]
                                         for row in filas_simca}
                                        | {f"Adulterated / other-class detected ({row['Class']})": row["Adulterated / other-class detected"]
                                           for row in filas_simca},
                            "entorno_software": mu.info_entorno_software(),
                        }
                        st.session_state.modelos_guardados[nombre_guardado_simca] = {
                            "tipo": "simca",
                            "modelo": modelos_simca,
                            "mascara_variables": eje_y_mascara_para_guardar("simca", None)[1],
                            "numeros_onda": eje_y_mascara_para_guardar("simca", None)[0],
                            "pasos_pretratamiento": st.session_state["simca_pasos_pretratamiento"], "tipo_senal": st.session_state.get("pret_tipo_aplicado", "NIR / MIR"),
                            **_extras_tabular(),
                            "ficha": ficha_simca,
                        }
                        st.success(f"SIMCA model set '{nombre_guardado_simca}' saved (traceability ID: "
                                   f"{id_trazabilidad_simca}). It is now available in the Prediction tab.")
    with tabs[13]:
        _frag_tab_10()


# TAB: REGRESSION
# -----------------------------------------------------------------------
if _abierta(tabs[14]):
    # Each tab re-runs ON ITS OWN when one of its widgets changes (fragment), instead of
    # re-executing the whole app. Buttons that must refresh other parts call st.rerun().
    @st.fragment
    def _frag_tab_11():
        st.subheader("Supervised regression")
        st.caption("**Model refinement.** Use this tab to refine the candidates that came out of the 🏁 Model screening: "
                   "train them with the settings you prefer (test set, cross-validation, variable selection, hyperparameter "
                   "optimization), inspect the details, and **save the model you will use to predict new samples** in the 🔮 Prediction tab.")
        gu.barra_guia("reg", "reg", valores=gu.valores_reg(), pros=("reg_algo", None))
        if st.button("🔄 Reset this tab", key="reset_reg",
                     help="Clears all trained models, metrics, and plots from this tab, so you can "
                          "start a completely fresh run without any leftover results from before."):
            resetear_prefijo("reg_")
            _rerun_tab()

        if st.session_state.valores_y is None:
            st.markdown("**Reference value (continuous Y variable)**")
            st.caption("Not loaded yet. Easiest: go back to the sidebar's 'Reference value column' "
                       "selector when (re-)loading your data file. Or define it here instead:")
            modo_y = st.radio("How do you want to load the reference values?",
                               ["Enter manually", "Upload file (id, value)"], horizontal=True,
                               help="The Y variable you want to predict (e.g. a lab-measured concentration) "
                                    "for each sample, in the same order as your spectra.", key="ac_ld_y_mode")

            if modo_y == "Enter manually":
                st.caption("One value per sample, comma-separated, in the same order as the IDs:")
                st.code(", ".join(st.session_state.ids[:8]) + (", ..." if len(st.session_state.ids) > 8 else ""))
                texto_y = st.text_area("Values (comma-separated)", key="texto_valores_y", help="Reference values for the samples, in the same order as the data, separated by commas.")
                if texto_y.strip():
                    try:
                        lista_y = [float(v.strip()) for v in texto_y.split(",")]
                        if len(lista_y) != len(st.session_state.ids):
                            st.error(f"You entered {len(lista_y)} values but there are {len(st.session_state.ids)} samples.")
                        else:
                            st.session_state.valores_y = np.array(lista_y, dtype=float)
                            st.rerun()
                    except ValueError:
                        st.error("Some value could not be parsed as a number.")
            else:
                archivo_y = st.file_uploader(
                    "File with columns: id, value", type=["csv", "xlsx", "xls"], key="valores_y_csv",
                help="Table with the sample ID and its reference value; matched to the spectra by ID.")
                if archivo_y is not None:
                    try:
                        if archivo_y.name.lower().endswith(".csv"):
                            df_y = pd.read_csv(archivo_y)
                        else:
                            df_y = pd.read_excel(archivo_y)
                        df_y = df_y.set_index(df_y.columns[0])
                        df_y.index = df_y.index.astype(str)
                        mapa_y = df_y.iloc[:, 0].to_dict()
                        valores_y = np.array([mapa_y.get(str(i), np.nan) for i in st.session_state.ids], dtype=float)
                        n_faltantes = int(np.isnan(valores_y).sum())
                        if n_faltantes > 0:
                            st.warning(f"{n_faltantes} sample(s) without a reference value in the file — "
                                       "they are automatically excluded from the regression.")
                        st.session_state.valores_y = valores_y
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error reading the file: {e}")
        else:
            col_y1, col_y2 = st.columns([4, 1])
            col_y1.success(f"✓ Reference values loaded for {len(st.session_state.valores_y)} samples.")
            if col_y2.button("Change", help="Clear the loaded reference values to enter or upload different ones."):
                st.session_state.valores_y = None
                st.rerun()

        if st.session_state.valores_y is None:
            st.info("Load the reference values to be able to train a regression model.")
        else:
            ids_activos, _, _ = datos_activos()
            X_modelado = st.session_state.X_pret[indice_activo()]
            eje_modelado = st.session_state.numeros_onda_pret
            X_modelado, eje_modelado, _crop_mask_reg, _eje_completo_reg = elegir_espectro_modelado(X_modelado, eje_modelado, "reg")
            y_activos = st.session_state.valores_y[indice_activo()]

            mask_validos = ~np.isnan(y_activos)
            if mask_validos.sum() < 5:
                st.error("Too few samples with a valid reference value (at least 5 are needed).")
            else:
                ids_reg = ids_activos[mask_validos]
                X_reg = X_modelado[mask_validos]
                y_reg = y_activos[mask_validos]
                st.caption(f"Using {mask_validos.sum()} of {len(y_activos)} active samples (with a reference value).")

                col1, col2 = st.columns(2)
                with col1:
                    modelos_elegidos_r = st.multiselect(
                        "Models to compare",
                        list(mu.crear_regresores().keys()),
                        default=["PLS", "Ridge", "Random Forest"],
                        help="Pick one or more algorithms to train and compare side by side in the same "
                             "run — the results table below will show every metric for each of them.",
                    key="ac_reg_models")
                    metodo_seleccion_r = st.selectbox(
                        "Variable selection", ["None", "Boruta", "Genetic Algorithm"], key="sel_var_reg", disabled=_BASICO,
                        help="Reduce the spectrum to just the most informative wavenumbers before "
                             "training. Boruta is conservative and keeps anything that helps at all; "
                             "the Genetic Algorithm explores combinations more freely but is slower and "
                             "a bit more random. 'None' uses the full spectrum.",
                    )
                    if _BASICO:
                        metodo_seleccion_r = "None"
                    gu.pros_cons("seleccion", metodo_seleccion_r, key="pc_sel_reg")
                with col2:
                    cv_folds_sel_r = st.selectbox(
                        "Cross-validation folds (internal)",
                        [3, 5, 10, "LOO (leave 1 sample out at a time)"], index=1, key="cv_reg",
                        help="How many folds to use for cross-validation. More folds = less biased but "
                             "slower and more variance between folds. LOO (Leave-One-Out) trains one "
                             "model per sample — most thorough, slowest, and typical for small datasets.",
                    )
                    cv_folds_r = "LOO" if cv_folds_sel_r == "LOO (leave 1 sample out at a time)" else cv_folds_sel_r
                    gu.pros_cons("cv", "Leave-one-out (LOO)" if cv_folds_r == "LOO" else "k-fold (5 or 10)", key="pc_cv_reg")
                    if cv_folds_r == "LOO" and X_reg.shape[0] > 150:
                        st.caption("⚠ LOO trains one model per sample — with many samples this can take a while.")
                    prop_test_r = st.slider("Proportion for independent test set (0 = use everything for CV)",
                                             0.0, 0.4, 0.2, step=0.05, key="test_reg",
                                             help="Fraction of samples set aside BEFORE training, never used "
                                                  "to choose anything — only to get one final, honest check "
                                                  "of how the model performs on unseen data. 0 means skip "
                                                  "this and use every sample for cross-validation instead.")
                    metodo_split_sel_r, metodo_split_r, semilla_reg = _ui_split(False, prop_test_r, "split_reg")
                    optimizar_r = st.checkbox(
                        "Optimize hyperparameters (slower)", key="opt_reg", disabled=_BASICO,
                        help="Automatically search for better settings for each model instead of using "
                             "the defaults. Usually improves results a bit, at the cost of extra "
                             "computation time.",
                    )
                    if _BASICO:
                        optimizar_r = False
                    if optimizar_r:
                        st.caption("Search ranges in use: **" + st.session_state.get("hp_modo_aplicado", HP_MODOS[1])
                                   + "** — change them in the 🎛️ Hyperparameters tab.")
                    preset_opt_r = st.selectbox(
                        "Optimization budget", list(PRESETS_OPTIMIZACION.keys()), index=1,
                        disabled=not optimizar_r, key="metodo_opt_reg",
                        help="How much effort to spend searching for good hyperparameters — see the "
                             "explanation below once you pick one.",
                    )
                    metodo_opt_r = PRESETS_OPTIMIZACION[preset_opt_r][0]
                    gu.pros_cons("optimizacion", preset_opt_r, key="pc_opt_reg")
                    st.caption("The ranges explored are shown (and editable) in the **🎛️ Hyperparameters** tab.")
                    cfg_aug_r = _ui_aumento("reg", False)

                if optimizar_r:
                    st.caption(f"ℹ️ **{preset_opt_r}**: {PRESETS_OPTIMIZACION[preset_opt_r][1]}")

                if optimizar_r and any(m in mu.ALGORITMOS_LENTOS_AL_OPTIMIZAR for m in modelos_elegidos_r):
                    lentos_elegidos_r = [m for m in modelos_elegidos_r if m in mu.ALGORITMOS_LENTOS_AL_OPTIMIZAR]
                    extra_r = " With LOO this can take several minutes." if cv_folds_r == "LOO" else ""
                    st.caption(f"⚠ Optimizing hyperparameters for {', '.join(lentos_elegidos_r)} can still take a "
                               f"while on a shared/limited CPU. If it's too slow, try 'Fast', or fewer folds.{extra_r}")

                if optimizar_r and len(modelos_elegidos_r) > 3:
                    st.warning(f"⚠️ You're optimizing hyperparameters for {len(modelos_elegidos_r)} models at once. "
                               "Each one runs its own search in parallel across CPU cores, so doing several "
                               "together can exhaust the machine's memory and cause the app to crash or freeze. "
                               "We recommend optimizing **up to 3 models at a time** — train the rest with "
                               "default settings first, then optimize the best candidates separately.")

                if metodo_seleccion_r == "Boruta":
                    c1, c2 = st.columns(2)
                    boruta_max_iter_r = c1.slider(
                        "Boruta iterations", 10, 500, 40, step=10, key="boruta_iter_reg",
                        help="How many comparison rounds to run against the random 'shadow' variables. "
                             "More iterations give a more stable decision but take longer.",
                    )
                    boruta_alpha_r = c2.select_slider(
                        "Boruta p-value (alpha)", options=[0.01, 0.02, 0.05, 0.1], value=0.05, key="boruta_alpha_reg",
                        help="Significance level for keeping a variable: smaller (e.g. 0.01) is stricter "
                             "and keeps fewer, more confidently relevant variables; larger (e.g. 0.1) is "
                             "more permissive.",
                    )
                if metodo_seleccion_r == "Genetic Algorithm":
                    c1, c2 = st.columns(2)
                    ga_poblacion_r = c1.slider("Population size", 10, 60, 20, key="ga_pob_reg", help="Number of candidate solutions per generation. Larger = better search but slower.")
                    ga_generaciones_r = c2.slider("Generations", 5, 50, 15, key="ga_gen_reg", help="How many generations the search runs. More = better but slower.")

                if st.session_state.get("reg_resultados") is not None:
                    insignia_estado("reg_firma", "This trained regression model", firma_actual=firma_modelos(_crop_mask_reg))

                if metodo_seleccion_r != "None" or optimizar_r:
                    st.caption("ℹ️ Variable selection and hyperparameter optimization use ONLY the training samples "
                               "(the test set is set aside first and never touched). Cross-validation figures can still be "
                               "slightly optimistic when selection is used; the independent test set is the honest estimate."
                               + (" ⚠ With no test set (0%), selection uses all the samples, so the CV figures are optimistic." if prop_test_r == 0 else ""))
                _run_reg = st.session_state.get("reg_run")
                _incompleto_reg = bool(_run_reg) and not _run_reg.get("completo", True)
                _valida_reg = _incompleto_reg and _run_reg.get("firma") == firma_modelos(_crop_mask_reg)
                if _incompleto_reg:
                    _hechos_reg = len([n_ for n_, r_ in _run_reg["resultados"].items() if "error" not in r_])
                    st.warning(f"⏸ Incomplete run: {_hechos_reg} of {len(_run_reg['modelos'])} models finished (the run was interrupted). "
                               + ("Click **Resume** to continue with the same settings, skipping the finished models."
                                  if _valida_reg else "The data, preprocessing or crop changed since then, so it can't be resumed — train again."))
                _rt1, _rt2, _ = st.columns([1.7, 1.6, 3])
                _ent_reg = _rt1.button("🚀 Train and evaluate (Regression)", disabled=len(modelos_elegidos_r) == 0)
                _rean_reg = _rt2.button("⏩ Resume interrupted run", key="reg_btn_resume", disabled=not _valida_reg,
                                        help="Continues the interrupted run with the SAME settings it started with, skipping "
                                             "the models that already finished (and not repeating the variable selection).")
                if _ent_reg or _rean_reg:
                    if _rean_reg:
                        _rr = _run_reg
                        modelos_elegidos_r = list(_rr["modelos"]); cv_folds_r = _rr["cv_folds"]; prop_test_r = _rr["prop_test"]
                        metodo_split_r = _rr["metodo_split"]; metodo_split_sel_r = _rr["metodo_split_sel"]; semilla_reg = _rr.get("semilla", 0)
                        optimizar_r = _rr["optimizar"]; metodo_opt_r = _rr["metodo_opt"]; metodo_seleccion_r = _rr["metodo_seleccion"]
                        boruta_max_iter_r, boruta_alpha_r = _rr["boruta_iter"], _rr["boruta_alpha"]
                        ga_poblacion_r, ga_generaciones_r = _rr["ga_pob"], _rr["ga_gen"]
                        _idx_split_reg = _rr["idx_split"]
                    else:
                        for _clave in ["reg_pvalores", "reg_puntajes_cv", "reg_comparacion_metodo",
                                        "reg_curva_aprendizaje", "reg_curva_modelo", "reg_ultima_ficha", "reg_rs_res"]:
                            st.session_state.pop(_clave, None)
                        _idx_split_reg = None
                        if prop_test_r > 0:
                            try:
                                _idx_split_reg = mu.dividir_train_test(X_reg, y_reg, ids_reg, prop_test_r, False, semilla_reg, metodo_split_r)
                            except Exception:
                                _idx_split_reg = None
                        st.session_state["reg_run"] = {
                            "completo": False, "firma": firma_modelos(_crop_mask_reg), "modelos": list(modelos_elegidos_r),
                            "cv_folds": cv_folds_r, "prop_test": prop_test_r, "metodo_split": metodo_split_r, "semilla": semilla_reg,
                            "metodo_split_sel": metodo_split_sel_r, "optimizar": optimizar_r, "metodo_opt": metodo_opt_r,
                            "metodo_seleccion": metodo_seleccion_r,
                            "boruta_iter": boruta_max_iter_r if metodo_seleccion_r == "Boruta" else None,
                            "boruta_alpha": boruta_alpha_r if metodo_seleccion_r == "Boruta" else None,
                            "ga_pob": ga_poblacion_r if metodo_seleccion_r == "Genetic Algorithm" else None,
                            "ga_gen": ga_generaciones_r if metodo_seleccion_r == "Genetic Algorithm" else None,
                            "idx_split": _idx_split_reg, "mascara": None, "seleccion_hecha": False,
                            "resultados": {}, "hp": {}, "desc": {},
                        }
                    _rr = st.session_state["reg_run"]
                    resultados_r, hiperparametros_optimos_r, descripcion_opt_usada_r = _rr["resultados"], _rr["hp"], _rr["desc"]
                    _idx_tr_reg = _idx_split_reg[0] if _idx_split_reg is not None else np.arange(len(y_reg))

                    def _guardar_reg():
                        st.session_state["reg_resultados"] = resultados_r
                        st.session_state["reg_mascara_variables"] = mascara_variables_r
                        st.session_state["reg_descripcion_opt"] = descripcion_opt_usada_r
                        st.session_state["reg_eje_usado"] = eje_modelado
                        st.session_state["reg_pasos_pretratamiento"] = st.session_state.pasos_pretratamiento
                        st.session_state["reg_hiperparametros"] = hiperparametros_optimos_r
                        st.session_state["reg_espectro_promedio"] = X_reg.mean(axis=0)
                        st.session_state["reg_cv_folds"] = cv_folds_r
                        st.session_state["reg_prop_test"] = prop_test_r
                        st.session_state["reg_metodo_split"] = metodo_split_sel_r
                        st.session_state["reg_metodo_seleccion"] = metodo_seleccion_r
                        st.session_state["reg_n_muestras"] = X_reg.shape[0]
                        st.session_state["reg_firma"] = firma_modelos(_crop_mask_reg)
                        st.session_state["reg_eje_completo"] = _eje_completo_reg
                        st.session_state["reg_crop_mask"] = _crop_mask_reg
                        st.session_state["reg_origen"] = "cropped" if _crop_mask_reg is not None else "full"
                        st.session_state["reg_crop_desc"] = st.session_state.get("crop_desc_aplicada")

                    mascara_variables_r = _rr["mascara"]
                    _msg_espera_r = "This may take a few minutes..." if (optimizar_r or metodo_seleccion_r in ("Boruta", "Genetic Algorithm")) else "Training..."
                    with st.spinner(f"Selecting variables ({_msg_espera_r})" if (metodo_seleccion_r != "None" and not _rr["seleccion_hecha"]) else _msg_espera_r):
                        if not _rr["seleccion_hecha"]:
                            if metodo_seleccion_r == "Boruta":
                                try:
                                    mascara_variables_r = mu.seleccionar_variables_boruta(
                                        X_reg[_idx_tr_reg], y_reg[_idx_tr_reg], es_clasificacion=False,
                                        max_iter=boruta_max_iter_r, alpha=boruta_alpha_r,
                                    )
                                    if mascara_variables_r.sum() == 0:
                                        st.warning("Boruta did not select any variable; using all of them.")
                                        mascara_variables_r = None
                                except Exception as e:
                                    st.error(f"Boruta failed ({e}); using all variables.")
                            elif metodo_seleccion_r == "Genetic Algorithm":
                                from sklearn.linear_model import LinearRegression as _LR
                                cv_ga_r = 3 if cv_folds_r == "LOO" else min(3, cv_folds_r)
                                ga_r = mu.SeleccionGenetica(
                                    _LR(), es_clasificacion=False,
                                    tam_poblacion=ga_poblacion_r, n_generaciones=ga_generaciones_r,
                                    cv=cv_ga_r, random_state=0,
                                )
                                ga_r.fit(X_reg[_idx_tr_reg], y_reg[_idx_tr_reg])
                                mascara_variables_r = ga_r.mejor_mascara_
                            _rr["mascara"], _rr["seleccion_hecha"] = mascara_variables_r, True

                        X_sel_r = X_reg[:, mascara_variables_r] if mascara_variables_r is not None else X_reg
                        try:
                            st.session_state["reg_dominio"] = cf.ajustar_dominio(X_sel_r[_idx_tr_reg])
                        except Exception:
                            st.session_state["reg_dominio"] = None
                        catalogo_r = mu.crear_regresores()
                        for nombre in modelos_elegidos_r:
                            if nombre in resultados_r and "error" not in resultados_r[nombre]:
                                continue                      # finished in the interrupted run: skip it
                            modelo = catalogo_r[nombre]
                            try:
                                if optimizar_r and nombre in mu.GRILLAS_REGRESION:
                                    modelo, mejores_params_r, _, desc_opt_r = mu.optimizar_hiperparametros(
                                        modelo, _grilla_activa(nombre, False, X_sel_r[_idx_tr_reg], y_reg[_idx_tr_reg], 0.0, cv_folds_r), X_sel_r[_idx_tr_reg], y_reg[_idx_tr_reg],
                                        es_clasificacion=False, cv=cv_folds_r, metodo=metodo_opt_r,
                                    )
                                    hiperparametros_optimos_r[nombre] = mejores_params_r
                                    descripcion_opt_usada_r[nombre] = desc_opt_r
                                st.session_state["reg_aug_desc"] = None
                                resultados_r[nombre] = mu.entrenar_evaluar_regresion(
                                    modelo, X_sel_r, y_reg, ids=ids_reg,
                                    cv=cv_folds_r, proporcion_test=prop_test_r, metodo_split=metodo_split_r, random_state=semilla_reg,
                                    indices_split=_idx_split_reg,
                                )
                            except Exception as e:
                                resultados_r[nombre] = {"error": str(e)}
                            _guardar_reg()
                    _rr["completo"] = True
                    _guardar_reg()
                    if _incompleto_reg:      # redraw: the 'incomplete run' banner above is now outdated
                        _rerun_tab()

                if "reg_resultados" in st.session_state:
                    resultados_r = st.session_state["reg_resultados"]
                    mascara_variables_r = st.session_state["reg_mascara_variables"]
                    hiperparametros_optimos_r = st.session_state.get("reg_hiperparametros", {})
                    descripcion_opt_usada_r = st.session_state.get("reg_descripcion_opt", {})

                    if mascara_variables_r is not None:
                        st.caption(f"Selected variables: {int(mascara_variables_r.sum())} of {len(mascara_variables_r)}.")
                        with st.expander("📍 View selected variables on the mean spectrum"):
                            fig_vars_r = go.Figure()
                            eje_rr = st.session_state["reg_eje_usado"]
                            fig_vars_r.add_trace(go.Scatter(x=eje_rr, y=st.session_state["reg_espectro_promedio"],
                                                             mode="lines", line=dict(color="#5F5E5A"), name="Mean spectrum"))
                            idx_sel_r = np.where(mascara_variables_r)[0]
                            bloques_r = np.split(idx_sel_r, np.where(np.diff(idx_sel_r) != 1)[0] + 1) if len(idx_sel_r) else []
                            for j, bloque in enumerate(bloques_r):
                                x0, x1 = eje_rr[bloque[0]], eje_rr[bloque[-1]]
                                fig_vars_r.add_vrect(x0=min(x0, x1), x1=max(x0, x1), fillcolor="#0F6E56",
                                                      opacity=0.25, line_width=0)
                            fig_vars_r.update_layout(height=380, xaxis_title="Wavenumber", yaxis_title="Signal")
                            if eje_rr[0] > eje_rr[-1]:
                                fig_vars_r.update_xaxes(autorange="reversed")
                            if not es_tabular():
                                _t_a, _m_a = _asig_pre(fig_vars_r, eje_rr, asg.regiones_desde_mascara(eje_rr, mascara_variables_r), "reg")
                                st.plotly_chart(fig_vars_r, width='stretch')
                                _asig_post(_t_a, _m_a, "reg")

                            st.markdown("**List of selected variables**")
                            df_vars_sel_r = df_variables_seleccionadas(eje_rr, mascara_variables_r)
                            st.dataframe(df_vars_sel_r, width='stretch', height=200)
                            st.download_button(
                                "⬇️ Download selected variables (Excel)",
                                data=df_a_excel_bytes({"selected_variables": df_vars_sel_r}),
                                file_name="selected_variables_regression.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="descargar_vars_reg",
                            )

                    filas = []
                    for nombre, res in resultados_r.items():
                        if "error" in res:
                            filas.append({"Model": nombre, "Error": res["error"]})
                            continue
                        fila = {
                            "Model": nombre,
                            "RMSE (CV)": round(res["cv"]["rmse_cv"], 4),
                            "R² (CV)": round(res["cv"]["r2_cv"], 3),
                            "RPD (CV)": round(res["cv"]["rpd_cv"], 2),
                            "RPIQ (CV)": round(res["cv"]["rpiq_cv"], 2),
                        }
                        if "test" in res:
                            fila["RMSE (test)"] = round(res["test"]["rmse"], 4)
                            fila["R² (test)"] = round(res["test"]["r2"], 3)
                            fila["RPD (test)"] = round(res["test"]["rpd"], 2)
                        filas.append(fila)
                    df_metricas_reg = pd.DataFrame(filas)
                    _rm1, _rm2, _rm3 = st.columns([1.5, 1.9, 3])
                    with _rm1:
                        gu.how_to_interpret("reg", valores=gu.valores_reg(), key="how_res_reg")
                    with _rm2:
                        gu.what_next("reg", key="next_res_reg")
                    st.dataframe(df_metricas_reg, width='stretch')
                    st.download_button(
                        "⬇️ Download metrics table (Excel)",
                        data=df_a_excel_bytes({"regression_metrics": df_metricas_reg}),
                        file_name="regression_metrics.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="descargar_metricas_reg",
                    )

                    modelos_exitosos_r = [n for n in resultados_r if "error" not in resultados_r[n]]
                    if len(modelos_exitosos_r) >= 2:
                        hay_test_reg = "test" in resultados_r[modelos_exitosos_r[0]]
                        if _BASICO:
                            st.caption(T("🔒 Compare models statistically — advanced, greyed out in First steps mode."))  # LOCKED
                        else:
                            with st.expander("📊 Compare models statistically"):
                                if hay_test_reg:
                                    st.caption("Compares models on the held-out TEST set predictions, sample by "
                                               "sample (paired t-test on squared errors) — this is what can "
                                               "actually reveal overfitting, since two models can look equally "
                                               "good in cross-validation yet behave very differently on genuinely "
                                               "unseen data. A p-value below 0.05 means the difference is unlikely "
                                               "to be just chance.")
                                else:
                                    st.caption("No independent test set was configured, so this compares models "
                                               "using a paired t-test on their per-fold cross-validation scores "
                                               "instead (same folds for every model). Note this can't detect "
                                               "overfitting the way a held-out test set would — consider setting "
                                               "aside a test proportion above 0% for a more trustworthy comparison.")
                                if st.button("Run comparison", key="comparar_modelos_reg"):
                                    if hay_test_reg:
                                        pvalores_r, puntajes_r = mu.comparar_modelos_en_test(
                                            {n: resultados_r[n] for n in modelos_exitosos_r}, es_clasificacion=False,
                                        )
                                    else:
                                        mascara_cmp_r = st.session_state["reg_mascara_variables"]
                                        ids_train_ref_r = resultados_r[modelos_exitosos_r[0]]["cv"]["ids"]
                                        mask_train_cmp_r = np.isin(ids_reg, ids_train_ref_r)
                                        X_cmp_r = X_reg[mask_train_cmp_r]
                                        y_cmp_r = y_reg[mask_train_cmp_r]
                                        if mascara_cmp_r is not None:
                                            X_cmp_r = X_cmp_r[:, mascara_cmp_r]
                                        modelos_para_comparar_r = {
                                            n: mu.clone(resultados_r[n]["modelo_final"]) for n in modelos_exitosos_r
                                        }
                                        with st.spinner("Running cross-validation for every model on the same folds..."):
                                            pvalores_r, puntajes_r = mu.comparar_modelos_estadisticamente(
                                                modelos_para_comparar_r, X_cmp_r, y_cmp_r, es_clasificacion=False,
                                                cv=st.session_state["reg_cv_folds"],
                                            )
                                    st.session_state["reg_pvalores"] = pvalores_r
                                    st.session_state["reg_puntajes_cv"] = puntajes_r
                                    st.session_state["reg_comparacion_metodo"] = (
                                        "Paired t-test on squared errors on the held-out test set" if hay_test_reg
                                        else "Paired t-test on cross-validation folds (no test set configured)"
                                    )
                                if "reg_pvalores" in st.session_state:
                                    interpretacion_r = mu.interpretar_comparacion_modelos(
                                        st.session_state["reg_pvalores"], st.session_state["reg_puntajes_cv"],
                                    )
                                    if interpretacion_r["hay_diferencias"]:
                                        for msg in interpretacion_r["mensajes"]:
                                            st.success(msg)
                                    else:
                                        st.info(interpretacion_r["mensajes"][0])
                                    st.markdown("**p-values:**")
                                    st.dataframe(st.session_state["reg_pvalores"].style.format("{:.4f}"), width='stretch')

                    _m_rs_r = st.session_state.get("reg_mascara_variables")
                    _ui_remuestreo("reg", {n_: mu.clone(r_["modelo_final"]) for n_, r_ in resultados_r.items() if "error" not in r_},
                                   X_reg[:, _m_rs_r] if _m_rs_r is not None else X_reg, y_reg, False)

                    nombre_detalle_r = st.selectbox(
                        "View details for", [n for n in resultados_r if "error" not in resultados_r[n]], key="detalle_reg",
                        help="Choose which of the trained models to inspect below: predicted vs. actual, "
                             "hyperparameters, learning curve, Williams plot, and the report/save options.",
                    )
                    if nombre_detalle_r:
                        res = resultados_r[nombre_detalle_r]

                        if hiperparametros_optimos_r.get(nombre_detalle_r):
                            with st.container(border=True):
                                st.markdown(f"**⚙️ Optimized hyperparameters — {nombre_detalle_r}**")
                                cols_hp_r = st.columns(len(hiperparametros_optimos_r[nombre_detalle_r]))
                                for col_hp, (k, v) in zip(cols_hp_r, hiperparametros_optimos_r[nombre_detalle_r].items()):
                                    col_hp.metric(k, str(v))
                                st.caption(f"🔧 Method: {descripcion_opt_usada_r.get(nombre_detalle_r, 'n/a')}")

                        hay_test_r = "test" in res
                        y_t_cv, y_p_cv, ids_cv = res["cv"]["y_true"], res["cv"]["y_pred"], res["cv"]["ids"]

                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(
                            x=y_t_cv, y=y_p_cv, mode="markers", name="Training (CV)",
                            marker=dict(color="#185FA5", size=8),
                            text=ids_cv, hovertemplate="%{text}<br>Actual: %{x}<br>Predicted: %{y}<extra></extra>",
                        ))
                        todos_valores = [y_t_cv, y_p_cv]
                        if hay_test_r:
                            y_t_test, y_p_test, ids_test = res["test"]["y_true"], res["test"]["y_pred"], res["test"]["ids"]
                            fig_pred.add_trace(go.Scatter(
                                x=y_t_test, y=y_p_test, mode="markers", name="Test",
                                marker=dict(color="#D97706", size=9, symbol="diamond"),
                                text=ids_test, hovertemplate="%{text}<br>Actual: %{x}<br>Predicted: %{y}<extra></extra>",
                            ))
                            todos_valores += [y_t_test, y_p_test]

                        lim_lo = min(arr.min() for arr in todos_valores)
                        lim_hi = max(arr.max() for arr in todos_valores)
                        fig_pred.add_trace(go.Scatter(x=[lim_lo, lim_hi], y=[lim_lo, lim_hi], mode="lines",
                                                       line=dict(dash="dash", color="gray"), name="Ideal (y=x)"))
                        fig_pred.update_layout(height=470, title=f"Predicted vs. actual — {nombre_detalle_r}",
                                                xaxis_title="Actual value", yaxis_title="Predicted value")
                        st.plotly_chart(fig_pred, width='stretch')
                        gu.nota("reg_pred", "predicted vs. actual plot", "reg")

                        df_exp_cv = mu.exportar_predicciones_regresion(ids_cv, y_t_cv, y_p_cv)
                        if hay_test_r:
                            col_dl_cv, col_dl_test = st.columns(2)
                        else:
                            col_dl_cv = st.container()
                        col_dl_cv.download_button(
                            f"⬇️ Download predictions ({nombre_detalle_r}, Training/CV)",
                            data=df_exp_cv.to_csv(index=False).encode("utf-8"),
                            file_name=f"predictions_{nombre_detalle_r}_CV.csv", mime="text/csv",
                            key="descargar_pred_reg_cv",
                        )
                        if hay_test_r:
                            df_exp_test = mu.exportar_predicciones_regresion(ids_test, y_t_test, y_p_test)
                            col_dl_test.download_button(
                                f"⬇️ Download predictions ({nombre_detalle_r}, Test)",
                                data=df_exp_test.to_csv(index=False).encode("utf-8"),
                                file_name=f"predictions_{nombre_detalle_r}_Test.csv", mime="text/csv",
                                key="descargar_pred_reg_test",
                            )

                        # Diagnostics further below (Williams plot, PDF report, model card) need a
                        # single dataset — prefer Test when available (a more honest check of
                        # generalization), otherwise fall back to the CV predictions.
                        if hay_test_r:
                            y_t, y_p, ids_f, fuente_r = y_t_test, y_p_test, ids_test, "Test"
                        else:
                            y_t, y_p, ids_f, fuente_r = y_t_cv, y_p_cv, ids_cv, "Training (CV)"

                        if _BASICO:
                            st.caption(T("🔒 Learning curve — advanced, greyed out in First steps mode."))  # LOCKED
                        else:
                            with st.expander(f"📈 Learning curve — {nombre_detalle_r}"):
                                st.caption("Shows performance vs. how many training samples were used. If both "
                                           "curves are still rising and far apart, more samples would likely help. "
                                           "If they're flat and close together, the model has plateaued.")
                                if st.button("Compute learning curve", key="curva_aprendizaje_reg"):
                                    mascara_lc_r = st.session_state["reg_mascara_variables"]
                                    X_lc_r = X_reg[:, mascara_lc_r] if mascara_lc_r is not None else X_reg
                                    with st.spinner("Training with increasing sample sizes (this may take a few minutes)..."):
                                        curva_r = mu.calcular_curva_aprendizaje(
                                            mu.clone(res["modelo_final"]), X_lc_r, y_reg, es_clasificacion=False,
                                            cv=min(5, st.session_state["reg_cv_folds"]) if st.session_state["reg_cv_folds"] != "LOO" else 5,
                                        )
                                    st.session_state["reg_curva_aprendizaje"] = curva_r
                                    st.session_state["reg_curva_modelo"] = nombre_detalle_r
                                if st.session_state.get("reg_curva_modelo") == nombre_detalle_r and "reg_curva_aprendizaje" in st.session_state:
                                    curva_r = st.session_state["reg_curva_aprendizaje"]
                                    fig_lc_r = go.Figure()
                                    fig_lc_r.add_trace(go.Scatter(x=curva_r["train_sizes"], y=curva_r["train_scores_mean"],
                                                                   mode="lines+markers", name="Training score",
                                                                   line=dict(color="#185FA5")))
                                    fig_lc_r.add_trace(go.Scatter(x=curva_r["train_sizes"], y=curva_r["val_scores_mean"],
                                                                   mode="lines+markers", name="Cross-validation score",
                                                                   line=dict(color="#B91C1C")))
                                    fig_lc_r.update_layout(height=380, xaxis_title="Training set size (samples)",
                                                            yaxis_title=curva_r["scoring"])
                                    st.plotly_chart(fig_lc_r, width='stretch')
                                    st.info(_diagnostico_curva(curva_r))


                        with st.expander(f"📐 Williams plot (leverage vs. residual) — {nombre_detalle_r}"):
                            st.caption("A classic applicability-domain diagnostic (common in QSAR/chemometrics): "
                                       "leverage (X-axis) measures how far a sample sits from the center of the "
                                       "spectral space — high leverage means an unusual spectrum. The standardized "
                                       "residual (Y-axis) measures how badly the model predicted that sample. "
                                       "Samples with HIGH leverage AND a large residual ('bad leverage points', top "
                                       "or bottom right) are the ones most likely to be distorting the model — "
                                       "worth double-checking. High leverage with a small residual ('good leverage "
                                       "points') are fine; they just extend the calibration range.")
                            indices_wp = [np.where(ids_reg == i)[0][0] for i in ids_f]
                            X_wp = X_reg[indices_wp]
                            mascara_wp_r = st.session_state["reg_mascara_variables"]
                            if mascara_wp_r is not None:
                                X_wp = X_wp[:, mascara_wp_r]
                            leverage_wp, n_comp_wp = cu.calcular_leverage(X_wp)
                            residuo_wp = cu.calcular_residuo_estandarizado(y_t, y_p)
                            lim_leverage_wp = cu.limite_leverage(len(leverage_wp), n_comp_wp)

                            fig_wp = px.scatter(x=leverage_wp, y=residuo_wp, hover_name=ids_f,
                                                 labels={"x": "Leverage", "y": "Standardized residual"})
                            fig_wp.add_vline(x=lim_leverage_wp, line_dash="dash", line_color="red")
                            fig_wp.add_hline(y=3, line_dash="dash", line_color="red")
                            fig_wp.add_hline(y=-3, line_dash="dash", line_color="red")
                            fig_wp.update_layout(height=450, title=f"Williams plot — {nombre_detalle_r} ({fuente_r})")
                            st.plotly_chart(fig_wp, width='stretch')
                            gu.nota("reg_williams", "Williams plot", "reg")

                        st.divider()
                        st.markdown("**📄 Report for this analysis**")
                        st.caption("Includes: dataset used, spectra, preprocessing, selected variables, "
                                   "a comparison table of all trained models, and the detail "
                                   f"({nombre_detalle_r}) with predicted vs. actual and hyperparameters.")
                        if st.button("🖨️ Generate report (PDF)", key="generar_reporte_reg"):
                            secciones_rep_r = [
                                {"tipo": "titulo", "texto": "1. Dataset"},
                                {"tipo": "clave_valor", "pares": [
                                    ("Samples used", int(X_reg.shape[0])),
                                    ("Original variables", int(X_reg.shape[1])),
                                    ("Y variable", f"n={len(y_reg)}, mean={y_reg.mean():.4g}, "
                                                   f"std={y_reg.std():.4g}, "
                                                   f"range=[{y_reg.min():.4g}, {y_reg.max():.4g}]"),
                                    ("Preprocessing", " -> ".join(
                                        p[0] for p in st.session_state["reg_pasos_pretratamiento"]) or "none"),
                                ]},
                                {"tipo": "imagen", "fig": ru.fig_espectros(eje_modelado, X_reg, titulo="Preprocessed spectra")},
                                {"tipo": "titulo", "texto": "2. Variable selection"},
                                {"tipo": "clave_valor", "pares": [
                                    ("Method", st.session_state["reg_metodo_seleccion"]),
                                    ("Selected variables", int(mascara_variables_r.sum()) if mascara_variables_r is not None
                                     else f"all ({X_reg.shape[1]})"),
                                ]},
                            ]
                            if mascara_variables_r is not None:
                                secciones_rep_r.append({"tipo": "imagen", "fig": ru.fig_variables_seleccionadas(
                                    eje_modelado, X_reg.mean(axis=0), mascara_variables_r)})
                                idx_sel_rep_r = np.where(mascara_variables_r)[0]
                                secciones_rep_r.append({"tipo": "tabla",
                                    "encabezados": ["Position", "Wavenumber"],
                                    "filas": [[str(i), f"{eje_modelado[i]:.2f}"] for i in idx_sel_rep_r[:60]],
                                    "anchos": [30, 70]})
                                if len(idx_sel_rep_r) > 60:
                                    secciones_rep_r.append({"tipo": "parrafo",
                                        "texto": f"(showing the first 60 of {len(idx_sel_rep_r)} variables — "
                                                 "the full list is in the Excel file downloadable from the app)"})
                            secciones_rep_r += [
                                {"tipo": "salto_pagina"},
                                {"tipo": "titulo", "texto": "3. Model comparison"},
                                {"tipo": "clave_valor", "pares": [
                                    ("Cross-validation", f"{res['cv']['cv_folds_usados']}-fold" if cv_folds_r != "LOO"
                                     else f"Leave-One-Out ({res['cv']['cv_folds_usados']} repetitions)"),
                                    ("Test proportion", f"{int(prop_test_r * 100)}% ({metodo_split_sel_r}, seed {semilla_reg})" if metodo_split_r != "kennard_stone" else f"{int(prop_test_r * 100)}% ({metodo_split_sel_r})"),
                                ]},
                                {"tipo": "tabla",
                                 "encabezados": list(df_metricas_reg.columns),
                                 "filas": df_metricas_reg.astype(str).values.tolist()},
                                {"tipo": "titulo", "texto": f"4. Detail: {nombre_detalle_r}"},
                                {"tipo": "imagen", "fig": ru.fig_predicho_vs_real(y_t, y_p)},
                            ]
                            if hiperparametros_optimos_r.get(nombre_detalle_r):
                                secciones_rep_r.append({"tipo": "subtitulo", "texto": "Optimal hyperparameters"})
                                secciones_rep_r.append({"tipo": "clave_valor",
                                    "pares": [("Optimization method", descripcion_opt_usada_r.get(nombre_detalle_r, "n/a"))]
                                    + list(hiperparametros_optimos_r[nombre_detalle_r].items())})
                            if st.session_state.get("reg_curva_modelo") == nombre_detalle_r and "reg_curva_aprendizaje" in st.session_state:
                                secciones_rep_r.append({"tipo": "subtitulo", "texto": f"Learning curve — {nombre_detalle_r}"})
                                secciones_rep_r.append({"tipo": "imagen",
                                    "fig": ru.fig_curva_aprendizaje(st.session_state["reg_curva_aprendizaje"], nombre_detalle_r)})
                            if "reg_pvalores" in st.session_state:
                                secciones_rep_r.append({"tipo": "salto_pagina"})
                                secciones_rep_r.append({"tipo": "titulo", "texto": "5. Statistical comparison between models"})
                                secciones_rep_r.append({"tipo": "parrafo",
                                    "texto": f"Method: {st.session_state.get('reg_comparacion_metodo', 'n/a')}. "
                                             "A p-value below 0.05 suggests the performance difference between "
                                             "that pair of models is unlikely to be due to chance alone."})
                                interpretacion_rep_r = mu.interpretar_comparacion_modelos(
                                    st.session_state["reg_pvalores"], st.session_state["reg_puntajes_cv"],
                                )
                                for msg in interpretacion_rep_r["mensajes"]:
                                    secciones_rep_r.append({"tipo": "parrafo", "texto": msg.replace("**", "")})
                                secciones_rep_r.append({"tipo": "imagen",
                                    "fig": ru.fig_comparacion_pvalores(st.session_state["reg_pvalores"])})

                            try:
                                secciones_rep_r += inf._b_remuestreo(st.session_state, "reg")
                            except Exception:
                                pass
                            pdf_reporte_reg = _generar_reporte(
                                "Regression Report", f"Models: {', '.join(modelos_elegidos_r)}", secciones_rep_r,
                            )
                            st.download_button(
                                "⬇️ Download full report (PDF)",
                                data=bytes(pdf_reporte_reg.output()),
                                file_name="regression_report.pdf", mime="application/pdf",
                                key="descargar_reporte_reg",
                            )
                        st.divider()

                        nombre_guardado_r = st.text_input("Name to save this model as", value=nombre_detalle_r,
                                                           key="nombre_guardado_reg", help="Name under which the model is kept in this session (and shown in Prediction).")
                        if st.button("💾 Save this model for the Prediction tab", key="guardar_reg"):
                            id_trazabilidad_r = mu.generar_id_trazabilidad()
                            y_ref_activos = y_reg
                            desc_y_r = (f"n={len(y_ref_activos)}, mean={y_ref_activos.mean():.4g}, "
                                        f"std={y_ref_activos.std():.4g}, "
                                        f"range=[{y_ref_activos.min():.4g}, {y_ref_activos.max():.4g}]")
                            cv_desc_r = (f"Leave-One-Out ({res['cv']['cv_folds_usados']} repetitions)"
                                         if cv_folds_r == "LOO" else f"{res['cv']['cv_folds_usados']}-fold")
                            metricas_ficha_r = {
                                "RMSE (CV)": round(res["cv"]["rmse_cv"], 4),
                                "R2 (CV)": round(res["cv"]["r2_cv"], 3),
                                "RPD (CV)": round(res["cv"]["rpd_cv"], 2),
                                "RPIQ (CV)": round(res["cv"]["rpiq_cv"], 2),
                            }
                            if "test" in res:
                                metricas_ficha_r.update({
                                    "RMSE (test)": round(res["test"]["rmse"], 4),
                                    "R2 (test)": round(res["test"]["r2"], 3),
                                    "RPD (test)": round(res["test"]["rpd"], 2),
                                })
                            ficha_r = {
                                "id_trazabilidad": id_trazabilidad_r,
                                "fecha_creacion": _fecha_hora_informe(con_segundos=True),
                                "nombre_modelo": nombre_guardado_r,
                                "tipo": "regression",
                                "algoritmo": nombre_detalle_r,
                                "n_muestras": int(X_reg.shape[0]),
                                "n_variables_totales": int(X_reg.shape[1]),
                                "n_variables_usadas": int(mascara_variables_r.sum()) if mascara_variables_r is not None else int(X_reg.shape[1]),
                                "descripcion_y": desc_y_r,
                                "pretratamiento_desc": desc_con_recorte(" -> ".join(p[0] for p in st.session_state["reg_pasos_pretratamiento"]) or "none", "reg"),
                                "seleccion_variables_desc": st.session_state["reg_metodo_seleccion"],
                                "hiperparametros": hiperparametros_optimos_r.get(nombre_detalle_r, {}),
                                "metodo_optimizacion": descripcion_opt_usada_r.get(nombre_detalle_r, "none"),
                                "aumento_datos": st.session_state.get("reg_aug_desc") or "none",
                                "cv_desc": cv_desc_r,
                                "prop_test_desc": f"{int(st.session_state['reg_prop_test'] * 100)}% ({st.session_state.get('reg_metodo_split', 'Random')})",
                                "metricas": metricas_ficha_r,
                                "entorno_software": mu.info_entorno_software(),
                            }
                            st.session_state.modelos_guardados[nombre_guardado_r] = {
                                "tipo": "regression",
                                "modelo": res["modelo_final"],
                                "mascara_variables": eje_y_mascara_para_guardar("reg", mascara_variables_r)[1],
                                "numeros_onda": eje_y_mascara_para_guardar("reg", mascara_variables_r)[0],
                                "pasos_pretratamiento": st.session_state["reg_pasos_pretratamiento"], "tipo_senal": st.session_state.get("pret_tipo_aplicado", "NIR / MIR"),
                                **_extras_tabular(),
                                **_extras_confianza("reg", res["cv"]),
                                "ficha": ficha_r,
                                # Cross-validation RMSE, kept to build an approximate 95% prediction
                                # interval later (point prediction ± 1.96 x RMSE_cv) — a standard,
                                # model-agnostic way to report uncertainty in chemometric calibration,
                                # independent of which regression algorithm was used.
                                "rmse_cv": res["cv"]["rmse_cv"],
                            }
                            st.session_state["reg_ultima_ficha"] = ficha_r
                            st.success(f"Model '{nombre_guardado_r}' saved (traceability ID: {id_trazabilidad_r}). "
                                       "It is now available in the Prediction tab.")

                        if st.session_state.get("reg_ultima_ficha", {}).get("nombre_modelo") == nombre_guardado_r:
                            pdf_ficha_r = _generar_pdf_ficha(
                                st.session_state["reg_ultima_ficha"],
                                fig_extra=ru.fig_predicho_vs_real(y_t, y_p),
                                titulo_fig_extra="Predicted vs. actual",
                            )
                            st.download_button(
                                "📄 Download model card (PDF)",
                                data=bytes(pdf_ficha_r.output()),
                                file_name=f"model_card_{nombre_guardado_r}.pdf", mime="application/pdf",
                                key="descargar_ficha_reg",
                            )
    with tabs[14]:
        _frag_tab_11()


# TAB: PREDICTION ON NEW SAMPLES
# -----------------------------------------------------------------------
if _abierta(tabs[15]):
    # Each tab re-runs ON ITS OWN when one of its widgets changes (fragment), instead of
    # re-executing the whole app. Buttons that must refresh other parts call st.rerun().
    @st.fragment
    def _frag_tab_12():
        st.subheader("Prediction on new samples")
        gu.barra_guia("pred", "pred")

        with st.expander("💾 Save/load models on your PC (to use them another day)"):
            st.caption("Models saved with the button in the Classification/Regression tabs "
                       "only live for as long as this browser session lasts. To keep them "
                       "for real, download them here as a file and upload them again when you need them.")

            if st.session_state.modelos_guardados:
                st.markdown("**Models available in this session:**")
                for nombre_m, bundle_m in st.session_state.modelos_guardados.items():
                    col_a, col_b = st.columns([3, 1])
                    col_a.write(f"🔹 **{nombre_m}** ({bundle_m['tipo']})")
                    buffer_descarga = io.BytesIO()
                    joblib.dump(bundle_m, buffer_descarga)
                    col_b.download_button(
                        "⬇️ Download", data=buffer_descarga.getvalue(),
                        file_name=f"model_{nombre_m}.joblib", mime="application/octet-stream",
                        key=f"descargar_{nombre_m}",
                    )

            archivo_modelo = st.file_uploader("Upload a saved model (.joblib)", type=["joblib"], key="subir_modelo", help="A model file saved with the 'Save this model' button in Classification/Regression.")
            if archivo_modelo is not None:
                nombre_cargado = st.text_input(
                    "Name for this model", value=archivo_modelo.name.replace(".joblib", ""), key="nombre_modelo_cargado",
                help="Name under which the model is kept in this session (and shown in Prediction).")
                if st.button("📤 Load this model"):
                    try:
                        bundle_cargado = joblib.load(archivo_modelo)
                        claves_esperadas = {"tipo", "modelo", "mascara_variables", "numeros_onda", "pasos_pretratamiento"}
                        if not claves_esperadas.issubset(bundle_cargado.keys()):
                            st.error("The file does not have the format expected for a model saved by this app.")
                        else:
                            st.session_state.modelos_guardados[nombre_cargado] = bundle_cargado
                            st.success(f"Model '{nombre_cargado}' loaded successfully.")
                            _rerun_tab()
                    except Exception as e:
                        st.error(f"Could not load the file: {e}")

        if not st.session_state.modelos_guardados:
            st.info("You haven't saved any model yet. Train one in the Classification or "
                     "Regression tab and use the 'Save this model' button, "
                     "or upload a previously saved one above.")
        else:
            nombre_modelo_usar = st.selectbox(
                "Model to use", list(st.session_state.modelos_guardados.keys()),
                help="Which saved model (or SIMCA model set) to apply to the new samples uploaded below.",
            )
            bundle = st.session_state.modelos_guardados[nombre_modelo_usar]
            st.caption(f"Type: {bundle['tipo']} | Preprocessing: "
                       + (" → ".join(p[0] for p in bundle["pasos_pretratamiento"]) or "none")
                       + f" | Variables used: "
                       + ("all" if bundle["mascara_variables"] is None else str(int(bundle["mascara_variables"].sum()))))

            _tipo_b = bundle.get("tipo_senal") or st.session_state.get("pret_tipo_aplicado")
            if _tipo_b:
                st.info(f"📡 Signal type of this model: **{_tipo_b}** (the one you applied in Preprocessing). "
                        "The new samples must come from the same technique, with the same axis and units.")
            ficha_bundle = bundle.get("ficha")
            if ficha_bundle:
                col_id, col_pdf = st.columns([3, 1])
                col_id.info(f"🔖 Traceability ID: **{ficha_bundle.get('id_trazabilidad', 'n/a')}** "
                            f"(trained on {ficha_bundle.get('fecha_creacion', 'n/a')})")
                pdf_ficha_pred = _generar_pdf_ficha(ficha_bundle)
                col_pdf.download_button(
                    "📄 Model card (PDF)",
                    data=bytes(pdf_ficha_pred.output()),
                    file_name=f"model_card_{nombre_modelo_usar}.pdf", mime="application/pdf",
                    key=f"descargar_ficha_pred_{nombre_modelo_usar}",
                )
                with st.expander("View full model card"):
                    for clave, valor in ficha_bundle.items():
                        if clave in ("hiperparametros", "metricas", "entorno_software") and isinstance(valor, dict):
                            st.markdown(f"**{clave}:**")
                            st.json(valor)
                        else:
                            st.markdown(f"**{clave}:** {valor}")
            else:
                st.caption("This model was saved before the traceability card existed "
                           "(or it was loaded from a .joblib from an earlier version of the app) — "
                           "it has no card attached.")

            archivo_nuevo = st.file_uploader(
                "File with the new samples (same format: ID + spectra)",
                type=["csv", "xlsx", "xls"], key="archivo_prediccion",
            help="Spectra of the samples to predict: first column = sample ID, other columns = the spectral axis (same technique as the model).")
            _firma_pred = ((nombre_modelo_usar, archivo_nuevo.name, getattr(archivo_nuevo, "size", 0))
                           if archivo_nuevo is not None else None)
            if archivo_nuevo is not None:
                if st.button("🔮 Predict", key="pred_btn_run", type="primary",
                             help="Runs the selected model on the samples of the file above."):
                    st.session_state["pred_firma_ok"] = _firma_pred
                if st.session_state.get("pred_firma_ok") != _firma_pred:
                    st.info("📄 File loaded. Click **Predict** to run the model on these samples "
                            "(if you change the model or the file, click it again).")
            if archivo_nuevo is not None and st.session_state.get("pred_firma_ok") == _firma_pred:
                try:
                    if archivo_nuevo.name.lower().endswith(".csv"):
                        df_nuevo = pd.read_csv(archivo_nuevo, index_col=0)
                    else:
                        df_nuevo = pd.read_excel(archivo_nuevo, index_col=0)
                    ids_nuevo = df_nuevo.index.astype(str).to_numpy()
                    if bundle.get("nombres_var") is not None:
                        # TABULAR model: match the new columns BY NAME, then apply the fitted preprocessor
                        nombres_ent = [str(n) for n in bundle["nombres_var"]]
                        cols_nuevo = {str(c).strip(): c for c in df_nuevo.columns}
                        eje_t = np.asarray(bundle["tab_eje"], dtype=float)
                        req = [nombres_ent[int(i) - 1] for i in eje_t]
                        faltan = [n for n in req if n not in cols_nuevo]
                        if faltan:
                            raise ValueError("these variables used by the model are missing in the new file: "
                                             + ", ".join(faltan[:10]) + ("..." if len(faltan) > 10 else ""))
                        X_nuevo_t = df_nuevo[[cols_nuevo[n] for n in req]].to_numpy(dtype=float)
                        X_interp = X_nuevo_t
                        fuera_de_rango = False
                        prep_t = bundle.get("tab_prep")
                        X_pret_nuevo = prep_t.transform(X_nuevo_t) if prep_t is not None else X_nuevo_t
                        if not np.array_equal(np.asarray(bundle["numeros_onda"], dtype=float), eje_t):
                            raise ValueError("this model was trained on a different set of variables than the saved axis.")
                    else:
                        # keep only the spectral columns (a class / reference column left in the file is ignored)
                        _cols_esp = []
                        for _c in df_nuevo.columns:
                            try:
                                float(str(_c).replace(",", ".").strip()); _cols_esp.append(_c)
                            except ValueError:
                                pass
                        if 0 < len(_cols_esp) < df_nuevo.shape[1]:
                            st.caption("Ignored non-spectral columns: " + ", ".join(str(c) for c in df_nuevo.columns if c not in _cols_esp))
                            df_nuevo = df_nuevo[_cols_esp]
                        eje_nuevo = ejes_a_float(df_nuevo.columns.tolist())
                        X_nuevo = df_nuevo.to_numpy(dtype=float)

                        # 1) Interpolate onto the exact axis the model was trained with
                        #    (supports the new instrument not measuring exactly the same
                        #    wavenumbers / bins as during calibration).
                        X_interp, fuera_de_rango = cu.interpolar_a_eje(X_nuevo, eje_nuevo, bundle["numeros_onda"])
                        if fuera_de_rango:
                            st.warning("The spectral axis of the new file does not fully cover the range "
                                       "used to train the model — there is extrapolation at the edges, "
                                       "results there may be less reliable.")

                        # 2) Apply exactly the same preprocessing (Step A/B) used during training
                        X_pret_nuevo = X_interp
                        for paso_tup in bundle["pasos_pretratamiento"]:
                            X_pret_nuevo = aplicar_paso(X_pret_nuevo, paso_tup)

                    # 3) Apply the same variable selection (if any)
                    if bundle["mascara_variables"] is not None:
                        X_final = X_pret_nuevo[:, bundle["mascara_variables"]]
                    else:
                        X_final = X_pret_nuevo

                    # 4) Predict
                    if bundle["tipo"] == "simca":
                        modelos_simca_bundle = bundle["modelo"]  # {class_name: class-model dict}
                        matriz_dentro_pred = pd.DataFrame(
                            index=ids_nuevo, columns=list(modelos_simca_bundle.keys()), dtype=bool,
                        )
                        for c, modelo_c in modelos_simca_bundle.items():
                            _, _, dentro_c = cu.evaluar_muestras_simca(X_final, modelo_c)
                            matriz_dentro_pred[c] = dentro_c

                        def _resumen_asignacion_simca(fila):
                            aceptadas = [c for c in matriz_dentro_pred.columns if fila[c]]
                            if len(aceptadas) == 0:
                                return "None (outlier)"
                            if len(aceptadas) == 1:
                                return aceptadas[0]
                            return "Ambiguous: " + ", ".join(aceptadas)

                        df_resultado = pd.DataFrame({"id": ids_nuevo})
                        for c in matriz_dentro_pred.columns:
                            df_resultado[f"in_class_{c}"] = matriz_dentro_pred[c].to_numpy()
                        df_resultado["assignment"] = [
                            _resumen_asignacion_simca(matriz_dentro_pred.loc[i]) for i in matriz_dentro_pred.index
                        ]
                        st.dataframe(df_resultado, width='stretch')
                    else:
                        predicciones = bundle["modelo"].predict(X_final)
                        if bundle["tipo"] == "classification":
                            df_resultado = pd.DataFrame({"id": ids_nuevo, "predicted_class": predicciones})
                            columnas_prob = []
                            if hasattr(bundle["modelo"], "predict_proba"):
                                try:
                                    proba = bundle["modelo"].predict_proba(X_final)
                                    clases_modelo = getattr(bundle["modelo"], "classes_", None)
                                    if clases_modelo is not None:
                                        for j, c in enumerate(clases_modelo):
                                            nombre_col = f"probability_{c}"
                                            df_resultado[nombre_col] = proba[:, j]
                                            columnas_prob.append(nombre_col)
                                        # The probability of whichever class was actually predicted —
                                        # a quick, single "how confident is the model?" number per sample.
                                        df_resultado["confidence"] = proba.max(axis=1)
                                except Exception:
                                    pass
                            st.caption("**confidence** is the model's probability for its own predicted "
                                       "class — closer to 100% means the model is more sure; values "
                                       "closer to 1/(number of classes) mean it's essentially guessing.")
                            if columnas_prob:
                                formato_cols = {c: "{:.1%}" for c in columnas_prob + ["confidence"]}
                                st.dataframe(df_resultado.style.format(formato_cols), width='stretch')
                            else:
                                st.dataframe(df_resultado, width='stretch')
                        else:
                            df_resultado = pd.DataFrame({"id": ids_nuevo, "predicted_value": predicciones})
                            rmse_cv_modelo = bundle.get("rmse_cv")
                            if rmse_cv_modelo is not None:
                                margen_95 = 1.96 * rmse_cv_modelo
                                df_resultado["uncertainty_95 (±)"] = margen_95
                                df_resultado["lower_95"] = predicciones - margen_95
                                df_resultado["upper_95"] = predicciones + margen_95
                                st.caption(
                                    f"The **95% interval** (`lower_95` to `upper_95`) is an approximate "
                                    f"prediction interval, built from this model's cross-validation RMSE "
                                    f"(±1.96 × {rmse_cv_modelo:.4g}) — the standard way to report "
                                    f"uncertainty in a chemometric calibration. It assumes the error is "
                                    f"roughly similar in size across the range the model was trained on; "
                                    f"treat it with extra caution for predictions near or beyond the "
                                    f"edges of that range."
                                )
                            else:
                                st.caption("⚠ This model was saved before uncertainty tracking was added "
                                           "(or loaded from an older .joblib) — no 95% interval available "
                                           "for it. Retrain and save it again to get one.")
                            st.dataframe(df_resultado, width='stretch')
                    gu.nota("pred_results", "predictions table", "pred")
                    _pp1, _pp2, _pp3 = st.columns([1.5, 1.9, 3])
                    with _pp1:
                        gu.how_to_interpret("pred", key="how_res_pred")
                    with _pp2:
                        gu.what_next("pred", key="next_res_pred")
                    st.session_state["pred_ultimo"] = {
                        "df": df_resultado.copy(), "modelo": nombre_modelo_usar,
                        "archivo": archivo_nuevo.name, "tipo": bundle.get("tipo"),
                        "ficha": ficha_bundle, "fecha": _fecha_hora_informe(),
                    }
                    st.download_button(
                        "⬇️ Download predictions",
                        data=df_resultado.to_csv(index=False).encode("utf-8"),
                        file_name=f"predictions_{nombre_modelo_usar}.csv", mime="text/csv",
                    )
                    st.download_button(
                        "⬇️ Download predictions (Excel)",
                        data=df_a_excel_bytes({"predictions": df_resultado}),
                        file_name=f"predictions_{nombre_modelo_usar}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="descargar_predicciones_excel",
                    )

                    _prediccion_confiable(bundle, X_final, ids_nuevo, nombre_modelo_usar)

                    st.divider()
                    st.markdown("**📄 Report for this prediction**")
                    if st.button("🖨️ Generate report (PDF)", key="generar_reporte_pred"):
                        secciones_rep_pred = [
                            {"tipo": "titulo", "texto": "1. Model used"},
                        ]
                        if ficha_bundle:
                            secciones_rep_pred.append({"tipo": "clave_valor", "pares": [
                                ("Traceability ID", ficha_bundle.get("id_trazabilidad", "n/a")),
                                ("Training date", ficha_bundle.get("fecha_creacion", "n/a")),
                                ("Algorithm", ficha_bundle.get("algoritmo", "n/a")),
                                ("Preprocessing", ficha_bundle.get("pretratamiento_desc", "n/a")),
                                ("Variables used", ficha_bundle.get("n_variables_usadas", "n/a")),
                            ]})
                        else:
                            secciones_rep_pred.append({"tipo": "clave_valor", "pares": [
                                ("Model name", nombre_modelo_usar),
                                ("Type", bundle["tipo"]),
                                ("Preprocessing", " -> ".join(p[0] for p in bundle["pasos_pretratamiento"]) or "none"),
                            ]})
                        secciones_rep_pred += [
                            {"tipo": "titulo", "texto": "2. New samples"},
                            {"tipo": "clave_valor", "pares": [
                                ("File", archivo_nuevo.name),
                                ("Number of samples", len(ids_nuevo)),
                            ]},
                            {"tipo": "titulo", "texto": "3. Predictions"},
                            {"tipo": "tabla",
                             "encabezados": list(df_resultado.columns),
                             "filas": df_resultado.astype(str).values.tolist()[:80]},
                        ]
                        if len(df_resultado) > 80:
                            secciones_rep_pred.append({"tipo": "parrafo",
                                "texto": f"(showing the first 80 of {len(df_resultado)} predictions — "
                                         "the full table is in the Excel file downloadable from the app)"})
                        pdf_reporte_pred = _generar_reporte(
                            "Prediction Report", f"Model: {nombre_modelo_usar}", secciones_rep_pred,
                        )
                        st.download_button(
                            "⬇️ Download full report (PDF)",
                            data=bytes(pdf_reporte_pred.output()),
                            file_name="prediction_report.pdf", mime="application/pdf",
                            key="descargar_reporte_pred",
                        )
                except Exception as e:
                    st.error(f"Could not predict on the new file: {e}")

            st.divider()
            if st.button("🗑️ Delete all saved models"):
                st.session_state.modelos_guardados = {}
                _rerun_tab()
    with tabs[15]:
        _frag_tab_12()


# -----------------------------------------------------------------------
# TAB: FINAL REPORT
# -----------------------------------------------------------------------
if _abierta(tabs[16]):
    # Each tab re-runs ON ITS OWN when one of its widgets changes (fragment), instead of
    # re-executing the whole app. Buttons that must refresh other parts call st.rerun().
    @st.fragment
    def _frag_tab_13():
        st.subheader("Final report")
        gu.barra_guia("report")
        if inf is None:
            st.error("The report module `informe_final_en.py` is missing. Upload it to the same "
                     "folder as `app_en.py` to enable this tab.")
        else:
            _firma_inf = firma_datos_activos()
            _estado_inf = inf.disponibilidad(st.session_state, _firma_inf)
            st.caption("Combine everything you have already computed into ONE PDF. Tick what you want "
                       "to include — nothing is recomputed, each result is laid out as it is stored right "
                       "now. Sections you haven't computed yet are greyed out. A result that no longer "
                       "matches the current data, preprocessing or excluded outliers is marked here and, "
                       "if you include it, flagged as OUT OF DATE inside the PDF.")

            _titulo_inf = st.text_input("Report title", value="Chemometric analysis report",
                                        key="inf_w_titulo", help="Title that appears on the first page of the report.")

            _claves_disp = [k for k in inf.TODAS_LAS_CLAVES if _estado_inf[k]["disponible"]]

            def _marcar_inf(valor, claves):
                for _k in claves:
                    st.session_state[f"inf_w_{_k}_1"] = valor

            _cb1, _cb2, _ = st.columns([1.4, 1.4, 3])
            _cb1.button("☑️ Select all available", on_click=_marcar_inf, args=(True, _claves_disp),
                        key="inf_btn_todo")
            _cb2.button("⬜ Select none", on_click=_marcar_inf, args=(False, _claves_disp),
                        key="inf_btn_nada")

            with st.expander("☑️ Choose what to include in the report", expanded=True):
                for _grupo, _items in inf.GRUPOS:
                    st.markdown(f"**{_grupo}**")
                    for _k, _etiqueta in _items:
                        _e = _estado_inf[_k]
                        _nota = ""
                        if not _e["disponible"]:
                            _nota = "  — not computed yet"
                        elif _e["desactualizado"]:
                            _nota = "  — ⚠ out of date (will be flagged in the PDF)"
                        st.checkbox(_etiqueta + _nota, value=_e["disponible"],
                                    disabled=not _e["disponible"],
                                    key=f"inf_w_{_k}_{int(_e['disponible'])}", help="Include this item in the report.")

                _res_clf = st.session_state.get("clf_resultados") or {}
                _mod_clf = [n for n, r_ in _res_clf.items() if "error" not in r_]
                _clf_det = []
                if _mod_clf:
                    _mejor_clf = max(_mod_clf, key=lambda n: _res_clf[n]["cv"]["balanced_accuracy"])
                    _clf_det = st.multiselect(
                        "Classification — models to show in detail (confusion matrix, hyperparameters)",
                        _mod_clf, default=[_mejor_clf],
                        key="inf_w_clf_det_" + hashlib.md5("|".join(_mod_clf).encode()).hexdigest()[:6], help="Detailed pages for these classification models are added to the report.")

                _res_reg = st.session_state.get("reg_resultados") or {}
                _mod_reg = [n for n, r_ in _res_reg.items() if "error" not in r_]
                _reg_det = []
                if _mod_reg:
                    _mejor_reg = min(_mod_reg, key=lambda n: _res_reg[n]["cv"]["rmse_cv"])
                    _reg_det = st.multiselect(
                        "Regression — models to show in detail (predicted vs. actual, hyperparameters)",
                        _mod_reg, default=[_mejor_reg],
                        key="inf_w_reg_det_" + hashlib.md5("|".join(_mod_reg).encode()).hexdigest()[:6], help="Detailed pages for these regression models are added to the report.")

                _guardados = [n for n, b_ in st.session_state.get("modelos_guardados", {}).items()
                              if b_.get("ficha")]
                _fichas_sel = []
                if _guardados:
                    _fichas_sel = st.multiselect(
                        "Model cards (traceability) of saved models to append", _guardados,
                        default=_guardados,
                        key="inf_w_fichas_" + hashlib.md5("|".join(_guardados).encode()).hexdigest()[:6], help="Adds the traceability sheet (preprocessing, parameters, validation) of these saved models.")

            _notas_dict = st.session_state.get("notas") or {}
            _inc_notas = st.checkbox(f"Print my observations on the plots in the report ({len(_notas_dict)} written)",
                                     value=True, key="inf_w_incl_notas", disabled=len(_notas_dict) == 0,
                                     help="The observations you wrote under each plot are printed in the section of that plot. "
                                          "Write them in the '📝 Your observations' boxes below every graph.")
            if _notas_dict:
                with st.expander("👀 Review my observations"):
                    for _pid, _n in _notas_dict.items():
                        st.markdown(f"**{_n['titulo']}**  \n{_n['texto']}")
            else:
                st.caption("Tip: below every plot there is a **📝 Your observations** box; what you write there can be printed in this report.")
            _notas_inf = st.text_area("Notes to add at the end of the report (optional)",
                                      key="inf_w_notas", height=90,
                                      placeholder="Objective, conclusions, observations...", help="Text added at the end of the report.")

            if st.button("📑 Build the final report (PDF)", type="primary", key="inf_btn_construir"):
                _incluir = {k for k in inf.TODAS_LAS_CLAVES if st.session_state.get(f"inf_w_{k}_1")}
                if not _incluir and not _fichas_sel and not _notas_inf.strip():
                    st.warning("Tick at least one section first.")
                else:
                    with st.spinner("Building the report..."):
                        _t, _s, _bloques, _indice = inf.construir_informe(
                            st.session_state,
                            {"titulo": _titulo_inf, "notas": _notas_inf, "incluir": _incluir, "notas_graficos": _inc_notas,
                             "clf_detalle": _clf_det, "reg_detalle": _reg_det, "fichas": _fichas_sel},
                            _firma_inf)
                        _pdf = _generar_reporte(_t, _s, _bloques)
                        st.session_state["informe_final"] = {
                            "pdf": bytes(_pdf.output()), "indice": _indice, "fecha": _fecha_hora_informe()}

            _inf_listo = st.session_state.get("informe_final")
            if _inf_listo is not None:
                st.success(f"Report ready — {len(_inf_listo['indice'])} section(s), built {_inf_listo['fecha']}.")
                st.caption("Contents: " + " · ".join(_inf_listo["indice"]))
                _d1, _d2, _ = st.columns([1.6, 1, 3])
                _d1.download_button("⬇️ Download final report (PDF)", data=_inf_listo["pdf"],
                                    file_name="final_report.pdf", mime="application/pdf",
                                    key="inf_dl_pdf")
                if _d2.button("🧹 Clear", key="inf_btn_clear",
                              help="Removes the built report from memory."):
                    st.session_state.pop("informe_final", None)
                    _rerun_tab()
    with tabs[16]:
        _frag_tab_13()


# -----------------------------------------------------------------------
# TAB: CROP
# -----------------------------------------------------------------------
if _abierta(tabs[3]):
    # Each tab re-runs ON ITS OWN when one of its widgets changes (fragment), instead of
    # re-executing the whole app. Buttons that must refresh other parts call st.rerun().
    @st.fragment
    def _frag_tab_3():
        st.subheader("Crop spectral regions")
        gu.barra_guia("crop")
        if es_tabular():
            st.info("Cropping spectral regions does not apply to tabular variables. To leave variables out, use "
                    "**Preprocessing** (exclusion list and missing-value filter) or the variable-selection options "
                    "of the Classification / Regression tabs.")
            return
        if sc is None:
            st.error("The module `screening_en.py` is missing. Upload it to the same folder as "
                     "`app_en.py` to enable this tab.")
        else:
            _eje_c = np.array(st.session_state.numeros_onda_pret, dtype=float)
            _X_c = st.session_state.X_pret[indice_activo()]
            _aplicado = st.session_state.get("crop_aplicado")

            st.caption(
                "Remove the regions that distort your spectra (for example the water band near "
                "1600 cm⁻¹ in FT-MIR) — or keep only the regions you trust. The crop is applied "
                "**after** the preprocessing, which is always computed on the full spectrum, so "
                "derivatives and smoothing are never disturbed by the cut edges. It then becomes an "
                "option in the model tabs (full vs. cropped spectrum) and in Model screening.")

            if "crop_borrador" not in st.session_state:
                st.session_state["crop_borrador"] = []
            _borrador = st.session_state["crop_borrador"]

            _modo = st.radio("The regions I define are the ones to…", ["Keep", "Remove"], horizontal=True,
                             key="crop_w_modo",
                             help="Keep: only these regions are used and everything else is dropped. "
                                  "Remove: these regions are dropped and the rest is used.")

            # ---- add a region ---------------------------------------------------------
            _rango = float(_eje_c.max() - _eje_c.min())
            _paso = max(_rango / 1000, 1e-6)
            _c1, _c2, _c3 = st.columns([1, 1, 1])
            _desde = _c1.number_input("From", min_value=float(_eje_c.min()), max_value=float(_eje_c.max()),
                                      value=float(_eje_c.min()), step=_paso, format="%.3f", key="crop_w_desde", help="Lower limit of the range you keep.")
            _hasta = _c2.number_input("To", min_value=float(_eje_c.min()), max_value=float(_eje_c.max()),
                                      value=float(_eje_c.min() + 0.1 * _rango), step=_paso, format="%.3f",
                                      key="crop_w_hasta", help="Upper limit of the range you keep.")
            _c3.markdown("&nbsp;")
            if _c3.button("➕ Add this region", key="crop_btn_agregar"):
                if _desde == _hasta:
                    st.warning("'From' and 'To' are the same value.")
                else:
                    _borrador.append([float(min(_desde, _hasta)), float(max(_desde, _hasta))])
                    _rerun_tab()

            # quick presets (only those that fall inside this axis)
            _atajos = [("Water bending band (1600–1700)", 1600, 1700), ("CO₂ (2280–2400)", 2280, 2400),
                       ("Water O–H stretch (3000–3700)", 3000, 3700)]
            _atajos = [a for a in _atajos if _eje_c.min() <= a[1] and a[2] <= _eje_c.max()]
            if _atajos:
                st.caption("Quick add (typical mid-infrared regions):")
                _cols_a = st.columns(len(_atajos))
                for _col, (_nom, _lo, _hi) in zip(_cols_a, _atajos):
                    if _col.button(_nom, key=f"crop_btn_atajo_{_lo}"):
                        _borrador.append([float(_lo), float(_hi)])
                        _rerun_tab()

            # ---- current list of regions ------------------------------------------------
            if _borrador:
                st.markdown("**Regions defined**")
                for _i, (_lo, _hi) in enumerate(list(_borrador)):
                    _cc1, _cc2 = st.columns([5, 1])
                    _cc1.markdown(f"{_i + 1}. {_lo:g} – {_hi:g}")
                    if _cc2.button("✖", key=f"crop_btn_del_{_i}", help="Remove this region from the list"):
                        _borrador.pop(_i)
                        _rerun_tab()
                if st.button("🗑️ Remove all regions from the list", key="crop_btn_vaciar"):
                    st.session_state["crop_borrador"] = []
                    _rerun_tab()
            else:
                st.info("No regions yet. Add them above, or draw a box over the plot below.")

            # ---- preview plot (drag a box on it to pick a region) ----------------------
            _propuesta = {"modo": _modo.lower(), "regiones": [tuple(r) for r in _borrador]}
            _mask_prev = sc.mascara_recorte(_eje_c, _propuesta)
            _fig_c = go.Figure()
            _idx_show = np.linspace(0, _X_c.shape[0] - 1, min(40, _X_c.shape[0])).astype(int)
            for _i in _idx_show:
                _fig_c.add_trace(_traza_espectro(_eje_c, _X_c[_i], mode="lines", showlegend=False,
                                            line=dict(width=0.6, color="rgba(120,120,120,0.35)"),
                                            hoverinfo="skip"))
            _fig_c.add_trace(go.Scatter(x=_eje_c, y=_X_c.mean(axis=0), mode="lines", name="Mean spectrum",
                                        line=dict(color="#0B3D54", width=2)))
            if _mask_prev is not None:
                _quitado = ~_mask_prev
                _lim = np.flatnonzero(np.diff(np.concatenate([[0], _quitado.astype(int), [0]])))
                for _a, _b in zip(_lim[::2], _lim[1::2]):
                    _fig_c.add_vrect(x0=_eje_c[_a], x1=_eje_c[_b - 1], fillcolor="red", opacity=0.18,
                                     line_width=0)
            _fig_c.update_layout(height=430, xaxis_title="Axis", yaxis_title="Signal",
                                 title="Spectra after preprocessing — red = removed by the crop (preview)",
                                 dragmode="select")
            if _eje_c[0] > _eje_c[-1]:
                _fig_c.update_xaxes(autorange="reversed")
            _cajas = []
            try:
                _evento = st.plotly_chart(_fig_c, on_select="rerun", selection_mode="box",
                                          key="crop_w_grafico", width='stretch')
                _cajas = list(getattr(getattr(_evento, "selection", None), "box", None) or [])
            except TypeError:                       # older Streamlit: plot without selection
                st.plotly_chart(_fig_c, width='stretch')
            if _cajas:
                _xs = _cajas[0].get("x") or []
                if len(_xs) == 2:
                    _s_lo, _s_hi = float(min(_xs)), float(max(_xs))
                    st.success(f"Selected on the plot: {_s_lo:.3f} – {_s_hi:.3f}")
                    if st.button("➕ Add the selected range as a region", key="crop_btn_agregar_sel"):
                        _borrador.append([_s_lo, _s_hi])
                        _rerun_tab()
            else:
                st.caption("Tip: drag a box over the plot to select a region, then click 'Add the selected range'.")

            # ---- summary + apply / reset ---------------------------------------------
            if _borrador and _mask_prev is None:
                st.warning("With these regions nothing would be cropped (all variables kept) or fewer "
                           "than 2 variables would remain. Adjust the regions or the Keep/Remove choice.")
            elif _mask_prev is not None:
                st.caption(f"Preview: **{int(_mask_prev.sum())}** of {len(_mask_prev)} variables kept "
                           f"({100 * _mask_prev.mean():.0f}%).")
                with st.expander("Preview of the cropped spectra (gaps = removed regions)"):
                    _fig_v = go.Figure()
                    for _i in _idx_show[:15]:
                        _fig_v.add_trace(_traza_espectro(_eje_c, np.where(_mask_prev, _X_c[_i], np.nan),
                                                    mode="lines", showlegend=False, line=dict(width=0.8)))
                    _fig_v.update_layout(height=320, xaxis_title="Axis", yaxis_title="Signal")
                    if _eje_c[0] > _eje_c[-1]:
                        _fig_v.update_xaxes(autorange="reversed")
                    st.plotly_chart(_fig_v, width='stretch')

            _ya = (_aplicado == _propuesta)
            if _aplicado:
                st.success(f"✅ Currently applied: {sc.describir_recorte(_aplicado)}.")
            else:
                st.info("No crop applied — the models use the full spectrum.")
            _a1, _a2, _ = st.columns([1.6, 1.4, 3])
            if _a1.button("✅ Apply crop", key="crop_btn_aplicar", type="secondary" if _ya else "primary",
                          disabled=_ya or _mask_prev is None,
                          help="Makes the cropped spectrum available in the model tabs and in Model "
                               "screening. Models already trained are flagged as out of date only if "
                               "they used the crop."):
                st.session_state["crop_aplicado"] = _propuesta
                st.session_state["crop_desc_aplicada"] = sc.describir_recorte(_propuesta)
                _rerun_tab()
            if _a2.button("↩ Reset (no crop)", key="crop_btn_reset", disabled=_aplicado is None):
                st.session_state.pop("crop_aplicado", None)
                st.session_state.pop("crop_desc_aplicada", None)
                _rerun_tab()
    with tabs[3]:
        _frag_tab_3()


# -----------------------------------------------------------------------
# TAB: MODEL SCREENING
# -----------------------------------------------------------------------
if _abierta(tabs[11]):
    # Each tab re-runs ON ITS OWN when one of its widgets changes (fragment), instead of
    # re-executing the whole app. Buttons that must refresh other parts call st.rerun().
    @st.fragment
    def _frag_tab_8():
        st.subheader("Model screening")
        gu.barra_guia("screening", "screening", pros=("seleccion", None))
        if sc is None:
            st.error("The module `screening_en.py` is missing. Upload it to the same folder as "
                     "`app_en.py` to enable this tab.")
        else:
            st.caption(
                "Try MANY combinations at once — spectrum (full / cropped) × preprocessing × variable "
                "selection × algorithm — with deliberately cheap settings, and get a ranking on the "
                "independent TEST set (AUC for classification, RMSE for regression, efficiency for SIMCA), "
                "with every training (CV) and test metric alongside. Use it to decide where to look, then "
                "refine the best candidates in the Classification / SIMCA / Regression tabs.")
            st.info("Fair comparison: every combination uses the SAME train/test split, and variable "
                    "selection (Boruta / genetic algorithm) is fitted on the training samples only — so "
                    "the test ranking is not inflated by selection done with test samples.", icon="ℹ️")

            _ids_s, _Xraw_s, _clases_s = datos_activos()
            _Xact_s = st.session_state.X_pret[indice_activo()]
            _valores_s = st.session_state.valores_y

            _tarea = st.selectbox("What do you want to screen?", ["Classification", "Regression", "SIMCA"],
                                  key="scr_w_tarea", help="Whether to compare pre-treatments, variable-selection strategies, or algorithms.")
            _datos_ok = True
            if _BASICO and _tarea == "SIMCA":
                st.info(T("🔒 SIMCA is an advanced tool. Switch to **🔬 I have experience** at the top of the page to unlock it."))
                _datos_ok = False
            if _datos_ok and _tarea in ("Classification", "SIMCA"):
                if _clases_s is None:
                    st.warning("Define classes in the left-hand panel first.")
                    _datos_ok = False
                elif pd.Series(_clases_s).value_counts().min() < 4:
                    st.error("Some class has fewer than 4 samples — too few to train, cross-validate and test.")
                    _datos_ok = False
            elif _datos_ok:
                if _valores_s is None:
                    st.warning("Load the reference values (left-hand panel) to screen regression models.")
                    _datos_ok = False

            if _datos_ok:
                _tipos = list(sc.RECETAS_POR_TIPO.keys())
                _tipo_def = st.session_state.get("pret_w_tipo", "NIR / MIR")
                _tipo_s = st.selectbox("Signal type", _tipos, index=_tipos.index(_tipo_def) if _tipo_def in _tipos else 0,
                                       key="scr_w_tipo", help="Decides which preprocessing combinations are offered.")

                # ---------------- spectrum
                _crop_ok = sc.mascara_recorte(np.array(st.session_state.numeros_onda), st.session_state.get("crop_aplicado")) is not None
                _opc_esp = ["Full spectrum"] + (["Cropped spectrum"] if _crop_ok else [])
                _esp_sel = st.multiselect("Spectrum", _opc_esp, default=_opc_esp, key=f"scr_w_esp_{len(_opc_esp)}",
                                          help="Cropped spectrum uses the regions defined in the Crop tab.")
                if not _crop_ok:
                    st.caption("To also screen a cropped spectrum, define and apply a crop in the ✂️ Crop tab first.")

                # ---------------- preprocessing combinations
                _recetas_cat = sc.RECETAS_POR_TIPO[_tipo_s]
                _recetas_sel = []
                with st.expander("☑️ Preprocessing combinations to test", expanded=True):
                    st.caption("Each is computed from the raw spectra on the full axis (smoothing / derivatives "
                               "with window 11, polynomial order 2).")
                    for _i, (_nom, _a, _b) in enumerate(_recetas_cat):
                        if st.checkbox(_nom, value=_i < sc.PREMARCADAS, key=f"scr_w_rec_{_tipo_s}_{_i}", help="Include this item in the report."):
                            _recetas_sel.append((_nom, _a, _b))
                    if st.checkbox(sc.RECETA_ACTUAL, value=False, key="scr_w_rec_actual",
                                   help="Uses the preprocessing currently applied in the Preprocessing tab as one more candidate."):
                        _recetas_sel.append(sc.RECETA_ACTUAL)

                # ---------------- selection + algorithms
                _sel_sel, _alg_sel, _var_sel = ["-"], ["SIMCA"], [0.95]
                if _tarea != "SIMCA":
                    _sel_sel = st.multiselect("Variable selection", ["None", "Boruta", "Genetic Algorithm"],
                                              default=["None"], key="scr_w_sel", disabled=_BASICO,
                                              help="Boruta / genetic algorithm are the slowest part of a screening.")
                    _cat = mu.crear_clasificadores() if _tarea == "Classification" else mu.crear_regresores()
                    _def_alg = ["LDA", "PLS-DA", "Random Forest", "SVM"] if _tarea == "Classification" \
                        else ["PLS", "Ridge", "Random Forest"]
                    _alg_sel = st.multiselect("Algorithms", list(_cat.keys()),
                                              default=[a for a in _def_alg if a in _cat],
                                              key=f"scr_w_alg_{_tarea}", help="Algorithms to compare. More algorithms = longer run.")
                else:
                    _var_sel = st.multiselect("Explained variance of each class model", [0.90, 0.95, 0.99],
                                              default=[0.95], format_func=lambda v: f"{v:.0%}", key="scr_w_var", help="Share of each class's variance retained. Higher = stricter class models: better detection of foreign samples, but more false rejections of genuine ones.")

                with st.expander("⚙️ Screening settings (defaults are chosen for speed)"):
                    _s1, _s2, _s3 = st.columns(3)
                    _cv_s = _s1.slider("Cross-validation folds", 3, 10, 5, key="scr_w_cv",
                                       help="Fewer folds = faster.") if _tarea != "SIMCA" else 5
                    _prop_s = _s2.slider("Independent test set (%)", 15, 40, 25, key="scr_w_prop", help="Share of samples kept apart and never used for training, to give an honest final test.") / 100
                    _clf_s = _tarea in ("Classification", "SIMCA")
                    _ms1, _ms2 = st.columns(2)
                    _opc_split_s = (["Random", "Random (stratified)", "Kennard-Stone (representative)"] if _clf_s
                                    else ["Random", "Random (stratified by y range)", "SPXY (extended Kennard-Stone)"])
                    _split_sel_s = _ms1.selectbox("Test set selection", _opc_split_s, index=1, key=f"scr_w_split_{_clf_s}",
                                                  help="Same three options as in the modelling tabs. In the screening ONE single split is made "
                                                       "(with the seed on the right) and EVERY pre-treatment, variable selection and algorithm "
                                                       "is trained and tested on exactly the same samples, so the comparison is fair.")
                    _seed_s = _ms2.number_input("Random seed", 0, 999999, 0, step=1, key="scr_w_seed",
                                                disabled=_split_sel_s.startswith(("Kennard", "SPXY")),
                                                help="Fixes the random split and the cross-validation folds. All candidates in the screening share this "
                                                     "same seed, so differences between them come from the method, not from a luckier split. "
                                                     "Kennard-Stone/SPXY are deterministic and ignore it.")
                    _split_m_s = ("kennard_stone" if _split_sel_s.startswith(("Kennard", "SPXY"))
                                  else ("random_strat" if "stratified" in _split_sel_s else "random"))
                    _opt_s = _s3.checkbox("Optimize hyperparameters (fastest preset)", value=False, key="scr_w_opt",
                                          disabled=(_tarea == "SIMCA") or _BASICO,
                                          help="Random search over 8 combinations — the quickest search option, but it still "
                                               "multiplies the time by several. Off by default: screen first, then "
                                               "optimize the best candidates in the Classification / Regression tabs.")
                    if _BASICO:
                        _opt_s = False
                    _hp_op = ["As applied in the 🎛️ Hyperparameters tab"] + HP_MODOS
                    _hp_sel = st.selectbox("Hyperparameter ranges for the optimization", _hp_op, key="scr_w_hpmodo",
                                           disabled=(not _opt_s) or _tarea == "SIMCA" or _BASICO,
                                           help="Only used when 'Optimize hyperparameters' is ticked. 'As applied' uses whatever you applied "
                                                "in the Hyperparameters tab; the other options override it for this screening only.")
                    _hp_modo_s = None if _hp_sel == _hp_op[0] else _hp_sel
                    _b1, _b2, _b3 = st.columns(3)
                    _bor_s = _b1.number_input("Boruta iterations", 10, 500, 50, step=10, key="scr_w_bor", disabled=_BASICO, help="More iterations give a more stable selection but take longer.")
                    _gap_s = _b2.number_input("Genetic algorithm: population", 6, 60, 12, step=2, key="scr_w_gap", disabled=_BASICO, help="Candidate solutions per generation. Larger = better search, slower.")
                    _gag_s = _b3.number_input("Genetic algorithm: generations", 3, 40, 6, step=1, key="scr_w_gag", disabled=_BASICO, help="Generations of the search. More = better, slower.")

                # ---------------- how big is this?
                _n_rec, _n_esp = len(_recetas_sel), len(_esp_sel)
                _total_s = sc.contar_combinaciones(_tarea, _n_rec, _n_esp, len(_sel_sel), len(_alg_sel), max(1, len(_var_sel)))
                _n_sel_runs = _n_rec * _n_esp * (int("Boruta" in _sel_sel) + int("Genetic Algorithm" in _sel_sel))
                st.markdown(f"**This will train {_total_s} model(s)**"
                            + (f", including {_n_sel_runs} variable-selection run(s) (the slow part)." if _n_sel_runs else "."))
                if _total_s > 150 or _n_sel_runs > 12:
                    st.warning("That is a big screening and may take a long time (minutes to hours, depending on "
                               "your data and the server). Consider ticking fewer preprocessing combinations, "
                               "algorithms or selection methods. Results are kept as they come in, so an "
                               "interrupted run can be resumed.")

                # ---------------- run / resume / clear
                _filas_s = st.session_state.get("scr_filas", [])
                _estado_s = st.session_state.get("scr_estado")
                _firma_s = (firma_datos_activos(), st.session_state.get("crop_desc_aplicada"), _tarea)
                _r1, _r2, _r3, _ = st.columns([1.5, 1.2, 1, 3])
                _iniciar = _r1.button("▶ Run screening" if not _filas_s else "🔁 Run again (replaces results)",
                                      type="primary", key="scr_btn_run",
                                      disabled=_total_s == 0 or not _alg_sel or not _esp_sel or not _sel_sel)
                _reanudar = _r2.button("⏩ Resume", key="scr_btn_resume",
                                       disabled=not (_estado_s and not _estado_s.get("completo")
                                                     and _estado_s.get("tarea") == _tarea),
                                       help="Continues an interrupted run, skipping what is already done.")
                if _r3.button("🧹 Clear", key="scr_btn_clear", disabled=not _filas_s):
                    st.session_state.pop("scr_filas", None)
                    st.session_state.pop("scr_estado", None)
                    _rerun_tab()

                if _iniciar or _reanudar:
                    if _iniciar:
                        st.session_state["scr_filas"] = []
                        _ya = set()
                    else:
                        _ya = {(f.get("Preprocessing"), f.get("Spectrum"), f.get("Variable selection"), f.get("Algorithm"))
                               for f in st.session_state.get("scr_filas", []) if not isinstance(f.get("Error"), str)}
                        st.session_state["scr_filas"] = [f for f in st.session_state.get("scr_filas", [])
                                                         if not isinstance(f.get("Error"), str)]
                    _mask_ok = np.ones(len(_Xraw_s), dtype=bool)
                    if _tarea == "Regression":
                        _y_s = np.asarray(_valores_s[indice_activo()], dtype=float)
                        _mask_ok = ~np.isnan(_y_s)
                    else:
                        _y_s = np.asarray(_clases_s)
                    _datos_s = {"X_raw": _Xraw_s[_mask_ok], "eje_raw": np.array(st.session_state.numeros_onda, dtype=float),
                                "X_actual": _Xact_s[_mask_ok], "eje_actual": np.array(st.session_state.numeros_onda_pret, dtype=float),
                                "ids": _ids_s[_mask_ok], "y": _y_s[_mask_ok],
                                "crop": st.session_state.get("crop_aplicado")}
                    _cfg_s = {"recetas": _recetas_sel,
                              "espectros": ["Full" if e.startswith("Full") else "Cropped" for e in _esp_sel],
                              "seleccion": _sel_sel, "algoritmos": _alg_sel, "varianzas": _var_sel,
                              "cv": _cv_s, "prop_test": _prop_s, "metodo_split": _split_m_s, "semilla": int(_seed_s), "optimizar": bool(_opt_s) and _tarea != "SIMCA",
                              "boruta_iter": int(_bor_s), "ga_pob": int(_gap_s), "ga_gen": int(_gag_s),
                              "grilla_fn": (lambda _nm, _X, _y, _clf=(_tarea == "Classification"), _pt=_prop_s, _cv=_cv_s, _hm=_hp_modo_s:
                                            _grilla_activa(_nm, _clf, _X, _y, 0.0, int(_cv), _hm))}
                    st.session_state["scr_estado"] = {"tarea": _tarea, "tipo": _tipo_s, "total": _total_s,
                                                      "completo": False, "firma": _firma_s,
                                                      "fecha": _fecha_hora_informe()}
                    _barra = st.progress(0.0, text="Starting...")
                    _t_ini = time.time()

                    def _progreso_scr(h, t, texto):
                        _el = time.time() - _t_ini
                        _eta = f" · ~{int(_el / h * (t - h))} s left" if h > 0 else ""
                        _barra.progress(min(h / max(t, 1), 1.0), text=f"{h}/{t} — {texto} · {int(_el)} s elapsed{_eta}")

                    sc.ejecutar(_tarea, _datos_s, _cfg_s, aplicar_paso,
                                al_terminar_fila=lambda f: st.session_state["scr_filas"].append(f),
                                progreso=_progreso_scr, ya_hechas=_ya)
                    st.session_state["scr_estado"]["completo"] = True
                    _rerun_tab()

                # ---------------- results
                if _filas_s:
                    _tarea_res = (_estado_s or {}).get("tarea", _tarea)
                    _df_rank = sc.tabla_ranking(_filas_s, _tarea_res)
                    _principal = sc.CLAVE_RANKING[_tarea_res][0]
                    if _estado_s and not _estado_s.get("completo"):
                        st.warning(f"⏸ Incomplete run: {len(_filas_s)} of {_estado_s['total']} combinations finished "
                                   "(the run was interrupted). The ranking below covers only those — click Resume.")
                    if _estado_s and _estado_s.get("firma") != _firma_s and _estado_s.get("tarea") == _tarea:
                        st.warning("⚠️ This ranking was computed with a data / preprocessing / outlier / crop state "
                                   "that is different from the current one.")
                    st.markdown(f"#### Ranking — {_estado_s['tarea'] if _estado_s else _tarea}")
                    st.caption(f"Sorted by **{_principal}** "
                               f"({'lower is better' if sc.CLAVE_RANKING[_tarea_res][1] else 'higher is better'}); ties are "
                               "broken by the next test metric and then by the simpler model (fewer variables). "
                               "All training (CV) and test metrics are shown.")
                    _fmt = {c: "{:.3f}" for c in _df_rank.columns
                            if c not in ("Rank", "n variables", "Time (s)") and pd.api.types.is_float_dtype(_df_rank[c])}
                    st.dataframe(_df_rank.style.format(_fmt, na_rep="–"), width='stretch', height=min(620, 60 + 35 * len(_df_rank)))
                    _d1, _d2, _ = st.columns([1.5, 1.5, 3])
                    _d1.download_button("⬇️ Download ranking (Excel)", data=df_a_excel_bytes({"screening_ranking": _df_rank}),
                                        file_name="model_screening_ranking.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key="scr_dl_xlsx")
                    _d2.download_button("⬇️ Download ranking (CSV)", data=_df_rank.to_csv(index=False).encode("utf-8"),
                                        file_name="model_screening_ranking.csv", mime="text/csv", key="scr_dl_csv")

                    # ---- take a candidate to the other tabs
                    _sc1, _sc2, _sc3 = st.columns([1.5, 1.9, 3])
                    with _sc1:
                        gu.how_to_interpret("screening", key="how_res_scr")
                    with _sc2:
                        gu.what_next("screening", key="next_res_scr")
                    gu.nota("scr_table", "ranking table", "screening")
                    st.markdown("#### 📊 What works best on average?")
                    _mets_g = sc.metricas_disponibles(_tarea_res, _df_rank)
                    _facs_g = sc.factores_con_variacion(_df_rank)
                    if not _mets_g or not _facs_g:
                        st.info("There is nothing to compare yet: only one level of every factor was tested, or no combination "
                                "finished with a usable metric.")
                    else:
                        _g1, _g2 = st.columns([1.2, 2])
                        _met_g = _g1.selectbox("Metric", _mets_g, key=f"scr_w_graf_met_{_tarea_res}",
                                               help="Shown for every chart. For RMSE, lower is better.")
                        _vista_g = _g2.radio("View", ["Bars (average)", "Pies (share of the top N)", "Heatmap (two factors)"],
                                             horizontal=True, key="scr_w_graf_vista", help="Choose how the results are displayed.")
                        _menor_g = sc.menor_es_mejor(_met_g)
                        _incompleto = bool(_estado_s) and not _estado_s.get("completo")
                        st.caption(("Lower is better. " if _menor_g else "Higher is better. ")
                                   + "Every bar averages ALL the combinations that include that level, so when the screening is "
                                     "complete each level is compared fairly across the other factors"
                                   + (" — ⚠ this run is incomplete, so the comparison may be unbalanced." if _incompleto else "."))
                        _verde, _gris = "#14B8A6", "#9DB4C0"
                        if _vista_g.startswith("Bars"):
                            _zoom = st.checkbox("Zoom the axis to the values (makes small differences easier to see)",
                                                value=not _menor_g, key="scr_w_graf_zoom", help="Narrows the axis to the data range so small differences are visible (bars no longer start at zero).")
                            _cols_g = st.columns(2)
                            for _i, _f in enumerate(_facs_g):
                                _res = sc.resumen_por_factor(_df_rank, _met_g, _f)
                                if _res.empty:
                                    continue
                                _fig_g = go.Figure(go.Bar(
                                    x=_res["mean"], y=_res[_f].astype(str), orientation="h",
                                    error_x=dict(type="data", array=_res["std"], color="#555555", thickness=1.2),
                                    marker_color=[_verde] + [_gris] * (len(_res) - 1),
                                    text=[f"{v:.3f}" for v in _res["mean"]], textposition="outside", cliponaxis=False,
                                    customdata=np.stack([_res["std"], _res["best"], _res["n"]], axis=-1),
                                    hovertemplate="<b>%{y}</b><br>average: %{x:.4f}<br>std: %{customdata[0]:.4f}"
                                                  "<br>best: %{customdata[1]:.4f}<br>combinations: %{customdata[2]:.0f}<extra></extra>"))
                                _fig_g.update_yaxes(autorange="reversed")
                                if _zoom:
                                    _lo = float((_res["mean"] - _res["std"]).min())
                                    _hi = float((_res["mean"] + _res["std"]).max())
                                    _pad = max((_hi - _lo) * 0.25, 1e-6)
                                    _fig_g.update_xaxes(range=[_lo - _pad, _hi + _pad])
                                _fig_g.update_layout(title=f"By {_f.lower()}  (best on top)", xaxis_title=f"{_met_g} — mean ± std",
                                                     height=max(230, 90 + 40 * len(_res)), margin=dict(l=10, r=30, t=50, b=40))
                                _cols_g[_i % 2].plotly_chart(_fig_g, width='stretch')
                        elif _vista_g.startswith("Pies"):
                            _n_ok = int(_df_rank[_met_g].notna().sum())
                            _n_top = st.slider("How many of the best combinations?", 3, max(3, min(30, _n_ok)),
                                               min(10, max(3, _n_ok)), key="scr_w_graf_topn",
                                               help="The pies show what share of the N best combinations (by the chosen metric) "
                                                    "uses each level.")
                            _cols_g = st.columns(2)
                            for _i, _f in enumerate(_facs_g):
                                _comp = sc.composicion_top(_df_rank, _met_g, _f, _n_top)
                                if _comp.empty:
                                    continue
                                _fig_g = go.Figure(go.Pie(labels=list(_comp.index), values=list(_comp.values), hole=0.4,
                                                          textinfo="label+percent", sort=False,
                                                          marker=dict(colors=px.colors.qualitative.Set2[:len(_comp)])))
                                _fig_g.update_layout(title=f"Top {_n_top} — by {_f.lower()}", height=330, showlegend=False,
                                                     margin=dict(l=10, r=10, t=50, b=10))
                                _cols_g[_i % 2].plotly_chart(_fig_g, width='stretch')
                        else:
                            if len(_facs_g) < 2:
                                st.info("A heatmap needs at least two factors with more than one level.")
                            else:
                                _h1, _h2 = st.columns(2)
                                _f_fil = _h1.selectbox("Rows", _facs_g, index=_facs_g.index("Preprocessing") if "Preprocessing" in _facs_g else 0,
                                                       key="scr_w_graf_fil", help="Which variable defines the rows of the table.")
                                _otros = [f for f in _facs_g if f != _f_fil]
                                _f_col = _h2.selectbox("Columns", _otros, index=_otros.index("Algorithm") if "Algorithm" in _otros else 0,
                                                       key="scr_w_graf_col", help="Which variable defines the columns of the table.")
                                _piv = _df_rank.pivot_table(index=_f_fil, columns=_f_col, values=_met_g, aggfunc="mean")
                                _fig_g = px.imshow(_piv, text_auto=".3f", aspect="auto",
                                                   color_continuous_scale="RdYlGn_r" if _menor_g else "RdYlGn",
                                                   labels=dict(color=_met_g))
                                _fig_g.update_layout(height=max(300, 90 + 48 * len(_piv)), title=f"Average {_met_g}: {_f_fil.lower()} × {_f_col.lower()}",
                                                     margin=dict(l=10, r=10, t=50, b=10))
                                st.plotly_chart(_fig_g, width='stretch')
                                gu.nota("scr_heat", "screening heat map", "screening")

                    st.markdown("#### Refine a candidate")
                    _top = _df_rank[_df_rank.get("Error", pd.Series([np.nan] * len(_df_rank))).isna()].head(20)
                    if len(_top):
                        _rk = st.selectbox("Candidate (rank)", list(_top["Rank"]), key="scr_w_rank",
                                           format_func=lambda r: f"#{r}  {_top[_top['Rank'] == r].iloc[0]['Preprocessing']} · "
                                           f"{_top[_top['Rank'] == r].iloc[0]['Spectrum']} · "
                                           f"{_top[_top['Rank'] == r].iloc[0]['Variable selection']} · "
                                           f"{_top[_top['Rank'] == r].iloc[0]['Algorithm']}", help="Which of the ranked candidates to inspect.")
                        _fila_top = _top[_top["Rank"] == _rk].iloc[0]
                        _tipo_run = (_estado_s or {}).get("tipo", _tipo_s)
                        _mapa_rec = {n: (a, b) for n, a, b in sc.RECETAS_POR_TIPO.get(_tipo_run, [])}
                        _rec_n = _fila_top["Preprocessing"]
                        _pasos_txt = (f"set Step A = **{_mapa_rec[_rec_n][0]}**, Step B = **{_mapa_rec[_rec_n][1]}** (A → B), then Apply"
                                      if _rec_n in _mapa_rec else "keep the preprocessing you already have applied")
                        st.markdown(
                            f"To reproduce it: in **Preprocessing** {_pasos_txt}; in the model tab choose "
                            f"**{'Cropped spectrum' if _fila_top['Spectrum'] == 'Cropped' else 'Full spectrum (no crop)'}**, "
                            f"variable selection **{_fila_top['Variable selection']}** and algorithm **{_fila_top['Algorithm']}**.")
                        if st.button("🧪 Use this candidate: set up Preprocessing and the model tab", key="scr_btn_prep",
                                     help="Fills in the Preprocessing steps of this recipe and leaves ONLY this algorithm "
                                          "(with its variable selection and spectrum choice) selected in the model tab, "
                                          "so you can train it straight away."):
                            _hecho = []
                            if _rec_n in _mapa_rec:
                                st.session_state["pret_w_tipo"] = _tipo_run
                                st.session_state[f"pret_w_a_{_tipo_run}"] = _mapa_rec[_rec_n][0]
                                st.session_state[f"pret_w_b_{_tipo_run}"] = _mapa_rec[_rec_n][1]
                                st.session_state["pret_w_orden"] = "A → B (recommended)"
                                _hecho.append("Preprocessing")
                            _alg_top, _vs_top = str(_fila_top["Algorithm"]), str(_fila_top["Variable selection"])
                            _pref_m = {"Classification": "clf", "Regression": "reg", "SIMCA": "simca"}.get(_tarea_res)
                            _ok_alg = False
                            if _tarea_res == "Classification" and _alg_top in mu.crear_clasificadores():
                                st.session_state["ac_clf_models"] = [_alg_top]
                                _ok_alg = True
                            elif _tarea_res == "Regression" and _alg_top in mu.crear_regresores():
                                st.session_state["ac_reg_models"] = [_alg_top]
                                _ok_alg = True
                            elif _tarea_res == "SIMCA":
                                _mv = re.search(r"(\d+)\s*%", _alg_top)
                                if _mv and 80 <= int(_mv.group(1)) <= 99:
                                    st.session_state["ac_simca_var"] = int(_mv.group(1)) / 100
                                    _ok_alg = True
                            if _ok_alg:
                                _hecho.append(f"{_tarea_res} → {_alg_top}")
                                if _vs_top in ("None", "Boruta", "Genetic Algorithm"):
                                    st.session_state[{"Classification": "ac_clf_varsel", "Regression": "sel_var_reg"}.get(_tarea_res, "_nada")] = _vs_top
                            if _pref_m and st.session_state.get("crop_aplicado"):
                                st.session_state[f"{_pref_m}_w_origen"] = ("Cropped spectrum" if _fila_top["Spectrum"] == "Cropped"
                                                                           else "Full spectrum (no crop)")
                            if _hecho:
                                st.success(T("Ready:") + " " + " · ".join(_hecho) + ". "
                                           + (T("Open the 🧪 Preprocessing tab and click Apply first, then open the model tab and train.")
                                              if _rec_n in _mapa_rec else T("Now open the model tab and train.")))
                            else:
                                st.info("Nothing could be set automatically for this candidate; follow the steps above.")
    with tabs[11]:
        _frag_tab_8()


# -----------------------------------------------------------------------
# TAB: HYPERPARAMETERS (search spaces: defaults, dataset-based, editable)
# -----------------------------------------------------------------------
if _abierta(tabs[30]):
    @st.fragment
    def _frag_tab_hp():
        st.subheader("Hyperparameter search spaces")
        gu.barra_guia("hyper", pros=("optimizacion", None))
        st.markdown(
            "A **hyperparameter** is a setting of an algorithm that you (or an automatic search) must choose — the number of PLS components, "
            "the regularisation strength, the number of neighbours… Here you can see **the range explored for every algorithm** when you tick "
            "*Optimize hyperparameters* (in Classification, Regression and Model screening), see a **suggestion adapted to your dataset**, and edit it.")

        modo = st.radio("Which ranges should the optimization use?", HP_MODOS, index=1, key="hp_modo", horizontal=True,
                        help="Library defaults: fixed grids, the same for every dataset.\n"
                             "Dataset-based: ranges computed from the number of samples, variables, classes and the spread of y (recalculated "
                             "automatically when your data change).\n"
                             "My edited ranges: you decide each value.")
        st.caption({
            HP_MODOS[0]: "Fixed grids — a safe, general-purpose choice.",
            HP_MODOS[1]: "Ranges scale with your data: e.g. at most one latent variable per 5 training samples, a KNN neighbourhood around √n, "
                         "SVM γ around 1/(p·Var(X)). Hover over each field to see the equation.",
            HP_MODOS[2]: "Edit the lists below (comma-separated). Use *Recalculate from my dataset* to restart from the recommendation.",
        }[modo])

        ids_a, _, clases_a = datos_activos()
        Xh = st.session_state.X_pret[indice_activo()]
        vy = st.session_state.get("valores_y")
        y_val = None
        if vy is not None:
            y_val = np.asarray(vy, dtype=float)[indice_activo()]
        tareas = []
        if clases_a is not None and len(set(clases_a)) >= 2:
            tareas.append("Classification")
        if y_val is not None and np.isfinite(y_val).sum() >= 10:
            tareas.append("Regression")
        if not tareas:
            st.info("Load classes or reference values (left-hand panel) so the recommendations can use them. "
                    "Meanwhile you can see the library defaults.")
            tareas = ["Classification", "Regression"]
            datos_ok = False
        else:
            datos_ok = True
        c1, c2, c3 = st.columns([2, 1, 1])
        tarea = c1.radio("Task", tareas, horizontal=True, key="hp_tarea")
        prop_t = c2.number_input("Test fraction", 0.0, 0.5, 0.2, 0.05, key="hp_prop",
                                 help="Fraction of samples kept out for the final test — the search only sees the rest. "
                                      "Used in the formulas (n_train = n × (1 − test fraction)).")
        cvk = c3.number_input("CV folds", 3, 10, 5, 1, key="hp_cv",
                              help="Folds used inside the search. Each model is fitted on n_train × (k−1)/k samples: n_fit.")
        es_clf = tarea == "Classification"
        base = mu.GRILLAS_CLASIFICACION if es_clf else mu.GRILLAS_REGRESION
        tag = "clf" if es_clf else "reg"

        if datos_ok:
            if es_clf:
                mask = np.ones(len(clases_a), dtype=bool)
                yh = np.asarray(clases_a)
            else:
                mask = np.isfinite(y_val)
                yh = y_val[mask]
            d = hu.resumen_datos(Xh[mask], yh, es_clf, float(prop_t), int(cvk))
            st.markdown(
                f"**Your data:** n = {d['n']} samples · p = {d['p']} variables" + (f" · {d['k']} classes (smallest: {d['min_c']})" if es_clf else
                f" · y: mean {d['y_mean']:.3g}, SD {d['y_sd']:.3g}") + f" → n_train ≈ {d['n_tr']}, n_fit ≈ {d['n_fit']}")
        else:
            d = None

        firma = (tarea, round(float(prop_t), 3), int(cvk), None if d is None else (d["n"], d["p"], d.get("k"), round(d.get("y_sd", 0.0), 6)))
        recalc = False
        cA, cB, _ = st.columns([1.6, 2.2, 3])
        if cA.button("🔄 Recalculate from my dataset", key="hp_recalc", disabled=d is None,
                     help="Replaces the edited values by the recommendation computed from the current data."):
            recalc = True
        auto = cB.checkbox("Recalculate automatically when the data change", value=True, key="hp_auto",
                           help="If the number of samples, variables, classes, the test fraction or the CV folds change, the edited ranges "
                                "are recomputed. Untick it to keep your edits fixed.")
        if auto and d is not None and st.session_state.get("hp_firma") != firma:
            recalc = True
        if d is not None:
            st.session_state["hp_firma"] = firma

        rec_all = {nm: hu.espacio_recomendado(nm, g, d) for nm, g in base.items()} if d is not None else {}
        edit_store = st.session_state.setdefault("hp_edit", {}).setdefault(tag, {})

        # ---------- Apply: the chosen ranges are the ones the optimization will use everywhere
        if "hp_modo_aplicado" not in st.session_state:
            st.session_state["hp_modo_aplicado"] = HP_MODOS[1]
        _aplicado = st.session_state["hp_modo_aplicado"]
        _ya_hp = (_aplicado == modo) and (modo != HP_MODOS[2]
                                          or st.session_state.get("hp_edit_aplicado") == st.session_state.get("hp_edit"))
        if _ya_hp:
            st.success(f"✅ Applied: **{_aplicado}**. These are the ranges used when you tick *Optimize hyperparameters* "
                       "(Model screening, Classification, Regression).")
        else:
            st.warning(f"⚠️ You are looking at **{modo}**, but the optimization is still using **{_aplicado}**. "
                       "Click **Apply** to use this one.")
        _ca1, _ca2, _ = st.columns([1.5, 2, 3])
        if _ca1.button("✅ Apply these ranges", key="hp_btn_aplicar", type="secondary" if _ya_hp else "primary",
                       disabled=_ya_hp, help="Makes this choice the one used by every optimization in the app."):
            import copy as _copy
            st.session_state["hp_modo_aplicado"] = modo
            st.session_state["hp_edit_aplicado"] = _copy.deepcopy(st.session_state.get("hp_edit", {}))
            _rerun_tab()
        _filas_ap = []
        for _nm in base:
            if d is not None:
                _g = _grilla_activa(_nm, es_clf, Xh[mask], yh, float(prop_t), int(cvk))
            else:
                _g = base[_nm]
            for _par, _vals in (_g or {}).items():
                _filas_ap.append({"Algorithm": _nm, "Hyperparameter": _par, "Values tried": hu.valores_a_texto(list(_vals))})
        with st.expander(f"📋 Ranges applied right now — {tarea} ({_aplicado})", expanded=True):
            st.dataframe(pd.DataFrame(_filas_ap), hide_index=True, width="stretch")
            st.caption("Below you can inspect and edit each algorithm's range. Nothing changes in the rest of the app until you click Apply.")
        n_mal = 0
        for nombre, grilla_def in base.items():
            rec = rec_all.get(nombre)
            if modo == HP_MODOS[0] or rec is None:
                fuente = {k: (list(v), "Library default.") for k, v in grilla_def.items()}
            elif modo == HP_MODOS[1]:
                fuente = rec
            else:
                fuente = rec or {k: (list(v), "Library default.") for k, v in grilla_def.items()}
            n_comb = hu.tamano_grilla({k: v[0] for k, v in fuente.items()})
            with st.expander(f"{nombre} — {n_comb} combination(s)"):
                vals_ed = {}
                ok_all = True
                for par, (vals, formula) in fuente.items():
                    kk = f"hp_txt_{tag}_{nombre}_{par}_{modo[:3]}"
                    if modo != HP_MODOS[2] or recalc or kk not in st.session_state:
                        st.session_state[kk] = hu.valores_a_texto(vals)
                    txt = st.text_input(par, key=kk, disabled=(modo != HP_MODOS[2]),
                                        help=f"Equation / rationale:\n{formula}\n\nLibrary default: {hu.valores_a_texto(grilla_def.get(par, []))}")
                    st.caption(f"ƒ {formula}")
                    if modo == HP_MODOS[2]:
                        try:
                            vals_ed[par] = hu.texto_a_valores(txt)
                        except ValueError as e:
                            st.error(f"{par}: {e}")
                            ok_all = False
                if modo == HP_MODOS[2]:
                    if ok_all and vals_ed:
                        edit_store[nombre] = vals_ed
                    else:
                        n_mal += 1
                if n_comb > 60:
                    st.warning("This grid is large: the optimization may be slow for this algorithm. Remove values or use the *Fast* or "
                               "*Bayesian* search.")
                st.caption("An optimization budget (Fast / Balanced / Thorough / Bayesian) decides how many of these combinations are actually tried.")
        if n_mal:
            st.warning(f"{n_mal} algorithm(s) have an invalid list: their previous valid ranges (or the recommendation) are used until you fix them.")

        # ---------- where did the last optimization land? (edge-of-range diagnosis)
        hp_last = st.session_state.get("clf_hiperparametros" if es_clf else "reg_hiperparametros") or {}
        if hp_last:
            st.markdown("#### Where did the last optimization land?")
            st.caption("If the best value is at the **edge** of the range, the true optimum may lie beyond it — widen that range and run again.")
            filas = []
            for nm, params in hp_last.items():
                for k, v in (params or {}).items():
                    g = (base.get(nm) or {}).get(k)
                    ef = None
                    if d is not None:
                        ef = (rec_all.get(nm) or {}).get(k, (None,))[0] if modo != HP_MODOS[0] else g
                    nums = [x for x in (ef or []) if isinstance(x, (int, float)) and not isinstance(x, bool)]
                    borde = ""
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and len(nums) >= 3:
                        if v <= min(nums) or v >= max(nums):
                            borde = "⚠️ at the edge — widen"
                    filas.append({"Algorithm": nm, "Hyperparameter": k, "Best value": str(v), "Check": borde})
            if filas:
                st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

        with st.expander("📘 How to choose a range by hand — rules of thumb"):
            st.markdown(
                "- **Number of components (PLS, PCR, PLS-DA):** from 1 up to about n_train/5, never more than the number of variables; "
                "the best value is usually where the cross-validated error stops improving.\n"
                "- **Regularisation (Ridge, Lasso, Elastic Net, Logistic C, MLP alpha):** search on a **logarithmic** scale (…0.01, 0.1, 1, 10…). "
                "More variables than samples → stronger regularisation.\n"
                "- **SVM:** C and γ interact; explore both on a log scale. γ around 1/(p·Var(X)) is a good centre.\n"
                "- **Trees / forests:** depth about log₂(n); the number of trees only costs time — more is never worse.\n"
                "- **KNN:** k around √n, odd numbers, never above the smallest class size.\n"
                "- **If the best value is at the edge of the range, widen it.** If it is in the middle, the range is fine.")

        with st.expander("📖 Glossary of hyperparameters"):
            _glos = [
                ("n_components", "PLS, PLS-DA, PCR", "Number of latent variables (or principal components). Too few: the model misses information; too many: it fits noise."),
                ("alpha (Ridge, Lasso, Elastic Net)", "Ridge, Lasso, Elastic Net", "Regularisation strength. Larger values shrink the coefficients more (simpler, more stable model). Lasso also sets some to exactly zero."),
                ("alpha (MLP)", "Neural Network (MLP)", "L2 penalty on the network weights. Larger values reduce overfitting."),
                ("l1_ratio", "Elastic Net", "Mix between Lasso (1) and Ridge (0) penalties."),
                ("C", "SVM, SVR, Logistic Regression", "Inverse of the regularisation strength (the opposite of alpha). Large C = fit the training data closely; small C = smoother, simpler model."),
                ("gamma", "SVM, SVR (RBF kernel)", "How far the influence of a single sample reaches. Large gamma = very local, wiggly boundary (risk of overfitting); small = smooth."),
                ("epsilon", "SVR", "Width of the tube around the prediction inside which errors are ignored."),
                ("n_estimators", "Random Forest, XGBoost", "Number of trees. More trees never hurt accuracy; they only cost time."),
                ("max_depth", "Random Forest, Decision Tree, XGBoost", "Maximum depth of each tree. Deep trees capture detail but overfit."),
                ("max_features", "Random Forest", "Variables considered at each split. Fewer = more diverse trees."),
                ("learning_rate", "XGBoost", "How much each new tree corrects the previous ones. Small values need more trees but generalise better."),
                ("n_neighbors", "KNN", "How many closest samples vote. Small k = noisy; large k = blurred boundaries."),
                ("hidden_layer_sizes", "Neural Network (MLP)", "Number of neurons in each hidden layer."),
                ("var_smoothing", "Naive Bayes", "Small value added to the variances for numerical stability; acts as regularisation."),
                ("solver", "LDA", "Algorithm used to fit the discriminant (e.g. SVD or shrinkage-based)."),
            ]
            st.dataframe(pd.DataFrame(_glos, columns=["Hyperparameter", "Used in", "What it does"]),
                         hide_index=True, width="stretch")
    with tabs[30]:
        _frag_tab_hp()
