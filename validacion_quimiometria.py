"""
validacion_quimiometria.py
--------------------------
Utilidades para validar modelos de regresión (PLS, Random Forest, etc.)
sobre datos espectroscópicos con RÉPLICAS, evitando la fuga de información
entre calibración y test.

Contenido:
  1. obtener_grupos()        -> ID de muestra a partir de una columna o del nombre
  2. particion_agrupada()    -> train/test sin separar réplicas (aleatorio o Kennard-Stone)
  3. resumen_y()             -> media, desvío y rango de y: total / train / test
  4. evaluar_modelo()        -> métricas CV (GroupKFold) + test
  5. particiones_repetidas() -> R² en N particiones aleatorias agrupadas (para el boxplot)
  6. boxplot_r2()            -> gráfico Plotly en español, marcando el split fijo
  7. proteger_terminos()     -> el navegador traduce la página, pero no los nombres
                               de algoritmos (Random Forest, PLS, SVM...); nt() para textos

Regla de oro: todo preprocesamiento que "aprende" de los datos (autoescalado,
centrado, selección de variables) va DENTRO del Pipeline, para que se ajuste
solo con train en cada fold.
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict


# ---------------------------------------------------------------------------
# 1. Grupos (ID de muestra)
# ---------------------------------------------------------------------------
# Quita sufijos típicos de réplica: "M12_R1", "M12-r2", "M12 rep3", "M12R1",
# "M12 (2)", "M12_1", "M12.2", "M12a"/"M12b" (letra tras un número).
# NO toca "M1" ni "Vino 7": un número sin separador _ - . no se considera réplica.
# Ante la duda, usar una columna con el ID de muestra (col_id).
_PATRON_REPLICA = re.compile(
    r"([_\-.](r|rep|replica|réplica)?\d+"
    r"|\s*\(\d+\)"
    r"|[\s_\-.]*(r|rep|replica|réplica)\s*\d+"
    r"|(?<=\d)[a-z])$",
    re.IGNORECASE,
)


def obtener_grupos(df, col_id=None, nombres=None):
    """Devuelve un array con el ID de muestra de cada fila.

    - Si `col_id` existe en df, se usa esa columna (opción más segura).
    - Si no, se deriva del nombre de la muestra (índice o `nombres`)
      quitando el sufijo de réplica.
    """
    if col_id is not None and col_id in df.columns:
        return df[col_id].astype(str).to_numpy()

    nombres = df.index if nombres is None else nombres
    return np.array([_PATRON_REPLICA.sub("", str(n)).strip() or str(n) for n in nombres])


def info_replicas(grupos):
    """Resumen rápido para mostrar en la app."""
    conteo = pd.Series(grupos).value_counts()
    return {
        "Espectros": int(conteo.sum()),
        "Muestras únicas": int(conteo.size),
        "Réplicas por muestra (mín–máx)": f"{conteo.min()}–{conteo.max()}",
        "Hay réplicas": bool(conteo.max() > 1),
    }


# ---------------------------------------------------------------------------
# 2. Partición train/test agrupada
# ---------------------------------------------------------------------------
def _kennard_stone(X, n_sel):
    """Índices seleccionados por Kennard-Stone (para calibración)."""
    X = np.asarray(X, dtype=float)
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    sel = [i, j]
    dmin = np.minimum(d[i], d[j])
    dmin[sel] = -1
    while len(sel) < n_sel:
        k = int(np.argmax(dmin))
        sel.append(k)
        dmin = np.minimum(dmin, d[k])
        dmin[sel] = -1
    return np.array(sel)


def particion_agrupada(X, y, grupos, test_size=0.25, metodo="aleatorio", seed=42):
    """Divide en train/test SIN separar réplicas de una misma muestra.

    metodo:
      "aleatorio"     -> GroupShuffleSplit (estimación honesta, recomendado)
      "kennard_stone" -> KS sobre el espectro medio de cada muestra.
                         Ojo: deja los extremos en calibración, por lo que el
                         test tiende a dar MEJOR que el CV (interpola).
    Devuelve (idx_train, idx_test) como arrays de posiciones.
    """
    X = np.asarray(X, dtype=float)
    grupos = np.asarray(grupos)

    if metodo == "aleatorio":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        idx_train, idx_test = next(gss.split(X, y, grupos))
        return idx_train, idx_test

    if metodo == "kennard_stone":
        unicos = pd.unique(grupos)
        X_medio = np.vstack([X[grupos == g].mean(axis=0) for g in unicos])
        n_cal = int(round(len(unicos) * (1 - test_size)))
        cal = set(unicos[_kennard_stone(X_medio, n_cal)])
        en_train = np.array([g in cal for g in grupos])
        return np.where(en_train)[0], np.where(~en_train)[0]

    raise ValueError("metodo debe ser 'aleatorio' o 'kennard_stone'")


# ---------------------------------------------------------------------------
# 3. Resumen de y: total / train / test
# Convención de idioma: lo que el navegador NO puede traducir (tablas de
# st.dataframe, tooltips de Plotly) va en inglés; el resto, en español.
# ---------------------------------------------------------------------------
def resumen_y(y, idx_train, idx_test, grupos=None, decimales=3):
    """Tabla con n, media, desvío, mín, máx, rango, CV% e IQR de y.
    Encabezados en inglés: st.dataframe no lo traduce el navegador."""
    y = np.asarray(y, dtype=float)
    conjuntos = {"Total": np.arange(len(y)), "Train": idx_train, "Test": idx_test}

    filas = {}
    for nombre, idx in conjuntos.items():
        v = y[idx]
        q1, q3 = np.percentile(v, [25, 75])
        fila = {"n spectra": len(v)}
        if grupos is not None:
            fila["n samples"] = len(np.unique(np.asarray(grupos)[idx]))
        fila.update({
            "Mean": v.mean(),
            "SD": v.std(ddof=1),
            "Min": v.min(),
            "Max": v.max(),
            "Range": v.max() - v.min(),
            "CV %": 100 * v.std(ddof=1) / v.mean() if v.mean() != 0 else np.nan,
            "IQR": q3 - q1,
        })
        filas[nombre] = fila

    tabla = pd.DataFrame(filas).T
    enteros = [c for c in ("n spectra", "n samples") if c in tabla]
    tabla[enteros] = tabla[enteros].astype(int)
    return tabla.round(decimales)


def alertas_particion(tabla):
    """Mensajes si el test no es representativo del total."""
    msgs = []
    t, tot = tabla.loc["Test"], tabla.loc["Total"]
    if t["SD"] < 0.75 * tot["SD"]:
        msgs.append(
            f"El desvío de y en test ({t['SD']:.3g}) es bastante menor que el "
            f"total ({tot['SD']:.3g}): el R² de test puede ser engañoso."
        )
    if t["Min"] > tabla.loc["Train", "Min"] and t["Max"] < tabla.loc["Train", "Max"]:
        msgs.append(
            "Todo el test cae dentro del rango de train (solo interpola): "
            "es esperable que el RMSE de test sea menor que el de CV."
        )
    if (t["Min"] < tabla.loc["Train", "Min"]) or (t["Max"] > tabla.loc["Train", "Max"]):
        msgs.append("Hay muestras de test fuera del rango de calibración (extrapolación).")
    return msgs


# ---------------------------------------------------------------------------
# 4. Métricas: CV agrupada + test
# ---------------------------------------------------------------------------
def _metricas(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real, float), np.asarray(y_pred, float)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    q1, q3 = np.percentile(y_real, [25, 75])
    return {
        "RMSE": rmse,
        "R²": r2_score(y_real, y_pred),
        "RPD": y_real.std(ddof=1) / rmse,
        "RPIQ": (q3 - q1) / rmse,
    }


def evaluar_modelo(modelo, X, y, grupos, idx_train, idx_test, n_folds=5):
    """CV con GroupKFold sobre train + evaluación en test.

    `modelo` debería ser un Pipeline que incluya el preprocesamiento
    (p. ej. make_pipeline(StandardScaler(), PLSRegression(8))).
    """
    X, y, grupos = np.asarray(X, float), np.asarray(y, float).ravel(), np.asarray(grupos)
    Xtr, ytr, gtr = X[idx_train], y[idx_train], grupos[idx_train]

    k = min(n_folds, len(np.unique(gtr)))
    y_cv = cross_val_predict(clone(modelo), Xtr, ytr, groups=gtr, cv=GroupKFold(n_splits=k))

    final = clone(modelo).fit(Xtr, ytr)
    y_test = np.asarray(final.predict(X[idx_test])).ravel()

    cv = {f"{m} (CV)": v for m, v in _metricas(ytr, np.ravel(y_cv)).items()}
    te = {f"{m} (test)": v for m, v in _metricas(y[idx_test], y_test).items()}
    return {**cv, **te}, final


def tabla_modelos(modelos, X, y, grupos, idx_train, idx_test, n_folds=5, decimales=3):
    """modelos: dict {"PLS": pipe_pls, "Random Forest": pipe_rf} -> DataFrame."""
    filas = {}
    for nombre, m in modelos.items():
        filas[nombre], _ = evaluar_modelo(m, X, y, grupos, idx_train, idx_test, n_folds)
    tabla = pd.DataFrame(filas).T
    tabla.index.name = "Model"
    return tabla.round(decimales)


# ---------------------------------------------------------------------------
# 5. Particiones repetidas (para el boxplot)
# ---------------------------------------------------------------------------
def particiones_repetidas(modelos, X, y, grupos, n_rep=20, test_size=0.25, seed=0):
    """R² de test en `n_rep` particiones aleatorias agrupadas.
    Devuelve DataFrame largo: columnas Model, Split, R²."""
    X, y, grupos = np.asarray(X, float), np.asarray(y, float).ravel(), np.asarray(grupos)
    gss = GroupShuffleSplit(n_splits=n_rep, test_size=test_size, random_state=seed)
    filas = []
    for i, (tr, te) in enumerate(gss.split(X, y, grupos), start=1):
        for nombre, m in modelos.items():
            pred = np.asarray(clone(m).fit(X[tr], y[tr]).predict(X[te])).ravel()
            filas.append({"Model": nombre, "Split": i, "R²": r2_score(y[te], pred)})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 6. Boxplot en español, marcando el split fijo
# ---------------------------------------------------------------------------
def boxplot_r2(df_rep, r2_split_fijo=None):
    """df_rep: salida de particiones_repetidas().
    r2_split_fijo: dict {"PLS": 0.952, "Random Forest": 0.546} (R² test del split usado)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for nombre, sub in df_rep.groupby("Model", sort=False):
        fig.add_trace(go.Box(
            y=sub["R²"], name=nombre, boxpoints="all", jitter=0.4, pointpos=0,
            hovertemplate="Split %{text}<br>R² = %{y:.3f}<extra></extra>",
            text=sub["Split"],
        ))
    if r2_split_fijo:
        fig.add_trace(go.Scatter(
            x=list(r2_split_fijo.keys()), y=list(r2_split_fijo.values()),
            mode="markers", name="Split usado (test)",
            marker=dict(symbol="diamond", size=14, color="crimson",
                        line=dict(width=1, color="white")),
            hovertemplate="%{x}<br>Test R² (current split) = %{y:.3f}<extra></extra>",
        ))
    n = df_rep["Split"].nunique()
    fig.update_layout(
        title=f"R² de test en {n} particiones aleatorias (agrupadas por muestra)",
        yaxis_title="R²", xaxis_title="", showlegend=bool(r2_split_fijo),
    )
    return fig


def percentil_split(df_rep, r2_split_fijo):
    """En qué percentil de las particiones aleatorias cae el split usado."""
    out = {}
    for nombre, r2 in r2_split_fijo.items():
        valores = df_rep.loc[df_rep["Model"] == nombre, "R²"].to_numpy()
        out[nombre] = 100 * np.mean(valores <= r2)
    return out


# ---------------------------------------------------------------------------
# 7. Traducción automática del navegador
# ---------------------------------------------------------------------------
# Nombres que NO se traducen nunca (se ven igual con la página en español,
# inglés o cualquier idioma). Agregá los que uses en la app.
TERMINOS_FIJOS = [
    "Random Forest", "PLS", "PLS-DA", "OPLS-DA", "PCA", "SVM", "SVR", "k-NN", "KNN",
    "LDA", "SIMCA", "ANN", "XGBoost", "Kennard-Stone", "Savitzky-Golay",
    "SNV", "MSC", "RMSE", "RMSEP", "RMSECV", "RPD", "RPIQ", "R²",
]


def nt(texto):
    """Envuelve un término para que el traductor no lo toque, dentro de
    st.markdown(..., unsafe_allow_html=True).
    Ej.: st.markdown(f"Modelo {vq.nt('Random Forest')} entrenado", unsafe_allow_html=True)"""
    return f'<span translate="no" class="notranslate">{texto}</span>'


def proteger_terminos(terminos=None):
    """Llamar UNA vez al inicio de la app Streamlit.

    Deja que el navegador traduzca la página, pero marca como 'no traducir'
    los elementos cuyo texto es exactamente uno de los términos fijos
    (ticks y leyendas de Plotly, opciones de selectbox/radio, tabs, etc.).
    Un observador vuelve a marcar cada vez que Streamlit redibuja.
    """
    import json

    import streamlit.components.v1 as components

    terminos = json.dumps(terminos or TERMINOS_FIJOS)
    components.html(
        f"""<script>
        (function () {{
          const doc = window.parent.document;
          if (doc.__terminosProtegidos) return;      // evitar duplicar en reruns
          doc.__terminosProtegidos = true;
          const fijos = new Set({terminos}.map(t => t.toLowerCase()));

          function marcar(raiz) {{
            raiz.querySelectorAll('*').forEach(el => {{
              if (el.childElementCount) return;      // solo elementos "hoja"
              const t = (el.textContent || '').trim().toLowerCase();
              if (t && fijos.has(t) && el.getAttribute('translate') !== 'no') {{
                el.setAttribute('translate', 'no');
                el.classList.add('notranslate');
              }}
            }});
          }}
          marcar(doc.body);
          new MutationObserver(muts => {{
            for (const m of muts) {{
              const n = m.target.nodeType === 1 ? m.target : m.target.parentElement;
              if (n) marcar(n.parentElement || n);
            }}
          }}).observe(doc.body, {{childList: true, subtree: true, characterData: true}});
        }})();
        </script>""",
        height=0,
    )
