"""
Página de la app: validación de modelos de regresión con réplicas.
Streamlit la muestra automáticamente en el menú lateral (carpeta pages/).
"""
import pandas as pd
import streamlit as st
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import validacion_quimiometria as vq

vq.proteger_terminos()  # traduce la página, pero "Random Forest", "PLS"... quedan igual

st.title("Validación de modelos de regresión")
st.markdown("Partición y validación cruzada **agrupadas por muestra** (las réplicas nunca "
            "quedan repartidas entre calibración y test).")

# --- Datos ------------------------------------------------------------------
archivo = st.file_uploader("Subí el Excel / CSV", type=["xlsx", "xls", "csv"])
if archivo is None:
    st.stop()
df = pd.read_csv(archivo, index_col=0) if archivo.name.endswith(".csv") \
    else pd.read_excel(archivo, index_col=0)

col_y = st.selectbox("Variable respuesta (y)", df.columns)
opciones_id = ["(derivar del nombre de la muestra)"] + [c for c in df.columns if c != col_y]
col_id = st.selectbox("Columna con el ID de muestra (réplicas)", opciones_id)
col_id = None if col_id.startswith("(") else col_id

cols_X = [c for c in df.select_dtypes("number").columns if c not in (col_y, col_id)]
X, y = df[cols_X].to_numpy(), df[col_y].to_numpy()
grupos = vq.obtener_grupos(df, col_id=col_id)

st.caption(" · ".join(f"{k}: {v}" for k, v in vq.info_replicas(grupos).items()))
with st.expander("Ver cómo se agruparon las réplicas"):
    st.dataframe(pd.DataFrame({"Spectrum": df.index, "Sample (group)": grupos}),
                 hide_index=True, use_container_width=True)

# --- Partición --------------------------------------------------------------
c1, c2 = st.columns(2)
metodo = c1.radio("Partición", ["aleatorio", "kennard_stone"],
                  format_func={"aleatorio": "Aleatoria agrupada",
                               "kennard_stone": "Kennard-Stone agrupada"}.get)
test_size = c2.slider("Proporción de test", 0.15, 0.40, 0.25, 0.05)
idx_tr, idx_te = vq.particion_agrupada(X, y, grupos, test_size, metodo)

st.subheader("Distribución de y")
tabla_y = vq.resumen_y(y, idx_tr, idx_te, grupos)
st.dataframe(tabla_y, use_container_width=True)
for msg in vq.alertas_particion(tabla_y):
    st.warning(msg)

# --- Modelos (preprocesamiento DENTRO del pipeline) ---------------------------
n_vl = st.slider("Variables latentes PLS", 1, 20, 8)
modelos = {
    "PLS": make_pipeline(StandardScaler(), PLSRegression(n_components=n_vl)),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
}

st.subheader("Métricas (CV agrupada por muestra + test)")
tabla = vq.tabla_modelos(modelos, X, y, grupos, idx_tr, idx_te)
st.dataframe(tabla, use_container_width=True)

st.subheader("Estabilidad frente a la partición")
rep = vq.particiones_repetidas(modelos, X, y, grupos, n_rep=20, test_size=test_size)
r2_fijo = tabla["R² (test)"].to_dict()
st.plotly_chart(vq.boxplot_r2(rep, r2_fijo), use_container_width=True)
for m, p in vq.percentil_split(rep, r2_fijo).items():
    st.markdown(f"<small>{vq.nt(m)}: el split usado queda en el percentil {p:.0f} "
                f"de las particiones aleatorias.</small>", unsafe_allow_html=True)
