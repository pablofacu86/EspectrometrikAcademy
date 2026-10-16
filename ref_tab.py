"""Reference spectra tab: a generic (illustrative) spectrum per technique with the typical regions of
functional groups / protons marked on top, and a table with the range, the structure (red = the atoms
that vibrate or the protons that resonate) and the name. Data: asignaciones.py (indicative values)."""
import base64
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import asignaciones as asg
from i18n import T

ROJO = "#D62828"

TECS = {  # label -> (key in asg.TECNICAS, base unit, selectable units)
    "¹H NMR": ("¹H NMR (ppm)", "ppm", ["ppm", "Hz"]),
    "Raman": ("Raman (cm⁻¹)", "cm⁻¹", ["cm⁻¹"]),
    "FT-MIR": ("FT-MIR (cm⁻¹)", "cm⁻¹", ["cm⁻¹", "µm"]),
    "NIR": ("NIR (nm)", "nm", ["nm", "cm⁻¹", "µm"]),
    "UV-Vis": ("UV-Vis (nm)", "nm", ["nm", "cm⁻¹", "eV"]),
}

REFERENCIAS = [
    "Silverstein, R. M., Webster, F. X., Kiemle, D. J., Bryce, D. L. *Spectrometric Identification of Organic Compounds*, 8th ed. Wiley, 2015.",
    "Pretsch, E., Bühlmann, P., Badertscher, M. *Structure Determination of Organic Compounds: Tables of Spectral Data*, 4th ed. Springer, 2009.",
    "Socrates, G. *Infrared and Raman Characteristic Group Frequencies: Tables and Charts*, 3rd ed. Wiley, 2001.",
    "Larkin, P. *Infrared and Raman Spectroscopy: Principles and Spectral Interpretation*. Elsevier, 2011.",
    "Workman, J., Weyer, L. *Practical Guide and Spectral Atlas for Interpretive Near-Infrared Spectroscopy*, 2nd ed. CRC Press, 2012.",
    "Osborne, B. G., Fearn, T., Hindle, P. H. *Practical NIR Spectroscopy with Applications in Food and Beverage Analysis*, 2nd ed. Longman, 1993.",
    "Skoog, D. A., Holler, F. J., Crouch, S. R. *Principles of Instrumental Analysis*, 7th ed. Cengage, 2018.",
    "Williams, D. H., Fleming, I. *Spectroscopic Methods in Organic Chemistry*, 6th ed. McGraw-Hill, 2008.",
]


def _conv(v, base, unit, mhz):
    if unit == base:
        return float(v)
    if base == "nm":
        if unit == "cm⁻¹":
            return 1e7 / max(float(v), 1e-9)
        if unit == "µm":
            return float(v) / 1000.0
        if unit == "eV":
            return 1239.84193 / max(float(v), 1e-9)
    if base == "cm⁻¹" and unit == "µm":
        return 1e4 / max(float(v), 1e-9)
    if base == "ppm" and unit == "Hz":
        return float(v) * mhz
    return float(v)


def _fmt(x):
    return f"{x:.4g}"


def _niveles(centros, textos, xmin, xmax, n_max=4):
    """Greedy staggering of the labels in rows so they do not overlap each other."""
    ancho = xmax - xmin
    fin = [-1e18] * n_max
    out = []
    for c, t in zip(centros, textos):
        mitad = 0.0085 * ancho * max(len(t), 3)
        lv = None
        for k in range(n_max):
            if c - mitad > fin[k]:
                lv = k
                break
        if lv is None:
            lv = int(np.argmin(fin))
        fin[lv] = c + mitad
        out.append(lv)
    return out


@st.fragment
def render():
    st.subheader("Reference spectra — where do functional groups and protons absorb?")
    st.caption(
        "Choose a technique to see a **generic, illustrative spectrum** with the typical regions of the functional groups (or protons) "
        "marked on top, and the table below with the range, the structure and the name. In the drawings, **red** marks the atoms or the "
        "bond that vibrate (IR, NIR, Raman), the protons that resonate (¹H NMR) or the chromophore (UV-Vis); R / R' is the rest of the molecule. "
        "The spectrum is a sketch built from the typical ranges, **not** a real measurement.")
    tipo = st.session_state.get("pret_w_tipo")
    defecto = {"NMR": "¹H NMR", "Raman": "Raman", "UV-Vis": "UV-Vis"}.get(tipo, "FT-MIR")
    opciones = list(TECS.keys())
    c1, c2, c3 = st.columns([3, 1.4, 1.4])
    tec = c1.radio("Technique", opciones, index=opciones.index(defecto), horizontal=True, key="ref_w_tec",
                   help="The default follows the signal type you chose in the Preprocessing tab.")
    clave, base, unidades = TECS[tec]
    unidad = c2.selectbox("Unit", unidades, key=f"ref_w_unit_{tec}",
                          help="Units commonly used for this technique. The ranges are converted automatically "
                               "(wavelength ↔ wavenumber: ν[cm⁻¹] = 10⁷ / λ[nm]).")
    mhz = 400.0
    if unidad == "Hz":
        mhz = c3.number_input("Spectrometer (MHz)", 60.0, 1200.0, 400.0, 20.0, key="ref_w_mhz",
                              help="Frequency in Hz = chemical shift (ppm) × spectrometer frequency (MHz).")
    q = st.text_input("Filter by name (optional)", key="ref_w_q", placeholder="e.g. carbonyl, water, aromatic, O–H…").strip().lower()

    entradas = asg.TECNICAS[clave][1]
    filas = []
    for i, e in enumerate(entradas):
        a, b = _conv(e["lo"], base, unidad, mhz), _conv(e["hi"], base, unidad, mhz)
        lo, hi = min(a, b), max(a, b)
        if q and q not in (e["grupo"] + " " + e["modo"]).lower():
            continue
        filas.append((i, e, lo, hi))
    if not filas:
        st.info("No group matches that filter.")
        return
    filas.sort(key=lambda r: (r[2] + r[3]) / 2)

    # ---------------- illustrative spectrum
    xmin = min(r[2] for r in filas); xmax = max(r[3] for r in filas)
    pad = 0.04 * (xmax - xmin)
    x = np.linspace(xmin - pad, xmax + pad, 4000)
    y = np.zeros_like(x)
    amp = [0.9, 0.55, 0.75, 0.4, 0.65, 0.5]
    rango = xmax - xmin
    for k, (_, e, lo, hi) in enumerate(filas):
        c, w = (lo + hi) / 2, hi - lo
        a0 = amp[k % len(amp)]
        if tec == "¹H NMR":
            # sharp Lorentzian lines; a few lines per region so it looks like a real 1H spectrum (singlets / multiplets)
            gam = max(rango / 2500.0, 1e-6)
            if w > 1.2 * (rango / 12.0) and w > 1.5:        # very broad exchangeable O-H / N-H
                y += 0.3 * (gam * 12) ** 2 / ((x - c) ** 2 + (gam * 12) ** 2)
            else:
                n = int(np.clip(round(w / max(rango / 120.0, 1e-9)) + 1, 1, 5))
                pos = np.linspace(lo + 0.12 * w, hi - 0.12 * w, n) if n > 1 else [c]
                pesos = np.array([1.0, 0.55, 0.8, 0.45, 0.6][:n]) * a0
                for q, h in zip(pos, pesos):
                    y += h * gam ** 2 / ((x - q) ** 2 + gam ** 2)
        elif tec == "Raman":
            sig = max(w / 6.0, rango / 450.0)
            y += a0 * np.exp(-0.5 * ((x - c) / sig) ** 2)
        elif tec == "FT-MIR":
            sig = max(w / 3.5, rango / 260.0)
            y += a0 * np.exp(-0.5 * ((x - c) / sig) ** 2)
        else:   # NIR and UV-Vis: broad, strongly overlapped bands
            sig = max(w / 1.7, rango / 60.0)
            fuerte = 1.0 if ("water" in e["grupo"].lower() or "aromatic" in e["grupo"].lower()) else 0.45 * a0 / 0.9
            y += fuerte * np.exp(-0.5 * ((x - c) / sig) ** 2)
    if tec in ("NIR", "UV-Vis"):
        y += 0.08 * (x - x.min()) / max(np.ptp(x), 1e-9)       # gently sloping baseline
    y += 0.01
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#1b1b1b", width=1.1 if tec == "¹H NMR" else 1.5), hoverinfo="skip", showlegend=False))
    centros = [(r[2] + r[3]) / 2 for r in filas]
    textos = [asg.etiqueta_corta(T(r[1]["grupo"]), 22) for r in filas]
    lv = _niveles(centros, textos, xmin, xmax)
    paso = 0.075
    for (i, e, lo, hi), c, t, l in zip(filas, centros, textos, lv):
        fig.add_vrect(x0=lo, x1=hi, fillcolor="rgba(214,40,40,0.13)", line_width=0, layer="below")
        fig.add_shape(type="line", xref="x", yref="paper", x0=c, x1=c, y0=1.0, y1=1.0 + paso * l + 0.01,
                      line=dict(color="rgba(214,40,40,0.45)", width=1))
        fig.add_annotation(x=c, y=1.0 + paso * l + 0.01, yref="paper", yanchor="bottom", text=t, showarrow=False,
                           font=dict(size=10, color=ROJO), bgcolor="rgba(255,255,255,0.75)")
    reverso = unidad in ("ppm", "Hz") or (unidad == "cm⁻¹" and tec in ("FT-MIR", "NIR", "UV-Vis"))
    fig.update_xaxes(title=f"{'Chemical shift' if tec == '¹H NMR' else ('Raman shift' if tec == 'Raman' else 'Position')} ({unidad})",
                     autorange="reversed" if reverso else True)
    fig.update_yaxes(title="Intensity (illustrative)", showticklabels=False)
    fig.update_layout(height=480, margin=dict(t=150, b=55, l=45, r=20))
    st.plotly_chart(fig, width="stretch")
    st.markdown(f":red[**⚠️ {T(asg.AVISO)}**]")

    # ---------------- table: range | structure | name
    celdas = []
    for i, e, lo, hi in filas:
        svg = asg.svg_estructura(e["estructura"], e["rojos"], e["enlaces"], px=30)
        b64 = base64.b64encode(svg.encode("utf-8")).decode()
        celdas.append(
            "<tr>"
            f"<td class='rf-r'>{_fmt(lo)} – {_fmt(hi)} <span class='rf-u'>{unidad}</span></td>"
            f"<td class='rf-i'><img src='data:image/svg+xml;base64,{b64}' alt='{e['estructura']}'></td>"
            f"<td><b>{T(e['grupo'])}</b><br><span class='rf-s'>{T(e['modo'])}</span></td></tr>")
    st.markdown(
        """<style>
        table.rf{width:100%;border-collapse:collapse;font-size:0.92rem}
        table.rf th{text-align:left;padding:8px 10px;border-bottom:2px solid rgba(128,128,128,.35);font-weight:600}
        table.rf td{padding:6px 10px;border-bottom:1px solid rgba(128,128,128,.2);vertical-align:middle}
        td.rf-r{white-space:nowrap;font-variant-numeric:tabular-nums;font-weight:600}
        .rf-u{font-weight:400;opacity:.7}
        td.rf-i img{height:78px;max-width:190px;border-radius:6px;background:#fff}
        .rf-s{opacity:.75;font-size:.85rem}
        </style>"""
        f"<table class='rf'><thead><tr><th>{T('Range')} ({unidad})</th><th>{T('Group')}</th><th>{T('Name')}</th></tr></thead><tbody>"
        + "".join(celdas) + "</tbody></table>", unsafe_allow_html=True)

    with st.expander("📚 References and where these values come from", expanded=False):
        st.markdown("Typical ranges compiled from standard spectroscopy handbooks and textbooks (values are rounded and "
                    "indicative; check the original tables for the exact ranges, sub-classes and the effect of solvent, matrix, "
                    "pH and hydrogen bonding):")
        st.markdown("\n".join(f"- {r}" for r in REFERENCIAS))
