# -*- coding: utf-8 -*-
"""Prominent, professional Contact tab."""
import urllib.parse
import streamlit as st
from i18n import T

TEMAS = [
    "Question about chemometrics (methods, interpretation of results)",
    "Help with my own data / analysis",
    "Help with my models / developing new models",
    "Request a meeting or online consultation",
    "Training, workshop or course request",
    "Collaboration or research project",
    "Feedback, bug report or suggestion for the app",
    "Other",
]


def render(enviar_fn, email):
    st.markdown("<style>a[kind=\"primary\"], a[kind=\"primary\"] * {color:#fff !important; text-decoration:none !important;}</style>",
                unsafe_allow_html=True)
    st.markdown(
        """
<div style="background:linear-gradient(135deg,#0B3D54 0%,#14607F 100%); color:#fff; border-radius:14px;
            padding:26px 30px; margin-bottom:18px;">
  <div style="font-size:1.55rem; font-weight:700; margin-bottom:6px;">Let's talk chemometrics</div>
  <div style="font-size:1.02rem; line-height:1.55; opacity:.95; max-width:820px;">
    Espectrometrika.edu is an open educational initiative. If something in your analysis is unclear, if you would like a second
    opinion on a model, or if you would like to arrange a meeting to discuss your data, you are very welcome to write.
    Every message is read personally.
  </div>
</div>
""", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### Want a hand with your models?")
        st.markdown(
            "Do you have a model you would like to improve, or a new one to build? We would be glad to help — write to us:")
        k1, k2 = st.columns(2)
        with k1:
            st.markdown(
                "- **Developing calibration and classification models** for your own spectral or tabular data "
                "(NIR, FT-MIR, Raman, NMR, chromatography, metabolomics…)\n"
                "- **Choosing and validating** the preprocessing, the algorithm and the validation scheme, so that results hold on new samples")
        with k2:
            st.markdown(
                "- **Reviewing a model you already have**: overfitting, outliers, applicability domain, uncertainty\n"
                "- **Training and workshops** in chemometrics for your team")

    c1, c2 = st.columns([3, 2])
    with c2:
        with st.container(border=True):
            st.markdown("#### Contact details")
            st.markdown("**Pablo Facundo Sánchez**")
            asunto = urllib.parse.quote(T("Espectrometrika.edu — query"))
            cuerpo = urllib.parse.quote(T(
                "Dear Pablo,\n\nI am writing from Espectrometrika.edu regarding:\n\n"
                "[Topic: chemometrics question / meeting request / data analysis help]\n\n"
                "Technique and sample type:\nNumber of samples:\nWhat I would like to achieve:\n\nKind regards,\n"))
            st.link_button(f"✉️ Write an email", f"mailto:{email}?subject={asunto}&body={cuerpo}",
                           use_container_width=True, type="primary")
            st.caption(f"Direct address: **{email}**")
            st.markdown("**Response time:** usually within a few working days.")
        with st.container(border=True):
            st.markdown("#### What helps me answer you well")
            st.markdown(
                "- The **technique** (NIR, FT-MIR, Raman, NMR, chromatography, tabular data…)\n"
                "- Number of **samples**, classes and/or the property to predict\n"
                "- What you have already tried in the app (and which tab you got stuck in)\n"
                "- For a meeting: your **time zone** and two or three possible time slots\n"
                "- *Please do not send confidential or personal data by e-mail.*")
    with c1:
        with st.container(border=True):
            st.markdown("#### Send a message from here")
            tema = st.selectbox("Topic", TEMAS, key="ct_tema",
                                help="Helps to route and prioritise your message.")
            colA, colB = st.columns(2)
            nombre = colA.text_input("Your name", key="ct_nombre", help="So that the reply can be addressed to you.")
            inst = colB.text_input("Institution / company (optional)", key="ct_inst")
            correo = st.text_input("Your e-mail (needed to receive a reply)", key="ct_email",
                                   help="Only used to answer your message.")
            msg = st.text_area("Message", key="ct_msg", height=180,
                               placeholder="Dear Pablo, I am working with … and I would like to ask …",
                               help="Describe your question or the purpose of the meeting.")
            if st.button("📨 Send message", type="primary", key="ct_enviar"):
                if not msg.strip():
                    st.warning("Please write a message before sending.")
                elif "@" not in correo or "." not in correo.split("@")[-1]:
                    st.warning("Please enter a valid e-mail address so that I can reply.")
                else:
                    texto = f"[{tema}]\n"
                    if inst.strip():
                        texto += f"Institution: {inst.strip()}\n"
                    texto += "\n" + msg.strip()
                    ok, detalle = enviar_fn(texto, nombre, correo)
                    if ok:
                        st.success(detalle)
                    else:
                        st.error(detalle + f" You can also write directly to {email}.")
    st.markdown("<div style='font-size:1.05rem; color:#3D4A52; line-height:1.55; margin-top:6px;'>"
                + T("Espectrometrika.edu is free for teaching and self-study. Suggestions for new tools, tutorials or datasets "
                    "are always welcome.") + "</div>", unsafe_allow_html=True)
