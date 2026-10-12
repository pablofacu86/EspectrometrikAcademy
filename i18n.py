# -*- coding: utf-8 -*-
"""
Language switch (English / Castellano) for Espectrometrika.edu.

Design goals (the app may be used by many students at once):
  * The language is stored PER USER SESSION (st.session_state["_lang"]); nothing global is mutated per user.
  * English mode costs one dictionary lookup per widget and changes nothing else.
  * Spanish mode translates the text *at display time* with a read-only dictionary shared by all sessions
    (built once per process, results cached). Values returned by widgets stay in English, so the application
    logic never depends on the language.
  * Reports / PDFs stay in English.
"""
import copy
import functools
import threading
import types

import streamlit as st
from streamlit.delta_generator import DeltaGenerator as _DG

LANG_LABELS = ["English", "Castellano"]
_STATE = {"installed": False, "ES": None, "IDX": None}
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------------------
def lang():
    """Language code of the current session ('en' or 'es')."""
    try:
        return st.session_state.get("_lang", "en")
    except Exception:
        return "en"


def _load():
    if _STATE["ES"] is not None:
        return
    with _LOCK:
        if _STATE["ES"] is not None:
            return
        from i18n_es_data import ES
        idx = {}
        for k in ES:
            if len(k) >= 4:
                idx.setdefault(k[:3], []).append(k)
        for v in idx.values():
            v.sort(key=len, reverse=True)
        _STATE["IDX"] = idx
        _STATE["ES"] = ES


@functools.lru_cache(maxsize=40000)
def _tr(s, frag):
    _load()
    ES, IDX = _STATE["ES"], _STATE["IDX"]
    r = ES.get(s)
    if r is not None:
        return r
    core = s.strip()
    if core != s and core:
        r = ES.get(core)
        if r is not None:
            return s[:len(s) - len(s.lstrip())] + r + s[len(s.rstrip()):]
    if not frag:
        return s
    out = []
    i, n, changed = 0, len(s), False
    while i < n:
        hit = None
        cands = IDX.get(s[i:i + 3])
        if cands:
            for k in cands:
                if s.startswith(k, i):
                    j = i + len(k)
                    if len(k) < 16:           # short keys must match whole words; long fragments may abut f-string values
                        if k[0].isalnum() and i > 0 and s[i - 1].isalnum():
                            continue
                        if k[-1].isalnum() and j < n and s[j].isalnum():
                            continue
                    hit = k
                    break
        if hit is not None:
            out.append(ES[hit])
            i += len(hit)
            changed = True
        else:
            out.append(s[i])
            i += 1
    return "".join(out) if changed else s


def T(s, frag=True):
    """Translate a visible string to the session language (identity in English)."""
    if not isinstance(s, str) or len(s) < 2 or lang() != "es":
        return s
    return _tr(s, frag)


# ---------------------------------------------------------------------------------------
# data frames and figures
def _df(df):
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            cols = list(df.columns)
            new = [T(c, False) if isinstance(c, str) else c for c in cols]
            if new != cols and len(set(map(str, new))) == len(new):
                d2 = df.copy(deep=False)
                d2.columns = new
                return d2
    except Exception:
        pass
    return df


def _walk_layout(o, depth=0):
    """Translate titles / annotations / legend entries of a plotly layout dict (in place)."""
    if depth > 6:
        return
    if isinstance(o, dict):
        for k, v in list(o.items()):
            if k in ("text", "label", "ticktext") and isinstance(v, str):
                o[k] = T(v)
            elif k == "title" and isinstance(v, str):
                o[k] = T(v)
            elif k in ("text", "ticktext") and isinstance(v, (list, tuple)):
                pass
            elif isinstance(v, (dict, list)):
                _walk_layout(v, depth + 1)
    elif isinstance(o, list):
        for it in o:
            _walk_layout(it, depth + 1)


def _fig(fig):
    try:
        import plotly.graph_objects as go
        if not isinstance(fig, go.Figure):
            return fig
        lay = fig.layout.to_plotly_json()
        lay0 = copy.deepcopy(lay)
        _walk_layout(lay)
        changed = lay != lay0
        names = []
        for tr in fig.data:
            nm = getattr(tr, "name", None)
            ht = getattr(tr, "hovertemplate", None)
            names.append((nm, T(nm) if isinstance(nm, str) else nm, ht, T(ht) if isinstance(ht, str) else ht))
            if names[-1][0] != names[-1][1] or names[-1][2] != names[-1][3]:
                changed = True
        if not changed:
            return fig
        d = fig.to_dict()
        d["layout"] = lay
        for tr, (nm, nm2, ht, ht2) in zip(d.get("data", []), names):
            if nm != nm2:
                tr["name"] = nm2
            if ht != ht2:
                tr["hovertemplate"] = ht2
        return go.Figure(d)
    except Exception:
        return fig


# ---------------------------------------------------------------------------------------
# wrappers
_TEXT = dict(pos=(0,), kws=("body", "help"))
_LAB = dict(pos=(0,), kws=("label", "help", "placeholder"))
_SPECS = {
    # body-type elements
    "markdown": _TEXT, "caption": _TEXT, "subheader": _TEXT, "header": _TEXT, "title": _TEXT,
    "info": dict(pos=(0,), kws=("body",)), "warning": dict(pos=(0,), kws=("body",)),
    "error": dict(pos=(0,), kws=("body",)), "success": dict(pos=(0,), kws=("body",)),
    "text": dict(pos=(0,), kws=("body",)),
    # widgets
    "button": _LAB, "download_button": _LAB, "link_button": _LAB, "form_submit_button": _LAB,
    "checkbox": _LAB, "toggle": _LAB, "slider": _LAB, "number_input": _LAB,
    "text_input": _LAB, "text_area": _LAB, "file_uploader": _LAB, "color_picker": _LAB,
    "radio": dict(pos=(0,), kws=("label", "help"), opts=1, captions=True),
    "selectbox": dict(pos=(0,), kws=("label", "help", "placeholder"), opts=1),
    "multiselect": dict(pos=(0,), kws=("label", "help", "placeholder"), opts=1),
    "select_slider": dict(pos=(0,), kws=("label", "help"), opts=1),
    # containers
    "expander": dict(pos=(0,), kws=("label",)), "popover": dict(pos=(0,), kws=("label", "help")),
    "status": dict(pos=(0,), kws=("label",)),
    "metric": dict(pos=(0, 2), kws=("label", "help", "delta")),
    "progress": dict(pos=(), kws=("text",)),
}


def _wrap_generic(name, spec):
    orig = getattr(_DG, name)

    @functools.wraps(orig)
    def w(self, *a, **k):
        if lang() == "es":
            a = list(a)
            for p in spec.get("pos", ()):
                if p < len(a) and isinstance(a[p], str):
                    a[p] = T(a[p])
            for n in spec.get("kws", ()):
                v = k.get(n)
                if isinstance(v, str):
                    k[n] = T(v)
            if spec.get("opts") is not None:
                ff = k.get("format_func")
                if ff is None:
                    k["format_func"] = lambda x: T(str(x), False)
                else:
                    k["format_func"] = lambda x, _ff=ff: T(str(_ff(x)), False)
            if spec.get("captions") and isinstance(k.get("captions"), (list, tuple)):
                k["captions"] = [T(c) if isinstance(c, str) else c for c in k["captions"]]
            a = tuple(a)
        return orig(self, *a, **k)
    return w


def _wrap_write():
    orig = _DG.write

    @functools.wraps(orig)
    def w(self, *a, **k):
        if lang() == "es":
            a = tuple(T(x) if isinstance(x, str) else x for x in a)
        return orig(self, *a, **k)
    return w


def _wrap_tabs():
    orig = _DG.tabs

    @functools.wraps(orig)
    def w(self, tabs, *a, **k):
        tabs = list(tabs)
        if lang() == "es":
            new = [T(x) for x in tabs]
        else:
            new = tabs
        key = k.get("key")
        if key:
            # keep the selected tab when the language changes (the stored value is the tab label)
            try:
                es_l = [T_es(x) for x in tabs]
                val = st.session_state.get(key)
                if val is not None and val not in new:
                    other = es_l if new is tabs else tabs
                    if val in other:
                        st.session_state[key] = new[other.index(val)]
            except Exception:
                pass
        return orig(self, new, *a, **k)
    return w


def T_es(s):
    """Spanish version of s regardless of the current language (used to remap tab selections)."""
    if not isinstance(s, str) or len(s) < 2:
        return s
    return _tr(s, True)


def _wrap_df(name):
    orig = getattr(_DG, name)

    @functools.wraps(orig)
    def w(self, data=None, *a, **k):
        if lang() == "es":
            data = _df(data)
        return orig(self, data, *a, **k)
    return w


def _wrap_plotly():
    orig = _DG.plotly_chart

    @functools.wraps(orig)
    def w(self, figure_or_data, *a, **k):
        if lang() == "es":
            figure_or_data = _fig(figure_or_data)
        return orig(self, figure_or_data, *a, **k)
    return w


def _rebind(name):
    setattr(st, name, types.MethodType(getattr(_DG, name), st._main))


def install():
    """Patch Streamlit once per process. In English mode the wrappers only check the session language."""
    if _STATE["installed"]:
        return
    with _LOCK:
        if _STATE["installed"]:
            return
        for name, spec in _SPECS.items():
            if hasattr(_DG, name):
                setattr(_DG, name, _wrap_generic(name, spec))
                _rebind(name)
        _DG.write = _wrap_write()
        _rebind("write")
        _DG.tabs = _wrap_tabs()
        _rebind("tabs")
        for n in ("dataframe", "table", "data_editor"):
            if hasattr(_DG, n):
                setattr(_DG, n, _wrap_df(n))
                _rebind(n)
        _DG.plotly_chart = _wrap_plotly()
        _rebind("plotly_chart")
        # module-level helpers that are not DeltaGenerator methods
        for fname in ("spinner", "toast"):
            f = getattr(st, fname, None)
            if f is not None:
                def _mk(f=f):
                    @functools.wraps(f)
                    def g(text="", *a, **k):
                        return f(T(text) if isinstance(text, str) else text, *a, **k)
                    return g
                setattr(st, fname, _mk())
        _STATE["installed"] = True


# ---------------------------------------------------------------------------------------
def language_switch():
    """Draw the English / Castellano selector and store the choice in the session."""
    ss = st.session_state
    if "lang_sel" not in ss:
        try:
            q = str(st.query_params.get("lang", "es")).lower()
        except Exception:
            q = "es"
        ss["lang_sel"] = "Castellano" if q.startswith("es") else "English"
    sel = st.radio("Language / Idioma", LANG_LABELS, horizontal=True, key="lang_sel",
                   label_visibility="collapsed")
    code = "es" if sel == "Castellano" else "en"
    ss["_lang"] = code
    try:
        if st.query_params.get("lang") != code:
            st.query_params["lang"] = code
    except Exception:
        pass
    return code
