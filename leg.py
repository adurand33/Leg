#
# wecov3r studio leg – FR/EN with toggle
#
import math
from typing import Optional, List, Tuple, Dict
import requests
import streamlit as st
import os
import uuid
import json
import base64
import pathlib
from collections import defaultdict
from streamlit.components.v1 import html
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# language (fr / en) and translation
# -----------------------------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state["lang"] = "fr"  # fr by default

def is_french() -> bool:
    return st.session_state.get("lang", "fr") == "fr"

def TR(fr: str, en: str) -> str:
    """simple fr/en translation based on session language"""
    return fr if is_french() else en

# -----------------------------------------------------------------------------
# simple helpers
# -----------------------------------------------------------------------------
def ConvertHexToRGB(h: str, a: float) -> str:
    h = (h or "#262730").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return f"rgba({r},{g},{b},{a})"

def GetExtents(vals):
    mn, mx = math.inf, -math.inf
    for v in vals or []:
        if isinstance(v, (int, float)) and v == v:
            mn = min(mn, v)
            mx = max(mx, v)
    if mn is math.inf:
        return (None, None)
    if mn == mx:
        return (mn - 0.5, mx + 0.5)
    return (mn, mx)

def Format1(x: Optional[float], decimals: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x:,.{decimals}f}".replace(",", " ")

def NodesToCentimeter(nodes_mm: List[float], is3d: bool = True):
    n = len(nodes_mm) // 3
    x = [nodes_mm[3 * i + 0] / 10.0 for i in range(n)]
    y = [nodes_mm[3 * i + 1] / 10.0 for i in range(n)]
    if is3d:
        z = [nodes_mm[3 * i + 2] / 10.0 for i in range(n)]
        return x, y, z
    return x, y

# -----------------------------------------------------------------------------
# PlotFrench / PlotEnglish + RenderFigure
# (modebar: 6 buttons, titles fr/en)
# -----------------------------------------------------------------------------
def _common_plot_layout(
    fig: go.Figure,
    axis_color: str = "#ffffff",
    grid_alpha: float = 0.18,
    pad_ratio: float = 0.12,
    margin_left: int = 70,
    margin_bot: int = 64,
) -> go.Figure:
    theme_text = st.get_option("theme.textColor") or axis_color
    axis_color = axis_color or theme_text
    grid_color = ConvertHexToRGB(axis_color, grid_alpha)

    f = go.Figure(fig)
    f.update_layout(
        margin=dict(l=margin_left, r=16, t=8, b=margin_bot),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    has_x = "xaxis" in f.layout
    has_y = "yaxis" in f.layout

    if has_x:
        f.update_xaxes(
            showline=True,
            linewidth=1,
            linecolor=axis_color,
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            zeroline=False,
            automargin=True,
            showticklabels=True,
            ticks="outside",
            ticklen=10,
            tickcolor="rgba(0,0,0,0)",
            tickwidth=0,
            ticklabelposition="outside",
            tickfont=dict(color=axis_color),
        )
    if has_y:
        f.update_yaxes(
            showline=False,
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            zeroline=False,
            automargin=True,
            showticklabels=True,
            ticks="outside",
            ticklen=10,
            tickcolor="rgba(0,0,0,0)",
            tickwidth=0,
            ticklabelposition="outside",
            tickfont=dict(color=axis_color),
        )

    if has_x or has_y:
        xmin = xmax = ymin = ymax = None
        for tr in f.data:
            xs = getattr(tr, "x", None)
            ys = getattr(tr, "y", None)
            if xs is not None:
                ex = GetExtents(xs)
                if ex[0] is not None:
                    tx_min, tx_max = ex
                    xmin = tx_min if xmin is None else min(xmin, tx_min)
                    xmax = tx_max if xmax is None else max(xmax, tx_max)
            if ys is not None:
                ey = GetExtents(ys)
                if ey[0] is not None:
                    ty_min, ty_max = ey
                    ymin = ty_min if ymin is None else min(ymin, ty_min)
                    ymax = ty_max if ymax is None else max(ymax, ty_max)

        if xmin is not None and xmax is not None and ymin is not None and ymax is not None:
            if xmin == xmax:
                xmax = xmin + 1
            if ymin == ymax:
                ymax = ymin + 1
            px = pad_ratio * (xmax - xmin)
            py = pad_ratio * (ymax - ymin)
            f.update_xaxes(range=[xmin - px, xmax + px])
            f.update_yaxes(range=[ymin - py, ymax + py])

    if "scene" in f.layout:
        f.update_scenes(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, showbackground=False, tickfont=dict(color=axis_color)),
            yaxis=dict(showgrid=False, showbackground=False, tickfont=dict(color=axis_color)),
            zaxis=dict(showgrid=False, showbackground=False, tickfont=dict(color=axis_color)),
        )

    return f

def PlotFrench(fig: go.Figure, height: int = 450):
    f = _common_plot_layout(fig)
    div_id = f"plot-fr-{uuid.uuid4().hex}"

    config = {
        "displaylogo": False,
        "locale": "fr",
        "modeBarButtons": [[
            "zoom3d",
            "pan3d",
            "orbitRotation",
            "tableRotation",
            "resetCameraDefault3d",
            "toImage",
        ]],
    }

    html_str = f.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id=div_id,
        config=config,
    )

    titles_fr = [
        "Zoom",
        "Translation",
        "Rotation orbitale",
        "Rotation planaire",
        "Recentrer",
        "Télécharger PNG",
    ]

    custom_html = f"""{html_str}
<script>
(function() {{
  var el = document.getElementById("{div_id}");
  if (!el) return;
  function setTitles() {{
    var titles = {json.dumps(titles_fr)};
    var btns = el.querySelectorAll('.modebar-btn');
    btns.forEach(function(btn, i) {{
      if (i < titles.length) {{
        btn.setAttribute('title', titles[i]);
        btn.setAttribute('data-title', titles[i]);
      }}
    }});
  }}
  setTimeout(setTitles, 500);
}})();
</script>
"""

    html(custom_html, height=height)

def PlotEnglish(fig: go.Figure, height: int = 450):
    f = _common_plot_layout(fig)
    div_id = f"plot-en-{uuid.uuid4().hex}"

    config = {
        "displaylogo": False,
        "locale": "en",
        "modeBarButtons": [[
            "zoom3d",
            "pan3d",
            "orbitRotation",
            "tableRotation",
            "resetCameraDefault3d",
            "toImage",
        ]],
    }

    html_str = f.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id=div_id,
        config=config,
    )

    titles_en = [
        "Zoom",
        "Translation",
        "Orbital Rotation",
        "Planar Rotation",
        "Reset View",
        "Download PNG",
    ]

    custom_html = f"""{html_str}
<script>
(function() {{
  var el = document.getElementById("{div_id}");
  if (!el) return;
  function setTitles() {{
    var titles = {json.dumps(titles_en)};
    var btns = el.querySelectorAll('.modebar-btn');
    btns.forEach(function(btn, i) {{
      if (i < titles.length) {{
        btn.setAttribute('title', titles[i]);
        btn.setAttribute('data-title', titles[i]);
      }}
    }});
  }}
  setTimeout(setTitles, 500);
}})();
</script>
"""

    html(custom_html, height=height)

def RenderFigure(fig: go.Figure, height: int = 450):
    if is_french():
        PlotFrench(fig, height=height)
    else:
        PlotEnglish(fig, height=height)

# -----------------------------------------------------------------------------
# styles and constants
# -----------------------------------------------------------------------------
BRAND_VIOLET = "#401189"
BRAND_VIOLET_RGBA = "rgba(64,17,137,0.25)"
SHADING_VIOLET = "#6B4AE6"

MESH_LIGHTING = dict(ambient=0.55, diffuse=1.0, specular=0.45, roughness=0.80)
MESH_LIGHTPOS = dict(x=1.4, y=0.2, z=2.0)

EDGE3D_BORDER_WIDTH = 8
EDGE3D_BORDER_COLOR = BRAND_VIOLET
EDGE2D_BORDER_WIDTH = 4
EDGE2D_BORDER_COLOR = BRAND_VIOLET

DEFAULTS = {"cheville": 25.0, "mollet": 38.0, "genou": 39.0, "cuisse": 49.0, "taille": 182.0}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "_do_raz" not in st.session_state:
    st.session_state["_do_raz"] = False
if st.session_state["_do_raz"]:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state["_do_raz"] = False

# -----------------------------------------------------------------------------
# http / wecov3r api
# -----------------------------------------------------------------------------
def CallAPI(url: str, payload: dict, bearer: Optional[str] = None, timeout_s: float = 30.0) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Referer": "https://yourleg.streamlit.app/",
        "X-Requested-With": "",
    }
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()

def RunPipeline(curves: list, api_key: str, api_secret: str, api_user: int, do_unroll: bool = True) -> dict:
    r1 = CallAPI("https://wecov3r.com/api/connect", {"k": api_key, "s": api_secret, "usr": api_user})
    data1 = r1.get("data")

    r2 = CallAPI("https://wecov3r.com/api/token", {"c": data1, "u": "mesh"})
    data2 = r2.get("data")
    token = data2.get("token") if isinstance(data2, dict) else data2

    op_wire = {
        "id": "meshWIRE",
        "params": [
            {"key": "WIRE_MODE", "type": "INT", "value": 3},
            {"key": "GROW_MODE", "type": "INT", "value": 0},
            {"key": "GROW_DIST", "type": "FLOAT", "value": 0},
            {"key": "ENABLE_CLOSECV", "type": "BOOL", "value": False},
            {"key": "ENABLE_AUTOORIENT", "type": "BOOL", "value": False},
            {"key": "ENABLE_AUTOPOLISH", "type": "BOOL", "value": True},
            {"key": "ENABLE_AUTOFITCVS", "type": "BOOL", "value": False},
            {"key": "ENABLE_AUTOCYCLE", "type": "BOOL", "value": False},
        ],
    }
    payload1 = {"globals": [], "data": curves, "operators": [op_wire]}
    r3 = CallAPI("https://wecov3r.com/api/mesh", payload1, bearer=token)
    if not do_unroll:
        return r3

    op_unroll = {
        "id": "meshUNROLL",
        "params": [
            {"key": "UNROLL_MODE", "type": "INT", "value": 1},
            {"key": "ADAPT_MODE", "type": "INT", "value": 0},
            {"key": "ENABLE_AUTOFILL", "type": "BOOL", "value": True},
            {"key": "ENABLE_MATCH", "type": "BOOL", "value": True},
            {"key": "ENABLE_RIGID", "type": "BOOL", "value": True},
            {"key": "RIGID_RATIO", "type": "FLOAT", "value": 0.5},
            {"key": "ENABLE_FULL", "type": "BOOL", "value": False},
            {"key": "ENABLE_REINDEX", "type": "BOOL", "value": False},
            {"key": "LAY_MODE", "type": "INT", "value": 1},
            {"key": "ATTACH_DIST", "type": "FLOAT", "value": 0.05},
            {"key": "SHIFT_DIST", "type": "FLOAT", "value": 0.01},
            {"key": "GROUP_DIST", "type": "FLOAT", "value": 0.05},
        ],
    }
    payload2 = {"globals": [], "data": r3.get("data"), "operators": [op_unroll]}
    r4 = CallAPI("https://wecov3r.com/api/mesh", payload2, bearer=token)
    return r4

# -----------------------------------------------------------------------------
# 2d: boundary edges
# -----------------------------------------------------------------------------
def ComputeBoundaryEdges(defn: dict) -> List[List[int]]:
    topo = int(defn.get("topology") or 0)
    elems = defn.get("elements") or []
    if topo not in (3, 4) or len(elems) < topo:
        return []

    edge_count: Dict[Tuple[int, int], int] = defaultdict(int)
    nb_elts = len(elems) // topo
    for e in range(nb_elts):
        idxs = [elems[topo * e + k] for k in range(topo)]
        for k in range(topo):
            a, b = idxs[k], idxs[(k + 1) % topo]
            if a > b:
                a, b = b, a
            edge_count[(a, b)] += 1

    adj: Dict[int, List[int]] = defaultdict(list)
    for (a, b), c in edge_count.items():
        if c == 1:
            adj[a].append(b)
            adj[b].append(a)

    visited = set()
    loops: List[List[int]] = []

    def PopNext(u: int) -> Optional[int]:
        while adj[u]:
            v = adj[u].pop()
            if (min(u, v), max(u, v)) not in visited:
                return v
        return None

    for start in list(adj.keys()):
        if not adj[start]:
            continue
        loop = [start]
        u = start
        v = PopNext(u)
        if v is None:
            continue
        visited.add((min(u, v), max(u, v)))
        loop.append(v)
        while True:
            u, v = v, PopNext(v)
            if v is None:
                break
            visited.add((min(u, v), max(u, v)))
            if v == start:
                loops.append(loop[:])
                break
            loop.append(v)

    return loops

# -----------------------------------------------------------------------------
# 2d: shaded / wireframe
# -----------------------------------------------------------------------------
def DrawMeshShade2D(mesh_obj: dict):
    defn = mesh_obj.get("definition", {})
    nodes = defn.get("nodes") or []
    topo = int(defn.get("topology") or 0)
    elems = defn.get("elements") or []
    elidx = defn.get("elementsLastIndex")

    fig = go.Figure()

    if elidx:
        first = 0
        x_all, y_all = NodesToCentimeter(nodes, False)
        for last in elidx:
            seq = elems[first:last + 1]
            if len(seq) >= 3:
                xseq = [x_all[i] for i in seq]
                yseq = [y_all[i] for i in seq]
                if seq[0] != seq[-1]:
                    xseq.append(xseq[0])
                    yseq.append(yseq[0])
                fig.add_trace(go.Scatter(
                    x=xseq,
                    y=yseq,
                    mode="lines",
                    line=dict(color=EDGE2D_BORDER_COLOR, width=EDGE2D_BORDER_WIDTH),
                    fill="toself",
                    fillcolor=BRAND_VIOLET_RGBA,
                    name="contour",
                ))
            first = last + 1
    elif topo in (3, 4) and len(elems) >= topo:
        loops = ComputeBoundaryEdges(defn)
        x_all, y_all = NodesToCentimeter(nodes, False)
        if not loops:
            return DrawMeshWire2D(mesh_obj)
        for seq in loops:
            if len(seq) >= 3:
                xseq = [x_all[i] for i in seq] + [x_all[seq[0]]]
                yseq = [y_all[i] for i in seq] + [y_all[seq[0]]]
                fig.add_trace(go.Scatter(
                    x=xseq,
                    y=yseq,
                    mode="lines",
                    line=dict(color=EDGE2D_BORDER_COLOR, width=EDGE2D_BORDER_WIDTH),
                    fill="toself",
                    fillcolor=BRAND_VIOLET_RGBA,
                    name="contour",
                ))
    else:
        return DrawMeshWire2D(mesh_obj)

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(title="cm", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="cm"),
        showlegend=False,
        hovermode=False,
    )
    RenderFigure(fig)

def DrawMeshWire2D(mesh_obj: dict):
    defn = mesh_obj.get("definition", {})
    nodes = defn.get("nodes") or []
    topo = int(defn.get("topology") or 0)
    elements = defn.get("elements") or []
    elidx = defn.get("elementsLastIndex")

    if len(nodes) < 6:
        st.warning(TR("Pas assez de points", "Not enough points"))
        return

    x, y = NodesToCentimeter(nodes, False)
    xl, yl = [], []

    def AddEdge(a, b):
        xl.extend([x[a], x[b], None])
        yl.extend([y[a], y[b], None])

    if topo and topo > 0:
        nb_elts = len(elements) // topo
        edges = set()
        for e in range(nb_elts):
            idxs = [elements[topo * e + k] for k in range(topo)]
            for k in range(topo):
                a, b = idxs[k], idxs[(k + 1) % topo]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        for (a, b) in edges:
            AddEdge(a, b)
    else:
        if not elidx:
            for i2 in range(len(elements)):
                a = elements[i2]
                b = elements[(i2 + 1) % len(elements)]
                AddEdge(a, b)
        else:
            first = 0
            for last in elidx:
                seq = elements[first:last + 1]
                for i2 in range(len(seq)):
                    a = seq[i2]
                    b = seq[(i2 + 1) % len(seq)]
                    AddEdge(a, b)
                first = last + 1

    fig = go.Figure(data=[go.Scatter(
        x=xl,
        y=yl,
        mode="lines",
        line=dict(width=EDGE2D_BORDER_WIDTH, color=EDGE2D_BORDER_COLOR),
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(title="cm", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="cm"),
        showlegend=False,
        hovermode=False,
    )
    RenderFigure(fig)

# -----------------------------------------------------------------------------
# 3d helpers and display
# -----------------------------------------------------------------------------
def AddWire3D(fig, x, y, z, elems, topo, elidx):
    xl, yl, zl = [], [], []

    def AddEdge(a, b):
        xl.extend([x[a], x[b], None])
        yl.extend([y[a], y[b], None])
        zl.extend([z[a], z[b], None])

    if topo and topo > 0:
        nb_elts = len(elems) // topo
        edges = set()
        for e in range(nb_elts):
            idxs = [elems[topo * e + k] for k in range(topo)]
            for k in range(topo):
                a, b = idxs[k], idxs[(k + 1) % topo]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        for (a, b) in edges:
            AddEdge(a, b)
    else:
        if not elidx:
            for i2 in range(len(elems)):
                a = elems[i2]
                b = elems[(i2 + 1) % len(elems)]
                AddEdge(a, b)
        else:
            first = 0
            for last in elidx:
                seq = elems[first:last + 1]
                for i2 in range(len(seq)):
                    a = seq[i2]
                    b = seq[(i2 + 1) % len(seq)]
                    AddEdge(a, b)
                first = last + 1

    if xl:
        fig.add_trace(go.Scatter3d(
            x=xl,
            y=yl,
            z=zl,
            mode="lines",
            name="edges",
            line=dict(width=EDGE3D_BORDER_WIDTH, color=EDGE3D_BORDER_COLOR),
        ))

def AddBoundaryEdges3D(fig, x, y, z, elems, topo):
    if topo not in (3, 4) or not elems:
        return

    edge_count = defaultdict(int)
    n = len(elems) // topo
    for e in range(n):
        idxs = [elems[topo * e + k] for k in range(topo)]
        for k in range(topo):
            a, b = idxs[k], idxs[(k + 1) % topo]
            if a > b:
                a, b = b, a
            edge_count[(a, b)] += 1

    xl, yl, zl = [], [], []
    for (a, b), c in edge_count.items():
        if c == 1:
            xl.extend([x[a], x[b], None])
            yl.extend([y[a], y[b], None])
            zl.extend([z[a], z[b], None])

    if xl:
        fig.add_trace(go.Scatter3d(
            x=xl,
            y=yl,
            z=zl,
            mode="lines",
            name="border",
            line=dict(width=EDGE3D_BORDER_WIDTH, color=EDGE3D_BORDER_COLOR),
        ))

def DrawMeshShade3D(mesh_obj: dict, edge_mode: str = "Bord"):
    defn = mesh_obj.get("definition", {})
    nodes = defn.get("nodes") or []
    topo = int(defn.get("topology") or 0)
    elems = defn.get("elements") or []
    elidx = defn.get("elementsLastIndex")

    if len(nodes) < 9:
        st.warning(TR("Pas assez de points", "Not enough points"))
        return

    x, y, z = NodesToCentimeter(nodes, True)
    fig = go.Figure()

    def AddEdgesChoice():
        if edge_mode in ("Bord", "Border"):
            AddBoundaryEdges3D(fig, x, y, z, elems, topo)
        elif edge_mode in ("Toutes", "All"):
            AddWire3D(fig, x, y, z, elems, topo, elidx)

    if topo in (3, 4) and len(elems) >= topo:
        i_idx, j_idx, k_idx = [], [], []

        if topo == 3 and len(elems) % 3 == 0:
            for t in range(len(elems) // 3):
                a, b, c = elems[3 * t:3 * t + 3]
                i_idx.append(a)
                j_idx.append(b)
                k_idx.append(c)
        elif topo == 4 and len(elems) % 4 == 0:
            for q in range(len(elems) // 4):
                a, b, c, d = elems[4 * q:4 * q + 4]
                i_idx.extend([a, a])
                j_idx.extend([b, c])
                k_idx.extend([c, d])

        if i_idx:
            fig.add_trace(go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i_idx,
                j=j_idx,
                k=k_idx,
                color=SHADING_VIOLET,
                opacity=1.0,
                flatshading=False,
                lighting=MESH_LIGHTING,
                lightposition=MESH_LIGHTPOS,
                showscale=False,
                hoverinfo="skip",
                hovertemplate=None,
            ))
            AddEdgesChoice()
        else:
            fig.add_trace(go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=SHADING_VIOLET,
                opacity=1.0,
                flatshading=False,
                alphahull=12,
                lighting=MESH_LIGHTING,
                lightposition=MESH_LIGHTPOS,
                showscale=False,
                hoverinfo="skip",
                hovertemplate=None,
            ))
            AddEdgesChoice()
    else:
        fig.add_trace(go.Mesh3d(
            x=x,
            y=y,
            z=z,
            color=SHADING_VIOLET,
            opacity=1.0,
            flatshading=False,
            alphahull=12,
            lighting=MESH_LIGHTING,
            lightposition=MESH_LIGHTPOS,
            showscale=False,
            hoverinfo="skip",
            hovertemplate=None,
        ))
        AddEdgesChoice()

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis_title="cm",
            yaxis_title="cm",
            zaxis_title="cm",
            aspectmode="data",
        ),
    )
    RenderFigure(fig)

def DrawMeshWire3D(mesh_obj: dict):
    defn = mesh_obj.get("definition", {})
    nodes = defn.get("nodes") or []
    topo = int(defn.get("topology") or 0)
    elems = defn.get("elements") or []
    elidx = defn.get("elementsLastIndex")

    if len(nodes) < 9:
        st.warning(TR("Pas assez de points", "Not enough points"))
        return

    x, y, z = NodesToCentimeter(nodes, True)
    fig = go.Figure()
    AddWire3D(fig, x, y, z, elems, topo, elidx)

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis_title="cm",
            yaxis_title="cm",
            zaxis_title="cm",
            aspectmode="data",
        ),
        showlegend=False,
        hovermode=False,
    )
    RenderFigure(fig)

# -----------------------------------------------------------------------------
# leg geometry
# -----------------------------------------------------------------------------
def ComputeCurves(
    ankle_cm: float,
    calf_cm: float,
    knee_cm: float,
    thigh_cm: float,
    height_cm: float,
    nb_point: int = 32,
):
    scale = 10.0
    v_ankle = scale * float(ankle_cm)
    v_calf = scale * float(calf_cm)
    v_knee = scale * float(knee_cm)
    v_thigh = scale * float(thigh_cm)
    v_height = scale * float(height_cm)

    h_ankle = 0.039 * v_height
    h_calf = 0.200 * v_height
    h_knee = 0.285 * v_height
    h_thigh = 0.400 * v_height
    h_crotch = 0.500 * v_height

    leg_height = h_crotch - h_ankle
    garment_bot_offset = 0.07 * leg_height
    garment_top_offset = 0.14 * leg_height

    h_garment_bot = h_ankle + garment_bot_offset
    h_garment_top = h_crotch - garment_top_offset

    tibia_length = h_knee - h_ankle
    femur_length = h_crotch - h_knee
    femur_length_real = 0.245 * v_height

    garment_length = 0.98 * (h_garment_top - h_garment_bot)

    def CirclePoints359(r, z, n=nb_point):
        pts = []
        if n <= 1:
            return [r, 0.0, z]
        end = math.radians(359.0)
        step = end / (n - 1)
        for i in range(n):
            a = i * step
            x = r * math.cos(a)
            y = r * math.sin(a)
            pts.extend([x, y, z])
        return pts

    r_ankle = v_ankle / (2.0 * math.pi)
    r_calf = v_calf / (2.0 * math.pi)
    r_knee = v_knee / (2.0 * math.pi)
    r_thigh = v_thigh / (2.0 * math.pi)

    curves_pts = [
        CirclePoints359(r_ankle, h_garment_bot),
        CirclePoints359(r_calf, h_calf),
        CirclePoints359(r_knee, h_knee),
        CirclePoints359(r_thigh, h_thigh),
        CirclePoints359(r_thigh, h_garment_top),
    ]

    curves = []
    for i, pts in enumerate(curves_pts):
        curves.append({
            "definition": {
                "points": pts,
                "open": False,
            },
            "properties": {
                "uuid": i,
                "name": f"curve_{i}",
                "type": "LINES",
                "scale": 1000,
            },
            "dimension": 3,
        })

    return {
        "curves": curves,
        "leg_height": leg_height,
        "tibia_length": tibia_length,
        "femur_length": femur_length,
        "femur_length_real": femur_length_real,
        "garment_length": garment_length,
        "garment_bot_offset_cm": garment_bot_offset / scale,
        "garment_top_offset_cm": garment_top_offset / scale,
    }

# -----------------------------------------------------------------------------
# streamlit page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Vêtement AI", page_icon="🤖", layout="wide")

_logo_path = pathlib.Path("logo.png")
try:
    _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode("ascii")
except Exception:
    _logo_b64 = ""

st.markdown(
    f"""
<style>
.wcvr-hero {{
  background: linear-gradient(135deg, #401189 0%, #c0bdc5 100%);
  border-radius: 16px;
  padding: 16px 18px;
  color: #fff;
  margin-bottom: 10px;
  box-shadow: 0 8px 28px rgba(0,0,0,.18);
  text-align: center;
}}
.wcvr-hero h1 {{
  position: relative;
  font-size: 28px;
  line-height: 1.1;
  padding-left: calc(2em + 4px);
}}
.wcvr-hero h1 .wcvr-logo {{
  height: 2em;
  width: auto;
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  display: block;
  filter: drop-shadow(0 1px 0 rgba(0,0,0,.15));
}}
.wcvr-hero h1 .robot {{
  font-size: 0.9em;
  line-height: 1;
  vertical-align: -0.12em;
  display: inline-block;
  filter: drop-shadow(0 1px 0 rgba(0,0,0,.15));
}}
.block {{
  border: 1px solid rgba(0,0,0,.06);
  border-radius: 12px;
  padding: 12px 14px;
  background: rgba(255,255,255,.85);
  backdrop-filter: blur(6px);
  margin-bottom: 12px;
}}
.small-input .stNumberInput>div>div>input {{ max-width: 7.5rem; }}
.small-input .stButton>button {{ padding: 0.35rem 0.8rem; border-radius: 10px; }}
div[data-testid="stCheckbox"] label {{ white-space: nowrap; }}
.stButton > button, .stButton > button * {{ white-space: nowrap !important; }}
.block:empty {{ display: none !important; padding: 0 !important; margin: 0 !important; border: 0 !important;
  box-shadow: none !important; background: transparent !important; }}
.block > :first-child {{ margin-top: 0 !important; }}
</style>

<div class="wcvr-hero">
  <h1>
    {f'<img class="wcvr-logo" src="data:image/png;base64,{_logo_b64}" alt="WeCov3r" />' if _logo_b64 else ''}
    <span class="robot">🔄</span>&thinsp;Vêtement AI&thinsp;<span class="robot">🔄</span>
  </h1>
</div>
""",
    unsafe_allow_html=True,
)

# language toggle button (label changes with language)
col_toggle, _ = st.columns([1, 4])
with col_toggle:
    label_toggle = "Switch to English" if is_french() else "Basculer en français"
    if st.button(label_toggle):
        st.session_state["lang"] = "en" if is_french() else "fr"
        st.rerun()

col_left, col_right = st.columns([1.0, 1.2], gap="large")

# api keys (secrets or env, then override for demo)
api_key    = st.secrets.get("WECOV3R_API_KEY", os.getenv("WECOV3R_API_KEY", ""))
api_secret = st.secrets.get("WECOV3R_API_SECRET", os.getenv("WECOV3R_API_SECRET", ""))
api_user   = st.secrets.get("WECOV3R_API_USER", os.getenv("WECOV3R_API_USER", "-1"))

if not api_key or not api_secret or api_user < 0:
    st.warning(TR(
        "Clés WeCov3r manquantes (WECOV3R_API_KEY, _SECRET, _USER)",
        "Missing WeCov3r Keys (WECOV3R_API_KEY, _SECRET, _USER)",
    ))

data = None
garment_length_cm = None
area_cm2 = None
perim_cm = None
vol_l = None
do_unroll = True

# left column: inputs
with col_left:
    st.markdown('<div class="block small-input">', unsafe_allow_html=True)
    st.subheader(TR("Jambe (cm)", "Leg (cm)"))

    r1a, r1b = st.columns([1, 1], gap="small")
    cheville = r1a.number_input(TR("Cheville", "Ankle"), 1.0, 1000.0, step=0.1, format="%.1f", key="cheville")
    mollet = r1b.number_input(TR("Mollet", "Calf"), 1.0, 1000.0, step=0.1, format="%.1f", key="mollet")

    r2a, r2b = st.columns([1, 1], gap="small")
    genou = r2a.number_input(TR("Genou", "Knee"), 1.0, 1000.0, step=0.1, format="%.1f", key="genou")
    cuisse = r2b.number_input(TR("Cuisse", "Thigh"), 1.0, 1000.0, step=0.1, format="%.1f", key="cuisse")

    r3a, r3b = st.columns([1, 1], gap="small")
    taille = r3a.number_input(TR("Taille", "Height"), 1.0, 300.0, step=0.1, format="%.1f", key="taille")

    g_run, g_raz, g_flat = r3b.columns([1, 1, 1], gap="small")
    run_btn = g_run.button(TR("Lancer", "Run"))
    raz_btn = g_raz.button(TR("RÀZ", "Reset"))
    do_unroll = g_flat.checkbox("2D", False, key="2d")

    if raz_btn:
        st.session_state["_do_raz"] = True
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# compute and api call
if run_btn:
    if not api_key or not api_secret or api_user < 0:
        with col_left:
            st.error(TR(
                "Clés WeCov3r manquantes, impossible d'appeler l'API",
                "Missing WeCov3r Keys, Cannot Call API",
            ))
    else:
        try:
            result = ComputeCurves(cheville, mollet, genou, cuisse, taille)
            leg_height = result["leg_height"]
            tibia_length = result["tibia_length"]
            femur_length = result["femur_length"]
            real_femur_length = result["femur_length_real"]
            garment_length = result["garment_length"]

            leg_height_cm = leg_height / 10.0
            tibia_length_cm = tibia_length / 10.0
            femur_length_cm = femur_length / 10.0
            real_femur_length_cm = real_femur_length / 10.0
            garment_length_cm = garment_length / 10.0

            curves = result["curves"]
            resp = RunPipeline(curves, api_key, api_secret, api_user, do_unroll=do_unroll)
            data = (resp or {}).get("data", [{}])[0]
            infos = data.get("infos", {})
            area_m2 = infos.get("area")
            perim_m = infos.get("perimeter")
            vol_m3 = infos.get("volume")

            area_cm2 = area_m2 * 10000.0 if isinstance(area_m2, (int, float)) else None
            perim_cm = perim_m * 100.0 if isinstance(perim_m, (int, float)) else None
            vol_l = vol_m3 * 1000.0 if isinstance(vol_m3, (int, float)) else None
        except requests.HTTPError as e:
            with col_left:
                st.error(TR(
                    f"HTTP Erreur : {e.response.status_code} — {e.response.text[:300]}",
                    f"HTTP Error: {e.response.status_code} — {e.response.text[:300]}",
                ))
        except Exception as ex:
            with col_left:
                st.error(TR(f"Erreur : {ex}", f"Error: {ex}"))

# results (left)
with col_left:
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader(TR("Calculs", "Calculations"))
    r0, r1, r2 = st.columns([1, 1, 1])

    if data is None:
        r0.write(TR("Pièce (cm) : —", "Part (cm): —"))
        r1.write(TR("Aire (cm²) : —", "Area (cm²): —"))
        if do_unroll:
            r2.write(TR("Bord (cm) : —", "Border (cm): —"))
        else:
            r2.write(TR("Volume (L) : —", "Volume (L): —"))
    else:
        r0.write(TR("**Pièce (cm)** : ", "**Part (cm)**: ") + Format1(garment_length_cm))
        sqrt_area = math.sqrt(area_cm2) if area_cm2 is not None else 0
        if is_french():
            r1.write(f"**Aire (cm²)** : {Format1(sqrt_area, 0)} × {Format1(sqrt_area, 0)}")
        else:
            r1.write(f"**Area (cm²)**: {Format1(sqrt_area, 0)} × {Format1(sqrt_area, 0)}")

        if do_unroll:
            label = TR("**Bord (cm)** : ", "**Border (cm)**: ")
            r2.write(label + Format1(perim_cm))
        else:
            label = TR("**Volume (L)** : ", "**Volume (L)**: ")
            r2.write(label + Format1(vol_l))

    st.markdown("</div>", unsafe_allow_html=True)

# display (right)
with col_right:
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader(TR("Affichage", "Display"))

    c_mode, c_edge = st.columns([1, 1], gap="small")

    if do_unroll:
        opts = [TR("Rempli", "Filled"), TR("Filaire", "Wireframe")]
        default_label = TR("Rempli", "Filled")
    else:
        opts = [TR("Ombré", "Shaded"), TR("Filaire", "Wireframe")]
        default_label = TR("Ombré", "Shaded")

    with c_mode:
        mode = st.selectbox(TR("Rendu", "Render"), opts, index=opts.index(default_label))

    with c_edge:
        if (mode == TR("Ombré", "Shaded")) and (not do_unroll):
            edge_opts = [TR("Aucun", "None"), TR("Bord", "Border"), TR("Toutes", "All")]
            edge_mode = st.selectbox(TR("Arêtes", "Edges"), edge_opts, index=1)
        else:
            edge_mode = "None"
            st.markdown("<div style='height:2.5em'></div>", unsafe_allow_html=True)

    if data is None:
        st.info(TR("Lancer pour calculer", "Run to compute"))
    else:
        if mode == TR("Rempli", "Filled") and do_unroll:
            DrawMeshShade2D(data)
        elif mode == TR("Filaire", "Wireframe") and do_unroll:
            DrawMeshWire2D(data)
        elif mode == TR("Ombré", "Shaded") and not do_unroll:
            DrawMeshShade3D(data, edge_mode=edge_mode)
        elif mode == TR("Filaire", "Wireframe") and not do_unroll:
            DrawMeshWire3D(data)

    st.markdown("</div>", unsafe_allow_html=True)
