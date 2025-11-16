# app.py
# WeCov3r Studio AI — Streamlit (axes en cm, 2D/3D options dynamiques, titre centré)
# Dépendances : streamlit, requests, plotly
# Lancer : streamlit run app.py

import math
from typing import Optional, List, Tuple, Dict
import requests
import streamlit as st
import plotly.graph_objects as go
from collections import defaultdict

# ---------------- Page / style ----------------
st.set_page_config(page_title="WeCov3r Studio AI", page_icon="🧩", layout="wide")
st.markdown(
    """
<style>
.wcvr-hero {
  background: linear-gradient(135deg, #401189 0%, #c0bdc5 100%);
  border-radius: 16px; padding: 16px 18px; color: #fff; margin-bottom: 10px;
  box-shadow: 0 8px 28px rgba(0,0,0,.18);
  text-align: center;
}
.wcvr-hero h1 { margin: 0; font-size: 30px; font-weight: 800; }

/* cartes */
.block {
  border: 1px solid rgba(0,0,0,.06);
  border-radius: 12px;
  padding: 12px 14px;
  background: rgba(255,255,255,.85);
  backdrop-filter: blur(6px);
  margin-bottom: 12px;
}

/* inputs compacts */
.small-input .stNumberInput>div>div>input { max-width: 7.5rem; }
.small-input .stButton>button { padding: 0.35rem 0.8rem; border-radius: 10px; }

/* éviter le retour à la ligne du libellé checkbox */
div[data-testid="stCheckbox"] label { white-space: nowrap; }
</style>
<div class="wcvr-hero"><h1>WeCov3r AI Engine</h1></div>
""",
    unsafe_allow_html=True,
)

st.markdown("""
<style>
/* si une .block est vide, on la cache */
.block:empty{display:none !important; padding:0 !important; margin:0 !important; border:0 !important; box-shadow:none !important; background:transparent !important;}
/* et on enlève l’espace avant le premier élément */
.block > :first-child{ margin-top:0 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers format ----------------
def fmt1(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:,.1f}".replace(",", " ")

# ---------------- HTTP helpers ----------------
def call_api(url: str, payload: dict, bearer: Optional[str] = None, timeout_s: float = 30.0) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Origin": "https://wecov3r.com",
        "Referer": "https://wecov3r.com/",
        "X-Requested-With": "XMLHttpRequest",
    }
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()

def run_pipeline(curves: list, api_key: str, api_secret: str, do_unroll: bool = True) -> dict:
    # 1) connect
    r1 = call_api("https://wecov3r.com/api/connect", {"k": api_key, "s": api_secret})
    ctx_code = r1.get("data")

    # 2) token
    r2 = call_api("https://wecov3r.com/api/token", {"c": ctx_code, "u": "mesh"})
    data2 = r2.get("data")
    token = data2.get("token") if isinstance(data2, dict) else data2

    # 3) meshWIRE
    op_wire = {
        "id": "meshWIRE",
        "params": [
            {"key":"WIRE_MODE","type":"INT","value":3},
            {"key":"GROW_MODE","type":"INT","value":0},
            {"key":"GROW_DIST","type":"FLOAT","value":0},
            {"key":"ENABLE_CLOSECV","type":"BOOL","value":False},
            {"key":"ENABLE_AUTOORIENT","type":"BOOL","value":False},
            {"key":"ENABLE_AUTOPOLISH","type":"BOOL","value":True},
            {"key":"ENABLE_AUTOFITCVS","type":"BOOL","value":False},
            {"key":"ENABLE_AUTOCYCLE","type":"BOOL","value":False},
        ]
    }
    payload = {"globals": [], "data": curves, "operators": [op_wire]}
    r3 = call_api("https://wecov3r.com/api/mesh", payload, bearer=token)

    if not do_unroll:
        return r3

    # 4) meshUNROLL
    op_unroll = {
        "id": "meshUNROLL",
        "params": [
            {"key":"UNROLL_MODE","type":"INT","value":1},
            {"key":"ENABLE_AUTOFILL","type":"BOOL","value":True},
            {"key":"ENABLE_MATCH","type":"BOOL","value":True},
            {"key":"ENABLE_RIGID","type":"BOOL","value":True},
            {"key":"RIGID_RATIO","type":"FLOAT","value":0.5},
            {"key":"ENABLE_FULL","type":"BOOL","value":False},
            {"key":"ENABLE_REINDEX","type":"BOOL","value":False},
            {"key":"LAY_MODE","type":"INT","value":1},
            {"key":"ATTACH_DIST","type":"FLOAT","value":0.05},
            {"key":"SHIFT_DIST","type":"FLOAT","value":0.01},
            {"key":"GROUP_DIST","type":"FLOAT","value":0.05},
        ]
    }
    payload2 = {"globals": [], "data": r3.get("data"), "operators": [op_unroll]}
    r4 = call_api("https://wecov3r.com/api/mesh", payload2, bearer=token)
    return r4

# ---------------- Géométrie paramétrique (mm côté data) ----------------
def compute_curves(cheville_cm, mollet_cm, genou_cm, cuisse_cm, taille_cm, nb_points=16):
    # on génère en mm pour le service, on affichera en cm en divisant par 10
    scale = 10.0  # cm -> mm
    v_cheville = scale * float(cheville_cm)
    v_mollet   = scale * float(mollet_cm)
    v_genou    = scale * float(genou_cm)
    v_cuisse   = scale * float(cuisse_cm)
    v_taille   = scale * float(taille_cm)

    # hauteurs (mm)
    h_cheville = 0.1 * v_taille
    h_mollet   = 0.2 * v_taille
    h_genou    = 0.3 * v_taille
    h_cuisse   = 0.4 * v_taille
    h_fourche  = 0.5 * v_taille

    angle = 2.0 * math.pi / nb_points

    def circle_points(r, z, n=nb_points):
        pts = []
        for i in range(n):
            x = r * math.cos(angle * i)
            y = r * math.sin(angle * i)
            pts.extend([x, y, z])
        return pts

    # r = P / (2π)
    r_cheville = v_cheville / (2.0 * math.pi)
    r_mollet   = v_mollet   / (2.0 * math.pi)
    r_genou    = v_genou    / (2.0 * math.pi)
    r_cuisse   = v_cuisse   / (2.0 * math.pi)

    curves_pts = [
        circle_points(r_cheville, h_cheville),
        circle_points(r_mollet,   h_mollet),
        circle_points(r_genou,    h_genou),
        circle_points(r_cuisse,   h_cuisse),
        circle_points(r_cuisse,   h_fourche),
    ]

    curves = []
    for i, pts in enumerate(curves_pts):
        curves.append({
            "definition": {"points": pts, "closed": False},
            "properties": {"uuid": i, "name": f"curve_{i}", "type": "LINES", "scale": 1000},
            "dimension": 3
        })
    return curves

# ---------------- Scaling pour affichage en cm ----------------
def _nodes_to_cm(nodes_mm: List[float]) -> Tuple[List[float], List[float], List[float]]:
    n = len(nodes_mm) // 3
    x = [nodes_mm[3*i]   / 10.0 for i in range(n)]
    y = [nodes_mm[3*i+1] / 10.0 for i in range(n)]
    z = [nodes_mm[3*i+2] / 10.0 for i in range(n)]
    return x, y, z

def _nodes2d_to_cm(nodes_mm: List[float]) -> Tuple[List[float], List[float]]:
    n = len(nodes_mm) // 3
    x = [nodes_mm[3*i]   / 10.0 for i in range(n)]
    y = [nodes_mm[3*i+1] / 10.0 for i in range(n)]
    return x, y

# ---------------- Utilitaires 2D ----------------
def _boundary_loops_from_elements(defn: dict) -> List[List[int]]:
    topo  = int(defn.get("topology") or 0)
    elems = defn.get("elements") or []
    if topo not in (3, 4) or len(elems) < topo:
        return []

    edge_count: Dict[Tuple[int,int], int] = defaultdict(int)
    nb_elts = len(elems) // topo
    for e in range(nb_elts):
        idxs = [elems[topo*e + k] for k in range(topo)]
        for k in range(topo):
            a, b = idxs[k], idxs[(k+1) % topo]
            if a > b: a, b = b, a
            edge_count[(a, b)] += 1

    adj: Dict[int, List[int]] = defaultdict(list)
    for (a, b), c in edge_count.items():
        if c == 1:  # frontière
            adj[a].append(b)
            adj[b].append(a)

    visited_edge = set()
    loops: List[List[int]] = []

    def pop_next_edge(u: int) -> Optional[int]:
        while adj[u]:
            v = adj[u].pop()
            if (min(u, v), max(u, v)) not in visited_edge:
                return v
        return None

    for start in list(adj.keys()):
        if not adj[start]:
            continue
        loop = [start]
        u = start
        v = pop_next_edge(u)
        if v is None:
            continue
        visited_edge.add((min(u, v), max(u, v)))
        loop.append(v)

        while True:
            u, v = v, pop_next_edge(v)
            if v is None:
                break
            visited_edge.add((min(u, v), max(u, v)))
            if v == start:
                loops.append(loop[:])
                break
            loop.append(v)

    return loops

# ---------------- Rendu 2D (rempli + filaire) ----------------
def draw_mesh_2d_filled(mesh_obj: dict):
    defn = mesh_obj.get("definition", {})
    nodes = defn.get("nodes") or []
    topo  = int(defn.get("topology") or 0)
    elems = defn.get("elements") or []
    elidx = defn.get("elementsLastIndex")

    fig = go.Figure()

    # Cas explicite : elementsLastIndex = listes de cycles
    if elidx:
        first = 0
        x_all, y_all = _nodes2d_to_cm(nodes)
        for last in elidx:
            seq = elems[first:last+1]
            if len(seq) >= 3:
                xseq = [x_all[i] for i in seq]
                yseq = [y_all[i] for i in seq]
                if seq[0] != seq[-1]:
                    xseq.append(xseq[0]); yseq.append(yseq[0])
                fig.add_trace(go.Scatter(x=xseq, y=yseq, mode="lines",
                                         fill="toself", name="contour"))
            first = last + 1

    # Sinon, tri/quad -> frontière
    elif topo in (3, 4) and len(elems) >= topo:
        loops = _boundary_loops_from_elements(defn)
        x_all, y_all = _nodes2d_to_cm(nodes)
        if not loops:
            return draw_mesh_2d_wire(mesh_obj)

        for seq in loops:
            if len(seq) >= 3:
                xseq = [x_all[i] for i in seq] + [x_all[seq[0]]]
                yseq = [y_all[i] for i in seq] + [y_all[seq[0]]]
                fig.add_trace(go.Scatter(x=xseq, y=yseq, mode="lines",
                                         fill="toself", name="contour"))
    else:
        return draw_mesh_2d_wire(mesh_obj)

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(title="cm", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="cm"),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def draw_mesh_2d_wire(mesh_obj: dict):
    defn = mesh_obj.get("definition", {})
    nodes = defn.get("nodes") or []
    topo  = int(defn.get("topology") or 0)
    elements = defn.get("elements") or []
    elidx = defn.get("elementsLastIndex")

    if len(nodes) < 6:
        st.warning("aucun nœud exploitable.")
        return

    x, y = _nodes2d_to_cm(nodes)
    xl, yl = [], []

    def add_edge(a, b):
        xl.extend([x[a], x[b], None])
        yl.extend([y[a], y[b], None])

    if topo and topo > 0:
        nb_elts = len(elements) // topo
        edges = set()
        for e in range(nb_elts):
            idxs = [elements[topo*e + k] for k in range(topo)]
            for k in range(topo):
                a, b = idxs[k], idxs[(k+1) % topo]
                if a > b: a, b = b, a
                edges.add((a, b))
        for a, b in edges:
            add_edge(a, b)
    else:
        if not elidx:
            for i2 in range(len(elements)):
                a = elements[i2]
                b = elements[(i2+1) % len(elements)]
                add_edge(a, b)
        else:
            first = 0
            for last in elidx:
                seq = elements[first:last+1]
                for i2 in range(len(seq)):
                    a = seq[i2]
                    b = seq[(i2+1) % len(seq)]
                    add_edge(a, b)
                first = last + 1

    fig = go.Figure(data=[go.Scatter(x=xl, y=yl, mode="lines")])
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(title="cm", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="cm"),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Rendu 3D (ombré lissé / filaire) ----------------
def _add_wire3d(fig, x, y, z, elems, topo, elidx):
    xl, yl, zl = [], [], []
    def add_edge(a, b):
        xl.extend([x[a], x[b], None])
        yl.extend([y[a], y[b], None])
        zl.extend([z[a], z[b], None])

    if topo and topo > 0:
        nb_elts = len(elems) // topo
        edges = set()
        for e in range(nb_elts):
            idxs = [elems[topo*e + k] for k in range(topo)]
            for k in range(topo):
                a, b = idxs[k], idxs[(k+1) % topo]
                if a > b: a, b = b, a
                edges.add((a, b))
        for a, b in edges:
            add_edge(a, b)
    else:
        if not elidx:
            for i2 in range(len(elems)):
                a = elems[i2]
                b = elems[(i2+1) % len(elems)]
                add_edge(a, b)
        else:
            first = 0
            for last in elidx:
                seq = elems[first:last+1]
                for i2 in range(len(seq)):
                    a = seq[i2]
                    b = seq[(i2+1) % len(seq)]
                    add_edge(a, b)
                first = last + 1

    if xl:
        fig.add_trace(go.Scatter3d(x=xl, y=yl, z=zl, mode="lines", name="edges"))

def draw_mesh_3d_shaded(mesh_obj: dict, add_edges: bool = True):
    defn = mesh_obj.get("definition", {})
    nodes = defn.get("nodes") or []
    topo  = int(defn.get("topology") or 0)
    elems = defn.get("elements") or []
    elidx = defn.get("elementsLastIndex")

    if len(nodes) < 9:
        st.warning("aucun nœud 3D exploitable.")
        return

    x_mm, y_mm, z_mm = _nodes_to_cm(nodes)  # déjà en cm via helper
    x, y, z = x_mm, y_mm, z_mm

    fig = go.Figure()
    lighting = dict(ambient=0.55, diffuse=0.85, specular=0.2, roughness=0.9)

    if topo in (3, 4) and len(elems) >= topo:
        i_idx, j_idx, k_idx = [], [], []
        if topo == 3 and len(elems) % 3 == 0:
            for t in range(len(elems)//3):
                a, b, c = elems[3*t:3*t+3]
                i_idx.append(a); j_idx.append(b); k_idx.append(c)
        elif topo == 4 and len(elems) % 4 == 0:
            for q in range(len(elems)//4):
                a, b, c, d = elems[4*q:4*q+4]
                i_idx.extend([a, a]); j_idx.extend([b, c]); k_idx.extend([c, d])

        if i_idx:
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i_idx, j=j_idx, k=k_idx,
                color="lightblue", opacity=1.0,
                flatshading=False,  # normales lissées (par nœud)
                lighting=lighting, showscale=False
            ))
            if add_edges:
                _add_wire3d(fig, x, y, z, elems, topo, elidx)
        else:
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                color="lightblue", opacity=1.0,
                flatshading=False, alphahull=12,
                lighting=lighting, showscale=False
            ))
            if add_edges:
                _add_wire3d(fig, x, y, z, elems, topo, elidx)
    else:
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            color="lightblue", opacity=1.0,
            flatshading=False, alphahull=12,
            lighting=lighting, showscale=False
        ))
        if add_edges:
            _add_wire3d(fig, x, y, z, elems, topo, elidx)

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis_title="cm", yaxis_title="cm", zaxis_title="cm",
            aspectmode="data"
        ),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def draw_mesh_3d_wire(mesh_obj: dict):
    defn = mesh_obj.get("definition", {})
    nodes = defn.get("nodes") or []
    topo  = int(defn.get("topology") or 0)
    elems = defn.get("elements") or []
    elidx = defn.get("elementsLastIndex")

    if len(nodes) < 9:
        st.warning("aucun nœud 3D exploitable.")
        return

    x, y, z = _nodes_to_cm(nodes)

    fig = go.Figure()
    _add_wire3d(fig, x, y, z, elems, topo, elidx)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(xaxis_title="cm", yaxis_title="cm", zaxis_title="cm", aspectmode="data"),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- UI : gauche (mesures + résultats) / droite (rendu) ----------------
col_left, col_right = st.columns([1.0, 1.2], gap="large")

api_key    = "bc1794e6d394e7ce19d3" #st.secrets.get("WECOV3R_API_KEY", "")
api_secret = "5f1da2aa59506f580a0f" #st.secrets.get("WECOV3R_API_SECRET", "")

data0 = None
area_cm2 = None
perim_cm = None
vol_l = None
do_unroll = True  # défaut

with col_left:
    # ---- MESURES ----
    st.markdown('<div class="block small-input">', unsafe_allow_html=True)
    st.subheader("Mesures (cm)")

    # AAA BBB
    r1a, r1b = st.columns([1, 1], gap="small", vertical_alignment="bottom")
    cheville = r1a.number_input("Cheville", 1.0, 1000.0, 25.0, 0.1, format="%.1f")
    mollet   = r1b.number_input("Mollet",   1.0, 1000.0, 38.0, 0.1, format="%.1f")

    # CCC DDD
    r2a, r2b = st.columns([1, 1], gap="small", vertical_alignment="bottom")
    genou  = r2a.number_input("Genou",  1.0, 1000.0, 39.0, 0.1, format="%.1f")
    cuisse = r2b.number_input("Cuisse", 1.0, 1000.0, 49.0, 0.1, format="%.1f")

    # EEE FFF (FFF=G H)
    r3a, r3b = st.columns([1, 1], gap="small", vertical_alignment="bottom")
    taille = r3a.number_input("Taille", 1.0, 300.0, 182.0, 0.1, format="%.1f")
    gcol, hcol = r3b.columns([1, 1], gap="small", vertical_alignment="bottom")
    run_btn   = gcol.button("Run AI")
    do_unroll = hcol.checkbox("Aplatir", True)

    st.markdown('</div>', unsafe_allow_html=True)

# calcul
if 'run_btn' in locals() and run_btn:
    if not api_key or not api_secret:
        with col_left:
            st.error("Clés API absentes. Ajoutez-les dans .streamlit/secrets.toml (local) ou dans les Secrets de Streamlit Cloud.")
    else:
        try:
            curves = compute_curves(cheville, mollet, genou, cuisse, taille)
            resp = run_pipeline(curves, api_key, api_secret, do_unroll=do_unroll)
            data0 = (resp or {}).get("data", [{}])[0]

            infos = data0.get("infos", {})
            area_m2     = infos.get("area")
            perimeter_m = infos.get("perimeter")
            volume_m3   = infos.get("volume")

            area_cm2 = area_m2 * 10000.0 if isinstance(area_m2, (int, float)) else None
            perim_cm = perimeter_m * 100.0 if isinstance(perimeter_m, (int, float)) else None
            vol_l    = volume_m3 * 1000.0 if isinstance(volume_m3, (int, float)) else None

        except requests.HTTPError as e:
            with col_left:
                st.error(f"Erreur HTTP: {e.response.status_code} — {e.response.text[:300]}")
        except Exception as ex:
            with col_left:
                st.error(f"Erreur: {ex}")

# ---- RÉSULTATS ----
with col_left:
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("Résultats")
    r1, r2 = st.columns([1, 1])
    if data0 is None:
        r1.write("Superficie (cm²) : —")
        r2.write(("Volume (L) : —") if (not do_unroll) else ("Périmètre (cm) : —"))
    else:
        r1.write(f"**Superficie (cm²)** : {fmt1(area_cm2)}")
        if do_unroll:
            r2.write(f"**Périmètre (cm)** : {fmt1(perim_cm)}")
        else:
            r2.write(f"**Volume (L)** : {fmt1(vol_l)}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---- RENDU + options dynamiques ----
with col_right:
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("Rendu")

    # options selon Aplatir
    if do_unroll:
        opts = ["Rempli", "Filaire"]
        default = "Rempli"
    else:
        opts = ["Ombré", "Filaire"]
        default = "Ombré"

    mode = st.selectbox("Affichage", opts, index=opts.index(default))

    if data0 is None:
        st.info("Run AI pour afficher le rendu")
    else:
        if mode == "Rempli":
            draw_mesh_2d_filled(data0)
        elif mode == "Filaire":
            draw_mesh_2d_wire(data0)
        elif mode == "Ombré":
            draw_mesh_3d_shaded(data0, add_edges=True)
        elif mode == "Filaire":
            draw_mesh_3d_wire(data0)

    st.markdown('</div>', unsafe_allow_html=True)
