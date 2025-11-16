# app.py
# WeCov3r Studio AI – Streamlit
# - mise en page : à gauche 2 rectangles (mesures / résultats), à droite 1 rectangle (rendu 2D/3D)
# - "Mesures (cm)" en 3 groupes de 2 colonnes (Cheville/Mollet, Genou/Cuisse, Taille/—)
# - dernière ligne : 2 colonnes (Calculer / Aplatir)
# - résultats :
#     * si 3D (Aplatir = False)  -> Superficie en cm², Volume en litres
#     * si 2D (Aplatir = True)   -> Superficie en cm², Périmètre en cm
# - booleans/params JSON conformes
# Dépendances : streamlit, requests, numpy, plotly
# Lancement : streamlit run app.py

import math
import requests
import streamlit as st
import plotly.graph_objects as go

# ---------------- page / style ----------------
st.set_page_config(page_title="WeCov3r Studio AI", page_icon="🧩", layout="wide")
st.markdown(
    """
<style>
/* bandeau titre simple */
.wcvr-hero {
  background: linear-gradient(135deg,#5B8CFF 0%,#B66DFF 50%,#FF8BD3 100%);
  border-radius: 16px; padding: 12px 16px; color: #fff; margin-bottom: 8px;
  box-shadow: 0 8px 28px rgba(0,0,0,.18);
}
.wcvr-hero h1 { margin: 0; font-size: 22px; }

/* rectangles (cartes) */
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
.small-input .stCheckbox>div { transform: scale(.95); }
.small-input .stButton>button { padding: 0.35rem 0.8rem; border-radius: 10px; }

/* titres compacts */
h3, .stSubheader { margin-top: 0.2rem; margin-bottom: 0.6rem; }

/* force le contenu à s’aligner en bas de la cellule droite de la 3e rangée */
.align-bottom {
  display: flex;
  flex-direction: column;
  min-height: 28px;      /* ajuste 80–100px selon ton thème */
  justify-content: flex-end;
}
</style>
<div class="wcvr-hero"><h1>Moteur WeCov3r Studio AI</h1></div>
""",
    unsafe_allow_html=True,
)

# ---------------- HTTP helpers ----------------
def call_api(url: str, payload: dict, bearer: str | None = None, timeout_s: float = 30.0) -> dict:
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

def run_pipeline(curves, api_key, api_secret, do_unroll=True):
    # 1) connect
    r1 = call_api("https://wecov3r.com/api/connect", {"k": api_key, "s": api_secret})
    ctx_code = r1.get("data")

    # 2) token
    r2 = call_api("https://wecov3r.com/api/token", {"c": ctx_code, "u": "mesh"})
    token = r2.get("data", r2).get("token") if isinstance(r2.get("data"), dict) else r2.get("data")

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

# ---------------- géométrie (mêmes formules que la version JS) ----------------
def compute_curves(cheville_cm, mollet_cm, genou_cm, cuisse_cm, taille_cm, nb_points=16):
    scale_factor = 10.0  # cm -> mm
    v_cheville = scale_factor * float(cheville_cm)
    v_mollet   = scale_factor * float(mollet_cm)
    v_genou    = scale_factor * float(genou_cm)
    v_cuisse   = scale_factor * float(cuisse_cm)
    v_taille   = scale_factor * float(taille_cm)

    # hauteurs (mm)
    h_cheville = 0.1 * v_taille
    h_mollet   = 0.2 * v_taille
    h_genou    = 0.3 * v_taille
    h_cuisse   = 0.4 * v_taille
    h_fourche  = 0.5 * v_taille

    angle_step = 2.0 * math.pi / nb_points

    def circle_points(radius, z, n=nb_points):
        pts = []
        for i in range(n):
            x = radius * math.cos(angle_step * i)
            y = radius * math.sin(angle_step * i)
            pts.extend([x, y, z])
        return pts

    # rayons (mm) à partir des périmètres (mm) -> r = P / 2π
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

# ---------------- rendu 2D / 3D ----------------
def draw_mesh_2d(mesh_obj: dict):
    defn = mesh_obj.get("definition", {})
    nodes = defn.get("nodes") or []
    topo  = int(defn.get("topology") or 0)
    elements = defn.get("elements") or []
    elementsLastIndex = defn.get("elementsLastIndex")

    if len(nodes) < 6:
        st.warning("aucun nœud exploitable.")
        return

    n = len(nodes) // 3
    x = [nodes[3*i] for i in range(n)]
    y = [nodes[3*i+1] for i in range(n)]

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
        if not elementsLastIndex:
            for i2 in range(len(elements)):
                a = elements[i2]
                b = elements[(i2+1) % len(elements)]
                add_edge(a, b)
        else:
            first = 0
            for last in elementsLastIndex:
                seq = elements[first:last+1]
                for i2 in range(len(seq)):
                    a = seq[i2]
                    b = seq[(i2+1) % len(seq)]
                    add_edge(a, b)
                first = last + 1

    fig = go.Figure(data=[go.Scatter(x=xl, y=yl, mode="lines")])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(scaleanchor="y", scaleratio=1))
    st.plotly_chart(fig, use_container_width=True)

def draw_mesh_3d(mesh_obj: dict):
    defn = mesh_obj.get("definition", {})
    nodes = defn.get("nodes") or []
    topo  = int(defn.get("topology") or 0)
    elements = defn.get("elements") or []
    elementsLastIndex = defn.get("elementsLastIndex")

    if len(nodes) < 6:
        st.warning("aucun nœud 3D exploitable.")
        return

    n = len(nodes) // 3
    x = [nodes[3*i] for i in range(n)]
    y = [nodes[3*i+1] for i in range(n)]
    z = [nodes[3*i+2] for i in range(n)]

    xl, yl, zl = [], [], []

    def add_edge(a, b):
        xl.extend([x[a], x[b], None])
        yl.extend([y[a], y[b], None])
        zl.extend([z[a], z[b], None])

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
        if not elementsLastIndex:
            for i2 in range(len(elements)):
                a = elements[i2]
                b = elements[(i2+1) % len(elements)]
                add_edge(a, b)
        else:
            first = 0
            for last in elementsLastIndex:
                seq = elements[first:last+1]
                for i2 in range(len(seq)):
                    a = seq[i2]
                    b = seq[(i2+1) % len(seq)]
                    add_edge(a, b)
                first = last + 1

    fig = go.Figure(data=[go.Scatter3d(x=xl, y=yl, z=zl, mode="lines")])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), scene=dict(aspectmode="data"))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- UI : gauche (mesures + résultats) / droite (rendu) ----------------
col_left, col_right = st.columns([1.0, 1.2], gap="large")

# secrets (connexion API masquée)
api_key    = "bc1794e6d394e7ce19d3" #st.secrets.get("WECOV3R_API_KEY", "")
api_secret = "5f1da2aa59506f580a0f" #st.secrets.get("WECOV3R_API_SECRET", "")

# valeurs calculées (pour affichage)
data0 = None
area_cm2 = None
perim_cm = None
vol_l = None

with col_left:
    st.markdown('<div class="block small-input">', unsafe_allow_html=True)
    st.subheader("Mesures (cm)")

    g1c1, g1c2 = st.columns([1, 1])
    g2c1, g2c2 = st.columns([1, 1])
    g3c1, g3c2 = st.columns([1, 1])

    cheville = g1c1.number_input("Cheville", min_value=1.0, max_value=1000.0,
                                 value=25.0, step=0.1, format="%.1f")
    mollet   = g1c2.number_input("Mollet",   min_value=1.0, max_value=1000.0,
                                 value=38.0, step=0.1, format="%.1f")

    genou    = g2c1.number_input("Genou",    min_value=1.0, max_value=1000.0,
                                 value=39.0, step=0.1, format="%.1f")
    cuisse   = g2c2.number_input("Cuisse",   min_value=1.0, max_value=1000.0,
                                 value=49.0, step=0.1, format="%.1f")

    taille   = g3c1.number_input("Taille",   min_value=1.0, max_value=300.0,
                                 value=182.0, step=0.1, format="%.1f")

    # --> bouton + checkbox dans g3c2, alignés en bas
    with g3c2:
        st.markdown('<div class="align-bottom">', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1], gap="small")
        run_btn   = c1.button("Calculer", use_container_width=True)
        do_unroll = c2.checkbox("Aplatir", value=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# calcul (si demandé)
if run_btn:
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
            volume_m3   = infos.get("volume")  # peut ne pas exister selon l'opérateur

            # conversions
            if isinstance(area_m2, (int, float)):
                area_cm2 = area_m2 * 10000.0
            if isinstance(perimeter_m, (int, float)):
                perim_cm = perimeter_m * 100.0
            if isinstance(volume_m3, (int, float)):
                vol_l = volume_m3 * 1000.0

        except requests.HTTPError as e:
            with col_left:
                st.error(f"Erreur HTTP: {e.response.status_code} — {e.response.text[:300]}")
        except Exception as ex:
            with col_left:
                st.error(f"Erreur: {ex}")

# helper à mettre au-dessus (ou juste avant le bloc résultats)
def fmt0(x):
    return "—" if x is None else f"{x:,.1f}".replace(",", " ")
   
# bloc résultats (toujours visible sous mesures)
with col_left:
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("Résultats")
    if data0 is None:
        r1, r2 = st.columns([1, 1])
        r1.write("Superficie (cm²) : —")
        if 'do_unroll' in locals() and not do_unroll:
            r2.write("Volume (L) : —")
        else:
            r2.write("Périmètre (cm) : —")
    else:
        r1, r2 = st.columns([1, 1])

        # 0 décimale
        r1.write(f"**Superficie (cm²)** : {fmt0(area_cm2)}")

        if do_unroll:
            r2.write(f"**Périmètre (cm)** : {fmt0(perim_cm)}")
        else:
            r2.write(f"**Volume (L)** : {fmt0(vol_l)}")

    st.markdown('</div>', unsafe_allow_html=True)

# bloc rendu à droite (2D si Aplatir, sinon 3D)
with col_right:
    st.markdown('<div class="block">', unsafe_allow_html=True)
    if data0 is None:
        st.subheader("Rendu")
        st.info("Lancez un calcul pour afficher le rendu.")
    else:
        if do_unroll:
            st.subheader("Rendu 2D")
            draw_mesh_2d(data0)
        else:
            st.subheader("Rendu 3D")
            draw_mesh_3d(data0)
    st.markdown('</div>', unsafe_allow_html=True)
