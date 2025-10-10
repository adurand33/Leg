#
# leg.py — WeCov3r Studio Leg
#
import math
from typing import Optional, List, Tuple, Dict
import requests
import streamlit as st
import os
import plotly.graph_objects as go
from collections import defaultdict

# ============ Page / Style ============
st.set_page_config(page_title="WeCov3r AI Leg", page_icon="🤖", layout="wide")
st.markdown(
  """
<style>
.wcvr-hero {
  background: linear-gradient(135deg, #401189 0%, #c0bdc5 100%);
  border-radius: 16px;
  padding: 16px 18px;
  color: #fff;
  margin-bottom: 10px;
  box-shadow: 0 8px 28px rgba(0,0,0,.18);
  text-align: center;
}
.wcvr-hero h1 {
  margin: 0;
  font-size: 40px;
  font-weight: 800;
}
.block {
  border: 1px solid rgba(0,0,0,.06);
  border-radius: 12px;
  padding: 12px 14px;
  background: rgba(255,255,255,.85);
  backdrop-filter: blur(6px);
  margin-bottom: 12px;
}
.small-input .stNumberInput>div>div>input {
  max-width: 7.5rem;
}
.small-input .stButton>button {
  padding: 0.35rem 0.8rem;
  border-radius: 10px;
}
div[data-testid="stCheckbox"] label {
  white-space: nowrap;
}
.stButton > button, .stButton > button * {
  white-space: nowrap !important;
}
.block:empty {
  display: none !important;
  padding: 0 !important;
  margin: 0 !important;
  border: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
}
.block > :first-child {
  margin-top: 0 !important;
}
.wcvr-hero h1 {
  font-size: 30px; /* Keeps the title size */
  line-height: 1.1;
}
.wcvr-hero h1 .robot {
  font-size: 1.2em; /* Increases only the robot emoji size */
  line-height: 1;
  vertical-align: -0.12em; /* Slight vertical centering adjustment for the emoji */
  display: inline-block;  /* Ensures vertical-align works properly */
  filter: drop-shadow(0 1px 0 rgba(0,0,0,.15)); /* Optional: Adds a subtle relief effect */
}
</style>
<div class="wcvr-hero">
  <h1><span class="robot">🔄</span>&nbsp;WeCov3r AI Leg <span class="robot">🔄</span></h1>
</div>
""",
  unsafe_allow_html=True,
)

# ============ Brand Colors & Lighting ============
BRAND_VIOLET = "#401189"                   # Brand color for lines (2D)
BRAND_VIOLET_RGBA = "rgba(64,17,137,0.25)" # 2D fill color with transparency
SHADING_VIOLET = "#6B4AE6"                 # Lighter 3D violet for shading

# Shading and light settings for 3D rendering
MESH_LIGHTING = dict(ambient=0.55, diffuse=1.0, specular=0.45, roughness=0.80)
MESH_LIGHTPOS = dict(x=1.4, y=0.2, z=2.0)

# ============ 3D Edge Styling ============
EDGE3D_BORDER_WIDTH = 8             # 3D edge width
EDGE3D_BORDER_COLOR = BRAND_VIOLET  # WeCov3r brand color
# ============ 2D Edge Styling ============
EDGE2D_BORDER_WIDTH = 4             # 2D edge width
EDGE2D_BORDER_COLOR = BRAND_VIOLET  # WeCov3r brand color

# ============ Defaults & Session State ============
DEFAULTS = {"cheville": 25.0, "mollet": 38.0, "genou": 39.0, "cuisse": 49.0, "taille": 182.0}

# Initialize session state with defaults
for key, value in DEFAULTS.items():
  if key not in st.session_state:
    st.session_state[key] = value

if "_do_raz" not in st.session_state:
  st.session_state["_do_raz"] = False

# Reset to defaults if requested
if st.session_state["_do_raz"]:
  for key, value in DEFAULTS.items():
    st.session_state[key] = value
  st.session_state["_do_raz"] = False

# ============ Format Helper ============
def fmt1(x: Optional[float], decimals: int = 1) -> str:
  """
  Format a number as a string with a specified number of decimal places, input is None, return "—".
  """
  if x is None:
    return "—"
  return f"{x:,.{decimals}f}".replace(",", " ")

# ============ HTTP Helpers ============
def call_api(url: str, payload: dict, bearer: Optional[str] = None, timeout_s: float = 30.0) -> dict:
  """
  Call the WeCov3r API with the given payload and optional bearer token.
  """
  headers = {
    "Content-Type": "application/json",
    "Origin": "https://wecov3r.com",
    "Referer": "https://wecov3r.com/",
    "X-Requested-With": "XMLHttpRequest",
  }
  if bearer:
    headers["Authorization"] = f"Bearer {bearer}"
  response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
  response.raise_for_status()
  return response.json()

def run_pipeline(curves: list, api_key: str, api_secret: str, do_unroll: bool = True) -> dict:
  """
  Run the WeCov3r pipeline: connect, get token, generate mesh, and optionally unroll.
  """
  # 1) Connect to the API
  r1 = call_api("https://wecov3r.com/api/connect", {"k": api_key, "s": api_secret})
  ctx_code = r1.get("data")

  # 2) Get token
  r2 = call_api("https://wecov3r.com/api/token", {"c": ctx_code, "u": "mesh"})
  data2 = r2.get("data")
  token = data2.get("token") if isinstance(data2, dict) else data2

  # 3) Generate mesh (wire mode)
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
    ]
  }
  payload = {"globals": [], "data": curves, "operators": [op_wire]}
  r3 = call_api("https://wecov3r.com/api/mesh", payload, bearer=token)

  if not do_unroll:
    return r3

  # 4) Unroll mesh
  op_unroll = {
    "id": "meshUNROLL",
    "params": [
      {"key": "UNROLL_MODE", "type": "INT", "value": 1},
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
    ]
  }
  payload2 = {"globals": [], "data": r3.get("data"), "operators": [op_unroll]}
  r4 = call_api("https://wecov3r.com/api/mesh", payload2, bearer=token)
  return r4

# ============ Geometry (last point at 359°) ============
def compute_curves(ankle_cm, calf_cm, knee_cm, thigh_cm, height_cm, nb_point=32):
    """
    Compute the curves for the leg geometry based on input measurements, returns curves and segment lengths in millimeters.
    """
    scale = 10.0 # Convert cm to mm

    v_ankle  = scale * float(ankle_cm)
    v_calf   = scale * float(calf_cm)
    v_knee   = scale * float(knee_cm)
    v_thigh  = scale * float(thigh_cm)
    v_height = scale * float(height_cm)

    # Heights in mm
    h_sole   = 0.000 * v_height # Ground level
    h_ankle  = 0.039 * v_height # Ankle joint center
    h_calf   = 0.200 * v_height # Lower leg max
    h_knee   = 0.285 * v_height # Knee joint center
    h_thigh  = 0.400 * v_height # Upper leg max
    h_crotch = 0.500 * v_height # Leg top

    # Calculate leg height (ankle to crotch)
    leg_height = h_crotch - h_ankle

    # Dynamic offsets as a percentage of leg height
    garment_bot_offset = 0.07 * leg_height # 7% of leg height above ankle
    garment_top_offset = 0.14 * leg_height # 14% of leg height below crotch

    # Adjusted heights for garment
    h_garment_bot = h_ankle + garment_bot_offset  # Slightly above ankle
    h_garment_top = h_crotch - garment_top_offset # Slightly below crotch

    # Calculated segment lengths in mm
    tibia_length      = h_knee - h_ankle
    femur_length      = h_crotch - h_knee
    femur_length_real = 0.245 * v_height # From hip joint to knee
    garment_length    = 0.98 * (h_garment_top - h_garment_bot) # Outer leg length

    def circle_points_359(r, z, n=nb_point):
        """
        Generate points for a circle at height z with radius r.
        """
        pts = []
        if n <= 1:
            pts.extend([r, 0.0, z])
            return pts
        end = math.radians(359.0)
        step = end / (n - 1)
        for i in range(n):
            a = i * step
            x = r * math.cos(a)
            y = r * math.sin(a)
            pts.extend([x, y, z])
        return pts

    # Radii in mm (circumference in cm → radius in mm)
    r_ankle = v_ankle / (2.0 * math.pi)
    r_calf  = v_calf  / (2.0 * math.pi)
    r_knee  = v_knee  / (2.0 * math.pi)
    r_thigh = v_thigh / (2.0 * math.pi)

    curves_pts = [
        circle_points_359(r_ankle, h_garment_bot),
        circle_points_359(r_calf,  h_calf),
        circle_points_359(r_knee,  h_knee),
        circle_points_359(r_thigh, h_thigh),
        circle_points_359(r_thigh, h_garment_top),
    ]

    curves = []
    for i, pts in enumerate(curves_pts):
        curves.append({
            "definition": {"points": pts, "open": False}, # Closed semantics on service side
            "properties": {"uuid": i, "name": f"curve_{i}", "type": "LINES", "scale": 1000},
            "dimension": 3
        })

    return {
        "curves": curves,
        "leg_height": leg_height,
        "tibia_length": tibia_length,
        "femur_length": femur_length,
        "femur_length_real": femur_length_real,
        "garment_length": garment_length,
        "garment_bot_offset_cm": garment_bot_offset / scale, # Return in cm
        "garment_top_offset_cm": garment_top_offset / scale, # Return in cm
    }

# ============ Scaling for cm Display ============
def _nodes_to_cm(nodes_mm: List[float]) -> Tuple[List[float], List[float], List[float]]:
  """
  Convert node coordinates from mm to cm.
  """
  n = len(nodes_mm) // 3
  x = [nodes_mm[3*i+0] / 10.0 for i in range(n)]
  y = [nodes_mm[3*i+1] / 10.0 for i in range(n)]
  z = [nodes_mm[3*i+2] / 10.0 for i in range(n)]
  return x, y, z

def _nodes2d_to_cm(nodes_mm: List[float]) -> Tuple[List[float], List[float]]:
  """
  Convert 2D node coordinates from mm to cm.
  """
  n = len(nodes_mm) // 3
  x = [nodes_mm[3*i+0] / 10.0 for i in range(n)]
  y = [nodes_mm[3*i+1] / 10.0 for i in range(n)]
  return x, y

# ============ 2D Helpers ============
def _boundary_loops_from_elements(defn: dict) -> List[List[int]]:
  """
  Extract boundary loops from mesh elements.
  """
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
    if c == 1:
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

# ============ 2D Rendering ============
def draw_mesh_2d_filled(mesh_obj: dict):
  """
  Render a filled 2D mesh.
  """
  defn = mesh_obj.get("definition", {})
  nodes = defn.get("nodes") or []
  topo  = int(defn.get("topology") or 0)
  elems = defn.get("elements") or []
  elidx = defn.get("elementsLastIndex")
  fig = go.Figure()

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
        fig.add_trace(go.Scatter(
          x=xseq, y=yseq, mode="lines",
          line=dict(color=EDGE2D_BORDER_COLOR, width=EDGE2D_BORDER_WIDTH),
          fill="toself", fillcolor=BRAND_VIOLET_RGBA,
          name="contour"
        ))
      first = last + 1
  elif topo in (3, 4) and len(elems) >= topo:
    loops = _boundary_loops_from_elements(defn)
    x_all, y_all = _nodes2d_to_cm(nodes)
    if not loops:
      return draw_mesh_2d_wire(mesh_obj)
    for seq in loops:
      if len(seq) >= 3:
        xseq = [x_all[i] for i in seq] + [x_all[seq[0]]]
        yseq = [y_all[i] for i in seq] + [y_all[seq[0]]]
        fig.add_trace(go.Scatter(
          x=xseq, y=yseq, mode="lines",
          line=dict(color=EDGE2D_BORDER_COLOR, width=EDGE2D_BORDER_WIDTH),
          fill="toself", fillcolor=BRAND_VIOLET_RGBA,
          name="contour"
        ))
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
  """
  Render a wireframe 2D mesh.
  """
  defn = mesh_obj.get("definition", {})
  nodes = defn.get("nodes") or []
  topo  = int(defn.get("topology") or 0)
  elements = defn.get("elements") or []
  elidx = defn.get("elementsLastIndex")

  if len(nodes) < 6:
    st.warning("Pas assez de points")
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

  fig = go.Figure(data=[go.Scatter(x=xl, y=yl, mode="lines",
                   line=dict(width=EDGE2D_BORDER_WIDTH, color=EDGE2D_BORDER_COLOR))])
  fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis=dict(title="cm", scaleanchor="y", scaleratio=1),
    yaxis=dict(title="cm"),
    showlegend=False
  )
  st.plotly_chart(fig, use_container_width=True)

# ============ 3D Edge Helpers ============
def _add_wire3d(fig, x, y, z, elems, topo, elidx):
  """
  Add all edges (wireframe) to the 3D figure.
  """
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
    fig.add_trace(go.Scatter3d(
      x=xl, y=yl, z=zl, mode="lines",
      name="edges",
      line=dict(width=EDGE3D_BORDER_WIDTH, color=EDGE3D_BORDER_COLOR)
    ))

def _add_boundary_edges3d(fig, x, y, z, elems, topo):
  """
  Add only boundary edges (belonging to a single face) to the 3D figure.
  """
  if topo not in (3, 4) or not elems:
    return

  edge_count = defaultdict(int)
  n = len(elems) // topo
  for e in range(n):
    idxs = [elems[topo*e + k] for k in range(topo)]
    for k in range(topo):
      a, b = idxs[k], idxs[(k+1) % topo]
      if a > b: a, b = b, a
      edge_count[(a, b)] += 1

  xl, yl, zl = [], [], []
  for (a, b), c in edge_count.items():
    if c == 1:
      xl.extend([x[a], x[b], None])
      yl.extend([y[a], y[b], None])
      zl.extend([z[a], z[b], None])

  if xl:
    fig.add_trace(go.Scatter3d(
      x=xl, y=yl, z=zl, mode="lines",
      name="border",
      line=dict(width=EDGE3D_BORDER_WIDTH, color=EDGE3D_BORDER_COLOR)
    ))

# ============ 3D Rendering ============
def draw_mesh_3d_shaded(mesh_obj: dict, edge_mode: str = "Bord"):
  """
  Render a shaded 3D mesh with optional edge display, edge_mode: "Aucune" | "Bord" | "Toutes"
  """
  defn = mesh_obj.get("definition", {})
  nodes = defn.get("nodes") or []
  topo  = int(defn.get("topology") or 0)
  elems = defn.get("elements") or []
  elidx = defn.get("elementsLastIndex")

  if len(nodes) < 9:
    st.warning("Pas assez de points")
    return

  x, y, z = _nodes_to_cm(nodes)
  fig = go.Figure()

  def add_edges_choice():
    if edge_mode == "Bord":
      _add_boundary_edges3d(fig, x, y, z, elems, topo)
    elif edge_mode == "Toutes":
      _add_wire3d(fig, x, y, z, elems, topo, elidx)
    # "Aucune" -> do nothing

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
        color=SHADING_VIOLET, opacity=1.0,
        flatshading=False,
        lighting=MESH_LIGHTING, lightposition=MESH_LIGHTPOS,
        showscale=False
      ))
      add_edges_choice()
    else:
      fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        color=SHADING_VIOLET, opacity=1.0,
        flatshading=False, alphahull=12,
        lighting=MESH_LIGHTING, lightposition=MESH_LIGHTPOS,
        showscale=False
      ))
      add_edges_choice()
  else:
    fig.add_trace(go.Mesh3d(
      x=x, y=y, z=z,
      color=SHADING_VIOLET, opacity=1.0,
      flatshading=False, alphahull=12,
      lighting=MESH_LIGHTING, lightposition=MESH_LIGHTPOS,
      showscale=False
    ))
    add_edges_choice()

  fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    scene=dict(xaxis_title="cm", yaxis_title="cm", zaxis_title="cm", aspectmode="data"),
    showlegend=False
  )
  st.plotly_chart(fig, use_container_width=True)

def draw_mesh_3d_wire(mesh_obj: dict):
  """
  Render a wireframe 3D mesh.
  """
  defn = mesh_obj.get("definition", {})
  nodes = defn.get("nodes") or []
  topo  = int(defn.get("topology") or 0)
  elems = defn.get("elements") or []
  elidx = defn.get("elementsLastIndex")

  if len(nodes) < 9:
    st.warning("Pas assez de points")
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

# ============ UI Layout ============
col_left, col_right = st.columns([1.0, 1.2], gap="large")

# Read API keys from Streamlit secrets (with fallback for local testing)
api_key    = st.secrets.get("WECOV3R_API_KEY", os.getenv("WECOV3R_API_KEY", ""))
api_secret = st.secrets.get("WECOV3R_API_SECRET", os.getenv("WECOV3R_API_SECRET", ""))

# Optionally, block Run if secrets are missing
if not api_key or not api_secret:
  st.warning("Clés WeCov3r manquantes ")

data = None
garment_length_cm = None
area_cm2 = None
perim_cm = None
vol_l = None
do_unroll = True

with col_left:
  st.markdown('<div class="block small-input">', unsafe_allow_html=True)
  st.subheader("Mesures (cm)")
  # Ankle and Calf
  r1a, r1b = st.columns([1, 1], gap="small", vertical_alignment="bottom")
  cheville = r1a.number_input("Cheville", 1.0, 1000.0, step=0.1, format="%.1f", key="cheville")
  mollet   = r1b.number_input("Mollet",   1.0, 1000.0, step=0.1, format="%.1f", key="mollet")
  # Knee and Thigh
  r2a, r2b = st.columns([1, 1], gap="small", vertical_alignment="bottom")
  genou  = r2a.number_input("Genou",  1.0, 1000.0, step=0.1, format="%.1f", key="genou")
  cuisse = r2b.number_input("Cuisse", 1.0, 1000.0, step=0.1, format="%.1f", key="cuisse")
  # Height and Buttons
  r3a, r3b = st.columns([1, 1], gap="small", vertical_alignment="bottom")
  taille = r3a.number_input("Taille", 1.0, 300.0, step=0.1, format="%.1f", key="taille")
  g_run, g_raz, g_flat = r3b.columns([1, 1, 1], gap="small", vertical_alignment="bottom")
  run_btn = g_run.button("Run")
  raz_btn = g_raz.button("RAZ")
  do_unroll = g_flat.checkbox("2D", True, key="2d")

  if raz_btn:
    st.session_state["_do_raz"] = True
    st.rerun()
  st.markdown('</div>', unsafe_allow_html=True)

# Compute and fetch results
if run_btn:
  if not api_key or not api_secret:
    with col_left:
      st.error("Clés WeCov3r manquantes !")
  else:
    try:
      result = compute_curves(cheville, mollet, genou, cuisse, taille)
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

      curves  = result["curves"]
      resp    = run_pipeline(curves, api_key, api_secret, do_unroll=do_unroll)
      data    = (resp or {}).get("data", [{}])[0]
      infos   = data.get("infos", {})
      area_m2 = infos.get("area")
      perim_m = infos.get("perimeter")
      vol_m3  = infos.get("volume")

      area_cm2 = area_m2 * 10000.0 if isinstance(area_m2, (int, float)) else None
      perim_cm = perim_m * 100.0 if isinstance(perim_m, (int, float)) else None
      vol_l  = vol_m3 * 1000.0 if isinstance(vol_m3, (int, float)) else None
    except requests.HTTPError as e:
      with col_left:
        st.error(f"HTTP Erreur: {e.response.status_code} — {e.response.text[:300]}")
    except Exception as ex:
      with col_left:
        st.error(f"Erreur: {ex}")

# Display results
with col_left:
  st.markdown('<div class="block">', unsafe_allow_html=True)
  st.subheader("Calculs")
  r0, r1, r2 = st.columns([1, 1, 1])
  if data is None:
    r0.write("Pièce (cm): —")
    r1.write("Aire (cm²): —")
    r2.write(("Volume (L): —") if (not do_unroll) else ("Bord (cm): —"))
  else:
    r0.write(f"**Pièce (cm)** : {fmt1(garment_length_cm)}")
    sqrt_area = math.sqrt(area_cm2) if area_cm2 is not None else 0
    r1.write(f"**Aire (cm²)** : {fmt1(sqrt_area, 0)} × {fmt1(sqrt_area, 0)}")
    if do_unroll:
      r2.write(f"**Bord (cm)** : {fmt1(perim_cm)}")
    else:
      r2.write(f"**Volume (L)** : {fmt1(vol_l)}")
  st.markdown('</div>', unsafe_allow_html=True)

# Render mesh
with col_right:
  st.markdown('<div class="block">', unsafe_allow_html=True)
  st.subheader("Affichage")
  # Side-by-side combos
  c_mode, c_edge = st.columns([1, 1], gap="small")
  # Display mode
  if do_unroll:
    opts = ["Rempli", "Filaire"]; default = "Rempli"
  else:
    opts = ["Ombré", "Filaire"]; default = "Ombré"
  with c_mode:
    mode = st.selectbox("Rendu", opts, index=opts.index(default))
  # Edge mode (only relevant for 3D shaded)
  with c_edge:
    if (mode == "Ombré") and (not do_unroll):
      edge_mode = st.selectbox("Arêtes", ["Aucun", "Bord", "Toutes"], index=1)
    else:
      edge_mode = "None"
      # Placeholder to keep the line at similar height
      st.markdown("<div style='height:2.6em'></div>", unsafe_allow_html=True)

  if data is None:
    st.info("Run pour lancer le calcul")
  else:
    if mode == "Rempli":
      draw_mesh_2d_filled(data)
    elif mode == "Filaire" and do_unroll:
      draw_mesh_2d_wire(data)
    elif mode == "Ombré" and not do_unroll:
      draw_mesh_3d_shaded(data, edge_mode=edge_mode)
    elif mode == "Filaire" and not do_unroll:
      draw_mesh_3d_wire(data)
  st.markdown('</div>', unsafe_allow_html=True)
