#
# wecov3r studio leg
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

# colors/util
def ConvertHexToRGB(h: str, a: float) -> str:
    # "#abc" or "#aabbcc" -> css rgba
    h = (h or "#262730").lstrip("#")
    if len(h) == 3: h = "".join(c*2 for c in h)
    r, g, b = int(h[:2],16), int(h[2:4],16), int(h[4:],16)
    return f"rgba({r},{g},{b},{a})"

def GetExtents(vals):
    # numeric min/max ignoring none/nan
    mn, mx = math.inf, -math.inf
    for v in vals or []:
        if isinstance(v, (int, float)) and v == v:
            mn = min(mn, v); mx = max(mx, v)
    return (None, None) if mn is math.inf else (mn, mx)

# unified plotly renderer (fr + ui tweaks)
def PlotFrench(
    fig,
    *,
    height=450,
    axis_color="#ffffff",  # same color for tick marks + labels
    ticks=None,            # auto ticks
    ticklen=10,            # label spacing
    grid_alpha=0.18,       # faint grid
    pad_ratio=0.12,        # slight zoom out
    margin_left=70,        # room so y labels never clip
    margin_bot=64,         # room for "cm"
):
    # theme/text colors
    theme_text = st.get_option("theme.textColor") or axis_color
    axis_color = axis_color or theme_text
    grid_color = ConvertHexToRGB(axis_color, grid_alpha)

    # clone + base style
    f = go.Figure(fig)
    f.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme_text),
        margin=dict(l=margin_left, r=16, t=12, b=margin_bot),
        hovermode=False
    )

    # x axis: no box/grid, readable ticks
    f.update_xaxes(
        showline=False,
        showgrid=False, gridcolor=grid_color, gridwidth=1,
        zeroline=False,
        automargin=True,
        showticklabels=True,
        ticks="outside",
        ticklen=ticklen,
        tickcolor="rgba(0,0,0,0)",  # hide tick mark
        tickwidth=0,
        ticklabelposition="outside",
        tickfont=dict(color=axis_color),
    )
    # y axis: only horizontal grid, no box
    f.update_yaxes(
        showline=False,
        showgrid=True, gridcolor=grid_color, gridwidth=1,
        zeroline=False,
        automargin=True,
        showticklabels=True,
        ticks="outside",
        ticklen=ticklen,
        tickcolor="rgba(0,0,0,0)",
        tickwidth=0,
        ticklabelposition="outside",
        tickfont=dict(color=axis_color),
    )

    # ensure ticks visible and labels never clip
    if ticks is not None:
        tickvals = list(ticks)
        ticktext = [str(v) for v in tickvals]
        f.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
        f.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

        xs, ys = [], []
        for tr in f.data:
            xs.extend(getattr(tr, "x", []) or [])
            ys.extend(getattr(tr, "y", []) or [])
        xmin, xmax = GetExtents(xs); ymin, ymax = GetExtents(ys)

        tx_min, tx_max = min(tickvals), max(tickvals)
        xmin = tx_min if xmin is None else min(xmin, tx_min)
        xmax = tx_max if xmax is None else max(xmax, tx_max)
        ymin = tx_min if ymin is None else min(ymin, tx_min)
        ymax = tx_max if ymax is None else max(ymax, tx_max)

        if xmin == xmax: xmax = xmin + 1
        if ymin == ymax: ymax = ymin + 1
        px = pad_ratio * (xmax - xmin)
        py = pad_ratio * (ymax - ymin)
        f.update_xaxes(range=[xmin - px, xmax + px])
        f.update_yaxes(range=[ymin - py, ymax + py])

    # 3d: minimal frame + readable ticks
    if "scene" in f.layout:
        f.update_scenes(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, showbackground=False,
                       tickfont=dict(color=axis_color)),
            yaxis=dict(showgrid=False, showbackground=False,
                       tickfont=dict(color=axis_color)),
            zaxis=dict(showgrid=False, showbackground=False,
                       tickfont=dict(color=axis_color)),
        )

    # embed with fr locale + custom modebar
    div_id = f"plotly-fr-{uuid.uuid4().hex}"
    cfg = {"locale": "fr", "displaylogo": False}
    html(f"""
<style>
/* transparent modebar only for this figure */
#{div_id} .modebar,
#{div_id} .modebar-group {{
  background: transparent !important;
  box-shadow: none !important;
}}
#{div_id} .modebar-btn {{
  background: transparent !important;
  border: 0 !important;
}}
</style>

<div id="{div_id}" style="width:100%;height:100%;background:transparent;overflow:visible;"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script src="https://cdn.plot.ly/plotly-locale-fr-latest.js"></script>
<script>
(function() {{
  const el   = document.getElementById("{div_id}");
  const spec = {f.to_json()};
  const cfg  = {json.dumps(cfg)}; // keep plotly config (locale, filename, etc.)

  // constants
  const DEFAULT_CAMERA_SCALE = 1.7; // 3d default camera scale
  const SCALE_2D = 1.2;             // 2d padded autoscale
  const MIN_EYE = 0.05, MAX_EYE = 1e6;

  try {{
    if (window.PlotlyLocales && PlotlyLocales.fr && Plotly.register) {{
      Plotly.register(PlotlyLocales.fr);
    }}
  }} catch (e) {{}}

  // remove unwanted buttons including official png, saved 3d cam, 2d autoscale/reset
  cfg.modeBarButtonsToRemove = (cfg.modeBarButtonsToRemove || []).concat([
    'toImage',
    'toggleSpikelines',
    'hoverClosestCartesian',
    'hoverCompareCartesian',
    'hoverClosest3d',
    'toggleHover',
    'resetCameraLastSave3d',
    'autoScale2d',
    'resetScale2d'
  ]);

  // helpers
  const is3dType = t => ['mesh3d','scatter3d','surface','cone','streamtube','isosurface','volume']
                        .includes((t||'').toLowerCase());

  function GetExtents(arr) {{
    let mn = Infinity, mx = -Infinity;
    for (const v of arr || []) {{
      const n = +v;
      if (!isNaN(n)) {{ if (n < mn) mn = n; if (n > mx) mx = n; }}
    }}
    if (!isFinite(mn)) return null;
    if (mn === mx) {{ mn -= 0.5; mx += 0.5; }}
    return [mn, mx];
  }}

  // 2d padded autoscale
  function AutoScale2D(gd, s) {{
    let xs = [], ys = [];
    (gd.data || []).forEach(tr => {{
      if (tr && !is3dType(tr.type) && !tr.scene && tr.x && tr.y) {{
        xs = xs.concat(tr.x);
        ys = ys.concat(tr.y);
      }}
    }});
    const ex = GetExtents(xs), ey = GetExtents(ys);
    if (!ex || !ey) return Promise.resolve();
    const cx = (ex[0] + ex[1]) / 2, cy = (ey[0] + ey[1]) / 2;
    let hx = (ex[1] - ex[0]) / 2, hy = (ey[1] - ey[0]) / 2;
    hx = Math.max(hx * s, 1e-6); hy = Math.max(hy * s, 1e-6);
    return Plotly.relayout(gd, {{
      'xaxis.autorange': false, 'yaxis.autorange': false,
      'xaxis.range': [cx - hx, cx + hx],
      'yaxis.range': [cy - hy, cy + hy]
    }});
  }}

  // capture baseline cameras once
  function SnapshotDefaultCameras(gd) {{
    const scenes = Object.keys(gd._fullLayout || {{}}).filter(k => /^scene\\d*$/.test(k));
    gd._wec_default_cams = gd._wec_default_cams || {{}};
    scenes.forEach(sc => {{
      if (!gd._wec_default_cams[sc]) {{
        const cam = (gd._fullLayout[sc] && gd._fullLayout[sc].camera) || {{}};
        gd._wec_default_cams[sc] = {{
          eye: {{
            x: (cam.eye && Number.isFinite(cam.eye.x)) ? cam.eye.x : 1.25,
            y: (cam.eye && Number.isFinite(cam.eye.y)) ? cam.eye.y : 1.25,
            z: (cam.eye && Number.isFinite(cam.eye.z)) ? cam.eye.z : 1.25
          }},
          up: {{
            x: (cam.up && Number.isFinite(cam.up.x)) ? cam.up.x : 0,
            y: (cam.up && Number.isFinite(cam.up.y)) ? cam.up.y : 0,
            z: (cam.up && Number.isFinite(cam.up.z)) ? cam.up.z : 1
          }},
          center: {{
            x: (cam.center && Number.isFinite(cam.center.x)) ? cam.center.x : 0,
            y: (cam.center && Number.isFinite(cam.center.y)) ? cam.center.y : 0,
            z: (cam.center && Number.isFinite(cam.center.z)) ? cam.center.z : 0
          }}
        }};
      }}
    }});
    return scenes;
  }}

  // apply scaled default camera
  function ApplyScaledDefaultCamera(gd, scale) {{
    const bases = gd._wec_default_cams || {{}};
    const ups = {{}};
    const scenes = Object.keys(gd._fullLayout || {{}}).filter(k => /^scene\\d*$/.test(k));
    scenes.forEach(sc => {{
      const base = bases[sc]; if (!base) return;
      const eye0 = base.eye || {{x:1.25, y:1.25, z:1.25}};
      const up0  = base.up  || {{x:0, y:0, z:1}};
      const ctr0 = base.center || {{x:0, y:0, z:0}};
      const len = Math.hypot(eye0.x||0, eye0.y||0, eye0.z||0) || 1e-9;
      const newLen = Math.min(MAX_EYE, Math.max(MIN_EYE, len * scale));
      const r = newLen / len;
      const eye = {{ x: (eye0.x||0)*r, y: (eye0.y||0)*r, z: (eye0.z||0)*r }};
      ups[`${{sc}}.camera.eye.x`] = eye.x;
      ups[`${{sc}}.camera.eye.y`] = eye.y;
      ups[`${{sc}}.camera.eye.z`] = eye.z;
      ups[`${{sc}}.camera.up.x`]  = up0.x;
      ups[`${{sc}}.camera.up.y`]  = up0.y;
      ups[`${{sc}}.camera.up.z`]  = up0.z;
      ups[`${{sc}}.camera.center.x`] = ctr0.x;
      ups[`${{sc}}.camera.center.y`] = ctr0.y;
      ups[`${{sc}}.camera.center.z`] = ctr0.z;
    }});
    if (Object.keys(ups).length) return Plotly.relayout(gd, ups);
    return Promise.resolve();
  }}

  // custom png export (forces black axes in export)
  const exportAxisColor = '#000000';
  const btnPngAxes = {{
    name: 'PNG (axes)',
    title: 'Télécharger PNG',
    icon: Plotly.Icons.camera,
    click: function(gd) {{
      const before = {{}}, on = {{}};

      // 2d axes
      Object.keys(gd.layout)
        .filter(k => /^xaxis\\d*$/.test(k) || /^yaxis\\d*$/.test(k))
        .forEach(axName => {{
          const ax = gd.layout[axName]; if (!ax) return;
          before[`${{axName}}.showline`]         = !!ax.showline;
          before[`${{axName}}.ticks`]            = ax.ticks || '';
          before[`${{axName}}.ticklen`]          = ax.ticklen || 0;
          before[`${{axName}}.linecolor`]        = ax.linecolor || null;
          before[`${{axName}}.tickcolor`]        = ax.tickcolor || null;
          before[`${{axName}}.tickfont.color`]   = (ax.tickfont && ax.tickfont.color) || null;
          before[`${{axName}}.title.font.color`] = (ax.title && ax.title.font && ax.title.font.color) || null;

          on[`${{axName}}.showline`]             = true;
          on[`${{axName}}.ticks`]                = 'outside';
          on[`${{axName}}.ticklen`]              = 6;
          on[`${{axName}}.linecolor`]            = exportAxisColor;
          on[`${{axName}}.tickcolor`]            = exportAxisColor;
          on[`${{axName}}.tickfont.color`]       = exportAxisColor;
          on[`${{axName}}.title.font.color`]     = exportAxisColor;
        }});

      // 3d axis text
      Object.keys(gd.layout)
        .filter(k => /^scene\\d*$/.test(k))
        .forEach(sceneName => {{
          const sc = gd.layout[sceneName]; if (!sc) return;
          ['xaxis','yaxis','zaxis'].forEach(sub => {{
            const ax = sc[sub] || {{}};
            const prefix = `${{sceneName}}.${{sub}}`;
            before[`${{prefix}}.tickfont.color`]   = (ax.tickfont && ax.tickfont.color) || null;
            before[`${{prefix}}.title.font.color`] = (ax.title && ax.title.font && ax.title.font.color) || null;
            on[`${{prefix}}.tickfont.color`]       = exportAxisColor;
            on[`${{prefix}}.title.font.color`]     = exportAxisColor;
          }});
        }});

      Plotly.relayout(gd, on)
        .then(() => Plotly.downloadImage(gd, {{ format: 'png', filename: 'wecov3r', scale: 2 }}))
        .finally(() => Plotly.relayout(gd, before));
    }}
  }};

  // cross-mode reset: 3d → scaled default camera, 2d → padded autoscale
  cfg.modeBarButtonsToRemove = cfg.modeBarButtonsToRemove.concat(['resetCameraDefault3d']);
  const btnResetCrossMode = {{
    name: 'resetCameraDefault3d',
    title: 'Recentrer',
    icon: Plotly.Icons.home,
    click: function(gd) {{
      const tasks = [];
      const has3d = Object.keys(gd._fullLayout || {{}}).some(k => /^scene\\d*$/.test(k));
      if (has3d) tasks.push(ApplyScaledDefaultCamera(gd, DEFAULT_CAMERA_SCALE));
      const has2d = (gd.data || []).some(tr => tr && !is3dType(tr.type) && !tr.scene && tr.x && tr.y);
      if (has2d) tasks.push(AutoScale2D(gd, SCALE_2D));
      return Promise.all(tasks);
    }}
  }};

  // add reset first, png last
  cfg.modeBarButtonsToAdd = (cfg.modeBarButtonsToAdd || []).concat([btnResetCrossMode, btnPngAxes]);

  // render
  Plotly.newPlot(el, spec.data, spec.layout, cfg).then(gd => {{
    const scenes = SnapshotDefaultCameras(gd);
    if (scenes.length) ApplyScaledDefaultCamera(gd, DEFAULT_CAMERA_SCALE);
    const has2d = (gd.data || []).some(tr => tr && !is3dType(tr.type) && !tr.scene && tr.x && tr.y);
    if (has2d) AutoScale2D(gd, SCALE_2D);
  }});

  // responsive
  window.addEventListener('resize', () => Plotly.Plots.resize(el));
}})();
</script>
""", height=height)

# page / style
st.set_page_config(page_title="Vêtement AI", page_icon="🤖", layout="wide")

# logo (base64 inline)
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
  position: relative;            /* absolute logo container */
  font-size: 28px;
  line-height: 1.1;
  padding-left: calc(2em + 4px); /* room for logo */
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
.block:empty {{ display: none !important; padding: 0 !important; margin: 0 !important; border: 0 !important; box-shadow: none !important; background: transparent !important; }}
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

# brand colors & lighting
BRAND_VIOLET = "#401189"
BRAND_VIOLET_RGBA = "rgba(64,17,137,0.25)"
SHADING_VIOLET = "#6B4AE6"

MESH_LIGHTING = dict(ambient=0.55, diffuse=1.0, specular=0.45, roughness=0.80)
MESH_LIGHTPOS = dict(x=1.4, y=0.2, z=2.0)

# edge styles
EDGE3D_BORDER_WIDTH = 8
EDGE3D_BORDER_COLOR = BRAND_VIOLET
EDGE2D_BORDER_WIDTH = 4
EDGE2D_BORDER_COLOR = BRAND_VIOLET

# defaults / session state
DEFAULTS = {"cheville": 25.0, "mollet": 38.0, "genou": 39.0, "cuisse": 49.0, "taille": 182.0}

for key, value in DEFAULTS.items():
  if key not in st.session_state:
    st.session_state[key] = value

if "_do_raz" not in st.session_state:
  st.session_state["_do_raz"] = False

# reset if requested
if st.session_state["_do_raz"]:
  for key, value in DEFAULTS.items():
    st.session_state[key] = value
  st.session_state["_do_raz"] = False

# format helper
def Format1(x: Optional[float], decimals: int = 1) -> str:
  # number → string with thin spacing
  if x is None:
    return "—"
  return f"{x:,.{decimals}f}".replace(",", " ")

# http helpers
def CallAPI(url: str, payload: dict, bearer: Optional[str] = None, timeout_s: float = 30.0) -> dict:
  # simple post wrapper
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

def RunPipeline(curves: list, api_key: str, api_secret: str, do_unroll: bool = True) -> dict:
  # connect
  r1 = CallAPI("https://wecov3r.com/api/connect", {"k": api_key, "s": api_secret})
  ctx_code = r1.get("data")

  # token
  r2 = CallAPI("https://wecov3r.com/api/token", {"c": ctx_code, "u": "mesh"})
  data2 = r2.get("data")
  token = data2.get("token") if isinstance(data2, dict) else data2

  # mesh (wire)
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
  r3 = CallAPI("https://wecov3r.com/api/mesh", payload, bearer=token)

  if not do_unroll:
    return r3

  # unroll
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
  r4 = CallAPI("https://wecov3r.com/api/mesh", payload2, bearer=token)
  return r4

# geometry (last point at 359°)
def ComputeCurves(ankle_cm, calf_cm, knee_cm, thigh_cm, height_cm, nb_point=32):
    # convert cm to mm
    scale = 10.0
    v_ankle  = scale * float(ankle_cm)
    v_calf   = scale * float(calf_cm)
    v_knee   = scale * float(knee_cm)
    v_thigh  = scale * float(thigh_cm)
    v_height = scale * float(height_cm)

    # heights in mm
    h_sole   = 0.000 * v_height
    h_ankle  = 0.039 * v_height
    h_calf   = 0.200 * v_height
    h_knee   = 0.285 * v_height
    h_thigh  = 0.400 * v_height
    h_crotch = 0.500 * v_height

    # leg height (ankle → crotch)
    leg_height = h_crotch - h_ankle

    # dynamic offsets (% of leg height)
    garment_bot_offset = 0.07 * leg_height
    garment_top_offset = 0.14 * leg_height

    # garment heights
    h_garment_bot = h_ankle + garment_bot_offset
    h_garment_top = h_crotch - garment_top_offset

    # lengths (mm)
    tibia_length      = h_knee - h_ankle
    femur_length      = h_crotch - h_knee
    femur_length_real = 0.245 * v_height
    garment_length    = 0.98 * (h_garment_top - h_garment_bot)

    # circle points (359°)
    def CirclePoints359(r, z, n=nb_point):
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

    # radii in mm
    r_ankle = v_ankle / (2.0 * math.pi)
    r_calf  = v_calf  / (2.0 * math.pi)
    r_knee  = v_knee  / (2.0 * math.pi)
    r_thigh = v_thigh / (2.0 * math.pi)

    curves_pts = [
        CirclePoints359(r_ankle, h_garment_bot),
        CirclePoints359(r_calf,  h_calf),
        CirclePoints359(r_knee,  h_knee),
        CirclePoints359(r_thigh, h_thigh),
        CirclePoints359(r_thigh, h_garment_top),
    ]

    curves = []
    for i, pts in enumerate(curves_pts):
        curves.append({
            "definition": {"points": pts, "open": False},  # closed on service side
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
        "garment_bot_offset_cm": garment_bot_offset / scale,
        "garment_top_offset_cm": garment_top_offset / scale,
    }

# mm → cm (2d/3d)
def NodesToCentimeter(nodes_mm: List[float], is3d: bool = True):
    n = len(nodes_mm) // 3
    x = [nodes_mm[3*i + 0] / 10.0 for i in range(n)]
    y = [nodes_mm[3*i + 1] / 10.0 for i in range(n)]
    if is3d:
        z = [nodes_mm[3*i + 2] / 10.0 for i in range(n)]
        return x, y, z
    return x, y

# 2d helpers
def ComputeBoundaryEdges(defn: dict) -> List[List[int]]:
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

  def PopNextEdge(u: int) -> Optional[int]:
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
    v = PopNextEdge(u)
    if v is None:
      continue
    visited_edge.add((min(u, v), max(u, v)))
    loop.append(v)
    while True:
      u, v = v, PopNextEdge(v)
      if v is None:
        break
      visited_edge.add((min(u, v), max(u, v)))
      if v == start:
        loops.append(loop[:])
        break
      loop.append(v)
  return loops

# 2d render (filled)
def DrawMeshShade2D(mesh_obj: dict):
  defn = mesh_obj.get("definition", {})
  nodes = defn.get("nodes") or []
  topo  = int(defn.get("topology") or 0)
  elems = defn.get("elements") or []
  elidx = defn.get("elementsLastIndex")
  fig = go.Figure()

  if elidx:
    first = 0
    x_all, y_all = NodesToCentimeter(nodes, False)
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
    loops = ComputeBoundaryEdges(defn)
    x_all, y_all = NodesToCentimeter(nodes, False)
    if not loops:
      return DrawMeshWire2D(mesh_obj)
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
    return DrawMeshWire2D(mesh_obj)

  fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis=dict(title="cm", scaleanchor="y", scaleratio=1),
    yaxis=dict(title="cm"),
    showlegend=False,
    hovermode=False
  )
  PlotFrench(fig)

# 2d render (wire)
def DrawMeshWire2D(mesh_obj: dict):
  defn = mesh_obj.get("definition", {})
  nodes = defn.get("nodes") or []
  topo  = int(defn.get("topology") or 0)
  elements = defn.get("elements") or []
  elidx = defn.get("elementsLastIndex")

  if len(nodes) < 6:
    st.warning("Pas assez de points")
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
      idxs = [elements[topo*e + k] for k in range(topo)]
      for k in range(topo):
        a, b = idxs[k], idxs[(k+1) % topo]
        if a > b: a, b = b, a
        edges.add((a, b))
    for a, b in edges:
      AddEdge(a, b)
  else:
    if not elidx:
      for i2 in range(len(elements)):
        a = elements[i2]
        b = elements[(i2+1) % len(elements)]
        AddEdge(a, b)
    else:
      first = 0
      for last in elidx:
        seq = elements[first:last+1]
        for i2 in range(len(seq)):
          a = seq[i2]
          b = seq[(i2+1) % len(seq)]
          AddEdge(a, b)
        first = last + 1

  fig = go.Figure(data=[go.Scatter(x=xl, y=yl, mode="lines",
                   line=dict(width=EDGE2D_BORDER_WIDTH, color=EDGE2D_BORDER_COLOR))])
  fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis=dict(title="cm", scaleanchor="y", scaleratio=1),
    yaxis=dict(title="cm"),
    showlegend=False,
    hovermode=False
  )
  PlotFrench(fig)

# 3d helpers
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
      idxs = [elems[topo*e + k] for k in range(topo)]
      for k in range(topo):
        a, b = idxs[k], idxs[(k+1) % topo]
        if a > b: a, b = b, a
        edges.add((a, b))
    for a, b in edges:
      AddEdge(a, b)
  else:
    if not elidx:
      for i2 in range(len(elems)):
        a = elems[i2]
        b = elems[(i2+1) % len(elems)]
        AddEdge(a, b)
    else:
      first = 0
      for last in elidx:
        seq = elems[first:last+1]
        for i2 in range(len(seq)):
          a = seq[i2]
          b = seq[(i2+1) % len(seq)]
          AddEdge(a, b)
        first = last + 1

  if xl:
    fig.add_trace(go.Scatter3d(
      x=xl, y=yl, z=zl, mode="lines",
      name="edges",
      line=dict(width=EDGE3D_BORDER_WIDTH, color=EDGE3D_BORDER_COLOR)
    ))

def AddBoundaryEdges3D(fig, x, y, z, elems, topo):
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

# 3d render (shaded)
def DrawMeshShade3D(mesh_obj: dict, edge_mode: str = "Bord"):
  defn = mesh_obj.get("definition", {})
  nodes = defn.get("nodes") or []
  topo  = int(defn.get("topology") or 0)
  elems = defn.get("elements") or []
  elidx = defn.get("elementsLastIndex")

  if len(nodes) < 9:
    st.warning("Pas assez de points")
    return

  x, y, z = NodesToCentimeter(nodes, True)
  fig = go.Figure()

  def AddEdgesChoice():
    if edge_mode == "Bord":
      AddBoundaryEdges3D(fig, x, y, z, elems, topo)
    elif edge_mode == "Toutes":
      AddWire3D(fig, x, y, z, elems, topo, elidx)

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
        showscale=False,
        hoverinfo="skip", hovertemplate=None
      ))
      AddEdgesChoice()
    else:
      fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        color=SHADING_VIOLET, opacity=1.0,
        flatshading=False, alphahull=12,
        lighting=MESH_LIGHTING, lightposition=MESH_LIGHTPOS,
        showscale=False,
        hoverinfo="skip", hovertemplate=None
      ))
      AddEdgesChoice()
  else:
    fig.add_trace(go.Mesh3d(
      x=x, y=y, z=z,
      color=SHADING_VIOLET, opacity=1.0,
      flatshading=False, alphahull=12,
      lighting=MESH_LIGHTING, lightposition=MESH_LIGHTPOS,
      showscale=False,
      hoverinfo="skip", hovertemplate=None
    ))
    AddEdgesChoice()

  fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    scene=dict(xaxis_title="cm", yaxis_title="cm", zaxis_title="cm", aspectmode="data"),
    showlegend=False,
    hovermode=False
  )
  PlotFrench(fig)

# 3d render (wire)
def DrawMeshWire3D(mesh_obj: dict):
  defn = mesh_obj.get("definition", {})
  nodes = defn.get("nodes") or []
  topo  = int(defn.get("topology") or 0)
  elems = defn.get("elements") or []
  elidx = defn.get("elementsLastIndex")

  if len(nodes) < 9:
    st.warning("Pas assez de points")
    return

  x, y, z = NodesToCentimeter(nodes, True)
  fig = go.Figure()
  AddWire3D(fig, x, y, z, elems, topo, elidx)
  fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    scene=dict(xaxis_title="cm", yaxis_title="cm", zaxis_title="cm", aspectmode="data"),
    showlegend=False,
    hovermode=False
  )
  PlotFrench(fig)

# ui layout
col_left, col_right = st.columns([1.0, 1.2], gap="large")

# api keys (fallback local)
api_key    = st.secrets.get("WECOV3R_API_KEY", os.getenv("WECOV3R_API_KEY", ""))
api_secret = st.secrets.get("WECOV3R_API_SECRET", os.getenv("WECOV3R_API_SECRET", ""))

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
  st.subheader("Jambe (cm)")
  # ankle/calf
  r1a, r1b = st.columns([1, 1], gap="small", vertical_alignment="bottom")
  cheville = r1a.number_input("Cheville", 1.0, 1000.0, step=0.1, format="%.1f", key="cheville")
  mollet   = r1b.number_input("Mollet",   1.0, 1000.0, step=0.1, format="%.1f", key="mollet")
  # knee/thigh
  r2a, r2b = st.columns([1, 1], gap="small", vertical_alignment="bottom")
  genou  = r2a.number_input("Genou",  1.0, 1000.0, step=0.1, format="%.1f", key="genou")
  cuisse = r2b.number_input("Cuisse", 1.0, 1000.0, step=0.1, format="%.1f", key="cuisse")
  # height + buttons
  r3a, r3b = st.columns([1, 1], gap="small", vertical_alignment="bottom")
  taille = r3a.number_input("Taille", 1.0, 300.0, step=0.1, format="%.1f", key="taille")
  g_run, g_raz, g_flat = r3b.columns([1, 1, 1], gap="small", vertical_alignment="bottom")
  run_btn = g_run.button("Run")
  raz_btn = g_raz.button("RÀZ")
  do_unroll = g_flat.checkbox("2D", False, key="2d")

  if raz_btn:
    st.session_state["_do_raz"] = True
    st.rerun()
  st.markdown('</div>', unsafe_allow_html=True)

# compute + fetch
if run_btn:
  if not api_key or not api_secret:
    with col_left:
      st.error("Clés WeCov3r manquantes !")
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

      curves  = result["curves"]
      resp    = RunPipeline(curves, api_key, api_secret, do_unroll=do_unroll)
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

# display results
with col_left:
  st.markdown('<div class="block">', unsafe_allow_html=True)
  st.subheader("Calculs")
  r0, r1, r2 = st.columns([1, 1, 1])
  if data is None:
    r0.write("Pièce (cm): —")
    r1.write("Aire (cm²): —")
    r2.write(("Volume (L): —") if (not do_unroll) else ("Bord (cm): —"))
  else:
    r0.write(f"**Pièce (cm)** : {Format1(garment_length_cm)}")
    sqrt_area = math.sqrt(area_cm2) if area_cm2 is not None else 0
    r1.write(f"**Aire (cm²)** : {Format1(sqrt_area, 0)} × {Format1(sqrt_area, 0)}")
    if do_unroll:
      r2.write(f"**Bord (cm)** : {Format1(perim_cm)}")
    else:
      r2.write(f"**Volume (L)** : {Format1(vol_l)}")
  st.markdown('</div>', unsafe_allow_html=True)

# render mesh
with col_right:
  st.markdown('<div class="block">', unsafe_allow_html=True)
  st.subheader("Affichage")
  c_mode, c_edge = st.columns([1, 1], gap="small")
  if do_unroll:
    opts = ["Rempli", "Filaire"]; default = "Rempli"
  else:
    opts = ["Ombré", "Filaire"]; default = "Ombré"
  with c_mode:
    mode = st.selectbox("Rendu", opts, index=opts.index(default))
  with c_edge:
    if (mode == "Ombré") and (not do_unroll):
      edge_mode = st.selectbox("Arêtes", ["Aucun", "Bord", "Toutes"], index=1)
    else:
      edge_mode = "None"
      st.markdown("<div style='height:2.5em'></div>", unsafe_allow_html=True)

  if data is None:
    st.info("Run pour lancer le calcul")
  else:
    if mode == "Rempli":
      DrawMeshShade2D(data)
    elif mode == "Filaire" and do_unroll:
      DrawMeshWire2D(data)
    elif mode == "Ombré" and not do_unroll:
      DrawMeshShade3D(data, edge_mode=edge_mode)
    elif mode == "Filaire" and not do_unroll:
      DrawMeshWire3D(data)
  st.markdown('</div>', unsafe_allow_html=True)
