# app.py
# WeCov3r Studio AI – Streamlit (affichage à droite des champs, JSON masqué, 3D si Unroll=off)
# pip install streamlit requests numpy plotly

import math
import json
import requests
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---------------- Page setup (titre simple, couleurs douces) ----------------
st.set_page_config(page_title="WeCov3r Studio AI", page_icon="🧩", layout="wide")
st.markdown("""
<style>
/* bandeau titre simple et coloré */
.wcvr-hero {background: linear-gradient(135deg,#5B8CFF 0%,#B66DFF 50%,#FF8BD3 100%);
border-radius:16px; padding:16px 20px; color:#fff; margin-bottom:10px; box-shadow:0 8px 28px rgba(0,0,0,.18);}
.wcvr-hero h1 {margin:0; font-size:24px; letter-spacing:.2px;}
/* inputs plus compacts */
.small-input .stNumberInput>div>div>input {max-width:8rem;}
.small-input .stCheckbox>div {transform:scale(.95);}
.small-input .stButton>button {padding:0.35rem 0.8rem; border-radius:10px;}
/* cartes sorties compactes */
.block {border:1px solid rgba(0,0,0,.06); border-radius:12px; padding:10px 12px; background:rgba(255,255,255,.8);}
</style>
<div class="wcvr-hero"><h1>Moteur WeCov3r Studio AI</h1></div>
""", unsafe_allow_html=True)

# ---------------- Helpers HTTP ----------------
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
	# 1) /api/connect
	r1 = call_api("https://wecov3r.com/api/connect", {"k": api_key, "s": api_secret})
	ctx_code = r1.get("data")

	# 2) /api/token
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

# ---------------- Géométrie (mêmes formules que ta version JS) ----------------
def compute_curves(cheville_cm, mollet_cm, genou_cm, cuisse_cm, taille_cm, nb_points=16):
	scale_factor = 10.0  # cm -> mm
	v_cheville = scale_factor * float(cheville_cm)
	v_mollet   = scale_factor * float(mollet_cm)
	v_genou    = scale_factor * float(genou_cm)
	v_cuisse   = scale_factor * float(cuisse_cm)
	v_taille   = scale_factor * float(taille_cm)

	# hauteurs
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

	curves_pts = []
	r_cheville = v_cheville / (2.0 * math.pi)
	r_mollet   = v_mollet   / (2.0 * math.pi)
	r_genou    = v_genou    / (2.0 * math.pi)
	r_cuisse   = v_cuisse   / (2.0 * math.pi)

	curves_pts.append(circle_points(r_cheville, h_cheville))
	curves_pts.append(circle_points(r_mollet,   h_mollet))
	curves_pts.append(circle_points(r_genou,    h_genou))
	curves_pts.append(circle_points(r_cuisse,   h_cuisse))
	curves_pts.append(circle_points(r_cuisse,   h_fourche))

	curves = []
	for i, pts in enumerate(curves_pts):
		curves.append({
			"definition": {
				"points": pts,
				"closed": False
			},
			"properties": {
				"uuid": i,
				"name": f"curve_{i}",
				"type": "LINES",
				"scale": 1000
			},
			"dimension": 3
		})
	return curves

# ---------------- Rendu 2D / 3D ----------------
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

# ---------------- UI: champs à gauche, affichage à droite ----------------
col_left, col_right = st.columns([0.9, 1.1], gap="large")

with col_left:
	st.markdown('<div class="small-input">', unsafe_allow_html=True)
	st.subheader("Mesures (cm)")
	# grille compacte de champs
	r1c1, r1c2 = st.columns([1, 1])
	r2c1, r2c2 = st.columns([1, 1])

	cheville = r1c1.number_input("Cheville", min_value=1.0, max_value=1000.0, value=25.0, step=0.5)
	mollet   = r1c2.number_input("Mollet",   min_value=1.0, max_value=1000.0, value=38.0, step=0.5)
	genou    = r2c1.number_input("Genou",    min_value=1.0, max_value=1000.0, value=39.0, step=0.5)
	cuisse   = r2c2.number_input("Cuisse",   min_value=1.0, max_value=1000.0, value=49.0, step=0.5)
	taille   = st.number_input("Taille",     min_value=1.0, max_value=300.0,  value=182.0, step=0.5)

	do_unroll = st.checkbox("Unroll (aplatir)", value=True)
	run_btn = st.button("Calculer")
	st.markdown('</div>', unsafe_allow_html=True)

with col_right:
	st.subheader("Affichage")

# On masque la section connexion API (utilisation des secrets)
api_key    = st.secrets.get("WECOV3R_API_KEY", "")
api_secret = st.secrets.get("WECOV3R_API_SECRET", "")

# Calcul + affichage à droite
with col_right:
	if run_btn:
		if not api_key or not api_secret:
			st.error("Clés API absentes. Ajoutez-les dans .streamlit/secrets.toml (local) ou dans les Secrets de Streamlit Cloud.")
		else:
			try:
				curves = compute_curves(cheville, mollet, genou, cuisse, taille)
				resp = run_pipeline(curves, api_key, api_secret, do_unroll=do_unroll)
				data0 = (resp or {}).get("data", [{}])[0]

				# métriques (conversion: m² -> cm² (x 10 000), m -> cm (x 100))
				infos = data0.get("infos", {})
				area_m2 = infos.get("area")
				perimeter_m = infos.get("perimeter")

				area_cm2 = area_m2 * 10000.0 if isinstance(area_m2, (int, float)) else None
				perim_cm = perimeter_m * 100.0 if isinstance(perimeter_m, (int, float)) else None

				m1, m2 = st.columns([1,1])
				with m1:
					st.metric("Aire (cm²)", f"{area_cm2:,.2f}".replace(",", " ") if area_cm2 is not None else "—")
				with m2:
					st.metric("Périmètre (cm)", f"{perim_cm:,.2f}".replace(",", " ") if perim_cm is not None else "—")

				st.markdown('<div class="block">', unsafe_allow_html=True)
				if do_unroll:
					st.caption("rendu 2D (aplati)")
					draw_mesh_2d(data0)
				else:
					st.caption("rendu 3D")
					draw_mesh_3d(data0)
				st.markdown('</div>', unsafe_allow_html=True)

			except requests.HTTPError as e:
				st.error(f"Erreur HTTP: {e.response.status_code} — {e.response.text[:300]}")
			except Exception as ex:
				st.error(f"Erreur: {ex}")
