# main.py
import io
import contextlib
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import openseespy.opensees as ops


# ============================================================
#  COSTANTI GLOBALI
# ============================================================

L = 4.0
H = 6.0

TARGET_DISP_MM = 15.0

CORDOLI_Y = [
    (2.7, 3.0),
    (5.7, 6.0),
]

MARGIN = 0.30
PIER_MIN = 0.30  # maschio orizzontale minimo

# Mesh conforming: passo massimo (più grande => mesh più grossa/veloce)
MAX_DX_DEFAULT = 0.10
MAX_DY_DEFAULT = 0.10


# ============================================================
#  HELPERS: parsing robusto da body/query
# ============================================================

def _as_int01(v: Any, default: int = 0) -> int:
    """Converte v in 0/1 robustamente (accetta bool, str, int)."""
    if v is None:
        return int(default)
    if isinstance(v, bool):
        return 1 if v else 0
    try:
        return 1 if int(v) == 1 else 0
    except Exception:
        return int(default)

def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


# ============================================================
#  GEOMETRIA / VALIDAZIONE
# ============================================================

def inside_opening(x: float, y: float,
                   openings: List[Tuple[float, float, float, float]]) -> bool:
    """True se il punto (x,y) è dentro una delle aperture (solo interno, NON il bordo)."""
    for (x1, x2, y1, y2) in openings:
        if (x > x1) and (x < x2) and (y > y1) and (y < y2):
            return True
    return False


def openings_valid(openings, cordoli_y, margin) -> bool:
    """
    Controlla:
      - aperture dentro parete con margin
      - distanza da cordoli >= margin
      - niente sovrapposizioni / troppo vicine
      - maschi orizzontali >= PIER_MIN quando overlap in y
      - maschi ai bordi >= PIER_MIN
    """
    # 1) limiti parete + maschi ai bordi
    for (x1, x2, y1, y2) in openings:
        if x1 < PIER_MIN or x2 > (L - PIER_MIN):
            return False

        if not (0.0 + margin <= x1 < x2 <= L - margin):
            return False
        if not (0.0 + margin <= y1 < y2 <= H - margin):
            return False

        # 2) distanza da cordoli
        for (yc1, yc2) in cordoli_y:
            if not (y2 <= yc1 - margin or y1 >= yc2 + margin):
                return False

    # 3) distanze tra aperture / maschi
    n = len(openings)
    for i in range(n):
        x1i, x2i, y1i, y2i = openings[i]
        for j in range(i + 1, n):
            x1j, x2j, y1j, y2j = openings[j]

            dx_gap = max(0.0, max(x1i, x1j) - min(x2i, x2j))
            dy_gap = max(0.0, max(y1i, y1j) - min(y2i, y2j))

            overlap_y = not (y2i <= y1j or y2j <= y1i)

            # stessi piani => maschio orizzontale >= PIER_MIN
            if overlap_y and dx_gap < PIER_MIN:
                return False

            # “diagonali” => separazione minima generica
            if dx_gap < margin and dy_gap < margin:
                return False

    return True


# ============================================================
#  MESH CONFORME (COME NEL NOTEBOOK)
# ============================================================

def _unique_sorted(vals: List[float], tol: float = 1e-6) -> List[float]:
    vals = sorted(vals)
    out: List[float] = []
    for v in vals:
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out


def _refine_intervals(coords: List[float], max_step: float) -> List[float]:
    """Spezza ogni intervallo in sotto-intervalli <= max_step."""
    refined = [coords[0]]
    for a, b in zip(coords[:-1], coords[1:]):
        seg = b - a
        if seg <= max_step:
            refined.append(b)
        else:
            n = int(np.ceil(seg / max_step))
            for k in range(1, n + 1):
                refined.append(a + seg * k / n)
    return _unique_sorted(refined)


def build_conforming_grid(openings: List[Tuple[float, float, float, float]],
                          max_dx: float,
                          max_dy: float) -> Tuple[List[float], List[float]]:
    """
    Costruisce una griglia che *allinea* nodi a:
      - bordi parete (0, L) e (0, H)
      - margini (MARGIN, L-MARGIN, ecc.)
      - bordi cordoli (y1c, y2c)
      - bordi aperture (x1,x2,y1,y2)
    Poi raffina gli intervalli con max_dx/max_dy.
    """
    xs = [0.0, L, MARGIN, L - MARGIN]
    ys = [0.0, H, MARGIN, H - MARGIN]

    for (yc1, yc2) in CORDOLI_Y:
        ys += [yc1, yc2]

    for (x1, x2, y1, y2) in openings:
        xs += [x1, x2]
        ys += [y1, y2]

    xs = _unique_sorted(xs)
    ys = _unique_sorted(ys)

    xs = _refine_intervals(xs, max_dx)
    ys = _refine_intervals(ys, max_dy)

    return xs, ys


# ============================================================
#  MATERIALI J2 (come nel tuo setup)
# ============================================================

def K_from_E_nu(E: float, nu: float) -> float:
    return E / (3.0 * (1.0 - 2.0 * nu))

def G_from_E_nu(E: float, nu: float) -> float:
    return E / (2.0 * (1.0 + nu))


# ============================================================
#  MODELLO NL: build + pushover
# ============================================================

def build_wall_J2_conforming(openings: List[Tuple[float, float, float, float]],
                             max_dx: float,
                             max_dy: float) -> Tuple[Dict[Tuple[int, int], int], int, List[float], List[float]]:
    """
    Modello NL J2 con mesh conforme.
    Ritorna:
      node_tags, control_node, xs, ys
    """
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 2)

    xs, ys = build_conforming_grid(openings, max_dx=max_dx, max_dy=max_dy)

    # -------------------------
    # NODI (no dentro aperture)
    # -------------------------
    node_tags: Dict[Tuple[int, int], int] = {}
    tag = 1
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if inside_opening(x, y, openings):
                continue
            ops.node(tag, x, y)
            node_tags[(i, j)] = tag
            tag += 1

    # Vincoli base
    for i in range(len(xs)):
        key = (i, 0)
        if key in node_tags:
            ops.fix(node_tags[key], 1, 1)

    # -------------------------
    # MATERIALI
    # -------------------------
    # Muratura
    E_mur, nu_mur = 1.5e9, 0.15
    sig0_m, sigInf_m, delta_m, H_m = 0.5e6, 2.0e6, 8.0, 0.0

    # Cordoli (c.a.)
    E_cord, nu_cord = 30e9, 0.20
    sig0_c, sigInf_c, delta_c, H_c = 6.0e6, 25.0e6, 6.0, 0.0

    K_m, G_m = K_from_E_nu(E_mur, nu_mur), G_from_E_nu(E_mur, nu_mur)
    K_c, G_c = K_from_E_nu(E_cord, nu_cord), G_from_E_nu(E_cord, nu_cord)

    matTag_mur_3D, matTag_cord_3D = 10, 20
    ops.nDMaterial('J2Plasticity', matTag_mur_3D,  K_m, G_m, sig0_m, sigInf_m, delta_m, H_m)
    ops.nDMaterial('J2Plasticity', matTag_cord_3D, K_c, G_c, sig0_c, sigInf_c, delta_c, H_c)

    matTag_mur, matTag_cord = 1, 2
    ops.nDMaterial('PlaneStress', matTag_mur,  matTag_mur_3D)
    ops.nDMaterial('PlaneStress', matTag_cord, matTag_cord_3D)

    t = 0.25  # spessore

    # -------------------------
    # ELEMENTI QUAD
    # -------------------------
    eleTag = 1
    for j in range(len(ys) - 1):
        y_low, y_up = ys[j], ys[j + 1]
        yc = 0.5 * (y_low + y_up)

        this_mat = matTag_mur
        for (y1c, y2c) in CORDOLI_Y:
            if (yc >= y1c) and (yc <= y2c):
                this_mat = matTag_cord
                break

        for i in range(len(xs) - 1):
            keys = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if not all(k in node_tags for k in keys):
                continue

            n1 = node_tags[keys[0]]
            n2 = node_tags[keys[1]]
            n3 = node_tags[keys[2]]
            n4 = node_tags[keys[3]]

            ops.element('quad', eleTag, n1, n2, n3, n4, t, 'PlaneStress', this_mat, 0.0, 0.0, 0.0)
            eleTag += 1

    # -------------------------
    # CARICO IN SOMMITÀ
    # -------------------------
    j_top = len(ys) - 1
    top_nodes = [node_tags[(i, j_top)] for i in range(len(xs)) if (i, j_top) in node_tags]
    if not top_nodes:
        raise RuntimeError("Nessun nodo in sommità: layout aperture troppo aggressivo.")

    control_node = top_nodes[len(top_nodes) // 2]

    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    Ptot = 100e3
    Pnode = Ptot / len(top_nodes)
    for nd in top_nodes:
        ops.load(nd, Pnode, 0.0)

    return node_tags, control_node, xs, ys


def shear_at_target_disp(disp_mm: np.ndarray,
                         shear_kN: np.ndarray,
                         target_mm: float = TARGET_DISP_MM) -> Optional[float]:
    if len(disp_mm) < 2:
        return None
    if float(np.max(disp_mm)) < target_mm:
        return None
    return float(np.interp(target_mm, disp_mm, shear_kN))


# ============================================================
#  LETTURA TENSIONI (TAU / SIGMA)
# ============================================================

def _compute_stress_grid_profiles(
    n_bins_x: int = 4,
    n_bins_y: int = 12
) -> Dict[str, Any]:
    """
    Legge tensioni (stress) dagli elementi quad (stato finale).
    Costruisce:
      - profilo verticale medio di tau e sigma_c
      - zone 2D (x_range, y_range) con tau/sigma.
    """
    ele_tags = ops.getEleTags()
    if not ele_tags:
        return {
            "tau_profile_y":  {"y": [], "tau_mean": []},
            "sigma_profile_y":{"y": [], "sigma_c_mean": []},
            "zones": []
        }

    y_edges = np.linspace(0.0, H, n_bins_y + 1)
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_edges = np.linspace(0.0, L, n_bins_x + 1)

    tau_sum_y   = np.zeros(n_bins_y)
    tau_count_y = np.zeros(n_bins_y)
    sigc_sum_y  = np.zeros(n_bins_y)
    sigc_count_y= np.zeros(n_bins_y)

    tau_max_2d  = np.full((n_bins_x, n_bins_y), -1e20)
    tau_sum_2d  = np.zeros((n_bins_x, n_bins_y))
    tau_count_2d= np.zeros((n_bins_x, n_bins_y))
    sigc_max_2d = np.full((n_bins_x, n_bins_y), -1e20)

    for ele in ele_tags:
        stress = ops.eleResponse(ele, 'stress')
        if stress is None:
            continue

        tau_vals  = []
        sigy_vals = []

        # quad: stress viene spesso a blocchi di 3 (sigx, sigy, tau)
        for i in range(0, len(stress), 3):
            sigx = stress[i]
            sigy = stress[i+1]
            tau  = stress[i+2]
            tau_vals.append(abs(float(tau)))
            sigy_vals.append(float(sigy))

        if not tau_vals or not sigy_vals:
            continue

        tau_max_el  = max(tau_vals)
        tau_mean_el = sum(tau_vals) / len(tau_vals)

        sigy_min = min(sigy_vals)         # più compressiva (negativa)
        sigma_c_el = abs(sigy_min)

        nds = ops.eleNodes(ele)
        xs, ys = [], []
        for nd in nds:
            x_nd, y_nd = ops.nodeCoord(nd)
            xs.append(float(x_nd))
            ys.append(float(y_nd))
        if not xs or not ys:
            continue

        xc = sum(xs) / len(xs)
        yc = sum(ys) / len(ys)

        j = int(np.searchsorted(y_edges, yc) - 1)
        if j < 0 or j >= n_bins_y:
            continue

        tau_sum_y[j]   += tau_mean_el
        tau_count_y[j] += 1

        sigc_sum_y[j]   += sigma_c_el
        sigc_count_y[j] += 1

        i = int(np.searchsorted(x_edges, xc) - 1)
        if i < 0 or i >= n_bins_x:
            continue

        tau_max_2d[i, j] = max(tau_max_2d[i, j], tau_max_el)
        tau_sum_2d[i, j] += tau_mean_el
        tau_count_2d[i, j] += 1

        sigc_max_2d[i, j] = max(sigc_max_2d[i, j], sigma_c_el)

    # profili 1D
    y_out, tau_mean_y = [], []
    for j in range(n_bins_y):
        if tau_count_y[j] > 0:
            y_out.append(float(y_centers[j]))
            tau_mean_y.append(float(tau_sum_y[j] / tau_count_y[j]))

    y_sig_out, sigc_mean_y = [], []
    for j in range(n_bins_y):
        if sigc_count_y[j] > 0:
            y_sig_out.append(float(y_centers[j]))
            sigc_mean_y.append(float(sigc_sum_y[j] / sigc_count_y[j]))

    # griglia 2D zone
    zones = []
    for i in range(n_bins_x):
        for j in range(n_bins_y):
            if tau_count_2d[i, j] <= 0:
                continue

            x_min, x_max = float(x_edges[i]), float(x_edges[i+1])
            y_min, y_max = float(y_edges[j]), float(y_edges[j+1])

            tau_mean_zone = float(tau_sum_2d[i, j] / tau_count_2d[i, j])
            tau_max_zone  = float(tau_max_2d[i, j]) if tau_max_2d[i, j] > -1e10 else 0.0
            sigma_c_zone  = float(sigc_max_2d[i, j]) if sigc_max_2d[i, j] > -1e10 else 0.0

            zones.append({
                "id": f"z_{i}_{j}",
                "x_range": [x_min, x_max],
                "y_range": [y_min, y_max],
                "tau_max": tau_max_zone,
                "tau_mean": tau_mean_zone,
                "sigma_c_max": sigma_c_zone
            })

    return {
        "tau_profile_y": {"y": y_out, "tau_mean": tau_mean_y},
        "sigma_profile_y": {"y": y_sig_out, "sigma_c_mean": sigc_mean_y},
        "zones": zones
    }


# ============================================================
#  ANALISI PUSHOVER
# ============================================================

def run_pushover_nonlinear_conforming(openings: List[Tuple[float, float, float, float]],
                                      max_dx: float,
                                      max_dy: float,
                                      target_mm: float = TARGET_DISP_MM,
                                      max_steps: int = 140,
                                      dU: float = 0.0002,
                                      verbose: bool = False,
                                      compute_stress_profiles: bool = False) -> Dict[str, Any]:
    """
    Pushover NL su mesh conforme.
    """
    try:
        if not openings_valid(openings, CORDOLI_Y, MARGIN):
            base = {
                "status": "error",
                "message": "openings_invalid",
                "disp_mm": [],
                "shear_kN": [],
                "V_target": None,
                "mesh": {"max_dx": float(max_dx), "max_dy": float(max_dy)},
            }
            if compute_stress_profiles:
                base["stress_profiles"] = {
                    "tau_profile_y": {"y": [], "tau_mean": []},
                    "sigma_profile_y": {"y": [], "sigma_c_mean": []},
                    "zones": []
                }
            return base

        node_tags, control_node, xs, ys = build_wall_J2_conforming(openings, max_dx=max_dx, max_dy=max_dy)

        ops.constraints('Plain')
        ops.numberer('RCM')
        ops.system('BandGeneral')
        ops.test('NormUnbalance', 1.0e-4, 15)
        ops.algorithm('Newton')
        ops.integrator('DisplacementControl', control_node, 1, dU)
        ops.analysis('Static')

        disp_mm: List[float] = []
        shear_kN: List[float] = []

        buf = io.StringIO()

        for step in range(max_steps):
            buf.truncate(0)
            buf.seek(0)

            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                ok = ops.analyze(1)

            log_text = buf.getvalue()

            if ok < 0:
                if verbose:
                    print(f"[NL] analyze failed step={step}\n{log_text.strip()}")
                break

            u = ops.nodeDisp(control_node, 1)  # m
            ops.reactions()

            Vb = 0.0
            for (i, j), nd in node_tags.items():
                if j == 0:
                    Vb += ops.nodeReaction(nd, 1)

            disp_mm.append(float(u * 1000.0))
            shear_kN.append(float(-Vb / 1000.0))

            if disp_mm[-1] >= target_mm * 1.0:
                break

        disp_arr = np.array(disp_mm, dtype=float)
        shear_arr = np.array(shear_kN, dtype=float)

        V_target = shear_at_target_disp(disp_arr, shear_arr, target_mm)

        result: Dict[str, Any] = {
            "status": "ok" if V_target is not None else "error",
            "message": None if V_target is not None else "analysis_not_reached_target",
            "disp_mm": disp_arr.tolist(),
            "shear_kN": shear_arr.tolist(),
            "V_target": V_target,
            "mesh": {
                "max_dx": float(max_dx),
                "max_dy": float(max_dy),
                "n_x_lines": int(len(xs)),
                "n_y_lines": int(len(ys)),
                "approx_nodes": int(ops.getNumNodes()) if hasattr(ops, "getNumNodes") else None,  # safe
                "approx_eles": int(ops.getNumElems()) if hasattr(ops, "getNumElems") else None,   # safe
            }
        }

        if compute_stress_profiles:
            result["stress_profiles"] = _compute_stress_grid_profiles(n_bins_x=4, n_bins_y=12)

        return result

    except Exception as e:
        if verbose:
            print(f"[run_pushover_nonlinear_conforming] errore: {e}")
        base = {
            "status": "error",
            "message": str(e),
            "disp_mm": [],
            "shear_kN": [],
            "V_target": None,
            "mesh": {"max_dx": float(max_dx), "max_dy": float(max_dy)},
        }
        if compute_stress_profiles:
            base["stress_profiles"] = {
                "tau_profile_y": {"y": [], "tau_mean": []},
                "sigma_profile_y": {"y": [], "sigma_c_mean": []},
                "zones": []
            }
        return base


# ============================================================
#  RUN DUE CASI (existing + project)
# ============================================================

def run_two_cases_from_dict(data: Dict[str, Any],
                            compute_profiles: bool,
                            max_dx: float,
                            max_dy: float) -> Dict[str, Any]:
    existing_openings = [tuple(o) for o in data.get("existing_openings", [])]
    project_openings  = [tuple(o) for o in data.get("project_openings", [])]

    if not existing_openings:
        raise ValueError("Chiave 'existing_openings' mancante o vuota.")
    if not project_openings:
        raise ValueError("Chiave 'project_openings' mancante o vuota.")

    res_existing = run_pushover_nonlinear_conforming(
        existing_openings,
        max_dx=max_dx,
        max_dy=max_dy,
        compute_stress_profiles=compute_profiles
    )
    res_project = run_pushover_nonlinear_conforming(
        project_openings,
        max_dx=max_dx,
        max_dy=max_dy,
        compute_stress_profiles=compute_profiles
    )

    return {"existing": res_existing, "project": res_project}


# ============================================================
#  FASTAPI APP
# ============================================================

app = FastAPI(
    title="Wall Pushover Service (Conforming Mesh)",
    description="Pushover muratura (Existing + Project) con mesh conforme alle aperture e stress opzionali",
    version="2.0.0"
)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Wall Pushover Service attivo (conforming mesh)"}


@app.post("/pushover")
async def pushover_endpoint(
    request: Request,
    max_dx: float = MAX_DX_DEFAULT,   # query override (opzionale)
    max_dy: float = MAX_DY_DEFAULT    # query override (opzionale)
):
    """
    Body JSON:
      - { "existing_openings": [...], "project_openings": [...], "stress": 0/1, "max_dx": ..., "max_dy": ... }
        oppure
      - [{ ... }] (wrapper n8n/lovable)

    Regole:
      - stress lo leggo dal BODY (0/1)
      - max_dx/max_dy: puoi metterli nel body, ma se li passi in query vincono quelli in query
    """
    payload = await request.json()

    # wrapper n8n: lista con 1 elemento
    if isinstance(payload, list):
        if not payload:
            return JSONResponse(status_code=400, content={"error": "Payload list vuota."})
        payload = payload[0]

    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"error": "Payload non valido: atteso oggetto JSON."})

    # stress: SOLO da body (come richiesto)
    compute_profiles = (_as_int01(payload.get("stress", 0), default=0) == 1)

    # max_dx/max_dy: body override se vuoi, ma query vince
    body_max_dx = _as_float(payload.get("max_dx", MAX_DX_DEFAULT), default=MAX_DX_DEFAULT)
    body_max_dy = _as_float(payload.get("max_dy", MAX_DY_DEFAULT), default=MAX_DY_DEFAULT)

    # se l’utente non passa query, FastAPI usa default => in quel caso prendiamo dal body
    # (per distinguere: consideriamo "query passed" se diverso da default oppure se nel body non c'è)
    # più semplice: se nel body c'è esplicitamente max_dx/max_dy, usali al posto dei default query.
    # ma se l’utente mette max_dx in query, lo lasciamo com’è.
    if "max_dx" in payload and float(max_dx) == float(MAX_DX_DEFAULT):
        max_dx = body_max_dx
    if "max_dy" in payload and float(max_dy) == float(MAX_DY_DEFAULT):
        max_dy = body_max_dy

    # clamp “ragionevole” (evita mesh assurde)
    max_dx = float(max(0.05, min(max_dx, 0.50)))
    max_dy = float(max(0.05, min(max_dy, 0.50)))

    try:
        result = run_two_cases_from_dict(payload, compute_profiles, max_dx, max_dy)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
