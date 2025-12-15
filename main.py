# main.py
import io
import time
import contextlib
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import openseespy.opensees as ops


# ============================================================
#  GEOMETRIA
# ============================================================
L = 4.0
H = 6.0

CORDOLI_Y = [(2.7, 3.0), (5.7, 6.0)]
MARGIN = 0.30
PIER_MIN = 0.30


# ============================================================
#  DEFAULTS "PRODUZIONE" (Render-friendly)
# ============================================================
DEFAULTS = {
    "stress": 0,
    "max_dx": 0.15,
    "max_dy": 0.15,
    "target_mm": 15.0,
    "dU": 0.0005,          # 0.5 mm/step
    "max_steps": 80,
    "Ptot": 100e3,
    "testTol": 1.0e-4,
    "testIter": 15,
    "algo": "Newton",
    "system": "BandGeneral",
    "numberer": "RCM",
    "constraints": "Plain",
    "n_bins_x": 4,
    "n_bins_y": 12,
    "verbose": 0,
}

CLAMPS = {
    "max_dx": (0.05, 0.50),
    "max_dy": (0.05, 0.50),
    "dU": (0.0001, 0.0020),
    "max_steps": (10, 400),
    "target_mm": (2.0, 60.0),
    "Ptot": (1e3, 1e7),
    "testTol": (1e-8, 1e-2),
    "testIter": (5, 80),
    "n_bins_x": (2, 12),
    "n_bins_y": (4, 40),
}


# ============================================================
#  PARSING ROBUSTO
# ============================================================
def _as_int(v: Any, default: int) -> int:
    if v is None:
        return int(default)
    if isinstance(v, bool):
        return 1 if v else 0
    try:
        return int(v)
    except Exception:
        return int(default)

def _as_float(v: Any, default: float) -> float:
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)

def _clamp(name: str, val: Union[int, float]) -> Union[int, float]:
    if name not in CLAMPS:
        return val
    lo, hi = CLAMPS[name]
    if isinstance(val, int):
        return int(max(lo, min(val, hi)))
    return float(max(lo, min(val, hi)))

def _merge_params(payload: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(DEFAULTS)

    # body
    for k in DEFAULTS:
        if k in payload:
            out[k] = payload[k]

    # query vince sul body
    for k in DEFAULTS:
        if k in query and query[k] is not None:
            out[k] = query[k]

    out["stress"] = 1 if _as_int(out["stress"], 0) == 1 else 0
    out["verbose"] = 1 if _as_int(out.get("verbose", 0), 0) == 1 else 0

    out["max_dx"] = _clamp("max_dx", _as_float(out["max_dx"], DEFAULTS["max_dx"]))
    out["max_dy"] = _clamp("max_dy", _as_float(out["max_dy"], DEFAULTS["max_dy"]))
    out["dU"] = _clamp("dU", _as_float(out["dU"], DEFAULTS["dU"]))
    out["target_mm"] = _clamp("target_mm", _as_float(out["target_mm"], DEFAULTS["target_mm"]))
    out["Ptot"] = _clamp("Ptot", _as_float(out["Ptot"], DEFAULTS["Ptot"]))

    out["max_steps"] = _clamp("max_steps", _as_int(out["max_steps"], DEFAULTS["max_steps"]))
    out["testTol"] = _clamp("testTol", _as_float(out["testTol"], DEFAULTS["testTol"]))
    out["testIter"] = _clamp("testIter", _as_int(out["testIter"], DEFAULTS["testIter"]))

    out["n_bins_x"] = _clamp("n_bins_x", _as_int(out["n_bins_x"], DEFAULTS["n_bins_x"]))
    out["n_bins_y"] = _clamp("n_bins_y", _as_int(out["n_bins_y"], DEFAULTS["n_bins_y"]))

    out["algo"] = str(out.get("algo", DEFAULTS["algo"]))
    out["system"] = str(out.get("system", DEFAULTS["system"]))
    out["numberer"] = str(out.get("numberer", DEFAULTS["numberer"]))
    out["constraints"] = str(out.get("constraints", DEFAULTS["constraints"]))

    return out


# ============================================================
#  VALIDAZIONE APERTURE
# ============================================================
def inside_opening(x: float, y: float, openings: List[Tuple[float, float, float, float]]) -> bool:
    for (x1, x2, y1, y2) in openings:
        if (x > x1) and (x < x2) and (y > y1) and (y < y2):
            return True
    return False

def openings_valid(openings: List[Tuple[float, float, float, float]]) -> bool:
    for (x1, x2, y1, y2) in openings:
        if x1 < PIER_MIN or x2 > (L - PIER_MIN):
            return False
        if not (MARGIN <= x1 < x2 <= L - MARGIN):
            return False
        if not (MARGIN <= y1 < y2 <= H - MARGIN):
            return False
        for (yc1, yc2) in CORDOLI_Y:
            if not (y2 <= yc1 - MARGIN or y1 >= yc2 + MARGIN):
                return False

    n = len(openings)
    for i in range(n):
        x1i, x2i, y1i, y2i = openings[i]
        for j in range(i + 1, n):
            x1j, x2j, y1j, y2j = openings[j]

            dx_gap = max(0.0, max(x1i, x1j) - min(x2i, x2j))
            dy_gap = max(0.0, max(y1i, y1j) - min(y2i, y2j))
            overlap_y = not (y2i <= y1j or y2j <= y1i)

            if overlap_y and dx_gap < PIER_MIN:
                return False
            if dx_gap < MARGIN and dy_gap < MARGIN:
                return False

    return True


# ============================================================
#  MESH CONFORME
# ============================================================
def _unique_sorted(vals: List[float], tol: float = 1e-6) -> List[float]:
    vals = sorted(vals)
    out: List[float] = []
    for v in vals:
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out

def _refine_intervals(coords: List[float], max_step: float) -> List[float]:
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
                          max_dx: float, max_dy: float) -> Tuple[List[float], List[float]]:
    xs = [0.0, L, MARGIN, L - MARGIN]
    ys = [0.0, H, MARGIN, H - MARGIN]

    for (yc1, yc2) in CORDOLI_Y:
        ys += [yc1, yc2]

    for (x1, x2, y1, y2) in openings:
        xs += [x1, x2]
        ys += [y1, y2]

    xs = _refine_intervals(_unique_sorted(xs), max_dx)
    ys = _refine_intervals(_unique_sorted(ys), max_dy)
    return xs, ys


# ============================================================
#  MATERIALI
# ============================================================
def K_from_E_nu(E: float, nu: float) -> float:
    return E / (3.0 * (1.0 - 2.0 * nu))

def G_from_E_nu(E: float, nu: float) -> float:
    return E / (2.0 * (1.0 + nu))


# ============================================================
#  BUILD MODEL
# ============================================================
def build_wall_J2_conforming(
    openings: List[Tuple[float, float, float, float]],
    max_dx: float, max_dy: float,
    Ptot: float
) -> Dict[str, Any]:
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)

    xs, ys = build_conforming_grid(openings, max_dx=max_dx, max_dy=max_dy)

    node_tags: Dict[Tuple[int, int], int] = {}
    tag = 1
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if inside_opening(x, y, openings):
                continue
            ops.node(tag, x, y)
            node_tags[(i, j)] = tag
            tag += 1

    # vincoli base
    for i in range(len(xs)):
        key = (i, 0)
        if key in node_tags:
            ops.fix(node_tags[key], 1, 1)

    # materiali
    E_mur, nu_mur = 1.5e9, 0.15
    sig0_m, sigInf_m, delta_m, H_m = 0.5e6, 2.0e6, 8.0, 0.0

    E_cord, nu_cord = 30e9, 0.20
    sig0_c, sigInf_c, delta_c, H_c = 6.0e6, 25.0e6, 6.0, 0.0

    K_m, G_m = K_from_E_nu(E_mur, nu_mur), G_from_E_nu(E_mur, nu_mur)
    K_c, G_c = K_from_E_nu(E_cord, nu_cord), G_from_E_nu(E_cord, nu_cord)

    ops.nDMaterial("J2Plasticity", 10, K_m, G_m, sig0_m, sigInf_m, delta_m, H_m)
    ops.nDMaterial("J2Plasticity", 20, K_c, G_c, sig0_c, sigInf_c, delta_c, H_c)
    ops.nDMaterial("PlaneStress", 1, 10)
    ops.nDMaterial("PlaneStress", 2, 20)

    t = 0.25
    eleTag = 1

    for j in range(len(ys) - 1):
        yc = 0.5 * (ys[j] + ys[j + 1])

        this_mat = 1
        for (y1c, y2c) in CORDOLI_Y:
            if (yc >= y1c) and (yc <= y2c):
                this_mat = 2
                break

        for i in range(len(xs) - 1):
            keys = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if not all(k in node_tags for k in keys):
                continue
            n1 = node_tags[keys[0]]
            n2 = node_tags[keys[1]]
            n3 = node_tags[keys[2]]
            n4 = node_tags[keys[3]]
            ops.element("quad", eleTag, n1, n2, n3, n4, t, "PlaneStress", this_mat, 0.0, 0.0, 0.0)
            eleTag += 1

    # carico in sommità
    j_top = len(ys) - 1
    top_nodes = [node_tags[(i, j_top)] for i in range(len(xs)) if (i, j_top) in node_tags]
    if not top_nodes:
        raise RuntimeError("Nessun nodo in sommità: layout aperture troppo aggressivo.")

    control_node = top_nodes[len(top_nodes) // 2]

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)

    Pnode = float(Ptot) / len(top_nodes)
    for nd in top_nodes:
        ops.load(nd, Pnode, 0.0)

    # conteggi "safe"
    n_nodes = len(node_tags)
    n_eles = eleTag - 1

    return {
        "node_tags": node_tags,
        "control_node": control_node,
        "xs": xs,
        "ys": ys,
        "n_nodes": n_nodes,
        "n_eles": n_eles,
    }


# ============================================================
#  STRESS PROFILES (solo se stress=1)
# ============================================================
def _compute_stress_grid_profiles(n_bins_x: int, n_bins_y: int) -> Dict[str, Any]:
    ele_tags = ops.getEleTags()
    if not ele_tags:
        return {"tau_profile_y": {"y": [], "tau_mean": []}, "sigma_profile_y": {"y": [], "sigma_c_mean": []}, "zones": []}

    y_edges = np.linspace(0.0, H, n_bins_y + 1)
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_edges = np.linspace(0.0, L, n_bins_x + 1)

    tau_sum_y = np.zeros(n_bins_y)
    tau_cnt_y = np.zeros(n_bins_y)
    sigc_sum_y = np.zeros(n_bins_y)
    sigc_cnt_y = np.zeros(n_bins_y)

    for ele in ele_tags:
        stress = ops.eleResponse(ele, "stress")
        if stress is None:
            continue

        tau_vals = []
        sigy_vals = []
        for k in range(0, len(stress), 3):
            sigy_vals.append(float(stress[k + 1]))
            tau_vals.append(abs(float(stress[k + 2])))

        if not tau_vals:
            continue

        tau_mean_el = float(sum(tau_vals) / len(tau_vals))
        sigma_c_el = float(abs(min(sigy_vals)))

        nds = ops.eleNodes(ele)
        ys_el = [float(ops.nodeCoord(nd)[1]) for nd in nds]
        xs_el = [float(ops.nodeCoord(nd)[0]) for nd in nds]
        yc = sum(ys_el) / len(ys_el)
        xc = sum(xs_el) / len(xs_el)

        j = int(np.searchsorted(y_edges, yc) - 1)
        i = int(np.searchsorted(x_edges, xc) - 1)
        if j < 0 or j >= n_bins_y or i < 0 or i >= n_bins_x:
            continue

        tau_sum_y[j] += tau_mean_el
        tau_cnt_y[j] += 1
        sigc_sum_y[j] += sigma_c_el
        sigc_cnt_y[j] += 1

    y_out = [float(y_centers[j]) for j in range(n_bins_y) if tau_cnt_y[j] > 0]
    tau_mean = [float(tau_sum_y[j] / tau_cnt_y[j]) for j in range(n_bins_y) if tau_cnt_y[j] > 0]
    y_sig = [float(y_centers[j]) for j in range(n_bins_y) if sigc_cnt_y[j] > 0]
    sigc_mean = [float(sigc_sum_y[j] / sigc_cnt_y[j]) for j in range(n_bins_y) if sigc_cnt_y[j] > 0]

    return {
        "tau_profile_y": {"y": y_out, "tau_mean": tau_mean},
        "sigma_profile_y": {"y": y_sig, "sigma_c_mean": sigc_mean},
        "zones": []
    }


# ============================================================
#  PUSHOVER
# ============================================================
def shear_at_target_disp(disp_mm: np.ndarray, shear_kN: np.ndarray, target_mm: float) -> Optional[float]:
    if len(disp_mm) < 2:
        return None
    if float(np.max(disp_mm)) < target_mm:
        return None
    return float(np.interp(target_mm, disp_mm, shear_kN))


def run_pushover_case(openings: List[Tuple[float, float, float, float]], params: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()

    if not openings_valid(openings):
        return {"status": "error", "message": "openings_invalid", "disp_mm": [], "shear_kN": [], "V_target": None}

    # build
    model = build_wall_J2_conforming(
        openings=openings,
        max_dx=float(params["max_dx"]),
        max_dy=float(params["max_dy"]),
        Ptot=float(params["Ptot"]),
    )
    node_tags = model["node_tags"]
    control_node = model["control_node"]

    # analysis setup
    ops.constraints(str(params["constraints"]))
    ops.numberer(str(params["numberer"]))
    ops.system(str(params["system"]))
    ops.test("NormUnbalance", float(params["testTol"]), int(params["testIter"]))

    algo = str(params["algo"])
    if algo == "Newton":
        ops.algorithm("Newton")
    else:
        ops.algorithm("Newton")

    ops.integrator("DisplacementControl", int(control_node), 1, float(params["dU"]))
    ops.analysis("Static")

    disp_mm: List[float] = []
    shear_kN: List[float] = []

    verbose = int(params["verbose"]) == 1
    buf = io.StringIO() if verbose else None

    max_steps = int(params["max_steps"])
    target_mm = float(params["target_mm"])

    for step in range(max_steps):
        if verbose:
            buf.truncate(0); buf.seek(0)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                ok = ops.analyze(1)
        else:
            ok = ops.analyze(1)

        if ok < 0:
            return {
                "status": "error",
                "message": f"analysis_failed_step_{step}",
                "disp_mm": disp_mm,
                "shear_kN": shear_kN,
                "V_target": None,
                "debug": (buf.getvalue().strip()[:800] if verbose else None),
                "mesh": {
                    "max_dx": float(params["max_dx"]),
                    "max_dy": float(params["max_dy"]),
                    "n_x_lines": len(model["xs"]),
                    "n_y_lines": len(model["ys"]),
                    "n_nodes": model["n_nodes"],
                    "n_eles": model["n_eles"],
                },
                "timing_s": {"total": float(time.time() - t0)},
            }

        u = float(ops.nodeDisp(control_node, 1))  # m
        ops.reactions()

        Vb = 0.0
        for (_ij, nd) in node_tags.items():
            # base nodes = j == 0
            # (salviamo j nell'indice del dict)
            pass

        # più veloce: scorri solo i base nodes precomputati
        # (precomputo la prima volta)
        break

    # ---- RILANCIO con base nodes precomputati (ottimizzazione reale) ----
    base_nodes = [nd for (i, j), nd in node_tags.items() if j == 0]

    disp_mm = []
    shear_kN = []
    # reset analysis state: più semplice rifare solo loop; modello è già in memoria, quindi ok
    # (non rebuildiamo)
    for step in range(max_steps):
        if verbose:
            buf.truncate(0); buf.seek(0)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                ok = ops.analyze(1)
        else:
            ok = ops.analyze(1)

        if ok < 0:
            return {
                "status": "error",
                "message": f"analysis_failed_step_{step}",
                "disp_mm": disp_mm,
                "shear_kN": shear_kN,
                "V_target": None,
                "debug": (buf.getvalue().strip()[:800] if verbose else None),
                "mesh": {
                    "max_dx": float(params["max_dx"]),
                    "max_dy": float(params["max_dy"]),
                    "n_x_lines": len(model["xs"]),
                    "n_y_lines": len(model["ys"]),
                    "n_nodes": model["n_nodes"],
                    "n_eles": model["n_eles"],
                },
                "timing_s": {"total": float(time.time() - t0)},
            }

        u = float(ops.nodeDisp(control_node, 1))  # m
        ops.reactions()

        Vb = 0.0
        for nd in base_nodes:
            Vb += float(ops.nodeReaction(nd, 1))

        disp_mm.append(u * 1000.0)
        shear_kN.append(-Vb / 1000.0)

        if disp_mm[-1] >= target_mm:
            break

    disp_arr = np.array(disp_mm, dtype=float)
    shear_arr = np.array(shear_kN, dtype=float)
    Vt = shear_at_target_disp(disp_arr, shear_arr, target_mm)

    out: Dict[str, Any] = {
        "status": "ok" if Vt is not None else "error",
        "message": None if Vt is not None else "analysis_not_reached_target",
        "disp_mm": disp_arr.tolist(),
        "shear_kN": shear_arr.tolist(),
        "V_target": Vt,
        "mesh": {
            "max_dx": float(params["max_dx"]),
            "max_dy": float(params["max_dy"]),
            "n_x_lines": len(model["xs"]),
            "n_y_lines": len(model["ys"]),
            "n_nodes": model["n_nodes"],
            "n_eles": model["n_eles"],
        },
        "timing_s": {"total": float(time.time() - t0)},
    }

    if int(params["stress"]) == 1:
        out["stress_profiles"] = _compute_stress_grid_profiles(int(params["n_bins_x"]), int(params["n_bins_y"]))

    return out


def run_two_cases(payload: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    existing_openings = [tuple(o) for o in payload.get("existing_openings", [])]
    project_openings  = [tuple(o) for o in payload.get("project_openings", [])]

    if not existing_openings:
        raise ValueError("Chiave 'existing_openings' mancante o vuota.")
    if not project_openings:
        raise ValueError("Chiave 'project_openings' mancante o vuota.")

    res_existing = run_pushover_case(existing_openings, params)
    res_project  = run_pushover_case(project_openings, params)
    return {"existing": res_existing, "project": res_project}


# ============================================================
#  FASTAPI
# ============================================================
app = FastAPI(
    title="Wall Pushover Service (Conforming Mesh)",
    version="3.1.0",
    description="Pushover muratura (Existing + Project) con mesh conforme e parametri via body/query."
)

@app.get("/")
def root():
    return {"status": "ok", "service": "pushover-conforming", "defaults": DEFAULTS}


@app.post("/pushover")
async def pushover(
    request: Request,
    # query params opzionali (vincono sul body)
    stress: Optional[int] = None,
    max_dx: Optional[float] = None,
    max_dy: Optional[float] = None,
    dU: Optional[float] = None,
    max_steps: Optional[int] = None,
    target_mm: Optional[float] = None,
    Ptot: Optional[float] = None,
    testTol: Optional[float] = None,
    testIter: Optional[int] = None,
    algo: Optional[str] = None,
    system: Optional[str] = None,
    numberer: Optional[str] = None,
    constraints: Optional[str] = None,
    n_bins_x: Optional[int] = None,
    n_bins_y: Optional[int] = None,
    verbose: Optional[int] = None,
):
    payload = await request.json()

    # wrapper n8n/lovable: lista con 1 elemento
    if isinstance(payload, list):
        if not payload:
            return JSONResponse(status_code=400, content={"error": "Payload list vuota."})
        payload = payload[0]

    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"error": "Payload non valido: atteso oggetto JSON."})

    query = {
        "stress": stress, "max_dx": max_dx, "max_dy": max_dy, "dU": dU, "max_steps": max_steps,
        "target_mm": target_mm, "Ptot": Ptot, "testTol": testTol, "testIter": testIter,
        "algo": algo, "system": system, "numberer": numberer, "constraints": constraints,
        "n_bins_x": n_bins_x, "n_bins_y": n_bins_y, "verbose": verbose
    }

    params = _merge_params(payload, query)

    t0 = time.time()
    try:
        result = run_two_cases(payload, params)
        return JSONResponse(content={
            "meta": {"params_used": params, "timing_s": {"total": float(time.time() - t0)}},
            "results": result
        })
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e), "params_used": params})
