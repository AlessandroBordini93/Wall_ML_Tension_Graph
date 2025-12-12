# main.py
import io
import contextlib
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import openseespy.opensees as ops

# ============================================================
#  COSTANTI GLOBALI
# ============================================================

L = 4.0
H = 6.0

# Mesh "default" (puoi sovrascriverla via query param nx, ny)
NX_DEFAULT = 15
NY_DEFAULT = 30

TARGET_DISP_MM = 15.0

CORDOLI_Y = [
    (2.7, 3.0),
    (5.7, 6.0),
]

MARGIN = 0.30
PIER_MIN = 0.30  # maschio orizzontale minimo


# ============================================================
#  FUNZIONI GEOMETRIA
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
    Controlla che:
      - le aperture siano tutte dentro la parete con margin
      - non siano troppo vicine ai cordoli (>= margin)
      - non si sovrappongano e non siano più vicine di 'margin' tra loro
      - i maschi orizzontali non siano più piccoli di PIER_MIN
    """
    # 1) limiti della parete + maschio ai bordi
    for (x1, x2, y1, y2) in openings:
        # bordo parete: voglio almeno PIER_MIN di muratura
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

    # 3) distanza tra aperture
    n = len(openings)
    for i in range(n):
        x1i, x2i, y1i, y2i = openings[i]
        for j in range(i + 1, n):
            x1j, x2j, y1j, y2j = openings[j]

            # gap in x e in y
            dx_gap = max(0.0, max(x1i, x1j) - min(x2i, x2j))
            dy_gap = max(0.0, max(y1i, y1j) - min(y2i, y2j))

            # sovrapposizione in quota?
            overlap_y = not (y2i <= y1j or y2j <= y1i)

            # se stanno sullo stesso piano (overlap in y), voglio un maschio >= PIER_MIN
            if overlap_y and dx_gap < PIER_MIN:
                return False

            # se sono "diagonali", tengo comunque un minimo di separazione generica = margin
            if dx_gap < margin and dy_gap < margin:
                return False

    return True


def K_from_E_nu(E: float, nu: float) -> float:
    return E / (3.0 * (1.0 - 2.0 * nu))


def G_from_E_nu(E: float, nu: float) -> float:
    return E / (2.0 * (1.0 + nu))


# ============================================================
#  MODELLO NON LINEARE J2
# ============================================================

def build_wall_J2(openings, nx: int, ny: int):
    """
    Modello NON lineare J2:
      - parete 4 x 6 m
      - quad4
      - aperture (input)
      - cordoli (materiale più rigido)
      - J2Plasticity -> PlaneStress per muratura e cordoli
    """
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 2)

    dx = L / nx
    dy = H / ny

    # -------------------------
    # NODI (NO dentro le aperture)
    # -------------------------
    node_tags: Dict[Tuple[int, int], int] = {}
    tag = 1

    for j in range(ny + 1):
        y = j * dy
        for i in range(nx + 1):
            x = i * dx

            if inside_opening(x, y, openings):
                continue

            ops.node(tag, x, y)
            node_tags[(i, j)] = tag
            tag += 1

    # Vincoli alla base
    for i in range(nx + 1):
        key = (i, 0)
        if key in node_tags:
            ops.fix(node_tags[key], 1, 1)

    # -------------------------
    # MATERIALI NON LINEARI – J2Plasticity -> PlaneStress
    # -------------------------
    # Muratura (valori indicativi)
    E_mur   = 1.5e9
    nu_mur  = 0.15
    sig0_m  = 0.5e6
    sigInf_m = 2.0e6
    delta_m  = 8.0
    H_m      = 0.0

    # Calcestruzzo cordoli (più rigido)
    E_cord   = 30e9
    nu_cord  = 0.20
    sig0_c   = 6.0e6
    sigInf_c = 25.0e6
    delta_c  = 6.0
    H_c      = 0.0

    K_m = K_from_E_nu(E_mur, nu_mur)
    G_m = G_from_E_nu(E_mur, nu_mur)
    K_c = K_from_E_nu(E_cord, nu_cord)
    G_c = G_from_E_nu(E_cord, nu_cord)

    # 3D J2Plasticity per muratura e c.a.
    matTag_mur_3D  = 10
    matTag_cord_3D = 20

    ops.nDMaterial('J2Plasticity', matTag_mur_3D,
                   K_m, G_m, sig0_m, sigInf_m, delta_m, H_m)
    ops.nDMaterial('J2Plasticity', matTag_cord_3D,
                   K_c, G_c, sig0_c, sigInf_c, delta_c, H_c)

    # Wrapper PlaneStress
    matTag_mur   = 1
    matTag_cord  = 2
    ops.nDMaterial('PlaneStress', matTag_mur,  matTag_mur_3D)
    ops.nDMaterial('PlaneStress', matTag_cord, matTag_cord_3D)

    t = 0.25  # spessore [m]

    # -------------------------
    # ELEMENTI QUAD4
    # -------------------------
    eleTag = 1
    for j in range(ny):
        for i in range(nx):
            keys = [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]

            if not all(k in node_tags for k in keys):
                continue

            yc = (j + 0.5) * dy

            this_mat = matTag_mur
            for (y1c, y2c) in CORDOLI_Y:
                if (yc >= y1c) and (yc <= y2c):
                    this_mat = matTag_cord
                    break

            n1 = node_tags[keys[0]]
            n2 = node_tags[keys[1]]
            n3 = node_tags[keys[2]]
            n4 = node_tags[keys[3]]

            ops.element(
                'quad', eleTag, n1, n2, n3, n4,
                t, 'PlaneStress', this_mat, 0.0, 0.0, 0.0
            )
            eleTag += 1

    # -------------------------
    # CARICO ORIZZONTALE (PUSHOVER)
    # -------------------------
    top_nodes = [node_tags[(i, ny)] for i in range(nx + 1)
                 if (i, ny) in node_tags]

    if not top_nodes:
        raise RuntimeError("Nessun nodo in sommità: layout aperture troppo aggressivo.")

    # nodo di controllo centrale tra quelli in sommità
    control_node = top_nodes[len(top_nodes) // 2]

    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    Ptot = 100e3      # [N]
    Pnode = Ptot / len(top_nodes)

    for nd in top_nodes:
        ops.load(nd, Pnode, 0.0)

    return node_tags, control_node


def shear_at_target_disp(disp_mm: np.ndarray,
                         shear_kN: np.ndarray,
                         target_mm: float = TARGET_DISP_MM) -> Union[float, None]:
    """
    Interpola il taglio alla base alla deformata target_mm.
    Restituisce None se la curva non arriva a target_mm.
    """
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
    Legge le tensioni nel CURRENT state di OpenSees (ultimo step analisi)
    e costruisce:
      - profilo verticale medio di tau e sigma_c
      - griglia 2D di zone con tau/sigma.
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

        for i in range(0, len(stress), 3):
            sigx = stress[i]
            sigy = stress[i+1]
            tau  = stress[i+2]
            tau_vals.append(abs(tau))
            sigy_vals.append(sigy)

        if not tau_vals or not sigy_vals:
            continue

        tau_max_el  = max(tau_vals)
        tau_mean_el = sum(tau_vals) / len(tau_vals)

        sigy_min = min(sigy_vals)      # più compressa (negativa)
        sigma_c_el = abs(sigy_min)

        node_tags = ops.eleNodes(ele)
        xs, ys = [], []
        for nd in node_tags:
            x_nd, y_nd = ops.nodeCoord(nd)
            xs.append(x_nd)
            ys.append(y_nd)
        if not xs or not ys:
            continue
        xc = sum(xs) / len(xs)
        yc = sum(ys) / len(ys)

        # profilo verticale
        j = int(np.searchsorted(y_edges, yc) - 1)
        if j < 0 or j >= n_bins_y:
            continue

        tau_sum_y[j]   += tau_mean_el
        tau_count_y[j] += 1

        sigc_sum_y[j]   += sigma_c_el
        sigc_count_y[j] += 1

        # griglia 2D
        i = int(np.searchsorted(x_edges, xc) - 1)
        if i < 0 or i >= n_bins_x:
            continue

        if tau_max_el > tau_max_2d[i, j]:
            tau_max_2d[i, j] = tau_max_el

        tau_sum_2d[i, j]  += tau_mean_el
        tau_count_2d[i, j]+= 1

        if sigma_c_el > sigc_max_2d[i, j]:
            sigc_max_2d[i, j] = sigma_c_el

    # profili 1D
    y_out = []
    tau_mean_y = []
    for j in range(n_bins_y):
        if tau_count_y[j] > 0:
            y_out.append(float(y_centers[j]))
            tau_mean_y.append(float(tau_sum_y[j] / tau_count_y[j]))

    y_sig_out = []
    sigc_mean_y = []
    for j in range(n_bins_y):
        if sigc_count_y[j] > 0:
            y_sig_out.append(float(y_centers[j]))
            sigc_mean_y.append(float(sigc_sum_y[j] / sigc_count_y[j]))

    # griglia 2D
    zones = []
    for i in range(n_bins_x):
        for j in range(n_bins_y):
            if tau_count_2d[i, j] <= 0:
                continue

            x_min = float(x_edges[i])
            x_max = float(x_edges[i+1])
            y_min = float(y_edges[j])
            y_max = float(y_edges[j+1])

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
        "tau_profile_y": {
            "y": y_out,
            "tau_mean": tau_mean_y
        },
        "sigma_profile_y": {
            "y": y_sig_out,
            "sigma_c_mean": sigc_mean_y
        },
        "zones": zones
    }


# ============================================================
#  ANALISI PUSHOVER (CURVA + STRESS OPZIONALE)
# ============================================================

def run_pushover_nonlinear(openings,
                           nx: int = NX_DEFAULT,
                           ny: int = NY_DEFAULT,
                           target_mm: float = TARGET_DISP_MM,
                           max_steps: int = 100,
                           dU: float = 0.0002,
                           verbose: bool = False,
                           compute_stress_profiles: bool = False) -> Dict[str, Any]:
    """
    Esegue pushover non lineare J2.
    Restituisce:
      {
        "status": "ok"/"error",
        "message": str o None,
        "disp_mm": [...],
        "shear_kN": [...],
        "V_target": float o None,
        "stress_profiles": {...}  (se richiesto)
      }
    """
    try:
        if not openings_valid(openings, CORDOLI_Y, MARGIN):
            base = {
                "status": "error",
                "message": "openings_invalid",
                "disp_mm": [],
                "shear_kN": [],
                "V_target": None,
            }
            if compute_stress_profiles:
                base["stress_profiles"] = {
                    "tau_profile_y": {"y": [], "tau_mean": []},
                    "sigma_profile_y": {"y": [], "sigma_c_mean": []},
                    "zones": []
                }
            return base

        node_tags, control_node = build_wall_J2(openings, nx, ny)

        ops.constraints('Plain')
        ops.numberer('RCM')
        ops.system('BandGeneral')

        ops.test('NormUnbalance', 1.0e-4, 15)
        ops.algorithm('Newton')

        ops.integrator('DisplacementControl', control_node, 1, dU)
        ops.analysis('Static')

        disp_mm: List[float] = []
        shear_kN: List[float] = []

        j2_problem = False
        lin_problem = False

        buf = io.StringIO()

        for step in range(max_steps):
            buf.truncate(0)
            buf.seek(0)

            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                ok = ops.analyze(1)

            log_text = buf.getvalue()

            if ("J2-plasticity" in log_text) and ("More than 25 iterations" in log_text):
                j2_problem = True
                if verbose:
                    print(f"  [NL] Problema J2-plasticity allo step {step}, analisi interrotta.")
                break

            if ("factorization failed" in log_text) or ("matrix singular" in log_text):
                lin_problem = True
                if verbose:
                    print(f"  [NL] Problema solver lineare allo step {step}, analisi interrotta.")
                    print(log_text.strip())
                break

            if ok < 0:
                if verbose:
                    print(f"  [NL] analyze failed allo step {step}")
                    print(log_text.strip())
                break

            u = ops.nodeDisp(control_node, 1)  # [m]
            ops.reactions()

            Vb = 0.0
            for (i, j), nd in node_tags.items():
                if j == 0:
                    Vb += ops.nodeReaction(nd, 1)

            disp_mm.append(u * 1000.0)
            shear_kN.append(-Vb / 1000.0)

            if disp_mm and disp_mm[-1] >= target_mm * 1.0:
                break

        disp_arr = np.array(disp_mm, dtype=float)
        shear_arr = np.array(shear_kN, dtype=float)

        if j2_problem or lin_problem:
            base = {
                "status": "error",
                "message": "analysis_not_converged",
                "disp_mm": disp_arr.tolist(),
                "shear_kN": shear_arr.tolist(),
                "V_target": None
            }
            if compute_stress_profiles:
                base["stress_profiles"] = {
                    "tau_profile_y": {"y": [], "tau_mean": []},
                    "sigma_profile_y": {"y": [], "sigma_c_mean": []},
                    "zones": []
                }
            return base

        V_target = shear_at_target_disp(disp_arr, shear_arr, target_mm)

        result: Dict[str, Any] = {
            "status": "ok",
            "message": None,
            "disp_mm": disp_arr.tolist(),
            "shear_kN": shear_arr.tolist(),
            "V_target": V_target
        }

        if compute_stress_profiles:
            stress_prof = _compute_stress_grid_profiles(
                n_bins_x=4,
                n_bins_y=12
            )
            result["stress_profiles"] = stress_prof

        return result

    except Exception as e:
        if verbose:
            print(f"[run_pushover_nonlinear] errore: {e}")
        base = {
            "status": "error",
            "message": str(e),
            "disp_mm": [],
            "shear_kN": [],
            "V_target": None
        }
        if compute_stress_profiles:
            base["stress_profiles"] = {
                "tau_profile_y": {"y": [], "tau_mean": []},
                "sigma_profile_y": {"y": [], "sigma_c_mean": []},
                "zones": []
            }
        return base


# ============================================================
#  FUNZIONE ALTO LIVELLO: EXISTING + PROJECT
# ============================================================

def run_two_cases_from_dict(
    data: Dict[str, Any],
    compute_profiles: bool,
    nx: int,
    ny: int
) -> Dict[str, Any]:
    """
    data deve contenere:
      {
        "existing_openings": [[x1,x2,y1,y2], ...],
        "project_openings":  [[x1,x2,y1,y2], ...]
      }
    """
    existing_openings = [tuple(o) for o in data.get("existing_openings", [])]
    project_openings  = [tuple(o) for o in data.get("project_openings", [])]

    if not existing_openings:
        raise ValueError("Chiave 'existing_openings' mancante o vuota.")
    if not project_openings:
        raise ValueError("Chiave 'project_openings' mancante o vuota.")

    res_existing = run_pushover_nonlinear(
        existing_openings,
        nx=nx,
        ny=ny,
        compute_stress_profiles=compute_profiles
    )
    res_project  = run_pushover_nonlinear(
        project_openings,
        nx=nx,
        ny=ny,
        compute_stress_profiles=compute_profiles
    )

    return {
        "existing": res_existing,
        "project": res_project
    }


# ============================================================
#  FASTAPI APP
# ============================================================

app = FastAPI(
    title="Wall Pushover Service",
    description="Servizio pushover muratura (Existing + Project) – curva e stress opzionali",
    version="1.0.0"
)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Wall Pushover Service attivo"}


@app.post("/pushover")
async def pushover_endpoint(
    request: Request,
    stress: int = 0,        # 0 = solo curva, 1 = curva + stress profiles
    nx: int = NX_DEFAULT,   # sovrascrivibile da query
    ny: int = NY_DEFAULT
):
    """
    Endpoint principale.

    Query param:
      - stress = 0/1           (0 solo curva, 1 anche stress)
      - nx, ny = dimensione mesh

    Body JSON (Lovable/n8n):
      - [{ "existing_openings": [...], "project_openings": [...] }]
        oppure
      - { "existing_openings": [...], "project_openings": [...] }
    """
    payload = await request.json()

    # gestisce sia lista che dict
    if isinstance(payload, list):
        if not payload:
            return JSONResponse(
                status_code=400,
                content={"error": "Payload list vuota."}
            )
        payload = payload[0]

    # clamp mesh per evitare richieste assurde
    nx = max(8, min(int(nx), 40))
    ny = max(16, min(int(ny), 80))

    compute_profiles = (int(stress) == 1)

    try:
        result = run_two_cases_from_dict(payload, compute_profiles, nx, ny)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
