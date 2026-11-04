"""Planificadores recientes basados en Voronoi (reimplementaciones para la revisión).

MVDRM (Ayawli et al., 2021)
---------------------------
Dilatación morfológica de los obstáculos + hoja de ruta de Voronoi + búsqueda en
el grafo.  Sólo se dispuso de la descripción general del método (diagrama de
Voronoi calculado con el algoritmo de Fortune sobre puntos de los obstáculos
dilatados, eliminación de las aristas que tocan obstáculos, búsqueda A*), así
que los detalles no publicados se fijaron así:

* dilatación: el espacio de configuración común (disco de radio r = D/2), que
  es exactamente la dilatación morfológica de los obstáculos por el robot;
* sitios: píxeles del contorno de los obstáculos dilatados (incluido el marco)
  tomados cada ``factor_escala`` píxeles a lo largo de cada contorno; este
  submuestreo hace el papel del factor de escala del artículo, que reduce y
  uniformiza los nodos;
* diagrama de Voronoi de los sitios (dual de Delaunay, cv2.Subdiv2D); se
  conservan las aristas cuyos dos extremos y cuyo segmento están libres;
* S y T se unen a los ``vecinos_ST`` vértices visibles más cercanos y la ruta se
  busca con A* (heurística euclidiana).  La ruta es la poligonal de vértices.

G2VD (Wen et al., 2025; versión arXiv 2201.12981v4)
---------------------------------------------------
1. GVD sobre la malla con la regla de los planificadores dinámicos de Voronoi:
   una celda libre es del GVD si el obstáculo más cercano de alguna vecina no es
   adyacente al suyo (transformada de distancia con etiquetas por píxel).
2. Búsqueda gruesa: BFS de S y T a la celda del GVD más cercana y A* sobre el
   GVD; el corredor de Voronoi es la unión de los cuadrados de lado 2·d_k
   centrados en cada celda de esa ruta (d_k = holgura de la celda).
3. Búsqueda fina: A* sobre una retícula de estados (x, y, θ) con 16 rumbos,
   restringida al corredor.  Primitivas: curvas de Bézier quínticas entre
   estados de la retícula (rumbo final = rumbo inicial o ±1) y giros en el sitio
   (robot diferencial).  Costo (ec. 3-4): t·(max ρ_V + 1), con
   t = max(L/v_max, |Δθ|/ω_max) y el campo de Voronoi
   ρ_V = d_V/(d_O + d_V) · (d_O − d_O^min)²/(d_O^min)² si d_O ≤ d_O^min.
   Heurística: distancia geodésica 2-D dentro del corredor (programación
   dinámica) dividida por v_max.
4. Suavizado por programación cuadrática (ec. 5-8): puntos de referencia cada
   0.1 m, min ω_s Σ‖x_{i+1} − 2x_i + x_{i−1}‖² + ω_r Σ‖x_i − x_i^ref‖², extremos
   fijos y cajas |x_i − x_i^ref| ≤ b_i con b_i = (√2/2)·d_i − r_c.  Como los
   planificadores reciben el espacio de configuración (robot puntual), r_c = 0,
   y las dos formas de b_i del artículo coinciden.  El QP se resuelve con ADMM
   (el mismo esquema que OSQP).  Se omite la interpolación final por spline
   cúbica, porque los vértices del QP ya distan 0.1 m.
"""
from __future__ import annotations

import heapq
import math
import time
from collections import deque

import cv2
import numpy as np

from common.geometry import polyline_free, polyline_length, segment_free
from common.problem import AlgorithmResult, Problem
from baselines.grid_search import distance_field

SQ2 = math.sqrt(2.0)


# --------------------------------------------------------------------------- MVDRM
def run_mvdrm(problem: Problem, factor_escala=4, vecinos_ST=6):
    t0 = time.perf_counter()
    occ = problem.occ
    H, W = occ.shape
    cnts, _ = cv2.findContours(occ.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    step = max(1, int(factor_escala))
    sites = [c.reshape(-1, 2)[::step] for c in cnts if len(c) >= 3]
    sites = np.unique(np.vstack(sites).astype(np.float32), axis=0) if sites else np.zeros((0, 2), np.float32)
    if len(sites) < 3:
        return AlgorithmResult("mvdrm", "MVDRM", False, None, time.perf_counter() - t0, "Sin obstáculos")
    sub = cv2.Subdiv2D((-2, -2, W + 4, H + 4))
    sub.insert([(float(x), float(y)) for x, y in sites])
    facets, _ = sub.getVoronoiFacetList([])
    key = {}
    pts = []

    def node(p):
        k = (round(float(p[0]), 2), round(float(p[1]), 2))
        if k not in key:
            key[k] = len(pts)
            pts.append((float(p[0]), float(p[1])))
        return key[k]

    def free_pt(p):
        x, y = p
        if not (0 <= x <= W - 1 and 0 <= y <= H - 1):
            return False
        return not occ[int(round(y)), int(round(x))]

    seen = set()
    adj: dict[int, list[tuple[int, float]]] = {}
    n_edges = 0
    for f in facets:
        f = np.asarray(f, float)
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            if not (free_pt(a) and free_pt(b)):
                continue
            ia, ib = node(a), node(b)
            if ia == ib or (min(ia, ib), max(ia, ib)) in seen:
                continue
            seen.add((min(ia, ib), max(ia, ib)))
            if not segment_free(occ, a, b):
                continue
            d = float(np.hypot(*(a - b)))
            adj.setdefault(ia, []).append((ib, d))
            adj.setdefault(ib, []).append((ia, d))
            n_edges += 1
    P = np.asarray(pts, float) if pts else np.zeros((0, 2))
    used = np.array(sorted(adj), int)
    if len(used) == 0:
        return AlgorithmResult("mvdrm", "MVDRM", False, None, time.perf_counter() - t0, "Hoja de ruta vacía")
    S, T = problem.start, problem.goal
    iS, iT = len(P), len(P) + 1
    P = np.vstack([P, S, T])

    def link(idx, p):
        d = np.hypot(*(P[used] - p).T)
        order = used[np.argsort(d)]
        k = 0
        for j in order[:60]:
            if segment_free(occ, p, P[j]):
                w = float(np.hypot(*(P[j] - p)))
                adj.setdefault(idx, []).append((int(j), w))
                adj.setdefault(int(j), []).append((idx, w))
                k += 1
                if k >= vecinos_ST:
                    break
        return k

    if segment_free(occ, S, T):
        w = float(np.hypot(*(S - T)))
        adj.setdefault(iS, []).append((iT, w))
        adj.setdefault(iT, []).append((iS, w))
    link(iS, S)
    link(iT, T)
    # A*
    g = {iS: 0.0}
    prev = {iS: -1}
    heap = [(float(np.hypot(*(S - T))), 0.0, iS)]
    closed = set()
    expanded = 0
    while heap:
        f, d, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        if u == iT:
            break
        for v, wgt in adj.get(u, ()):
            nd = d + wgt
            if nd < g.get(v, math.inf):
                g[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd + float(np.hypot(*(P[v] - T))), nd, v))
    stats = {"nodos_expandidos": expanded, "sitios": int(len(sites)), "vertices_hoja": int(len(used)),
             "aristas_hoja": n_edges}
    if iT not in closed:
        return AlgorithmResult("mvdrm", "MVDRM", False, None, time.perf_counter() - t0,
                               "La hoja de ruta de Voronoi no conecta S y T", stats)
    chain = [iT]
    while prev[chain[-1]] >= 0:
        chain.append(prev[chain[-1]])
    path = P[chain[::-1]]
    return AlgorithmResult("mvdrm", "MVDRM", True, path, time.perf_counter() - t0, "Ruta encontrada", stats)


# --------------------------------------------------------------------------- G2VD
def grid_gvd(occ: np.ndarray) -> np.ndarray:
    """Celdas libres del GVD: el obstáculo más cercano de una vecina no es adyacente al propio."""
    free = (~occ).astype(np.uint8)
    _, lab = cv2.distanceTransformWithLabels(free, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(occ)
    ox = np.zeros(int(lab.max()) + 1, np.int32)
    oy = np.zeros_like(ox)
    ox[lab[ys, xs]] = xs
    oy[lab[ys, xs]] = ys
    NX, NY = ox[lab], oy[lab]
    gvd = np.zeros(occ.shape, bool)
    for dy, dx in ((0, 1), (1, 0)):
        a = (slice(0, occ.shape[0] - dy), slice(0, occ.shape[1] - dx))
        b = (slice(dy, None), slice(dx, None))
        m = (~occ[a]) & (~occ[b]) & ((np.abs(NX[a] - NX[b]) > 1) | (np.abs(NY[a] - NY[b]) > 1))
        gvd[a] |= m
        gvd[b] |= m
    return gvd


_NB8 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2),
        (1, 1, SQ2)]


def _bfs_to(mask_target: np.ndarray, occ: np.ndarray, src) -> list[tuple[int, int]] | None:
    """Camino de celdas (r, c) desde src hasta la celda de mask_target más cercana (BFS 8-conexa)."""
    H, W = occ.shape
    r0, c0 = int(round(src[1])), int(round(src[0]))
    prev = {(r0, c0): None}
    q = deque([(r0, c0)])
    while q:
        r, c = q.popleft()
        if mask_target[r, c]:
            out = [(r, c)]
            while prev[out[-1]] is not None:
                out.append(prev[out[-1]])
            return out[::-1]
        for dr, dc, _ in _NB8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and not occ[rr, cc] and (rr, cc) not in prev:
                prev[(rr, cc)] = (r, c)
                q.append((rr, cc))
    return None


def _astar_cells(allowed: np.ndarray, a, b) -> list[tuple[int, int]] | None:
    H, W = allowed.shape
    g = {a: 0.0}
    prev = {a: None}
    heap = [(math.hypot(a[0] - b[0], a[1] - b[1]), 0.0, a)]
    closed = set()
    while heap:
        _, d, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        if u == b:
            out = [u]
            while prev[out[-1]] is not None:
                out.append(prev[out[-1]])
            return out[::-1]
        r, c = u
        for dr, dc, w in _NB8:
            v = (r + dr, c + dc)
            if not (0 <= v[0] < H and 0 <= v[1] < W) or not allowed[v]:
                continue
            nd = d + w
            if nd < g.get(v, math.inf):
                g[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd + math.hypot(v[0] - b[0], v[1] - b[1]), nd, v))
    return None


# 16 rumbos de la retícula: direcciones de los vectores enteros (1,0), (2,1), (1,1), (1,2), ...
_DIRS16 = [(1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1),
           (-1, 0), (-2, -1), (-1, -1), (-1, -2), (0, -1), (1, -2), (1, -1), (2, -1)]
_ANG16 = [math.atan2(dy, dx) for dx, dy in _DIRS16]


def _bezier5(p0, th0, p1, th1, n):
    """Bézier quíntica de p0 a p1 tangente a th0 y th1 (curvatura nula en los extremos)."""
    L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    u0 = np.array([math.cos(th0), math.sin(th0)])
    u1 = np.array([math.cos(th1), math.sin(th1)])
    k = L / 5.0
    P = np.array([p0, p0 + k * u0, p0 + 2 * k * u0, p1 - 2 * k * u1, p1 - k * u1, p1], float)
    t = np.linspace(0, 1, n)[:, None]
    B = [(1 - t) ** 5, 5 * t * (1 - t) ** 4, 10 * t ** 2 * (1 - t) ** 3, 10 * t ** 3 * (1 - t) ** 2,
         5 * t ** 4 * (1 - t), t ** 5]
    return sum(b * P[i] for i, b in enumerate(B))


def _qp_box_admm(ref: np.ndarray, b: np.ndarray, ws: float, wr: float, rho: float = 1.0,
                 iters: int = 400) -> np.ndarray:
    """min ws·Σ‖Δ²x‖² + wr·Σ‖x − ref‖², x_1, x_n fijos, |x_i − ref_i| ≤ b_i (por coordenada, ADMM)."""
    n = len(ref)
    if n < 3:
        return ref.copy()
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i:i + 3] = (1.0, -2.0, 1.0)
    Hm = 2 * ws * D.T @ D + 2 * wr * np.eye(n)
    lo = np.empty(n)
    hi = np.empty(n)
    out = np.empty_like(ref)
    K = np.linalg.inv(Hm + rho * np.eye(n))
    for c in range(2):
        r = ref[:, c]
        lo[:] = r - b
        hi[:] = r + b
        lo[[0, -1]] = hi[[0, -1]] = r[[0, -1]]
        q = -2 * wr * r
        z = r.copy()
        y = np.zeros(n)
        for _ in range(iters):
            x = K @ (rho * z - y - q)
            z_new = np.clip(x + y / rho, lo, hi)
            y += rho * (x - z_new)
            if np.max(np.abs(z_new - z)) < 1e-4 and np.max(np.abs(x - z_new)) < 1e-4:
                z = z_new
                break
            z = z_new
        out[:, c] = z
    return out


def run_g2vd(problem: Problem, d_seguridad_m=0.5, w_s=10.0, w_r=1.0, paso_ref_m=0.1, celda_reticula=3,
             v_max=1.0, w_max_deg=90.0, max_expansiones=200000):
    t0 = time.perf_counter()
    occ = problem.occ
    H, W = occ.shape
    mpp = problem.mpp
    S, T = problem.start, problem.goal
    clear = problem.clearance                                  # d_O [px] (espacio de configuración)
    gvd = grid_gvd(occ)
    stats = {"celdas_gvd": int(gvd.sum())}
    # --- búsqueda gruesa sobre el GVD
    l1 = _bfs_to(gvd, occ, S)
    l2 = _bfs_to(gvd, occ, T)
    if l1 is None or l2 is None:
        return AlgorithmResult("g2vd", "G2VD", False, None, time.perf_counter() - t0, "S o T no alcanzan el GVD", stats)
    l3 = _astar_cells(gvd, l1[-1], l2[-1])
    if l3 is None:
        return AlgorithmResult("g2vd", "G2VD", False, None, time.perf_counter() - t0,
                               "El GVD no conecta S y T", stats)
    coarse = l1 + l3[1:] + l2[::-1][1:]
    # --- corredor de Voronoi: cuadrados de lado 2·d_k
    corr = np.zeros(occ.shape, bool)
    for r, c in coarse:
        d = int(math.floor(clear[r, c]))
        corr[max(0, r - d):r + d + 1, max(0, c - d):c + d + 1] = True
    corr &= ~occ
    stats["celdas_corredor"] = int(corr.sum())
    # --- campo de Voronoi ρ_V (ec. 2), distancias en metros
    dV = cv2.distanceTransform((~gvd).astype(np.uint8), cv2.DIST_L2, 5) * mpp
    dO = clear * mpp
    dmin = d_seguridad_m
    rho_v = np.where(dO <= dmin, dV / np.maximum(dO + dV, 1e-9) * (dO - dmin) ** 2 / dmin ** 2, 0.0)
    rho_v[~corr] = np.inf
    # --- heurística 2-D dentro del corredor
    cell = max(1, int(celda_reticula))
    Dh, dcell = distance_field(~corr, T, cell=1)
    w_max = math.radians(w_max_deg)

    def heur(x, y):
        r = min(max(int(round(y)), 0), H - 1)
        c = min(max(int(round(x)), 0), W - 1)
        g = Dh[r, c]
        return (g if math.isfinite(g) else math.hypot(x - T[0], y - T[1])) * mpp / v_max

    # primitivas (relativas): por rumbo h, finales con rumbo h-1, h, h+1
    prims = {}
    for h in range(16):
        lst = []
        for dh in (-1, 0, 1):
            h2 = (h + dh) % 16
            dx, dy = _DIRS16[h2]
            end = np.array([dx * cell, dy * cell], float) * (1 if dh == 0 else 2)
            n = max(4, int(math.hypot(*end)) + 2)
            curve = _bezier5(np.zeros(2), _ANG16[h], end, _ANG16[h2], n)
            L = float(np.sum(np.hypot(*np.diff(curve, axis=0).T)))
            dth = abs((_ANG16[h2] - _ANG16[h] + math.pi) % (2 * math.pi) - math.pi)
            lst.append((h2, end, curve, max(L * mpp / v_max, dth / w_max)))
        for dh in (-1, 1):                                 # giro en el sitio
            h2 = (h + dh) % 16
            dth = abs((_ANG16[h2] - _ANG16[h] + math.pi) % (2 * math.pi) - math.pi)
            lst.append((h2, np.zeros(2), None, dth / w_max))
        prims[h] = lst
    # rumbo inicial: dirección del primer tramo del camino grueso
    k0 = min(len(coarse) - 1, 6)
    th0 = math.atan2(coarse[k0][0] - S[1], coarse[k0][1] - S[0])
    h0 = int(np.argmin([abs((a - th0 + math.pi) % (2 * math.pi) - math.pi) for a in _ANG16]))
    sx, sy = S
    start = (int(round(sx)), int(round(sy)), h0)
    g = {start: 0.0}
    prev = {start: None}
    pos = {start: np.array([sx, sy], float)}
    heap = [(heur(sx, sy), 0.0, start)]
    closed = set()
    goal_state = None
    expanded = 0
    tol = 1.5 * cell
    while heap and expanded < max_expansiones:
        _, d, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        p = pos[u]
        if math.hypot(p[0] - T[0], p[1] - T[1]) <= tol and segment_free(occ, p, T):
            goal_state = u
            break
        for h2, end, curve, t_cost in prims[u[2]]:
            if curve is None:
                q = p
                cmax = 0.0
            else:
                pts = curve + p
                xi = np.round(pts[:, 0]).astype(int)
                yi = np.round(pts[:, 1]).astype(int)
                if xi.min() < 0 or yi.min() < 0 or xi.max() >= W or yi.max() >= H:
                    continue
                vals = rho_v[yi, xi]
                if not np.all(np.isfinite(vals)) or not polyline_free(occ, pts):
                    continue
                cmax = float(vals.max())
                q = pts[-1]
            v = (int(round(q[0])), int(round(q[1])), h2)
            if v in closed:
                continue
            nd = d + t_cost * (cmax + 1.0)
            if nd < g.get(v, math.inf):
                g[v] = nd
                prev[v] = (u, curve)
                pos[v] = q
                heapq.heappush(heap, (nd + heur(q[0], q[1]), nd, v))
    stats["estados_expandidos"] = expanded
    if goal_state is None:
        return AlgorithmResult("g2vd", "G2VD", False, None, time.perf_counter() - t0,
                               "La búsqueda en la retícula no llegó a T dentro del corredor", stats)
    segs = [np.array([T], float)]
    u = goal_state
    while prev[u] is not None:
        pu, curve = prev[u]
        if curve is not None:
            segs.append(curve[1:] + pos[pu])
        u = pu
    segs.append(np.array([S], float))
    raw = np.vstack(segs[::-1])
    keep = np.concatenate([[True], np.hypot(*np.diff(raw, axis=0).T) > 1e-9])
    raw = raw[keep]
    # --- suavizado QP (ec. 5-8)
    L = polyline_length(raw)
    step = max(paso_ref_m / mpp, 1.0)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(raw, axis=0).T))])
    ts = np.append(np.arange(0.0, L, step), L)
    ref = np.stack([np.interp(ts, s, raw[:, 0]), np.interp(ts, s, raw[:, 1])], axis=1)
    from common.geometry import bilinear
    di = bilinear(clear, ref)
    b = np.maximum(SQ2 / 2 * di - 0.0, 0.0)                 # r_c = 0 en el espacio de configuración
    path = _qp_box_admm(ref, b, w_s, w_r)
    smoothed = True
    if not polyline_free(occ, path):
        path, smoothed = ref, False
    stats["suavizado_qp"] = smoothed
    return AlgorithmResult("g2vd", "G2VD", True, path, time.perf_counter() - t0, "Ruta encontrada", stats)
