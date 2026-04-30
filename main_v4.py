"""
Orbit Wars – v4 Stable: Wave Sync + Speed Meta + Safe Fallback
"""

import math
import sys
import traceback
from collections import defaultdict

try:
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
except ImportError:
    from collections import namedtuple
    Planet = namedtuple("Planet", "id owner x y radius ships production")
    Fleet  = namedtuple("Fleet",  "id owner x y angle from_planet_id ships")

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
CX, CY       = 50.0, 50.0
SUN_R        = 10.0
MAX_SPEED    = 6.0
TOTAL_TURNS  = 500
ROT_LIMIT    = 50.0

def _dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def _fleet_speed(n):
    if n <= 1: return 1.0
    ratio = math.log(n) / math.log(1000)
    return 1.0 + (MAX_SPEED - 1.0) * min(1.0, ratio ** 1.5)

def _travel_turns(d, n):
    s = _fleet_speed(max(1, n))
    return d / s if s > 0 else 1e9

def _seg_pt_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    len2 = dx*dx + dy*dy
    if len2 < 1e-12: return _dist(px, py, ax, ay)
    t = max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy) / len2))
    return _dist(px, py, ax + t*dx, ay + t*dy)

def _crosses_sun(x1, y1, x2, y2, margin=1.5):
    return _seg_pt_dist(CX, CY, x1, y1, x2, y2) < SUN_R + margin

# ═══════════════════════════════════════════════════════════════════
# High‑Precision Interception
# ═══════════════════════════════════════════════════════════════════
def _predict_pos_iter(target, init_map, ang_vel, travel_t):
    init = init_map.get(target.id)
    if init is None: return target.x, target.y
    orb_r = _dist(init.x, init.y, CX, CY)
    if orb_r + init.radius >= ROT_LIMIT:
        return target.x, target.y
    cur_ang = math.atan2(target.y - CY, target.x - CX)
    fut_ang = cur_ang + ang_vel * travel_t
    return CX + orb_r * math.cos(fut_ang), CY + orb_r * math.sin(fut_ang)

def _intercept(source, target, init_map, ang_vel, n_ships, is_comet, comet_data):
    tx, ty = target.x, target.y
    tt = 0
    for _ in range(5):
        d = _dist(source.x, source.y, tx, ty)
        tt = _travel_turns(d, n_ships)
        if is_comet and target.id in comet_data:
            path = comet_data[target.id]["path"]
            idx = int(comet_data[target.id]["index"] + tt)
            if idx >= len(path):
                return None, None, tt
            tx, ty = path[idx]
        else:
            tx, ty = _predict_pos_iter(target, init_map, ang_vel, tt)
    return tx, ty, tt

# ═══════════════════════════════════════════════════════════════════
# Wave Planner (simplified but robust)
# ═══════════════════════════════════════════════════════════════════
def _plan_wave(target, my_planets, available, init_map, ang_vel, comet_data, comet_ids):
    arrivals = []
    for src in my_planets:
        avail = available.get(src.id, 0)
        if avail < 5: continue
        # Try a moderate fleet size
        send = min(avail, max(10, target.ships + 5))
        tx, ty, tt = _intercept(src, target, init_map, ang_vel, send,
                                target.id in comet_ids, comet_data)
        if tx is None or _crosses_sun(src.x, src.y, tx, ty):
            continue
        angle = math.atan2(ty - src.y, tx - src.x)
        arrivals.append((src.id, send, angle, tt))
    if len(arrivals) < 2:
        return []
    # Group by rounded arrival turn (allow ±1 tolerance)
    groups = defaultdict(list)
    for sid, send, ang, tt in arrivals:
        groups[int(tt)].append((sid, send, ang))
    best_group = max(groups.values(), key=lambda g: sum(s for _,s,_ in g), default=[])
    # Only return if total ships > target garrison * 1.5
    total = sum(s for _,s,_ in best_group)
    if total < target.ships * 1.2:
        return []
    return [[sid, send, ang] for sid, send, ang in best_group]

# ═══════════════════════════════════════════════════════════════════
# Over‑send Optimizer
# ═══════════════════════════════════════════════════════════════════
def _optimize_send(source, target, base_needed, max_avail, step, init_map, ang_vel, comet_data, is_comet):
    best_send = base_needed
    best_score = -1e9
    for mult in (1.0, 1.3, 1.6, 2.0):
        send = min(max_avail, int(base_needed * mult))
        if send < base_needed: continue
        tx, ty, tt = _intercept(source, target, init_map, ang_vel, send, is_comet, comet_data)
        if tx is None or _crosses_sun(source.x, source.y, tx, ty):
            continue
        remaining = TOTAL_TURNS - (step + tt)
        if remaining <= 0: continue
        net = target.production * remaining - send * 0.25
        if net > best_score:
            best_score = net
            best_send = send
    return best_send

# ═══════════════════════════════════════════════════════════════════
# Agent Entry Point with Safety Fallback
# ═══════════════════════════════════════════════════════════════════
_state = {"step": 0, "init_map": None}

def agent(obs):
    try:
        return _agent_impl(obs)
    except Exception as e:
        # In lỗi ra stderr để debug
        print(f"[v4 ERROR] {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Fallback: gửi tất cả tàu từ hành tinh mạnh nhất đến mục tiêu gần nhất
        return _fallback_agent(obs)

def _fallback_agent(obs):
    get = (lambda k, d=None: obs.get(k, d)) if isinstance(obs, dict) else \
          (lambda k, d=None: getattr(obs, k, d))
    player = get("player", 0)
    planets = [Planet(*p) for p in (get("planets") or [])]
    my = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    if not my or not targets: return []
    src = max(my, key=lambda p: p.ships)
    tgt = min(targets, key=lambda t: _dist(src.x, src.y, t.x, t.y))
    angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
    send = min(src.ships - 5, tgt.ships + 10)
    if send <= 0: return []
    return [[src.id, angle, send]]

def _agent_impl(obs):
    get = (lambda k, d=None: obs.get(k, d)) if isinstance(obs, dict) else \
          (lambda k, d=None: getattr(obs, k, d))

    player   = get("player", 0)
    planets  = [Planet(*p) for p in (get("planets") or [])]
    fleets   = [Fleet(*f)  for f in (get("fleets")  or [])]
    ang_vel  = get("angular_velocity", 0.0) or 0.0
    raw_init = get("initial_planets") or []
    comet_ids = set(get("comet_planet_ids") or [])
    step = _state["step"]
    _state["step"] += 1

    if _state["init_map"] is None and raw_init:
        _state["init_map"] = {p[0]: Planet(*p) for p in raw_init}
    init_map = _state["init_map"] or {}

    comets_raw = get("comets") or []
    comet_data = {}
    for c_group in comets_raw:
        p_ids = c_group.get("planet_ids", [])
        paths = c_group.get("paths", [])
        p_idx = c_group.get("path_index", 0)
        for i, cid in enumerate(p_ids):
            if i < len(paths):
                comet_data[cid] = {"path": paths[i], "index": p_idx}

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    if not my_planets or not targets:
        return []

    # ─── Phòng thủ: giữ 15% tàu ────────────────────────────────────
    available = {p.id: int(p.ships * 0.85) for p in my_planets}

    # ─── Sắp xếp mục tiêu theo giá trị ─────────────────────────────
    def target_value(t):
        dist = min([_dist(p.x, p.y, t.x, t.y) for p in my_planets], default=999)
        return t.production / (dist + 1)
    targets.sort(key=target_value, reverse=True)

    moves = []
    used = defaultdict(int)

    # ─── Wave Sync cho 2 mục tiêu hàng đầu ─────────────────────────
    for target in targets[:2]:
        wave = _plan_wave(target, my_planets, available, init_map, ang_vel, comet_data, comet_ids)
        for src_id, send, angle in wave:
            avail = available.get(src_id, 0) - used[src_id]
            if send <= avail:
                moves.append([src_id, angle, send])
                used[src_id] += send

    # ─── Greedy Over‑send cho các mục tiêu còn lại ─────────────────
    for target in targets:
        for src in my_planets:
            avail = available.get(src.id, 0) - used[src.id]
            if avail < 5: continue
            is_comet = target.id in comet_ids
            base = target.ships + 1
            if target.owner >= 0:
                base += target.production * 3
            send = _optimize_send(src, target, base, avail, step, init_map, ang_vel, comet_data, is_comet)
            if send <= 0: continue
            tx, ty, _ = _intercept(src, target, init_map, ang_vel, send, is_comet, comet_data)
            if tx is None or _crosses_sun(src.x, src.y, tx, ty):
                continue
            angle = math.atan2(ty - src.y, tx - src.x)
            moves.append([src.id, angle, send])
            used[src.id] += send
            break  # Mỗi target chỉ đánh từ 1 nguồn trong pha này

    return moves