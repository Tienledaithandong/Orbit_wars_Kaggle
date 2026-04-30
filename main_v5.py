"""
Orbit Wars – v5 Champion
Wave Sync + Over‑send + Dynamic Garrison + Coordinated Strikes + Backline Reinforcement
"""

import math
import sys
import traceback
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CX, CY = 50.0, 50.0
SUN_R = 10.0
MAX_SPEED = 6.0
TOTAL_TURNS = 500
ROT_LIMIT = 50.0

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def _fleet_speed(n):
    if n <= 1: return 1.0
    ratio = math.log(n) / math.log(1000)
    return 1.0 + (MAX_SPEED - 1.0) * min(1.0, ratio ** 1.5)

def _travel_turns(d, n):
    s = _fleet_speed(max(1, n))
    return d / s if s > 0 else 1e9

def _crosses_sun(x1, y1, x2, y2, margin=1.5):
    dx, dy = x2 - x1, y2 - y1
    a = dx*dx + dy*dy
    if a < 1e-9:
        return _dist(x1, y1, CX, CY) < SUN_R + margin
    fx, fy = x1 - CX, y1 - CY
    b = 2 * (fx*dx + fy*dy)
    c = fx*fx + fy*fy - (SUN_R + margin)**2
    disc = b*b - 4*a*c
    if disc < 0:
        return False
    disc = math.sqrt(disc)
    t1 = (-b - disc) / (2*a)
    t2 = (-b + disc) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)

# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────
def _predict_pos(target, init_map, ang_vel, travel_t):
    init = init_map.get(target.id)
    if init is None:
        return target.x, target.y
    orb_r = _dist(init.x, init.y, CX, CY)
    if orb_r + init.radius >= ROT_LIMIT:
        return target.x, target.y
    cur_ang = math.atan2(target.y - CY, target.x - CX)
    fut_ang = cur_ang + ang_vel * travel_t
    return CX + orb_r * math.cos(fut_ang), CY + orb_r * math.sin(fut_ang)

def _intercept(source, target, n_ships, init_map, ang_vel, comet_data, is_comet):
    tx, ty = target.x, target.y
    for _ in range(5):
        d = _dist(source.x, source.y, tx, ty)
        tt = _travel_turns(d, n_ships)
        if is_comet:
            data = comet_data.get(target.id)
            if not data:
                return None, None, tt
            idx = int(data["index"] + tt)
            if idx >= len(data["path"]):
                return None, None, tt
            tx, ty = data["path"][idx]
        else:
            tx, ty = _predict_pos(target, init_map, ang_vel, tt)
    return tx, ty, tt

# ─────────────────────────────────────────────────────────────────────────────
# Fleet targeting
# ─────────────────────────────────────────────────────────────────────────────
def _fleet_target(f, planets):
    dx, dy = math.cos(f.angle), math.sin(f.angle)
    best_id, best_t = None, 1e9
    for p in planets:
        vx, vy = p.x - f.x, p.y - f.y
        t = vx*dx + vy*dy
        if t <= 0: continue
        perp = abs(vx*dy - vy*dx)
        if perp < p.radius + 2.0 and t < best_t:
            best_t, best_id = t, p.id
    return best_id

# ─────────────────────────────────────────────────────────────────────────────
# Wave Planner (core of v5)
# ─────────────────────────────────────────────────────────────────────────────
def _plan_wave(target, my_planets, available, init_map, ang_vel, comet_data, comet_ids):
    arrivals = []
    for src in my_planets:
        avail = available.get(src.id, 0)
        if avail < 8:
            continue
        send = min(avail, max(12, target.ships + 8))
        is_comet = target.id in comet_ids
        res = _intercept(src, target, send, init_map, ang_vel, comet_data, is_comet)
        if res[0] is None or _crosses_sun(src.x, src.y, res[0], res[1]):
            continue
        tx, ty, tt = res
        angle = math.atan2(ty - src.y, tx - src.x)
        arrivals.append((src.id, send, angle, int(round(tt))))

    if len(arrivals) < 2:
        return []

    # Group by arrival turn (±1)
    groups = defaultdict(list)
    for sid, send, ang, tt in arrivals:
        for key in (tt-1, tt, tt+1):
            groups[key].append((sid, send, ang))

    best_group = max(groups.values(), key=lambda g: sum(s for _, s, _ in g), default=[])
    total = sum(s for _, s, _ in best_group)
    if total < target.ships * 1.2:
        return []
    return [(sid, s, ang) for sid, s, ang in best_group]

# ─────────────────────────────────────────────────────────────────────────────
# Over‑send Optimiser
# ─────────────────────────────────────────────────────────────────────────────
def _optimize_send(src, target, base_needed, max_avail, step, init_map, ang_vel, comet_data, is_comet):
    best_send = base_needed
    best_score = -1e9
    for mult in (1.0, 1.3, 1.6, 2.0):
        send = min(max_avail, int(base_needed * mult))
        if send < base_needed:
            continue
        res = _intercept(src, target, send, init_map, ang_vel, comet_data, is_comet)
        if res[0] is None or _crosses_sun(src.x, src.y, res[0], res[1]):
            continue
        tx, ty, tt = res
        remaining = TOTAL_TURNS - (step + tt)
        if remaining <= 0:
            continue
        net = target.production * remaining - send * 0.25
        if net > best_score:
            best_score = net
            best_send = send
    return best_send

# ─────────────────────────────────────────────────────────────────────────────
# Coordinated Multi‑planet Strike (enhanced v3)
# ─────────────────────────────────────────────────────────────────────────────
def _plan_coordinated(target, my_planets, available, init_map, ang_vel, comet_data, comet_ids, en_route):
    """Find best combination of planets to hit target on same turn."""
    strikes = []
    for src in my_planets:
        avail = available.get(src.id, 0)
        if avail < 5:
            continue
        # Try different fleet sizes
        for ratio in (0.6, 0.8, 1.0):
            send = int(avail * ratio)
            if send < 5:
                continue
            is_comet = target.id in comet_ids
            res = _intercept(src, target, send, init_map, ang_vel, comet_data, is_comet)
            if res[0] is None or _crosses_sun(src.x, src.y, res[0], res[1]):
                continue
            tx, ty, tt = res
            angle = math.atan2(ty - src.y, tx - src.x)
            strikes.append((tt, src.id, send, angle))

    if len(strikes) < 2:
        return []

    # Group by rounded turn
    by_turn = defaultdict(list)
    for tt, sid, send, ang in strikes:
        by_turn[int(tt)].append((sid, send, ang, tt))

    best_turn, best_group = max(by_turn.items(), key=lambda kv: sum(s for _, s, _, _ in kv[1]))
    total = sum(s for _, s, _, _ in best_group)
    already = en_route.get(target.id, 0)
    pred_garrison = target.ships + (target.production * best_turn if target.owner >= 0 else 0)

    if total + already < pred_garrison + 5:
        return []

    return [(sid, send, ang) for sid, send, ang, _ in best_group]

# ─────────────────────────────────────────────────────────────────────────────
# Backline Reinforcement
# ─────────────────────────────────────────────────────────────────────────────
def _reinforce_backline(my_planets, targets, available, init_map, ang_vel, comet_data):
    """Send idle ships from safe back planets to frontline."""
    if len(my_planets) < 2:
        return []
    # Find frontline planet (closest to enemy)
    front_dist = {}
    for p in my_planets:
        min_d = min([_dist(p.x, p.y, t.x, t.y) for t in targets] + [999])
        front_dist[p.id] = min_d
    front = min(my_planets, key=lambda p: front_dist[p.id])

    candidates = []
    for src in my_planets:
        if src.id == front.id:
            continue
        # Only reinforce if source is significantly further from enemy
        if front_dist[src.id] < front_dist[front.id] * 1.3:
            continue
        avail = available.get(src.id, 0)
        if avail < 15:
            continue
        send = int(avail * 0.7)
        if send < 10:
            continue
        res = _intercept(src, front, send, init_map, ang_vel, comet_data, False)
        if res[0] is None or _crosses_sun(src.x, src.y, res[0], res[1]):
            continue
        tx, ty, tt = res
        if tt > 35:
            continue
        angle = math.atan2(ty - src.y, tx - src.x)
        candidates.append((src.id, send, angle, tt))

    # Sort by shortest travel time
    candidates.sort(key=lambda x: x[3])
    # Take up to 2 best reinforcements
    return [(sid, send, ang) for sid, send, ang, _ in candidates[:2]]

# ─────────────────────────────────────────────────────────────────────────────
# Agent Entry
# ─────────────────────────────────────────────────────────────────────────────
_state = {"step": 0, "init_map": None}

def agent(obs):
    try:
        return _agent_impl(obs)
    except Exception as e:
        print(f"[v5 ERROR] {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return _fallback_agent(obs)

def _fallback_agent(obs):
    get = (lambda k, d=None: obs.get(k, d)) if isinstance(obs, dict) else \
          (lambda k, d=None: getattr(obs, k, d))
    player = get("player", 0)
    planets = [Planet(*p) for p in (get("planets") or [])]
    my = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    if not my or not targets:
        return []
    src = max(my, key=lambda p: p.ships)
    tgt = min(targets, key=lambda t: _dist(src.x, src.y, t.x, t.y))
    angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
    send = min(src.ships - 5, tgt.ships + 10)
    if send <= 0:
        return []
    return [[src.id, angle, send]]

def _agent_impl(obs):
    get = (lambda k, d=None: obs.get(k, d)) if isinstance(obs, dict) else \
          (lambda k, d=None: getattr(obs, k, d))

    player = get("player", 0)
    step = _state["step"]
    _state["step"] += 1

    raw_planets = get("planets") or []
    raw_fleets  = get("fleets")  or []
    ang_vel     = get("angular_velocity", 0.0) or 0.0
    raw_init    = get("initial_planets") or []
    comets_raw  = get("comets") or []
    comet_ids   = set(get("comet_planet_ids") or [])

    try:
        from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
    except ImportError:
        from collections import namedtuple
        Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
        Fleet  = namedtuple("Fleet",  ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])

    planets = [Planet(*p) for p in raw_planets]
    fleets  = [Fleet(*f)  for f in raw_fleets]

    if _state["init_map"] is None and raw_init:
        _state["init_map"] = {p[0]: Planet(*p) for p in raw_init}
    init_map = _state["init_map"] or {}

    comet_data = {}
    for c in comets_raw:
        pids = c.get("planet_ids", [])
        paths = c.get("paths", [])
        idx = c.get("path_index", 0)
        for i, pid in enumerate(pids):
            if i < len(paths):
                comet_data[pid] = {"path": paths[i], "index": idx}

    my_planets = [p for p in planets if p.owner == player]
    if not my_planets:
        return []

    remaining_steps = max(1, TOTAL_TURNS - step)
    is_late = remaining_steps < 60

    # ── Threat detection (incoming enemy fleets) ─────────────────────────────
    threats = {}
    enemy_fleets = [f for f in fleets if f.owner != player and f.owner != -1]
    for f in enemy_fleets:
        tid = _fleet_target(f, my_planets)
        if tid is not None:
            threats[tid] = threats.get(tid, 0) + f.ships

    # ── En‑route friendly ships ──────────────────────────────────────────────
    en_route = {}
    my_fleets = [f for f in fleets if f.owner == player]
    for f in my_fleets:
        tid = _fleet_target(f, planets)
        if tid is not None:
            en_route[tid] = en_route.get(tid, 0) + f.ships

    # ── Available ships (smart garrison) ────────────────────────────────────
    available = {}
    for p in my_planets:
        # Base reserve: 15% for safety
        reserve = int(p.ships * 0.15)
        # If enemies are coming, reserve enough to defend
        threat = threats.get(p.id, 0)
        if threat > 0:
            reserve = max(reserve, threat + 5)
        available[p.id] = max(0, p.ships - reserve)

    # ── Target sorting ───────────────────────────────────────────────────────
    enemy_planets = [p for p in planets if p.owner != player and p.owner != -1]
    neutral_planets = [p for p in planets if p.owner == -1]

    def target_score(t):
        # Simple production / distance score
        d = min([_dist(p.x, p.y, t.x, t.y) for p in my_planets], default=999)
        prod = t.production
        if t.id in comet_ids:
            prod *= 0.2  # comets are low value
        if t.owner != -1:
            prod *= 1.8  # enemy planets are high priority
        return prod / (d + 1)

    all_targets = enemy_planets + neutral_planets
    all_targets.sort(key=target_score, reverse=True)

    moves = []
    used = defaultdict(int)
    taken_targets = set()

    # ── Phase 1: Wave Sync on top 2 high‑value targets ──────────────────────
    if not is_late:
        for tgt in enemy_planets[:2]:
            if tgt.id in taken_targets:
                continue
            wave = _plan_wave(tgt, my_planets, available, init_map, ang_vel, comet_data, comet_ids)
            for sid, send, ang in wave:
                avail = available[sid] - used[sid]
                if send <= avail:
                    moves.append([sid, ang, send])
                    used[sid] += send
            if wave:
                taken_targets.add(tgt.id)

    # ── Phase 2: Coordinated strikes on remaining enemy planets ─────────────
    if not is_late:
        for tgt in enemy_planets[2:]:
            if tgt.id in taken_targets:
                continue
            coord = _plan_coordinated(tgt, my_planets, available, init_map, ang_vel, comet_data, comet_ids, en_route)
            for sid, send, ang in coord:
                avail = available[sid] - used[sid]
                if send <= avail:
                    moves.append([sid, ang, send])
                    used[sid] += send
            if coord:
                taken_targets.add(tgt.id)

    # ── Phase 3: Over‑send greedy attacks on any remaining targets ───────────
    for tgt in all_targets:
        if tgt.id in taken_targets:
            continue
        for src in my_planets:
            avail = available[src.id] - used[src.id]
            if avail < 8:
                continue
            d = _dist(src.x, src.y, tgt.x, tgt.y)
            if d > 140:
                continue
            base_needed = tgt.ships + 1
            if tgt.owner != -1:
                base_needed += tgt.production * 3  # rough arrival time
            is_comet = tgt.id in comet_ids
            send = _optimize_send(src, tgt, base_needed, avail, step, init_map, ang_vel, comet_data, is_comet)
            if send <= 0:
                continue
            res = _intercept(src, tgt, send, init_map, ang_vel, comet_data, is_comet)
            if res[0] is None or _crosses_sun(src.x, src.y, res[0], res[1]):
                continue
            tx, ty, tt = res
            angle = math.atan2(ty - src.y, tx - src.x)
            moves.append([src.id, angle, send])
            used[src.id] += send
            taken_targets.add(tgt.id)
            break  # one attack per source in this phase

    # ── Phase 4: Backline reinforcement ──────────────────────────────────────
    if not is_late and enemy_planets:
        reinforcements = _reinforce_backline(my_planets, enemy_planets, available, init_map, ang_vel, comet_data)
        for sid, send, ang in reinforcements:
            avail = available[sid] - used[sid]
            if send <= avail:
                moves.append([sid, ang, send])
                used[sid] += send

    return moves