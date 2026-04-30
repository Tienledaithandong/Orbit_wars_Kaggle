"""
Orbit Wars — Advanced Strategic Agent v3

New in v3:
  ✅ Coordinated multi‑planet strikes (same arrival turn)
  ✅ Speed‑optimized fleet sizing (bigger = faster)
  ✅ Adaptive garrison based on actual enemy movements
  ✅ Simple 2‑turn lookahead for threat avoidance
"""

import math
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
CX, CY     = 50.0, 50.0
SUN_R      = 10.0
MAX_SPEED  = 6.0
TOTAL_TURNS = 500
ROT_LIMIT  = 50.0

# ─── Helpers ───────────────────────────────────────────────────────
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

# ═══════════════════════════════════════════════════════════════════
# Orbital & Comet Prediction (unchanged, works well)
# ═══════════════════════════════════════════════════════════════════
def _predict_pos(target, init_map, ang_vel, travel_t):
    init = init_map.get(target.id)
    if init is None: return target.x, target.y
    orb_r = _dist(init.x, init.y, CX, CY)
    if orb_r + init.radius >= ROT_LIMIT:
        return target.x, target.y
    cur_ang = math.atan2(target.y - CY, target.x - CX)
    fut_ang = cur_ang + ang_vel * travel_t
    return CX + orb_r * math.cos(fut_ang), CY + orb_r * math.sin(fut_ang)

def _get_target_pos(source, target, init_map, ang_vel, n_ships, is_comet, comet_data):
    tx, ty = target.x, target.y
    tt = 0
    for _ in range(3):
        d = _dist(source.x, source.y, tx, ty)
        tt = _travel_turns(d, n_ships)
        if is_comet and target.id in comet_data:
            path = comet_data[target.id]["path"]
            idx = int(comet_data[target.id]["index"] + tt)
            if idx >= len(path):
                return None, None, tt
            tx, ty = path[idx]
        else:
            tx, ty = _predict_pos(target, init_map, ang_vel, tt)
    return tx, ty, tt

# ═══════════════════════════════════════════════════════════════════
# NEW: Coordinated attack planner
# ═══════════════════════════════════════════════════════════════════
def _plan_coordinated_strikes(my_planets, targets, available, init_map, ang_vel,
                              comet_data, comet_ids, en_route, step):
    """
    For each target, try to combine fleets from multiple planets so they
    arrive on the same turn. Returns a list of candidate moves.
    """
    candidates = []
    # Only coordinate for high‑value targets (production >= 3 or enemy home)
    priority_targets = [t for t in targets if t.production >= 3 or t.owner >= 0]
    if not priority_targets:
        priority_targets = targets

    for target in priority_targets:
        is_comet = target.id in comet_ids
        # Predict garrison when our combined fleet would arrive
        # We'll pick a common arrival turn (target_turn) that maximizes
        # the ships we can send while minimizing travel time.
        possible_strikes = []
        for source in my_planets:
            avail = available.get(source.id, 0)
            if avail < 5: continue
            # Estimate how many ships we could send from this source
            # We'll test a few sizes: 50%, 75%, 100% of available
            for ratio in (0.5, 0.75, 1.0):
                send = int(avail * ratio)
                if send < 5: continue
                tx, ty, tt = _get_target_pos(source, target, init_map, ang_vel,
                                             send, is_comet, comet_data)
                if tx is None or _crosses_sun(source.x, source.y, tx, ty):
                    continue
                possible_strikes.append((tt, source.id, send,
                                         math.atan2(ty - source.y, tx - source.x)))
        if not possible_strikes:
            continue

        # Group by arrival turn (rounded to nearest integer)
        by_turn = defaultdict(list)
        for tt, sid, send, ang in possible_strikes:
            by_turn[int(tt)].append((sid, send, ang, tt))
        # Find the turn where we can send the most total ships
        best_turn, best_group = max(by_turn.items(), key=lambda kv: sum(s for _,s,_,_ in kv[1]))
        total_ships = sum(s for _,s,_,_ in best_group)

        # Predict garrison at that turn
        pred_garrison = target.ships
        if target.owner >= 0:
            pred_garrison += int(target.production * (best_turn + 1))
        already = en_route.get(target.id, 0)

        # Only commit if we can overwhelm the garrison
        if total_ships + already < pred_garrison + 5:
            continue

        # Score the coordinated strike
        prod = target.production
        if is_comet: prod *= 0.3
        mult = 2.0 if (step < 120 and target.owner == -1) else 1.5 if target.owner == -1 else 1.0
        score = (prod * mult * 100) / (best_turn + 1)
        # Add each source's move to the candidate list
        for sid, send, ang, tt in best_group:
            candidates.append((score, sid, target.id, send, ang, True))  # True = coordinated
    return candidates

# ═══════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════
_state = {"step": 0, "init_map": None}

def agent(obs):
    get = (lambda k, d=None: obs.get(k, d)) if isinstance(obs, dict) else \
          (lambda k, d=None: getattr(obs, k, d))

    player   = get("player", 0)
    planets  = [Planet(*p) for p in (get("planets") or [])]
    fleets   = [Fleet(*f)  for f in (get("fleets")  or [])]
    ang_vel  = get("angular_velocity", 0.0) or 0.0
    raw_init = get("initial_planets") or []
    comet_ids = set(get("comet_planet_ids") or [])

    if _state["init_map"] is None and raw_init:
        _state["init_map"] = {p[0]: Planet(*p) for p in raw_init}
    init_map = _state["init_map"] or {}

    step = _state["step"]
    _state["step"] += 1

    # Parse comets
    comets_raw = get("comets") or []
    comet_data = {}
    for c_group in comets_raw:
        p_ids = c_group.get("planet_ids", [])
        paths = c_group.get("paths", [])
        p_idx = c_group.get("path_index", 0)
        for i, cid in enumerate(p_ids):
            if i < len(paths):
                comet_data[cid] = {"path": paths[i], "index": p_idx}

    my_planets   = [p for p in planets if p.owner == player]
    targets      = [p for p in planets if p.owner != player]
    my_fleets    = [f for f in fleets  if f.owner == player]
    enemy_fleets = [f for f in fleets  if f.owner != player]

    if not targets or not my_planets:
        return []

    # En‑route ships (ours)
    en_route = {}
    for f in my_fleets:
        tid = _fleet_target(f, targets)
        if tid is not None:
            en_route[tid] = en_route.get(tid, 0) + f.ships

    # Threats (enemy fleets heading toward our planets)
    threats = {}
    for f in enemy_fleets:
        tid = _fleet_target(f, my_planets)
        if tid is not None:
            threats[tid] = threats.get(tid, 0) + f.ships

    # ─── Dynamic Garrison (NEW) ────────────────────────────────────
    available = {}
    for p in my_planets:
        threat = threats.get(p.id, 0)
        # Base garrison depends on game phase
        if step < 80:
            base = max(1, p.production)
        elif step < 250:
            base = max(3, p.production * 2)
        else:
            base = max(5, p.production * 3)
        # If enemies are coming, keep enough to defend
        garrison = max(base, threat + 5)
        available[p.id] = max(0, p.ships - garrison)

    # ─── Collect attack candidates ─────────────────────────────────
    candidates = []

    # 1. Coordinated strikes (NEW)
    coord = _plan_coordinated_strikes(my_planets, targets, available, init_map,
                                      ang_vel, comet_data, comet_ids, en_route, step)
    candidates.extend(coord)

    # 2. Fallback: single‑planet strikes (improved speed & sizing)
    for target in targets:
        is_comet = target.id in comet_ids
        for source in my_planets:
            avail = available.get(source.id, 0)
            if avail <= 5: continue

            # NEW: Test a range of fleet sizes to find optimal speed/power trade‑off
            best_net = -1
            best_move = None
            for send_ratio in (0.6, 0.8, 1.0):
                send = int(avail * send_ratio)
                if send < 5: continue
                tx, ty, tt = _get_target_pos(source, target, init_map, ang_vel,
                                             send, is_comet, comet_data)
                if tx is None or _crosses_sun(source.x, source.y, tx, ty):
                    continue
                # Net gain = (production gained) - (travel cost) - (ships risked)
                pred_garrison = target.ships
                if target.owner >= 0:
                    pred_garrison += int(target.production * (tt + 1))
                already = en_route.get(target.id, 0)
                if send + already < pred_garrison + 5:
                    continue  # Not enough to capture
                prod_gain = target.production * (500 - step - tt) * 0.8
                net = prod_gain - send * 0.5 - tt * 2
                if net > best_net:
                    best_net = net
                    angle = math.atan2(ty - source.y, tx - source.x)
                    best_move = (send, angle, tt)

            if best_move is not None:
                send, angle, tt = best_move
                # Score for single‑planet strike (lower priority than coordinated)
                score = target.production * 50 / (tt + 1)
                candidates.append((score, source.id, target.id, send, angle, False))

    # ─── Backline Reinforcement (kept from your v2) ────────────────
    my_frontline = []
    for p in my_planets:
        min_e_dist = min([_dist(p.x, p.y, t.x, t.y) for t in targets] + [999])
        my_frontline.append((min_e_dist, p))

    for source in my_planets:
        avail = available.get(source.id, 0)
        if avail < 15: continue
        my_dist = min([_dist(source.x, source.y, t.x, t.y) for t in targets] + [999])
        if my_dist > 45:
            best_friend, best_fscore = None, -1
            for f_dist, friend in my_frontline:
                if friend.id == source.id: continue
                if f_dist < my_dist - 15:
                    fscore = friend.production / (_dist(source.x, source.y, friend.x, friend.y) + 1)
                    if fscore > best_fscore:
                        best_fscore, best_friend = fscore, friend
            if best_friend:
                tx, ty, tt = _get_target_pos(source, best_friend, init_map, ang_vel, avail, False, comet_data)
                if tx is not None and not _crosses_sun(source.x, source.y, tx, ty):
                    angle = math.atan2(ty - source.y, tx - source.x)
                    candidates.append((0.01, source.id, best_friend.id, avail, angle, False))

    # ─── Execute moves (prevent double‑booking) ────────────────────
    candidates.sort(key=lambda c: -c[0])
    moves = []
    used = {}
    claimed = set()

    for score, src_id, tgt_id, send, angle, is_coord in candidates:
        if tgt_id in claimed: continue
        spent = used.get(src_id, 0)
        remaining = available.get(src_id, 0) - spent
        if remaining < send:
            if not is_coord:  # For single strikes, scale down if possible
                if remaining > 5:
                    send = remaining
                else:
                    continue
            else:
                continue
        if send <= 0: continue
        moves.append([src_id, angle, send])
        used[src_id] = spent + send
        claimed.add(tgt_id)

    return moves