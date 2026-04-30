"""
Orbit Wars — Advanced Strategic Agent v2

Strategy layers:
  1. Sun collision avoidance   – skip paths that cross the sun
  2. Travel-time compensation  – add target production × travel turns
  3. Decisive Strikes          – forces full-fleet attacks; prevents trickle-feeding
  4. True Comet Tracking       – utilizes full orbit paths to predict comet positions
  5. Backline Reinforcement    – forwards idle ships from safe planets to the front
  6. Multi-phase strategy      – expand → consolidate → defend
"""

import math

# ─── Named-tuple imports (with fallback) ───────────────────────────
try:
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
except ImportError:
    from collections import namedtuple
    Planet = namedtuple("Planet", "id owner x y radius ships production")
    Fleet  = namedtuple("Fleet",  "id owner x y angle from_planet_id ships")

# ═══════════════════════════════════════════════════════════════════
# Constants & Helpers
# ═══════════════════════════════════════════════════════════════════
CX, CY     = 50.0, 50.0   # Sun center
SUN_R      = 10.0         # Sun radius
MAX_SPEED  = 6.0          # Maximum fleet speed
TOTAL_TURNS = 500
ROT_LIMIT  = 50.0         # orbital_radius + planet_radius < this → orbiting

def _dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def _fleet_speed(n):
    if n <= 1: return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * min(1.0, (math.log(n) / math.log(1000)) ** 1.5)

def _travel_turns(d, n):
    s = _fleet_speed(max(1, n))
    return d / s if s > 0 else 1e9

def _seg_pt_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 < 1e-12: return _dist(px, py, ax, ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len2))
    return _dist(px, py, ax + t * dx, ay + t * dy)

def _crosses_sun(x1, y1, x2, y2, margin=1.5):
    return _seg_pt_dist(CX, CY, x1, y1, x2, y2) < SUN_R + margin

def _fleet_target(f, planets):
    dx, dy = math.cos(f.angle), math.sin(f.angle)
    best_id, best_t = None, 1e9
    for p in planets:
        vx, vy = p.x - f.x, p.y - f.y
        t = vx * dx + vy * dy
        if t <= 0: continue
        perp = abs(vx * dy - vy * dx)
        if perp < p.radius + 2.0 and t < best_t:
            best_t, best_id = t, p.id
    return best_id

# ═══════════════════════════════════════════════════════════════════
# Orbital & Comet Prediction
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
    for _ in range(3): # Iterative lead
        d = _dist(source.x, source.y, tx, ty)
        tt = _travel_turns(d, n_ships)
        
        if is_comet and target.id in comet_data:
            path = comet_data[target.id]["path"]
            idx = int(comet_data[target.id]["index"] + tt)
            if idx >= len(path):
                return None, None, tt # Comet leaves the board before we arrive
            tx, ty = path[idx]
        else:
            tx, ty = _predict_pos(target, init_map, ang_vel, tt)
    return tx, ty, tt

# ═══════════════════════════════════════════════════════════════════
# Agent entry point
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

    # ── Parse Comets ───────────────────────────────────────────────
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

    if not targets or not my_planets: return []

    en_route = {}
    for f in my_fleets:
        tid = _fleet_target(f, targets)
        if tid is not None:
            en_route[tid] = en_route.get(tid, 0) + f.ships

    threats = {}
    for f in enemy_fleets:
        tid = _fleet_target(f, my_planets)
        if tid is not None:
            threats[tid] = threats.get(tid, 0) + f.ships

    # ── Available ships (Garrison Logic) ───────────────────────────
    available = {}
    for p in my_planets:
        threat = threats.get(p.id, 0)
        if step < 80:     base_g = max(1, p.production)
        elif step < 250:  base_g = max(3, p.production * 2)
        else:             base_g = max(5, p.production * 3)
        
        garrison = max(base_g, threat + 1) if threat else base_g
        available[p.id] = max(0, p.ships - garrison)

    # ── Evaluate Attacks ───────────────────────────────────────────
    candidates = []
    
    for target in targets:
        is_comet = target.id in comet_ids

        for source in my_planets:
            avail = available.get(source.id, 0)
            if avail <= 0: continue

            rough_ships = min(avail, max(1, target.ships + 1))
            tx, ty, tt = _get_target_pos(source, target, init_map, ang_vel, rough_ships, is_comet, comet_data)
            
            if tx is None or _crosses_sun(source.x, source.y, tx, ty):
                continue
            
            pred_dist = _dist(source.x, source.y, tx, ty)
            if pred_dist < 0.5: continue

            angle = math.atan2(ty - source.y, tx - source.x)

            # Garrison Prediction
            pred_garrison = target.ships
            if target.owner >= 0:
                pred_garrison += int(target.production * (tt + 1))

            already = en_route.get(target.id, 0)
            
            # If we've already sent enough to crush them, skip.
            if already > pred_garrison:
                continue
                
            # DO NOT subtract 'already' from needed. This forces a single massive fleet,
            # preventing trickle-feed suicide against planetary regeneration.
            needed = pred_garrison + 1

            if avail < needed:
                continue

            send = min(avail, int(needed * 1.15) + 3) # Send a bit extra for speed

            # Value Scoring
            prod = target.production
            if is_comet: prod *= 0.1 # Comets are mostly distractions
            
            mult = 2.5 if (step < 100 and target.owner == -1) else 1.2 if target.owner == -1 else 1.0
            
            # Penalize travel time heavily to prioritize fast strikes
            score = (prod * mult * 100.0) / (needed + tt * 2.0 + 1)
            candidates.append((score, source.id, target.id, send, angle))

    # ── Backline Reinforcement ─────────────────────────────────────
    # If a friendly planet has excess ships but is too far to attack effectively,
    # forward those ships to the nearest friendly frontline planet.
    my_frontline = []
    for p in my_planets:
        min_e_dist = min([_dist(p.x, p.y, t.x, t.y) for t in targets] + [999])
        my_frontline.append((min_e_dist, p))

    for source in my_planets:
        avail = available.get(source.id, 0)
        if avail < 15: continue
        
        my_dist = min([_dist(source.x, source.y, t.x, t.y) for t in targets] + [999])
        
        # Am I in the backline?
        if my_dist > 45:
            best_friend, best_score = None, -1
            for f_dist, friend in my_frontline:
                if friend.id == source.id: continue
                # Friend must be significantly closer to the enemy
                if f_dist < my_dist - 15:
                    f_score = friend.production / (_dist(source.x, source.y, friend.x, friend.y) + 1)
                    if f_score > best_score:
                        best_score, best_friend = f_score, friend
            
            if best_friend:
                tx, ty, tt = _get_target_pos(source, best_friend, init_map, ang_vel, avail, False, comet_data)
                if tx is not None and not _crosses_sun(source.x, source.y, tx, ty):
                    angle = math.atan2(ty - source.y, tx - source.x)
                    r_score = 0.01 * best_score # Extremely low score so attacks always take priority
                    candidates.append((r_score, source.id, best_friend.id, avail, angle))

    # ── Execute Moves ──────────────────────────────────────────────
    candidates.sort(key=lambda c: -c[0])

    moves = []
    used  = {}
    claimed = set()

    for score, src_id, tgt_id, send, angle in candidates:
        if tgt_id in claimed: continue
        
        spent = used.get(src_id, 0)
        remaining = available.get(src_id, 0) - spent
        
        if remaining < send:
            # Only downgrade the send amount if it's a reinforcement move.
            # If it's an attack, lowering it means we won't pierce the garrison!
            if score < 0.1: send = remaining 
            else: continue
                
        if send <= 0: continue
            
        moves.append([src_id, angle, send])
        used[src_id] = spent + send
        claimed.add(tgt_id)

    return moves