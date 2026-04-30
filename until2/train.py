# train.py
import cma
import numpy as np
import json
import sys
import os
import kaggle_environments

# ------------------------------------------------------------
# Import your learning agent and its default config
# ------------------------------------------------------------
from agent import agent as learning_agent, DEFAULT_CFG

# ------------------------------------------------------------
# Baseline agent (v7.5 without opponent prediction)
# ------------------------------------------------------------
def baseline_agent(obs, config=None):
    cfg = DEFAULT_CFG.copy()
    cfg["OPP_PREDICT_HORIZON"] = 0
    if config is None:
        config = {}
    config["weights"] = cfg
    return learning_agent(obs, config)

# ------------------------------------------------------------
# Map pool – varied Orbit Wars layouts
# ------------------------------------------------------------
MAP_POOL = [
    "default",
    "ring",
    "spiral",
    "four_corners",
    "two_player_static",
    "four_player_rotating",
    "comet_map",
    "dense_clusters",
    "many_neutrals",
    "big_ring"
]

# ------------------------------------------------------------
# Which keys are optimizable (all numeric non‑boolean)
# ------------------------------------------------------------
OPTIMIZABLE_KEYS = []
for k, v in DEFAULT_CFG.items():
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if not any(k.startswith(p) for p in [
            "BOARD", "CENTER", "SUN_R", "MAX_SPEED", "ROTATION_LIMIT",
            "TOTAL_STEPS", "SIM_HORIZON", "HORIZON", "LAUNCH_CLEARANCE"
        ]):
            OPTIMIZABLE_KEYS.append(k)

OPTIMIZABLE_KEYS_SORTED = sorted(OPTIMIZABLE_KEYS)

# ------------------------------------------------------------
# Evaluation function
# ------------------------------------------------------------
def evaluate_weights(weights, num_games=6, verbose=False):
    cfg = DEFAULT_CFG.copy()
    for k, v in zip(OPTIMIZABLE_KEYS_SORTED, weights):
        cfg[k] = v

    total_advantage = 0.0
    for game_idx in range(num_games):
        map_name = np.random.choice(MAP_POOL)
        env = kaggle_environments.make(
            "orbit-wars",   # Correct environment name
            configuration={
                "episodeSteps": 500,
                "map": map_name,
                "num_players": 2,
            }
        )
        agents = [
            lambda obs, config: learning_agent(obs, {**config, "weights": cfg}),
            baseline_agent,
        ]
        state = env.run(agents)
        final_frame = state[-1]
        obs0 = final_frame[0]["observation"]
        my_ships = sum(p["ships"] for p in obs0["planets"] if p["owner"] == 0)
        opp_ships = sum(p["ships"] for p in obs0["planets"] if p["owner"] == 1)
        advantage = my_ships - opp_ships
        total_advantage += advantage
        if verbose:
            print(f"Game {game_idx+1}: map={map_name}, adv={advantage:.1f}")
    return total_advantage / num_games