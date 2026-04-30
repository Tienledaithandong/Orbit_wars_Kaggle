"""
Orbit Wars Agent - Refactored Version

A reinforcement learning agent for the Orbit Wars game using PPO.
This module provides feature encoding, neural network policy, and action selection.
"""

import math
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =============================================================================
# PyTorch Import with Fallback
# =============================================================================

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False
    from unittest.mock import MagicMock
    dummy_torch = MagicMock()
    dummy_torch.Tensor = Any
    sys.modules['torch'] = dummy_torch
    sys.modules['torch.nn'] = dummy_torch.nn
    sys.modules['torch.optim'] = dummy_torch.optim
    torch = dummy_torch
    nn = dummy_torch.nn
    
    class Categorical:
        pass


# =============================================================================
# Mock Module Setup for Checkpoint Compatibility
# =============================================================================

def _setup_mock_modules():
    """Setup mock module structure for loading checkpoints saved with different module paths."""
    _mock_src = types.ModuleType('src')
    _mock_src.rl_template = types.ModuleType('src.rl_template')
    _mock_src.rl_template.config = types.ModuleType('src.rl_template.config')
    
    @dataclass
    class _MockEnvConfig:
        board_size: float = 100.0
        episode_steps: int = 500
        candidate_count: int = 8
        ship_bucket_count: int = 8
        max_planets: int = 48
        max_ships: float = 400.0
        max_production: float = 5.0

    @dataclass
    class _MockModelConfig:
        hidden_size: int = 128

    @dataclass
    class _MockPPOConfig:
        rollout_steps: int = 32
        num_envs: int = 4
        total_updates: int = 200
        epochs: int = 4
        minibatch_size: int = 512
        gamma: float = 0.99
        clip_coef: float = 0.2
        ent_coef: float = 0.01
        vf_coef: float = 0.5
        lr: float = 3e-4
        max_grad_norm: float = 0.5

    @dataclass
    class _MockTrainConfig:
        seed: int = 42
        run_name: str = "orbit_wars_template_ppo"
        device: str = "auto"
        save_dir: str = "artifacts/rl_template"
        checkpoint_every: int = 10
        log_every: int = 1
        opponent: str = "random"
        self_play_update_interval: int = 10
        self_play_deterministic: bool = False
        alternate_player_sides: bool = True
        env: Any = field(default_factory=_MockEnvConfig)
        model: Any = field(default_factory=_MockModelConfig)
        ppo: Any = field(default_factory=_MockPPOConfig)

    _mock_src.rl_template.config.EnvConfig = _MockEnvConfig
    _mock_src.rl_template.config.ModelConfig = _MockModelConfig
    _mock_src.rl_template.config.PPOConfig = _MockPPOConfig
    _mock_src.rl_template.config.TrainConfig = _MockTrainConfig
    sys.modules['src'] = _mock_src
    sys.modules['src.rl_template'] = _mock_src.rl_template
    sys.modules['src.rl_template.config'] = _mock_src.rl_template.config

_setup_mock_modules()


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass(slots=True)
class EnvConfig:
    """Environment configuration for Orbit Wars."""
    board_size: float = 100.0
    episode_steps: int = 500
    candidate_count: int = 8
    ship_bucket_count: int = 8
    max_planets: int = 48
    max_ships: float = 400.0
    max_production: float = 5.0


@dataclass(slots=True)
class ModelConfig:
    """Neural network model configuration."""
    hidden_size: int = 128


@dataclass(slots=True)
class PPOConfig:
    """Proximal Policy Optimization hyperparameters."""
    rollout_steps: int = 32
    num_envs: int = 4
    total_updates: int = 200
    epochs: int = 4
    minibatch_size: int = 512
    gamma: float = 0.99
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5


@dataclass(slots=True)
class TrainConfig:
    """Training configuration."""
    seed: int = 42
    run_name: str = "orbit_wars_template_ppo"
    device: str = "auto"
    save_dir: str = "artifacts/rl_template"
    checkpoint_every: int = 10
    log_every: int = 1
    opponent: str = "random"
    self_play_update_interval: int = 10
    self_play_deterministic: bool = False
    alternate_player_sides: bool = True
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)


# =============================================================================
# Configuration Loading Utilities
# =============================================================================

def default_train_config_path() -> Path:
    """Return the default path to the training config YAML file."""
    return Path(__file__).resolve().parent / "configs" / "default.yaml"


def load_train_config(path: str | Path) -> TrainConfig:
    """Load training configuration from a YAML file."""
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return train_config_from_dict(data)


def train_config_from_dict(data: dict[str, Any]) -> TrainConfig:
    """Create TrainConfig from a dictionary."""
    cfg = TrainConfig()
    _update_dataclass(cfg, data, skip={"env", "model", "ppo"})
    _update_dataclass(cfg.env, data.get("env", {}))
    _update_dataclass(cfg.model, data.get("model", {}))
    _update_dataclass(cfg.ppo, data.get("ppo", {}))
    return cfg


def _update_dataclass(instance: Any, values: dict[str, Any], skip: set[str] | None = None) -> None:
    """Update a dataclass instance with values from a dictionary."""
    if not isinstance(values, dict):
        return
    skip = skip or set()
    for key, value in values.items():
        if key in skip or not hasattr(instance, key):
            continue
        default = getattr(instance, key)
        setattr(instance, key, _coerce_value(value, default))


def _coerce_value(value: Any, default: Any) -> Any:
    """Coerce a value to match the type of the default."""
    if isinstance(default, bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


# =============================================================================
# Game State Representation
# =============================================================================

@dataclass(slots=True)
class PlanetState:
    """Represents the state of a planet."""
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


@dataclass(slots=True)
class FleetState:
    """Represents the state of a fleet."""
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int


@dataclass(slots=True)
class GameState:
    """Complete game state at a given step."""
    step: int
    player: int
    planets: list[PlanetState]
    fleets: list[FleetState]


def parse_observation(observation: Any) -> GameState:
    """Parse raw observation into a structured GameState."""
    def obs_get(key: str, default: Any) -> Any:
        if isinstance(observation, dict):
            return observation.get(key, default)
        return getattr(observation, key, default)

    planets = [
        PlanetState(
            id=int(row[0]),
            owner=int(row[1]),
            x=float(row[2]),
            y=float(row[3]),
            radius=float(row[4]),
            ships=int(row[5]),
            production=int(row[6]),
        )
        for row in obs_get("planets", [])
    ]
    fleets = [
        FleetState(
            id=int(row[0]),
            owner=int(row[1]),
            x=float(row[2]),
            y=float(row[3]),
            angle=float(row[4]),
            from_planet_id=int(row[5]),
            ships=int(row[6]),
        )
        for row in obs_get("fleets", [])
    ]
    return GameState(
        step=int(obs_get("step", 0)),
        player=int(obs_get("player", 0)),
        planets=planets,
        fleets=fleets,
    )


# =============================================================================
# Feature Encoding
# =============================================================================

BOARD_CENTER = (50.0, 50.0)
ROTATION_RADIUS_LIMIT = 50.0
SUN_RADIUS = 10.0
PLANET_LAUNCH_RADIUS_OFFSET = 0.1


@dataclass(slots=True)
class DecisionContext:
    """Context for a single decision (source planet)."""
    env_index: int
    source_id: int
    candidate_ids: list[int]
    candidate_mask: np.ndarray
    ship_counts: list[int]
    target_angles: list[float]


@dataclass(slots=True)
class TurnBatch:
    """Encoded features for all decisions in a turn."""
    self_features: np.ndarray
    candidate_features: np.ndarray
    global_features: np.ndarray
    candidate_mask: np.ndarray
    contexts: list[DecisionContext]
    state: GameState


def self_feature_dim() -> int:
    """Dimension of self (source planet) features."""
    return 11


def candidate_feature_dim() -> int:
    """Dimension of candidate target features."""
    return 14


def global_feature_dim() -> int:
    """Dimension of global game state features."""
    return 8


def encode_turn(observation: Any, env_cfg: EnvConfig, *, env_index: int = 0) -> TurnBatch:
    """Encode game observation into ML-ready features."""
    state = observation if isinstance(observation, GameState) else parse_observation(observation)
    my_planets = sorted(
        (planet for planet in state.planets if planet.owner == state.player),
        key=lambda planet: planet.id
    )
    
    if not my_planets:
        return TurnBatch(
            self_features=np.zeros((0, self_feature_dim()), dtype=np.float32),
            candidate_features=np.zeros((0, env_cfg.candidate_count, candidate_feature_dim()), dtype=np.float32),
            global_features=np.zeros((0, global_feature_dim()), dtype=np.float32),
            candidate_mask=np.zeros((0, env_cfg.candidate_count), dtype=bool),
            contexts=[],
            state=state,
        )

    global_feat = build_global_features(state, env_cfg)
    self_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    candidate_masks: list[np.ndarray] = []
    contexts: list[DecisionContext] = []

    for src in my_planets:
        candidates = build_candidates(src, state, env_cfg)
        cand_feat, cand_mask, ship_counts, candidate_ids, target_angles = build_candidate_features(
            src, candidates, state, env_cfg,
        )
        self_rows.append(build_self_features(src, state, env_cfg))
        candidate_rows.append(cand_feat)
        candidate_masks.append(cand_mask)
        contexts.append(
            DecisionContext(
                env_index=env_index,
                source_id=src.id,
                candidate_ids=candidate_ids,
                candidate_mask=cand_mask,
                ship_counts=ship_counts,
                target_angles=target_angles,
            )
        )

    return TurnBatch(
        self_features=np.asarray(self_rows, dtype=np.float32),
        candidate_features=np.asarray(candidate_rows, dtype=np.float32),
        global_features=np.repeat(global_feat[None, :], len(self_rows), axis=0),
        candidate_mask=np.asarray(candidate_masks, dtype=bool),
        contexts=contexts,
        state=state,
    )


def build_candidates(src: PlanetState, state: GameState, env_cfg: EnvConfig) -> list[PlanetState]:
    """Select candidate target planets for a source planet."""
    others = [planet for planet in state.planets if planet.id != src.id]
    enemy_quota = env_cfg.candidate_count // 3
    neutral_quota = env_cfg.candidate_count // 3
    friendly_quota = env_cfg.candidate_count - enemy_quota - neutral_quota

    enemies = sorted(
        (planet for planet in others if planet.owner not in {-1, state.player}),
        key=lambda planet: (distance(src, planet), planet.id),
    )[:enemy_quota]
    neutrals = sorted(
        (planet for planet in others if planet.owner == -1),
        key=lambda planet: (distance(src, planet), planet.id),
    )[:neutral_quota]
    friendlies = sorted(
        (planet for planet in others if planet.owner == state.player),
        key=lambda planet: (distance(src, planet), planet.id),
    )[:friendly_quota]

    selected_ids = {planet.id for planet in enemies + neutrals + friendlies}
    candidates = enemies + neutrals + friendlies
    
    if len(candidates) >= env_cfg.candidate_count:
        return candidates[:env_cfg.candidate_count]

    fallback = sorted(
        (planet for planet in others if planet.id not in selected_ids),
        key=lambda planet: (distance(src, planet), planet.id),
    )
    candidates.extend(fallback[:env_cfg.candidate_count - len(candidates)])
    return candidates


def build_self_features(src: PlanetState, state: GameState, env_cfg: EnvConfig) -> np.ndarray:
    """Build feature vector for the source planet."""
    my_planets = [planet for planet in state.planets if planet.owner == state.player]
    enemy_planets = [planet for planet in state.planets if planet.owner not in {-1, state.player}]
    return np.asarray(
        [
            1.0,
            src.x / env_cfg.board_size,
            src.y / env_cfg.board_size,
            src.radius / 5.0,
            min(src.ships, env_cfg.max_ships) / env_cfg.max_ships,
            src.production / env_cfg.max_production,
            1.0 if is_rotating_planet(src) else 0.0,
            len(my_planets) / env_cfg.max_planets,
            len(enemy_planets) / env_cfg.max_planets,
            total_ships(my_planets) / (env_cfg.max_planets * env_cfg.max_ships),
            total_ships(enemy_planets) / (env_cfg.max_planets * env_cfg.max_ships),
        ],
        dtype=np.float32,
    )


def build_candidate_features(
    src: PlanetState,
    candidates: list[PlanetState],
    state: GameState,
    env_cfg: EnvConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[float]]:
    """Build feature vectors for candidate targets."""
    features = np.zeros((env_cfg.candidate_count, candidate_feature_dim()), dtype=np.float32)
    candidate_mask = np.zeros((env_cfg.candidate_count,), dtype=bool)
    ship_counts = [0] * env_cfg.candidate_count
    candidate_ids = [-1] * env_cfg.candidate_count
    target_angles = [0.0] * env_cfg.candidate_count
    candidate_mask[0] = True

    for idx, tgt in enumerate(candidates, start=1):
        if idx >= env_cfg.candidate_count:
            break
        dx = tgt.x - src.x
        dy = tgt.y - src.y
        angle = math.atan2(dy, dx)
        crosses_sun = shot_crosses_sun(src, angle, tgt)
        ships_needed = fixed_ship_count(src, tgt)
        features[idx] = np.asarray(
            [
                1.0,
                1.0 if tgt.owner == -1 else 0.0,
                1.0 if tgt.owner == state.player else 0.0,
                1.0 if tgt.owner not in {-1, state.player} else 0.0,
                tgt.x / env_cfg.board_size,
                tgt.y / env_cfg.board_size,
                dx / env_cfg.board_size,
                dy / env_cfg.board_size,
                distance(src, tgt) / env_cfg.board_size,
                min(tgt.ships, env_cfg.max_ships) / env_cfg.max_ships,
                tgt.production / env_cfg.max_production,
                1.0 if is_rotating_planet(tgt) else 0.0,
                1.0 if crosses_sun else 0.0,
                min(src.ships, env_cfg.max_ships) / env_cfg.max_ships,
            ],
            dtype=np.float32,
        )
        ship_counts[idx] = ships_needed
        candidate_mask[idx] = ships_needed > 0 and not crosses_sun and src.ships >= ships_needed
        candidate_ids[idx] = tgt.id
        target_angles[idx] = angle

    return features, candidate_mask, ship_counts, candidate_ids, target_angles


def build_global_features(state: GameState, env_cfg: EnvConfig) -> np.ndarray:
    """Build global game state features."""
    my_planets = [planet for planet in state.planets if planet.owner == state.player]
    enemy_planets = [planet for planet in state.planets if planet.owner not in {-1, state.player}]
    neutral_planets = [planet for planet in state.planets if planet.owner == -1]
    my_fleets = [fleet for fleet in state.fleets if fleet.owner == state.player]
    enemy_fleets = [fleet for fleet in state.fleets if fleet.owner != state.player]
    return np.asarray(
        [
            state.step / env_cfg.episode_steps,
            len(my_planets) / env_cfg.max_planets,
            len(enemy_planets) / env_cfg.max_planets,
            len(neutral_planets) / env_cfg.max_planets,
            total_ships(my_planets) / (env_cfg.max_planets * env_cfg.max_ships),
            total_ships(enemy_planets) / (env_cfg.max_planets * env_cfg.max_ships),
            sum(fleet.ships for fleet in my_fleets) / (env_cfg.max_planets * env_cfg.max_ships),
            sum(fleet.ships for fleet in enemy_fleets) / (env_cfg.max_planets * env_cfg.max_ships),
        ],
        dtype=np.float32,
    )


# =============================================================================
# Geometry & Combat Utilities
# =============================================================================

def fixed_ship_count(src: PlanetState, tgt: PlanetState) -> int:
    """Calculate ships needed to capture target planet."""
    return max(tgt.ships + 1, 3)


def distance(a: PlanetState, b: PlanetState) -> float:
    """Euclidean distance between two planets."""
    return math.hypot(a.x - b.x, a.y - b.y)


def total_ships(planets: list[PlanetState]) -> float:
    """Total ships across multiple planets."""
    return float(sum(planet.ships for planet in planets))


def is_rotating_planet(planet: PlanetState) -> bool:
    """Check if planet orbits the sun (vs stationary)."""
    dx = planet.x - BOARD_CENTER[0]
    dy = planet.y - BOARD_CENTER[1]
    orbital_radius = math.hypot(dx, dy)
    return orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT


def shot_crosses_sun(src: PlanetState, angle: float, tgt: PlanetState) -> bool:
    """Check if shot trajectory passes through the sun."""
    start_x = src.x + math.cos(angle) * (src.radius + PLANET_LAUNCH_RADIUS_OFFSET)
    start_y = src.y + math.sin(angle) * (src.radius + PLANET_LAUNCH_RADIUS_OFFSET)
    return point_to_segment_distance(BOARD_CENTER, (start_x, start_y), (tgt.x, tgt.y)) < SUN_RADIUS


def point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    """Minimum distance from point to line segment."""
    segment_len_sq = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2
    if segment_len_sq == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        ((point[0] - start[0]) * (end[0] - start[0]) + (point[1] - start[1]) * (end[1] - start[1]))
        / segment_len_sq
    )
    projection = max(0.0, min(1.0, projection))
    closest_x = start[0] + projection * (end[0] - start[0])
    closest_y = start[1] + projection * (end[1] - start[1])
    return math.hypot(point[0] - closest_x, point[1] - closest_y)


# =============================================================================
# Neural Network Policy
# =============================================================================

@dataclass(slots=True)
class PolicyOutput:
    """Output from policy network forward pass."""
    target_logits: torch.Tensor
    value: torch.Tensor


class PlanetPolicy(nn.Module):
    """Neural network policy for Orbit Wars."""
    
    def __init__(
        self,
        self_dim: int,
        candidate_dim: int,
        global_dim: int,
        candidate_count: int,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.candidate_count = candidate_count
        
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.target_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        self_features: torch.Tensor,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> PolicyOutput:
        """Forward pass through the policy network."""
        self_hidden = self.self_encoder(self_features)
        global_hidden = self.global_encoder(global_features)
        candidate_hidden = self.candidate_encoder(candidate_features)
        
        expanded_self = self_hidden.unsqueeze(1).expand(-1, self.candidate_count, -1)
        expanded_global = global_hidden.unsqueeze(1).expand(-1, self.candidate_count, -1)
        joint = torch.cat([expanded_self, expanded_global, candidate_hidden], dim=-1)
        
        target_logits = self.target_head(joint).squeeze(-1)
        target_logits = target_logits.masked_fill(~candidate_mask, torch.finfo(target_logits.dtype).min)
        
        pooled_candidates = candidate_hidden.mean(dim=1)
        value = self.value_head(torch.cat([self_hidden, global_hidden, pooled_candidates], dim=-1)).squeeze(-1)
        
        return PolicyOutput(target_logits=target_logits, value=value)


# =============================================================================
# Action Sampling & PPO Training
# =============================================================================

@dataclass(slots=True)
class SampledAction:
    """Sampled action from policy."""
    target_index: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


@dataclass(slots=True)
class TransitionBatch:
    """Batch of transitions for PPO training."""
    self_features: torch.Tensor
    candidate_features: torch.Tensor
    global_features: torch.Tensor
    candidate_mask: torch.Tensor
    target_index: torch.Tensor
    log_prob: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def sample_actions(outputs: PolicyOutput, deterministic: bool) -> SampledAction:
    """Sample actions from policy output."""
    target_logits = safe_target_logits(outputs.target_logits)
    target_dist = Categorical(logits=target_logits)
    target_index = target_logits.argmax(dim=-1) if deterministic else target_dist.sample()
    log_prob, entropy = action_log_prob_and_entropy(outputs=outputs, target_index=target_index)
    return SampledAction(target_index=target_index, log_prob=log_prob, entropy=entropy)


def action_log_prob_and_entropy(
    outputs: PolicyOutput,
    target_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log probability and entropy of actions."""
    target_logits = safe_target_logits(outputs.target_logits)
    target_dist = Categorical(logits=target_logits)
    target_log_prob = target_dist.log_prob(target_index)
    target_entropy = target_dist.entropy()
    return target_log_prob, target_entropy


def safe_target_logits(target_logits: torch.Tensor) -> torch.Tensor:
    """Ensure logits are valid for categorical distribution."""
    invalid_rows = ~torch.isfinite(target_logits).any(dim=-1)
    if not invalid_rows.any():
        return target_logits
    safe_logits = target_logits.clone()
    safe_logits[invalid_rows, 0] = 0.0
    return safe_logits


def ppo_update(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TransitionBatch,
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    epochs: int,
    minibatch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Perform PPO update on a batch of transitions."""
    if batch.self_features.shape[0] == 0:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    
    self_features = batch.self_features.to(device)
    candidate_features = batch.candidate_features.to(device)
    global_features = batch.global_features.to(device)
    candidate_mask = batch.candidate_mask.to(device).bool()
    old_log_prob = batch.log_prob.to(device)
    target_index = batch.target_index.to(device)
    returns = batch.returns.to(device)
    advantages = batch.advantages.to(device)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    
    size = self_features.shape[0]
    minibatch_size = min(size, max(1, minibatch_size))
    metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0
    
    for _ in range(epochs):
        order = torch.randperm(size, device=device)
        for start in range(0, size, minibatch_size):
            idx = order[start : start + minibatch_size]
            outputs = policy(
                self_features[idx],
                candidate_features[idx],
                global_features[idx],
                candidate_mask[idx],
            )
            new_log_prob, entropy = action_log_prob_and_entropy(outputs, target_index[idx])
            
            ratio = (new_log_prob - old_log_prob[idx]).exp()
            policy_loss = torch.maximum(
                -advantages[idx] * ratio,
                -advantages[idx] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
            ).mean()
            value_loss = 0.5 * (returns[idx] - outputs.value).pow(2).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            
            metrics["loss"] += float(loss.detach().cpu())
            metrics["policy_loss"] += float(policy_loss.detach().cpu())
            metrics["value_loss"] += float(value_loss.detach().cpu())
            metrics["entropy"] += float(entropy_mean.detach().cpu())
            updates += 1
    
    return {key: value / max(updates, 1) for key, value in metrics.items()}


# =============================================================================
# Agent Interface
# =============================================================================

_policy = None
_device = None
_cfg = TrainConfig()


def agent(obs, config=None):
    """
    Main agent entry point called by the environment.
    
    Args:
        obs: Raw observation from the environment
        config: Optional configuration override
        
    Returns:
        List of moves, each move is [source_id, angle, ship_count]
    """
    global _policy, _device
    
    if not TORCH_AVAILABLE:
        return []
        
    if _policy is None:
        _device = torch.device("cpu")
        _policy = PlanetPolicy(
            self_dim=self_feature_dim(),
            candidate_dim=candidate_feature_dim(),
            global_dim=global_feature_dim(),
            candidate_count=_cfg.env.candidate_count,
            hidden_size=_cfg.model.hidden_size,
        ).to(_device)
        
        # Load pre-trained weights
        weight_path = Path(__file__).parent / "weights" / "ckpt_fixed.pt"
        
        if not weight_path.exists():
            weight_path = Path("/kaggle_simulations/agent/ckpt_002000.pt")
        if not weight_path.exists():
            weight_path = Path(__file__).parent / "weights" / "ckpt_002000.pt"
            
        if weight_path.exists():
            checkpoint = torch.load(weight_path, map_location=_device, weights_only=False)
            _policy.load_state_dict(checkpoint.get("policy", checkpoint))
            
        _policy.eval()
    
    batch = encode_turn(obs, _cfg.env, env_index=0)
    if batch.self_features.shape[0] == 0:
        return []
        
    with torch.inference_mode():
        outputs = _policy(
            torch.from_numpy(batch.self_features).to(_device),
            torch.from_numpy(batch.candidate_features).to(_device),
            torch.from_numpy(batch.global_features).to(_device),
            torch.from_numpy(batch.candidate_mask).to(_device).bool(),
        )
        sampled = sample_actions(outputs, deterministic=True)
        
    target_indices = sampled.target_index.detach().cpu().numpy()
    moves = []
    
    for row_idx, context in enumerate(batch.contexts):
        target_idx = int(target_indices[row_idx])
        if target_idx == 0:
            continue
        if target_idx >= len(context.candidate_ids):
            continue
        if not context.candidate_mask[target_idx]:
            continue
        ships = int(context.ship_counts[target_idx])
        if ships <= 0:
            continue
        moves.append([context.source_id, float(context.target_angles[target_idx]), ships])
        
    return moves
