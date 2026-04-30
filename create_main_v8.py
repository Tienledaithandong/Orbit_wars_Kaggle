import os
import re

with open('scratch_extracted.py', 'r', encoding='utf-8') as f:
    text = f.read()

# We need to extract the contents of the files created with %%writefile
files_to_extract = [
    'src/config.py',
    'src/game_types.py',
    'src/features.py',
    'src/policy.py',
    'src/ppo.py',
]

combined = []

# Add some global imports
combined.append('''import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import sys
import types
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

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
    
    class Categorical: pass
''')

for filename in files_to_extract:
    # Find the block starting with %%writefile filename
    # and ending before the next %% or EOF
    pattern = r'%%writefile ' + re.escape(filename) + r'\s*\n(.*?)(?=\n%%|\Z)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1)
        # Remove imports from the content since we already have them or will handle them
        lines = content.split('\n')
        filtered_lines = []
        for line in lines:
            if line.startswith('from __future__') or line.startswith('import ') or line.startswith('from '):
                # Don't add relative imports
                if not line.startswith('from .'):
                    # Add to top imports if not there
                    import_statement = line.strip()
                    if import_statement not in combined:
                        pass # Actually just let python figure it out, or append them
            else:
                filtered_lines.append(line)
        combined.append('\n'.join(filtered_lines))
    else:
        print(f"Could not find {filename}")

agent_code = """
# Initialize global policy
_policy = None
_device = None
_cfg = TrainConfig()

def agent(obs, config=None):
    global _policy, _device
    if not TORCH_AVAILABLE:
        return []
        
    if _policy is None:
        _device = torch.device("cpu") # default to CPU for agent
        _policy = PlanetPolicy(
            self_dim=self_feature_dim(),
            candidate_dim=candidate_feature_dim(),
            global_dim=global_feature_dim(),
            candidate_count=_cfg.env.candidate_count,
            hidden_size=_cfg.model.hidden_size,
        ).to(_device)
        
        # Load pre-trained weights if bundled with the submission
        weight_path = Path("/kaggle_simulations/agent/ckpt_002000.pt")
        # Fallback to local path if running locally
        if not weight_path.exists():
            weight_path = Path(__file__).parent / "weights" / "ckpt_002000.pt"
            
        if weight_path.exists():
            checkpoint = torch.load(weight_path, map_location=_device)
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
"""
combined.append(agent_code)

with open('main_v8.py', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(combined))

print("Created main_v8.py")
