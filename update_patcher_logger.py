#!/usr/bin/env python3
"""
Update OLMoERouterPatcher class in notebook to support logger parameter.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "OLMoE_BH_Routing_Experiments.ipynb"

# Load notebook
with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

cells = nb['cells']

# Find the OLMoERouterPatcher class cell
patcher_cell_idx = None
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'class OLMoERouterPatcher:' in source and 'def patch_with_bh' in source:
            patcher_cell_idx = i
            break

if patcher_cell_idx is None:
    print("❌ Could not find OLMoERouterPatcher class")
    exit(1)

print(f"Found OLMoERouterPatcher at cell index {patcher_cell_idx}")

# Get the source lines
lines = cells[patcher_cell_idx]['source']

# Make modifications
new_lines = []
in_init = False
in_patch_with_bh_signature = False
in_bh_routing_call = False
init_modified = False
patch_signature_modified = False
bh_call_modified = False

i = 0
while i < len(lines):
    line = lines[i]

    # MODIFICATION 1: Add self.logger = None in __init__
    if 'def __init__' in line:
        in_init = True

    if in_init and 'self.patched = False' in line and not init_modified:
        new_lines.append(line)
        new_lines.append('        self.logger = None  # Store logger instance for BH routing\n')
        new_lines.append('        self.token_counter = 0  # Track tokens for logging\n')
        in_init = False
        init_modified = True
        i += 1
        continue

    # MODIFICATION 2: Add logger parameter to patch_with_bh signature
    if 'def patch_with_bh(' in line:
        in_patch_with_bh_signature = True

    if in_patch_with_bh_signature and 'collect_stats: bool = True' in line and not patch_signature_modified:
        new_lines.append(line.rstrip(',\n') + ',\n')
        new_lines.append('        logger: Optional[\'BHRoutingLogger\'] = None\n')
        in_patch_with_bh_signature = False
        patch_signature_modified = True
        i += 1
        continue

    # MODIFICATION 3: Store logger after unpatch()
    if 'self.unpatch()' in line and 'patch_with_bh' in ''.join(lines[max(0, i-20):i]):
        new_lines.append(line)
        # Add logger storage after unpatch() and stats.clear()
        if i + 1 < len(lines) and 'self.stats.clear()' in lines[i + 1]:
            new_lines.append(lines[i + 1])
            new_lines.append('        self.logger = logger  # Store logger instance\n')
            new_lines.append('        self.token_counter = 0  # Reset token counter\n')
            new_lines.append('\n')
            i += 2
            continue

    # MODIFICATION 4: Add logging message after KDE load
    if 'print(f"   ⚠️  No KDE models found' in line:
        new_lines.append(line)
        new_lines.append('        \n')
        new_lines.append('        if logger is not None:\n')
        new_lines.append('            print(f"   📝 Logging enabled: {logger.experiment_name}")\n')
        i += 1
        continue

    # MODIFICATION 5: Update benjamini_hochberg_routing call in custom forward
    if 'routing_weights, selected_experts, expert_counts = benjamini_hochberg_routing(' in line:
        in_bh_routing_call = True

    if in_bh_routing_call and 'kde_models=kde_models' in line and not bh_call_modified:
        # This is the line with kde_models parameter
        # Add logger parameters after it
        new_lines.append(line.rstrip(',\n') + ',\n')
        new_lines.append('                    logger=self.logger,        # Pass logger for detailed logging\n')
        new_lines.append('                    log_every_n_tokens=100,    # Sample every 100 tokens\n')
        new_lines.append('                    sample_idx=0,              # Default sample index\n')
        new_lines.append('                    token_idx=self.token_counter  # Tracked token index\n')
        in_bh_routing_call = False
        bh_call_modified = True
        i += 1
        # Increment token counter after BH call
        # Look ahead to find the end of the benjamini_hochberg_routing call
        while i < len(lines) and ')' not in lines[i]:
            i += 1
        if i < len(lines):
            new_lines.append(lines[i])  # The closing )
            new_lines.append('                \n')
            new_lines.append('                # Increment token counter for next logging\n')
            new_lines.append('                if self.logger is not None:\n')
            new_lines.append('                    self.token_counter += 1\n')
            i += 1
        continue

    # Keep all other lines unchanged
    new_lines.append(line)
    i += 1

# Update the cell with modified source
cells[patcher_cell_idx]['source'] = new_lines

# Save updated notebook
with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print("\n" + "=" * 70)
print("✅ PATCHER CLASS UPDATE COMPLETE")
print("=" * 70)
print(f"\nModified: {NOTEBOOK_PATH}")
print("\nChanges made:")
print(f"  1. Added self.logger instance variable: {init_modified}")
print(f"  2. Added logger parameter to patch_with_bh(): {patch_signature_modified}")
print(f"  3. Updated benjamini_hochberg_routing() call: {bh_call_modified}")
print("\n" + "=" * 70)
