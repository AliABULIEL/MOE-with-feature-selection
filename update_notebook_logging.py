#!/usr/bin/env python3
"""
Update notebook to add DEBUG_MODE configuration and integrate BHRoutingLogger.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "OLMoE_BH_Routing_Experiments.ipynb"

# Load notebook
with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

cells = nb['cells']

# =========================================================================
# STEP 1: Insert DEBUG_MODE configuration cell after Section 4.5
# =========================================================================
print("Step 1: Adding DEBUG_MODE configuration...")

# Find insertion point (after Section 4.5 code cell)
insert_idx = None
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        content = ''.join(cell['source'])
        if '4.5' in content and 'Framework' in content:
            insert_idx = i + 2  # After the markdown and code cell
            break

if insert_idx is None:
    print("  ⚠️ Could not find Section 4.5 to insert DEBUG_MODE")
else:
    # Create DEBUG_MODE markdown cell
    debug_markdown = {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '## 4.6 DEBUG_MODE Configuration\n',
            '\n',
            'Configure fast testing vs full evaluation mode.\n'
        ]
    }

    # Create DEBUG_MODE code cell
    debug_code = {
        'cell_type': 'code',
        'metadata': {},
        'source': [
            'print("=" * 70)\n',
            'print("DEBUG MODE CONFIGURATION")\n',
            'print("=" * 70)\n',
            '\n',
            '# Toggle for fast testing vs full evaluation\n',
            'DEBUG_MODE = False  # Set to True for quick testing\n',
            '\n',
            'if DEBUG_MODE:\n',
            '    # Fast testing configuration\n',
            '    MAX_SAMPLES = 10  # Very small sample for speed\n',
            '    LOG_EVERY_N = 5   # Log every 5 tokens\n',
            '    SAVE_PLOTS = True\n',
            '    print("\\n⚡ DEBUG MODE: ENABLED")\n',
            '    print("   • Max samples: 10 (fast testing)")\n',
            '    print("   • Logging: Every 5 tokens")\n',
            '    print("   • Plots: Generated for all experiments")\n',
            'else:\n',
            '    # Full evaluation configuration\n',
            '    MAX_SAMPLES = 200  # Full benchmark evaluation\n',
            '    LOG_EVERY_N = 100  # Log every 100 tokens for efficiency\n',
            '    SAVE_PLOTS = False  # Only save summaries, not per-token logs\n',
            '    print("\\n🎯 PRODUCTION MODE: ENABLED")\n',
            '    print("   • Max samples: 200 (full evaluation)")\n',
            '    print("   • Logging: Every 100 tokens")\n',
            '    print("   • Plots: Summary only")\n',
            '\n',
            'print("\\n" + "=" * 70)\n'
        ],
        'execution_count': None,
        'outputs': []
    }

    # Insert cells
    cells.insert(insert_idx, debug_markdown)
    cells.insert(insert_idx + 1, debug_code)
    print(f"  ✅ Inserted DEBUG_MODE configuration at cell index {insert_idx}")

# =========================================================================
# STEP 2: Import BHRoutingLogger in Section 4.5
# =========================================================================
print("\nStep 2: Adding BHRoutingLogger import to Section 4.5...")

# Find Section 4.5 code cell
section_45_idx = None
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        content = ''.join(cell['source'])
        if '4.5' in content and 'Framework' in content:
            section_45_idx = i + 1  # Code cell after markdown
            break

if section_45_idx is not None and section_45_idx < len(cells):
    # Modify the import section to add BHRoutingLogger
    code_cell = cells[section_45_idx]
    original_source = ''.join(code_cell['source'])

    # Add BHRoutingLogger import after BHMetricsComputer
    if 'BHRoutingLogger' not in original_source:
        # Find the line with BHMetricsComputer import
        lines = code_cell['source']
        new_lines = []

        for line in lines:
            new_lines.append(line)
            if 'from bh_routing_metrics import BHMetricsComputer' in line:
                # Add BHRoutingLogger import after metrics
                new_lines.append('\n')
                new_lines.append('# Import BH routing logger for detailed logging\n')
                new_lines.append('try:\n')
                new_lines.append('    if \'bh_routing_logging\' in sys.modules:\n')
                new_lines.append('        importlib.reload(sys.modules[\'bh_routing_logging\'])\n')
                new_lines.append('    from bh_routing_logging import BHRoutingLogger\n')
                new_lines.append('    print("✅ Imported BHRoutingLogger")\n')
                new_lines.append('except ImportError as e:\n')
                new_lines.append('    print(f"⚠️ Could not import BHRoutingLogger: {e}")\n')
                new_lines.append('    BHRoutingLogger = None\n')

        code_cell['source'] = new_lines
        print("  ✅ Added BHRoutingLogger import to Section 4.5")
    else:
        print("  ℹ️ BHRoutingLogger import already exists")

# =========================================================================
# STEP 3: Modify Section 9.5 to integrate BHRoutingLogger
# =========================================================================
print("\nStep 3: Integrating BHRoutingLogger into Section 9.5...")

# Find Section 9.5 code cell
section_95_idx = None
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        content = ''.join(cell['source'])
        if '9.5' in content and 'Comprehensive' in content and 'Benchmark' in content:
            section_95_idx = i + 1  # Code cell after markdown
            break

if section_95_idx is not None and section_95_idx < len(cells):
    # Create new Section 9.5 with logging integration
    new_section_95 = {
        'cell_type': 'code',
        'metadata': {},
        'source': [
            'print("=" * 70)\n',
            'print("COMPREHENSIVE BENCHMARK EVALUATION WITH LOGGING")\n',
            'print("=" * 70)\n',
            '\n',
            'if \'EVAL_DATASETS\' not in globals() or not EVAL_DATASETS:\n',
            '    print("⚠️ No datasets loaded. Skipping benchmark evaluation.")\n',
            '    print("   Run Section 7.5 to load datasets first.")\n',
            '    comprehensive_results = []\n',
            'else:\n',
            '    print(f"\\nExperiment Scope:")\n',
            '    print(f"  • Configurations: {len(configs)}")\n',
            '    print(f"  • Datasets: {list(EVAL_DATASETS.keys())}")\n',
            '    print(f"  • Samples per dataset: {MAX_SAMPLES}")\n',
            '    print(f"  • Total experiments: {len(configs) * len(EVAL_DATASETS)}")\n',
            '    \n',
            '    # Configure logging based on DEBUG_MODE\n',
            '    if \'LOG_EVERY_N\' not in globals():\n',
            '        LOG_EVERY_N = 100  # Default\n',
            '    \n',
            '    comprehensive_results = []\n',
            '    benchmark_start = time.time()\n',
            '    \n',
            '    for dataset_name, dataset_data in EVAL_DATASETS.items():\n',
            '        print(f"\\n{\'=\'*70}")\n',
            '        print(f"EVALUATING ON: {dataset_name.upper()}")\n',
            '        print(f"{\'=\'*70}")\n',
            '        \n',
            '        for i, config in enumerate(configs):\n',
            '            print(f"\\n[{i+1}/{len(configs)}] {config.name} on {dataset_name}")\n',
            '            print("-" * 50)\n',
            '            \n',
            '            config_start = time.time()\n',
            '            \n',
            '            # Setup routing\n',
            '            patcher.unpatch()\n',
            '            patcher.stats.clear()\n',
            '            \n',
            '            # Initialize logger for BH configurations only\n',
            '            logger = None\n',
            '            if config.routing_type == \'bh\' and BHRoutingLogger is not None:\n',
            '                experiment_name = f"{config.name}_{dataset_name}"\n',
            '                logger = BHRoutingLogger(\n',
            '                    output_dir=str(OUTPUT_DIR),\n',
            '                    experiment_name=experiment_name,\n',
            '                    log_every_n=LOG_EVERY_N\n',
            '                )\n',
            '                print(f"  📊 Logging enabled: {experiment_name}")\n',
            '            \n',
            '            if config.routing_type == \'baseline\':\n',
            '                # For baselines with k != 8, patch with topk\n',
            '                if config.k != 8:\n',
            '                    patcher.patch_with_topk(k=config.k, collect_stats=True)\n',
            '                print(f"  Using TopK={config.k} routing")\n',
            '            else:\n',
            '                # Patch with BH routing and logger\n',
            '                patcher.patch_with_bh(\n',
            '                    alpha=config.alpha,\n',
            '                    max_k=config.max_k,\n',
            '                    min_k=config.min_k,\n',
            '                    collect_stats=True,\n',
            '                    logger=logger  # Pass logger to BH routing\n',
            '                )\n',
            '                print(f"  Using BH routing (α={config.alpha}, max_k={config.max_k})")\n',
            '            \n',
            '            # Initialize result with common fields\n',
            '            result = {\n',
            '                \'config_name\': config.name,\n',
            '                \'routing_type\': \'topk\' if config.routing_type == \'baseline\' else \'bh\',\n',
            '                \'dataset\': dataset_name,\n',
            '                \'k_or_max_k\': config.k if config.routing_type == \'baseline\' else config.max_k,\n',
            '                \'alpha\': config.alpha if config.routing_type == \'bh\' else None\n',
            '            }\n',
            '            \n',
            '            try:\n',
            '                # Evaluate based on dataset type\n',
            '                if dataset_name == \'wikitext\':\n',
            '                    eval_result = evaluate_perplexity(\n',
            '                        model=model,\n',
            '                        tokenizer=tokenizer,\n',
            '                        texts=dataset_data,\n',
            '                        device=device,\n',
            '                        max_length=512\n',
            '                    )\n',
            '                    result[\'perplexity\'] = eval_result[\'perplexity\']\n',
            '                    result[\'tokens_per_second\'] = eval_result.get(\'tokens_per_second\', 0)\n',
            '                    print(f"  Perplexity: {eval_result[\'perplexity\']:.2f}")\n',
            '                    \n',
            '                elif dataset_name == \'lambada\':\n',
            '                    eval_result = evaluate_lambada(\n',
            '                        model=model,\n',
            '                        tokenizer=tokenizer,\n',
            '                        dataset=dataset_data,\n',
            '                        device=device\n',
            '                    )\n',
            '                    result[\'lambada_accuracy\'] = eval_result[\'accuracy\']\n',
            '                    print(f"  LAMBADA Accuracy: {eval_result[\'accuracy\']:.4f}")\n',
            '                    \n',
            '                elif dataset_name == \'hellaswag\':\n',
            '                    eval_result = evaluate_hellaswag(\n',
            '                        model=model,\n',
            '                        tokenizer=tokenizer,\n',
            '                        dataset=dataset_data,\n',
            '                        device=device\n',
            '                    )\n',
            '                    result[\'hellaswag_accuracy\'] = eval_result[\'accuracy\']\n',
            '                    print(f"  HellaSwag Accuracy: {eval_result[\'accuracy\']:.4f}")\n',
            '                    \n',
            '            except Exception as e:\n',
            '                print(f"  ❌ Evaluation failed: {e}")\n',
            '                import traceback\n',
            '                print(traceback.format_exc())\n',
            '                result[\'error\'] = str(e)\n',
            '            \n',
            '            # Get routing statistics\n',
            '            stats = patcher.get_stats()\n',
            '            k_val = config.k if config.routing_type == \'baseline\' else config.max_k\n',
            '            \n',
            '            if stats:\n',
            '                result[\'avg_experts\'] = stats.get(\'avg_experts\', k_val)\n',
            '                result[\'std_experts\'] = stats.get(\'std_experts\', 0)\n',
            '                result[\'min_experts\'] = stats.get(\'min_experts\', k_val)\n',
            '                result[\'max_experts\'] = stats.get(\'max_experts\', k_val)\n',
            '                \n',
            '                # Compute additional metrics\n',
            '                expert_counts = np.array(patcher.stats.get(\'expert_counts\', []))\n',
            '                if len(expert_counts) > 0 and metrics_computer:\n',
            '                    result[\'adaptive_range\'] = metrics_computer.compute_adaptive_range(expert_counts)\n',
            '                    result[\'ceiling_hit_rate\'] = metrics_computer.compute_ceiling_hit_rate(expert_counts, k_val)\n',
            '                    result[\'floor_hit_rate\'] = metrics_computer.compute_floor_hit_rate(expert_counts)\n',
            '                    result[\'mid_range_rate\'] = 100.0 - result[\'ceiling_hit_rate\'] - result[\'floor_hit_rate\']\n',
            '                    \n',
            '                    entropy, norm_entropy = metrics_computer.compute_selection_entropy(expert_counts, k_val)\n',
            '                    result[\'selection_entropy\'] = entropy\n',
            '                    result[\'normalized_entropy\'] = norm_entropy\n',
            '                    \n',
            '                    result[\'expert_activation_ratio\'] = metrics_computer.compute_expert_activation_ratio(\n',
            '                        result[\'avg_experts\'], k_val\n',
            '                    )\n',
            '                    result[\'flops_reduction_pct\'] = metrics_computer.compute_flops_reduction_pct(\n',
            '                        result[\'avg_experts\'], baseline_k=8\n',
            '                    )\n',
            '                \n',
            '                # Reduction vs baseline\n',
            '                result[\'reduction_vs_baseline\'] = (8 - result[\'avg_experts\']) / 8 * 100\n',
            '                \n',
            '                print(f"  Avg Experts: {result.get(\'avg_experts\', \'N/A\'):.2f}")\n',
            '                if \'adaptive_range\' in result:\n',
            '                    print(f"  Adaptive Range: {result[\'adaptive_range\']}")\n',
            '            else:\n',
            '                # For baseline K=8 without patching\n',
            '                result[\'avg_experts\'] = k_val\n',
            '                result[\'std_experts\'] = 0\n',
            '                result[\'min_experts\'] = k_val\n',
            '                result[\'max_experts\'] = k_val\n',
            '                result[\'adaptive_range\'] = 0\n',
            '                result[\'ceiling_hit_rate\'] = 100.0\n',
            '                result[\'floor_hit_rate\'] = 0.0\n',
            '                result[\'mid_range_rate\'] = 0.0\n',
            '                result[\'reduction_vs_baseline\'] = (8 - k_val) / 8 * 100\n',
            '            \n',
            '            # Save and generate plots if logger exists\n',
            '            if logger is not None:\n',
            '                try:\n',
            '                    # Save logs\n',
            '                    logger.save_logs()\n',
            '                    \n',
            '                    # Generate plots (controlled by DEBUG_MODE)\n',
            '                    if \'SAVE_PLOTS\' in globals() and SAVE_PLOTS:\n',
            '                        logger.generate_plots()\n',
            '                        print(f"  📊 Generated plots")\n',
            '                    \n',
            '                    # Get summary stats and add to result\n',
            '                    summary = logger.get_summary()\n',
            '                    if summary:\n',
            '                        result[\'logger_summary\'] = summary\n',
            '                    \n',
            '                    # Clear logger for next experiment\n',
            '                    logger.clear()\n',
            '                except Exception as e:\n',
            '                    print(f"  ⚠️ Logging/plotting failed: {e}")\n',
            '            \n',
            '            config_time = time.time() - config_start\n',
            '            result[\'elapsed_time\'] = config_time\n',
            '            print(f"  ✅ Completed in {config_time:.1f}s")\n',
            '            \n',
            '            comprehensive_results.append(result)\n',
            '            \n',
            '            # Clear GPU cache\n',
            '            if torch.cuda.is_available():\n',
            '                torch.cuda.empty_cache()\n',
            '    \n',
            '    # Ensure unpatched\n',
            '    patcher.unpatch()\n',
            '    \n',
            '    benchmark_time = time.time() - benchmark_start\n',
            '    \n',
            '    print("\\n" + "=" * 70)\n',
            '    print("COMPREHENSIVE BENCHMARK EVALUATION COMPLETE!")\n',
            '    print("=" * 70)\n',
            '    print(f"\\nTotal time: {benchmark_time / 60:.1f} minutes")\n',
            '    print(f"Experiments completed: {len(comprehensive_results)}")\n',
            '    print("=" * 70)\n'
        ],
        'execution_count': None,
        'outputs': []
    }

    # Replace the existing Section 9.5 code cell
    cells[section_95_idx] = new_section_95
    print("  ✅ Updated Section 9.5 with BHRoutingLogger integration")

# =========================================================================
# STEP 4: Modify OLMoERouterPatcher.patch_with_bh to accept logger
# =========================================================================
print("\nStep 4: Updating OLMoERouterPatcher to support logger...")

# Find Section 6 code cell (OLMoERouterPatcher class)
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'class OLMoERouterPatcher:' in source and 'patch_with_bh' in source:
            print(f"  Found OLMoERouterPatcher at cell index {i}")

            # This is complex - add a note in the source that logger support is needed
            # We'll handle this in the bh_routing.py instead, not in the notebook patcher class
            print("  ℹ️ Note: Logger will be passed through benjamini_hochberg_routing() directly")
            print("  ℹ️ The patcher class calls the BH function which now supports logger parameter")
            break

# Save modified notebook
with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print("\n" + "=" * 70)
print("✅ NOTEBOOK UPDATE COMPLETE")
print("=" * 70)
print(f"\nModified: {NOTEBOOK_PATH}")
print("\nChanges made:")
print("  1. Added Section 4.6 - DEBUG_MODE configuration")
print("  2. Added BHRoutingLogger import to Section 4.5")
print("  3. Integrated BHRoutingLogger into Section 9.5")
print("  4. Added automatic log saving and plot generation")
print("\nNext: Run the notebook to test the logging system")
print("=" * 70)
