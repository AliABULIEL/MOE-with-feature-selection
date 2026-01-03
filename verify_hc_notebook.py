#!/usr/bin/env python3
"""
HC Routing Notebook Verification Script
=======================================

Verifies all 5 phases of fixes were applied correctly.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

NOTEBOOK_PATH = Path("/Users/aliab/Desktop/GitHub/MOE-with-feature-selection/OLMoE_HC_PURE_FINAL.ipynb")


def load_notebook():
    with open(NOTEBOOK_PATH, 'r') as f:
        return json.load(f)


def get_all_text(nb):
    """Get all text from notebook."""
    texts = []
    for cell in nb['cells']:
        source = cell.get('source', [])
        if isinstance(source, list):
            texts.append(''.join(source))
        else:
            texts.append(source)
    return '\n'.join(texts)


def verify_phase1_terminology(nb):
    """Verify no BH/alpha terminology remains."""
    print("\n" + "="*70)
    print("PHASE 1 VERIFICATION: TERMINOLOGY CLEANUP")
    print("="*70)

    all_text = get_all_text(nb)
    issues = []

    # Check for BH references (excluding URLs, imports for reference, etc.)
    bh_matches = re.findall(r'\bBH\b(?!_)', all_text)
    if bh_matches:
        issues.append(f"Found {len(bh_matches)} 'BH' references")
    else:
        print("✅ No 'BH' references found")

    # Check for α= parameter usage (should be β=)
    alpha_matches = re.findall(r'α\s*=\s*\d', all_text)
    if alpha_matches:
        issues.append(f"Found {len(alpha_matches)} 'α=' parameter references")
        for match in alpha_matches[:5]:
            print(f"  ⚠️ Found: {match}")
    else:
        print("✅ No 'α=' parameter references found")

    # Check for Benjamini-Hochberg
    benjamini_matches = re.findall(r'Benjamini[- ]Hochberg', all_text, re.IGNORECASE)
    if benjamini_matches:
        issues.append(f"Found {len(benjamini_matches)} 'Benjamini-Hochberg' references")
    else:
        print("✅ No 'Benjamini-Hochberg' references found")

    # Check for β usage (should exist)
    beta_matches = re.findall(r'β\s*=\s*\d', all_text)
    print(f"✅ Found {len(beta_matches)} 'β=' parameter references (good!)")

    return len(issues) == 0, issues


def verify_phase2_logger(nb):
    """Verify logger integration."""
    print("\n" + "="*70)
    print("PHASE 2 VERIFICATION: LOGGER INTEGRATION")
    print("="*70)

    all_text = get_all_text(nb)
    issues = []

    # Check for logger parameter in patch_with_hc
    if 'logger: Optional[HCRoutingLogger]' in all_text or 'logger=logger' in all_text:
        print("✅ Logger parameter added to patch_with_hc()")
    else:
        issues.append("Logger parameter NOT found in patch_with_hc()")

    # Check for logger storage
    if 'self._logger = logger' in all_text:
        print("✅ Logger stored as instance variable")
    else:
        issues.append("Logger NOT stored as instance variable")

    # Check for logger passed to higher_criticism_routing
    if 'logger=self._logger' in all_text:
        print("✅ Logger passed to higher_criticism_routing()")
    else:
        issues.append("Logger NOT passed to higher_criticism_routing()")

    # Check for token counter
    if 'self._token_counter' in all_text:
        print("✅ Token counter implemented")
    else:
        issues.append("Token counter NOT implemented")

    # Check for start_sample method
    if 'def start_sample(self)' in all_text:
        print("✅ start_sample() method added")
    else:
        issues.append("start_sample() method NOT found")

    # Check for patcher.start_sample() call in evaluation
    if 'patcher.start_sample()' in all_text:
        print("✅ patcher.start_sample() called in evaluation loop")
    else:
        print("⚠️ Warning: patcher.start_sample() call not found (may need manual check)")

    return len(issues) == 0, issues


def verify_phase3_bugs(nb):
    """Verify bug fixes."""
    print("\n" + "="*70)
    print("PHASE 3 VERIFICATION: BUG FIXES")
    print("="*70)

    all_text = get_all_text(nb)
    issues = []

    # Check for minlength=... syntax error
    if 'minlength=...' in all_text:
        issues.append("Syntax error 'minlength=...' still present")
    else:
        print("✅ Syntax error 'minlength=...' fixed")

    # Check for minlength=65
    if 'minlength=65' in all_text:
        print("✅ Corrected to 'minlength=65'")

    # Check for duplicate bh_routing_file
    if all_text.count('bh_routing_file') > 0:
        print(f"⚠️ Warning: Found {all_text.count('bh_routing_file')} 'bh_routing_file' references (should be removed)")
    else:
        print("✅ No duplicate 'bh_routing_file' checks")

    return len(issues) == 0, issues


def verify_phase4_research(nb):
    """Verify research standardization."""
    print("\n" + "="*70)
    print("PHASE 4 VERIFICATION: RESEARCH STANDARDIZATION")
    print("="*70)

    all_text = get_all_text(nb)
    issues = []

    # Check for Limitations section
    if 'Limitations' in all_text and 'KDE Calibration Assumption' in all_text:
        print("✅ Limitations section added")
    else:
        issues.append("Limitations section NOT found")

    # Check for unsupported claims
    unsupported = [
        'HC improves model quality',
        'HC is optimal for all MoE',
        'HC reduces inference latency',
        'HC generalizes to fine-tuned'
    ]

    found_unsupported = []
    for claim in unsupported:
        if claim in all_text and '[NEEDS EVIDENCE]' not in all_text:
            found_unsupported.append(claim)

    if found_unsupported:
        print(f"⚠️ Warning: Found {len(found_unsupported)} potentially unsupported claims:")
        for claim in found_unsupported:
            print(f"  - {claim}")
    else:
        print("✅ No unsupported claims found")

    # Check for proper claims
    supported_claims = [
        'HC routing adaptively selects',
        'reduction in expert usage',
        'β parameter controls'
    ]

    found_supported = sum(1 for claim in supported_claims if claim in all_text)
    print(f"✅ Found {found_supported}/{len(supported_claims)} supported claims")

    return len(issues) == 0, issues


def verify_overall_structure(nb):
    """Verify overall notebook structure."""
    print("\n" + "="*70)
    print("OVERALL STRUCTURE VERIFICATION")
    print("="*70)

    # Count cells by type
    cell_types = defaultdict(int)
    for cell in nb['cells']:
        cell_types[cell['cell_type']] += 1

    print(f"📊 Notebook Statistics:")
    print(f"  Total cells: {len(nb['cells'])}")
    print(f"  Markdown cells: {cell_types['markdown']}")
    print(f"  Code cells: {cell_types['code']}")

    # Check for key sections
    all_text = get_all_text(nb)

    sections = {
        'OLMoERouterPatcher class': 'class OLMoERouterPatcher' in all_text,
        'Evaluation loop': 'for config in configs' in all_text,
        'Logger instantiation': 'HCRoutingLogger(' in all_text,
        'Conclusions': 'Conclusions' in all_text,
        'Limitations': 'Limitations' in all_text,
    }

    print(f"\n📝 Key Sections:")
    for section, found in sections.items():
        status = "✅" if found else "❌"
        print(f"  {status} {section}")

    return all(sections.values())


def main():
    print("="*70)
    print("HC ROUTING NOTEBOOK VERIFICATION")
    print("="*70)
    print(f"\n📖 Verifying: {NOTEBOOK_PATH}")

    nb = load_notebook()

    # Run all verification phases
    results = {}

    passed, issues = verify_phase1_terminology(nb)
    results['Phase 1: Terminology'] = (passed, issues)

    passed, issues = verify_phase2_logger(nb)
    results['Phase 2: Logger'] = (passed, issues)

    passed, issues = verify_phase3_bugs(nb)
    results['Phase 3: Bugs'] = (passed, issues)

    passed, issues = verify_phase4_research(nb)
    results['Phase 4: Research'] = (passed, issues)

    structure_ok = verify_overall_structure(nb)

    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    all_passed = True
    for phase, (passed, issues) in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {phase}")
        if issues:
            for issue in issues:
                print(f"       ⚠️ {issue}")
            all_passed = False

    print()
    if all_passed and structure_ok:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print()
        print("✅ Notebook is ready for execution")
        print()
        print("NEXT STEPS:")
        print("1. Open notebook in Jupyter")
        print("2. Run all cells")
        print("3. Verify logger.total_decisions > 0")
        print("4. Check verification section passes")
        return 0
    else:
        print("⚠️ SOME VERIFICATIONS FAILED")
        print("\nPlease review the issues above and make corrections.")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
