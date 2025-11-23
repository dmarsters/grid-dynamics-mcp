#!/usr/bin/env python
"""
Grid Dynamics MCP - Example Usage & Workflow Demonstration

This script demonstrates the three-layer olog workflow for analyzing
image composition and synthesizing enhanced prompts.
"""

import json
from pathlib import Path
from grid_dynamics_mcp import (
    # Types
    AlignmentType,
    StructuralAnalysis,
    MoodAnalysis,
    SeedInterpretation,
    AlignmentProposal,
)


# ============================================================================
# EXAMPLE 1: Birthday Party Analysis (from spec)
# ============================================================================

def example_birthday_party_workflow():
    """
    Workflow: Analyze birthday photo with melancholic mood + celebratory intent
    Shows conflict detection and bittersweet register emergence
    """
    print("=" * 80)
    print("EXAMPLE 1: Birthday Party - Bittersweet Celebration")
    print("=" * 80)

    # Layer 2 Output (from vision models)
    structural = StructuralAnalysis(
        edge_tension=0.8,
        asymmetric_balance=0.5,
        negative_space=0.7,
        rhythm=0.5,
        density=0.6,
        scale_hierarchy=0.4,
        diagonal_dominance=0.5,
        containment=0.3,
        summary="Tight composition with subjects crowding frame edges; generous spacing abandoned for intimate closeness; moderate regular rhythm; medium element density creates visual weight without overwhelming."
    )

    mood = MoodAnalysis(
        dimensions={
            "melancholy": 0.6,
            "intimacy": 0.7,
            "tension": 0.4,
            "stillness": 0.3,
            "energy": 0.3,
            "warmth": 0.2,
            "unease": 0.4,
        },
        summary="Deeply intimate moment shadowed by melancholic undertone; uneasy tension tempers the potential warmth; moment feels frozen, still despite subject activity."
    )

    seed = SeedInterpretation(
        subject="children's birthday party",
        intent="celebration and joy"
    )

    # Layer 2 Alignment Proposal (rule-based)
    alignment = AlignmentProposal(
        relationship_type=AlignmentType.PRODUCTIVE_TENSION,
        register_description="bittersweet celebration",
        conflicts=[
            "Melancholic mood (0.6) contradicts celebratory intent",
            "Low warmth (0.2) tempers joyful subject matter",
            "Uneasy tension (0.4) adds shadow to festivity",
            "Structural compression (negative_space 0.7) amplifies psychological unease"
        ]
    )

    print("\n📊 LAYER 2: INTERPRETATION OUTPUT")
    print("\nStructural Analysis:")
    print(f"  Edge Tension:        {structural.edge_tension:.1f}")
    print(f"  Negative Space:      {structural.negative_space:.1f} (0=generous, 1=compressed)")
    print(f"  Density:             {structural.density:.1f}")
    print(f"  Summary: {structural.summary}\n")

    print("Mood Analysis:")
    for dimension, value in mood.dimensions.items():
        print(f"  {dimension.capitalize():15s} {value:.1f}")
    print(f"  Summary: {mood.summary}\n")

    print("Seed Context:")
    print(f"  Subject: {seed.subject}")
    print(f"  Intent:  {seed.intent}\n")

    print("Alignment Proposal:")
    print(f"  Type:     {alignment.relationship_type.value}")
    print(f"  Register: '{alignment.register_description}'")
    print(f"  Conflicts detected: {len(alignment.conflicts)}")
    for i, conflict in enumerate(alignment.conflicts, 1):
        print(f"    {i}. {conflict}")

    # Layer 3: User modifies and synthesizes
    print("\n" + "-" * 80)
    print("👤 USER MODIFICATION (Layer 2 → Layer 3)")
    print("-" * 80)
    print("\nUser reviews interpretation and decides:")
    print("  ✓ Approve 'bittersweet celebration' register")
    print("  ✓ Reduce melancholy from 0.6 → 0.3 (soften sadness)")
    print("  ✓ Keep structural compression high (emotional intensity)")
    print("  ✓ Keep intimacy high (warmth through closeness)")

    modified_mood = {
        "melancholy": 0.3,  # User adjusted
        "intimacy": 0.7,    # Kept
        "tension": 0.4,
        "stillness": 0.3,
        "energy": 0.3,
        "warmth": 0.2,
        "unease": 0.4,
    }

    print("\n📝 LAYER 3: SYNTHESIZED PROMPT")
    print("-" * 80)
    prompt = """Frame the birthday scene with tight, edge-crowding composition that leaves 
no breathing room—the compression paradoxically creates intimacy rather than claustrophobia. 
Subjects clustered at frame edges suggest time passing, moments slipping away even as they're 
being lived. Moderate rhythm keeps the eye moving without settling; let colors be muted, 
lighting slightly overcast despite festive activity. The register is bittersweet: joy present 
but tinged with awareness of transience, childhood slipping toward adulthood, this moment 
never to be repeated. Structure enforces the emotional subtext—celebration framed by 
impermanence."""

    print(prompt)

    print("\n" + "=" * 80)
    return {
        "structural": structural,
        "mood_original": mood,
        "mood_modified": modified_mood,
        "alignment": alignment,
        "register": alignment.register_description,
        "enhanced_prompt": prompt,
    }


# ============================================================================
# EXAMPLE 2: Energetic vs. Serene (Reinforcing Relationship)
# ============================================================================

def example_reinforcing_composition():
    """
    Workflow: Analyze composition where structure AND mood reinforce each other
    (high energy structure + energetic mood = coherent, harmonious aesthetic)
    """
    print("\n" * 2)
    print("=" * 80)
    print("EXAMPLE 2: Dynamic Composition - Reinforcing Structure & Mood")
    print("=" * 80)

    structural = StructuralAnalysis(
        edge_tension=0.3,
        asymmetric_balance=0.7,
        negative_space=0.2,
        rhythm=0.8,
        density=0.7,
        scale_hierarchy=0.6,
        diagonal_dominance=0.8,
        containment=0.6,
        summary="Dynamic, energetic composition with strong diagonals creating movement; generous density and irregular rhythm; clear scale hierarchy draws eye through frame; elements escape containment, suggesting action beyond frame edges."
    )

    mood = MoodAnalysis(
        dimensions={
            "melancholy": 0.1,
            "intimacy": 0.2,
            "tension": 0.8,
            "stillness": 0.1,
            "energy": 0.9,
            "warmth": 0.4,
            "unease": 0.6,
        },
        summary="Vibrant, kinetic energy dominates; tension and dynamism create palpable sense of movement; low stillness reinforces constant motion; unease stems from visual instability, not darkness."
    )

    seed = SeedInterpretation(
        subject="urban street scene during rush hour",
        intent="capture momentum and chaos of city life"
    )

    alignment = AlignmentProposal(
        relationship_type=AlignmentType.REINFORCING,
        register_description="kinetic urban energy",
        conflicts=[]  # No conflicts—perfect alignment
    )

    print("\n📊 LAYER 2: INTERPRETATION OUTPUT")
    print("\nStructural-Mood Alignment:")
    print(f"  Diagonal dominance:  {structural.diagonal_dominance:.1f}")
    print(f"  Energy:              {mood.dimensions['energy']:.1f}")
    print(f"  → Both HIGH: Movement reinforces movement")
    print(f"  Rhythm:              {structural.rhythm:.1f}")
    print(f"  Tension:             {mood.dimensions['tension']:.1f}")
    print(f"  → Both HIGH: Irregular structure echoes tense mood\n")

    print("Seed Context:")
    print(f"  Subject: {seed.subject}")
    print(f"  Intent:  {seed.intent}\n")

    print("Alignment Proposal:")
    print(f"  Type:     {alignment.relationship_type.value}")
    print(f"  Register: '{alignment.register_description}'")
    print(f"  Conflicts: None—structure and mood work in perfect harmony\n")

    print("📝 SYNTHESIZED PROMPT DIRECTION:")
    print("-" * 80)
    prompt = """Embrace visual chaos as compositional virtue. Use harsh diagonals, competing 
textures, and overlapping subjects to create kinetic energy. High-contrast lighting with 
dramatic shadows amplifies the sense of urgency. Colors should feel saturated but slightly 
discordant—not harmonious, but alive. Frame elements bleeding off edges; let the composition 
feel unstable, caught mid-motion. Rhythm should be jarring: repeating elements interrupted by 
sudden breaks, creating visual staccato. Tension is the point—the viewer should feel the 
momentum of the moment."""

    print(prompt)
    print("=" * 80)

    return {
        "structural": structural,
        "mood": mood,
        "alignment": alignment,
        "enhanced_prompt": prompt,
    }


# ============================================================================
# EXAMPLE 3: Contrasting Composition
# ============================================================================

def example_contrasting_composition():
    """
    Workflow: Deliberate contrast where structure + mood create productive tension
    (minimal structure + energetic mood = visceral, unsettling aesthetic)
    """
    print("\n" * 2)
    print("=" * 80)
    print("EXAMPLE 3: Minimalist Geometry - Productive Tension")
    print("=" * 80)

    structural = StructuralAnalysis(
        edge_tension=0.2,
        asymmetric_balance=0.3,
        negative_space=0.9,
        rhythm=0.1,
        density=0.1,
        scale_hierarchy=0.8,
        diagonal_dominance=0.0,
        containment=0.9,
        summary="Sparse, geometric composition; vast negative space; single dominant element; horizontal/vertical lines only; perfectly regular rhythm; everything contained, nothing escapes frame. Visual restraint maximized."
    )

    mood = MoodAnalysis(
        dimensions={
            "melancholy": 0.4,
            "intimacy": 0.7,
            "tension": 0.7,
            "stillness": 0.9,
            "energy": 0.2,
            "warmth": 0.3,
            "unease": 0.8,
        },
        summary="Profound stillness mixed with deep unease; minimal energy but maximum tension; intimate in scale but cold in feeling; stillness creates a vacuum that tension rushes to fill."
    )

    seed = SeedInterpretation(
        subject="single figure in empty room",
        intent="loneliness and containment"
    )

    alignment = AlignmentProposal(
        relationship_type=AlignmentType.PRODUCTIVE_TENSION,
        register_description="existential isolation: minimalist architecture amplifying human smallness",
        conflicts=[
            "Restraint of composition (density 0.1) conflicts with tension (0.7) of mood",
            "High stillness (0.9) creates void that emotional unease rushes to fill",
            "Scale hierarchy (0.8) emphasizes human insignificance against space"
        ]
    )

    print("\n📊 LAYER 2: INTERPRETATION OUTPUT")
    print("\nStructure-Mood Tension:")
    print(f"  Negative space:      {structural.negative_space:.1f} (maximal emptiness)")
    print(f"  Density:             {structural.density:.1f} (minimal content)")
    print(f"  Stillness:           {mood.dimensions['stillness']:.1f} (extreme quietness)")
    print(f"  BUT Tension:         {mood.dimensions['tension']:.1f} (underlying unease)")
    print(f"  → Visual emptiness amplifies emotional discomfort\n")

    print("Seed Context:")
    print(f"  Subject: {seed.subject}")
    print(f"  Intent:  {seed.intent}\n")

    print("Alignment Proposal:")
    print(f"  Type:     {alignment.relationship_type.value}")
    print(f"  Register: '{alignment.register_description}'")
    for i, conflict in enumerate(alignment.conflicts, 1):
        print(f"    {i}. {conflict}\n")

    print("📝 SYNTHESIZED PROMPT DIRECTION:")
    print("-" * 80)
    prompt = """Let emptiness become the subject. Frame the figure small within vast, 
rectilinear space—architecture as psychological weight. Horizontal and vertical lines create 
order, but the proportions should feel slightly off; make the space seem too vast relative to 
human scale. Lighting cold and flat; no shadows to offer comfort or visual interest. Negative 
space becomes emotional content: the void around the figure speaks louder than the figure itself. 
Tension emerges from this contradiction—visual silence concealing psychological turmoil. The 
composition says: I am here, but vastly, invisibly alone."""

    print(prompt)
    print("=" * 80)

    return {
        "structural": structural,
        "mood": mood,
        "alignment": alignment,
        "enhanced_prompt": prompt,
    }


# ============================================================================
# Runner
# ============================================================================

def main():
    """Run all examples and demonstrate workflow"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  GRID DYNAMICS MCP: Three-Layer Olog Workflow Demonstration".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")

    results = []

    # Run all examples
    results.append(("Birthday Party", example_birthday_party_workflow()))
    results.append(("Energetic Composition", example_reinforcing_composition()))
    results.append(("Minimalist Isolation", example_contrasting_composition()))

    # Summary
    print("\n" * 2)
    print("=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    print("\nAll three examples demonstrate the three-layer olog cycle:\n")
    print("  1. Layer 1 (Domain):         Types define categorical structure")
    print("  2. Layer 2 (Interpretation): Vision models extract + propose alignment")
    print("  3. Layer 3 (Synthesis):      LLM reconciles layers into coherent prompt\n")
    print("Key insight: Conflicts are productive. Contrasting structure/mood create")
    print("emergent aesthetic registers (bittersweet, existential, kinetic) that")
    print("neither layer produces alone.\n")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
