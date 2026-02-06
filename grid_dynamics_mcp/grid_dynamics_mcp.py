"""
Grid Dynamics MCP Server
Analyzes compositional structure and mood, proposes alignments, and synthesizes enhanced prompts.
Three-layer olog architecture: Domain → Interpretation → Synthesis

STATELESS DESIGN: This MCP receives pre-extracted structural and mood parameters
(from Claude's vision model or other sources) and performs deterministic alignment 
proposal + synthesis. Claude orchestrates the image analysis; Grid Dynamics handles 
the interpretation and synthesis layers.
"""

import json
import math
import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

import numpy as np
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("grid-dynamics")


# ============================================================================
# LAYER 1: Domain Types (Categorical Structure)
# ============================================================================

class AlignmentType(str, Enum):
    REINFORCING = "reinforcing"
    CONTRASTING = "contrasting"
    PRODUCTIVE_TENSION = "productive_tension"


class PromptLength(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    DETAILED = "detailed"


@dataclass
class StructuralAnalysis:
    """Structural composition parameters extracted from image"""
    edge_tension: float  # 0-1: subjects crowding frame boundaries
    asymmetric_balance: float  # 0-1: weight distribution
    negative_space: float  # 0-1: 0=generous, 1=compressed
    rhythm: float  # 0-1: 0=regular, 1=irregular
    density: float  # 0-1: sparse to overflowing
    scale_hierarchy: float  # 0-1: 0=uniform, 1=strong dominant element
    diagonal_dominance: float  # 0-1: movement/instability
    containment: float  # 0-1: 0=contained, 1=bleeding/cropped
    summary: str  # Natural language description


@dataclass
class MoodAnalysis:
    """Mood/sentiment dimensions extracted from image"""
    dimensions: dict[str, float]  # e.g., {"melancholy": 0.6, "intimacy": 0.7}
    summary: str  # Natural language description


@dataclass
class SeedInterpretation:
    """Parsed intent from optional seed text"""
    subject: str
    intent: str


@dataclass
class AlignmentProposal:
    """Proposed relationship between structure, mood, and seed"""
    relationship_type: AlignmentType
    register_description: str  # e.g., "claustrophobic tenderness"
    conflicts: list[str]  # Detected tensions between dimensions


@dataclass
class ExtractionResult:
    """Complete Layer 2 interpretation output"""
    structural_analysis: StructuralAnalysis
    mood_analysis: MoodAnalysis
    seed_interpretation: Optional[SeedInterpretation]
    proposed_alignment: Optional[AlignmentProposal]
    editable: bool = True





def propose_alignment(
    structural: StructuralAnalysis,
    mood: MoodAnalysis,
    seed: Optional[SeedInterpretation],
) -> AlignmentProposal:
    """
    Alignment Proposal: Deterministic rule-based inference.
    Suggests how structure, mood, and seed interact expressively.
    
    Rules:
    - If structure and mood align (high/low in same direction): REINFORCING
    - If structure and mood oppose (one high, one low): CONTRASTING or PRODUCTIVE_TENSION
    - If seed intent conflicts with mood: PRODUCTIVE_TENSION
    """
    conflicts = []
    
    # Rule 1: Detect structural-mood alignment
    high_energy_structure = (structural.diagonal_dominance > 0.6 or 
                             structural.rhythm > 0.6)
    low_energy_structure = (structural.diagonal_dominance < 0.4 and 
                            structural.rhythm < 0.4)
    
    high_energy_mood = mood.dimensions.get("energy", 0) > 0.6
    low_energy_mood = mood.dimensions.get("energy", 0) < 0.4
    
    structural_compressed = structural.negative_space > 0.6
    mood_uneasy = mood.dimensions.get("unease", 0) > 0.5
    
    # Rule 2: Check seed intent vs mood
    has_intent_conflict = False
    if seed:
        celebratory_intent = any(word in seed.intent.lower() 
                                 for word in ["celebr", "joyful", "happy", "festive", "fun"])
        melancholic_mood = mood.dimensions.get("melancholy", 0) > 0.5
        
        if celebratory_intent and melancholic_mood:
            conflicts.append("Melancholic mood contradicts celebratory intent")
            has_intent_conflict = True
        
        lonely_intent = any(word in seed.intent.lower() 
                            for word in ["lone", "isolat", "alone"])
        warm_mood = mood.dimensions.get("warmth", 0) > 0.6
        
        if lonely_intent and warm_mood:
            conflicts.append("Warm mood contradicts isolated intent")
    
    # Rule 3: Check structural-mood tension
    if structural_compressed and mood_uneasy:
        conflicts.append("Structural compression amplifies emotional unease")
    
    if high_energy_structure and high_energy_mood:
        relationship = AlignmentType.REINFORCING
        register = "kinetic and dynamic"
    elif low_energy_structure and low_energy_mood:
        relationship = AlignmentType.REINFORCING
        register = "still and contained"
    elif (high_energy_structure and low_energy_mood) or (low_energy_structure and high_energy_mood):
        relationship = AlignmentType.CONTRASTING
        register = "contradictory visual-emotional tension"
    else:
        relationship = AlignmentType.PRODUCTIVE_TENSION
        register = "compositionally complex"
    
    # If we have conflicts, upgrade to productive tension (where applicable)
    if conflicts and relationship != AlignmentType.REINFORCING:
        relationship = AlignmentType.PRODUCTIVE_TENSION
    
    # Add seed-based register if available
    if seed and has_intent_conflict:
        register = f"{ seed.intent.lower()} shadowed by melancholy"
    
    return AlignmentProposal(
        relationship_type=relationship,
        register_description=register,
        conflicts=conflicts,
    )


# ============================================================================
# LAYER 3: Synthesis (Enhanced Prompt Generation)
# ============================================================================

def synthesize_prompt(
    structural_values: dict,
    mood_values: dict,
    alignment_override: Optional[str] = None,
    seed_text: Optional[str] = None,
    target_length: PromptLength = PromptLength.MEDIUM,
) -> str:
    """
    Synthesis: Deterministic function that translates structural and mood parameters
    into compositional prompt language.
    
    Uses rules-based parameter-to-language mapping rather than LLM generation
    to maintain reproducibility and control.
    """
    
    # Extract key values
    edge_tension = structural_values.get("edge_tension", 0.5)
    negative_space = structural_values.get("negative_space", 0.5)
    density = structural_values.get("density", 0.5)
    diagonal_dominance = structural_values.get("diagonal_dominance", 0.5)
    rhythm = structural_values.get("rhythm", 0.5)
    containment = structural_values.get("containment", 0.5)
    
    melancholy = mood_values.get("melancholy", 0.5)
    intimacy = mood_values.get("intimacy", 0.5)
    energy = mood_values.get("energy", 0.5)
    warmth = mood_values.get("warmth", 0.5)
    tension = mood_values.get("tension", 0.5)
    
    # Build compositional language from parameters
    parts = []
    
    # Spatial/containment description
    if negative_space > 0.7:
        parts.append("Frame with generous negative space, allowing subjects to breathe")
    elif negative_space > 0.4:
        parts.append("Compose with moderate spacing between elements")
    else:
        parts.append("Create a compressed composition with subjects clustered tightly")
    
    # Edge tension description
    if edge_tension > 0.7:
        parts.append("Position subjects crowding frame edges, creating visual tension")
    elif edge_tension > 0.4:
        parts.append("Place elements with moderate edge presence")
    else:
        parts.append("Center subjects with breathing room from frame boundaries")
    
    # Diagonal/rhythm description
    if diagonal_dominance > 0.7:
        parts.append("Use strong diagonal lines to create dynamic movement and visual urgency")
    elif diagonal_dominance > 0.3:
        parts.append("Mix diagonal and horizontal-vertical lines for balanced composition")
    else:
        parts.append("Emphasize horizontal and vertical lines for stability and order")
    
    # Rhythm/regularity
    if rhythm > 0.7:
        parts.append("Create irregular, chaotic rhythm with unpredictable element placement")
    elif rhythm > 0.3:
        parts.append("Establish moderate rhythm balancing pattern and variation")
    else:
        parts.append("Maintain regular, predictable rhythm and repetition")
    
    # Emotional tone from mood
    if warmth > 0.6:
        parts.append("Use warm tones and soft lighting to create comfort")
    elif warmth < 0.4:
        parts.append("Employ cool tones and harsh lighting for distance")
    
    if intimacy > 0.6:
        parts.append("Frame with close proximity, making the scene feel personal and immediate")
    elif intimacy < 0.4:
        parts.append("Distance the viewer from subjects for objectivity")
    
    if energy > 0.7:
        parts.append("Infuse the scene with kinetic motion and palpable momentum")
    elif energy < 0.3:
        parts.append("Emphasize stillness and quiet presence")
    
    if melancholy > 0.6:
        parts.append("Let a melancholic undertone permeate the image, suggesting impermanence")
    elif melancholy > 0.3:
        parts.append("Introduce subtle sadness or nostalgia")
    
    if tension > 0.6:
        parts.append("Create palpable visual tension that unsettles the viewer")
    elif tension < 0.3:
        parts.append("Establish harmony and visual resolution")
    
    # Combine with register description if provided
    register_phrase = alignment_override or "the compositional-emotional interaction"
    parts.insert(0, f"Create an image that embodies {register_phrase}.")
    
    # Join based on target length
    if target_length == PromptLength.SHORT:
        # Take first 2-3 parts
        prompt = " ".join(parts[:3])
    elif target_length == PromptLength.DETAILED:
        # Use all parts
        prompt = " ".join(parts)
    else:  # MEDIUM
        # Use 4-5 parts
        prompt = " ".join(parts[:5])
    
    # Add seed context if provided
    if seed_text:
        prompt += f" The subject is {seed_text}."
    
    return prompt


def extract_impl(
    structural_values: dict,
    mood_values: dict,
    seed_text: Optional[str] = None,
) -> dict:
    """
    Core extraction implementation (testable).
    Receives pre-extracted structural and mood parameters (from Claude's vision model)
    and performs alignment proposal + conflict detection.
    
    This MCP is STATELESS—Claude handles image analysis, Grid Dynamics handles
    Layer 2 interpretation (alignment proposal) and Layer 3 synthesis.
    
    Args:
        structural_values: Dict with structural parameters (8 keys, values 0-1)
        mood_values: Dict with mood dimensions (7 keys, values 0-1)
        seed_text: Optional seed text describing subject/intent
    
    Returns:
        Dictionary with structural_analysis, mood_analysis, proposed_alignment,
        and editable flag set to true for user modification
    """
    # Reconstruct dataclasses from input dictionaries
    structural = StructuralAnalysis(
        edge_tension=structural_values.get("edge_tension", 0.5),
        asymmetric_balance=structural_values.get("asymmetric_balance", 0.5),
        negative_space=structural_values.get("negative_space", 0.5),
        rhythm=structural_values.get("rhythm", 0.5),
        density=structural_values.get("density", 0.5),
        scale_hierarchy=structural_values.get("scale_hierarchy", 0.5),
        diagonal_dominance=structural_values.get("diagonal_dominance", 0.5),
        containment=structural_values.get("containment", 0.5),
        summary=structural_values.get("summary", "Compositional analysis from Claude")
    )
    
    mood = MoodAnalysis(
        dimensions=mood_values,
        summary=mood_values.get("summary", "Mood analysis from Claude")
    )
    
    # Parse seed if provided
    seed_interp = None
    if seed_text:
        # Simple inline seed parsing (no Claude call)
        seed_interp = SeedInterpretation(
            subject=seed_text.split(',')[0].strip() if ',' in seed_text else seed_text,
            intent=seed_text.split(',')[1].strip() if ',' in seed_text else "aesthetic intent"
        )
    
    # Layer 2: Propose alignment (deterministic rule-based)
    alignment = propose_alignment(structural, mood, seed_interp)
    
    result = ExtractionResult(
        structural_analysis=structural,
        mood_analysis=mood,
        seed_interpretation=seed_interp,
        proposed_alignment=alignment,
        editable=True,
    )
    
    # Convert to dict for JSON serialization
    return {
        "structural_analysis": {
            "values": {
                "edge_tension": structural.edge_tension,
                "asymmetric_balance": structural.asymmetric_balance,
                "negative_space": structural.negative_space,
                "rhythm": structural.rhythm,
                "density": structural.density,
                "scale_hierarchy": structural.scale_hierarchy,
                "diagonal_dominance": structural.diagonal_dominance,
                "containment": structural.containment,
            },
            "summary": structural.summary,
        },
        "mood_analysis": {
            "dimensions": mood.dimensions,
            "summary": mood.summary,
        },
        "seed_interpretation": (
            asdict(seed_interp) if seed_interp else None
        ),
        "proposed_alignment": {
            "relationship_type": alignment.relationship_type.value,
            "register_description": alignment.register_description,
            "conflicts": alignment.conflicts,
        },
        "editable": True,
    }


def synthesize_impl(
    structural_values: dict,
    mood_values: dict,
    alignment_override: Optional[str] = None,
    seed_text: Optional[str] = None,
    target_length: str = "medium",
) -> dict:
    """
    Core synthesis implementation (testable).
    Synthesize enhanced prompt from modified structural and mood parameters.
    
    Args:
        structural_values: Modified structural analysis dictionary
        mood_values: Modified mood analysis dictionary
        alignment_override: Optional override for register description
        seed_text: Optional seed text context
        target_length: "short", "medium", or "detailed"
    
    Returns:
        Dictionary with enhanced_prompt string
    """
    prompt_length = PromptLength(target_length)

    enhanced_prompt = synthesize_prompt(
        structural_values=structural_values,
        mood_values=mood_values,
        alignment_override=alignment_override,
        seed_text=seed_text,
        target_length=prompt_length,
    )

    return {
        "enhanced_prompt": enhanced_prompt,
    }



# ============================================================================
# STRATEGIC ANALYSIS: Tomographic Domain Projection
# ============================================================================

# Strategic pattern vocabulary (8 structural dimensions)
STRATEGIC_PATTERNS = {
    "edge_tension": {
        "boundary_contact": {
            "pattern": r"(external|outside|beyond|outside of|partner|stakeholder|community|public)",
            "threshold": 5,
            "confidence": 0.8,
        },
        "boundary_safe": {
            "pattern": r"(internal|within|inside|member|staff|employee|committee)",
            "threshold": 5,
            "confidence": 0.7,
        },
    },
    "asymmetric_balance": {
        "radial_dispersed": {
            "pattern": r"(board|council|committee|chair|assembly|president|director|officer)",
            "threshold": 4,
            "confidence": 0.75,
            "referral_pattern": r"(can refer|may refer|or\s+\w+\s+can|council or board)",
        },
        "bilateral_symmetric": {
            "pattern": r"(chair (and|with) vice|co-chair|dual leadership|shared authority)",
            "threshold": 2,
            "confidence": 0.8,
        },
    },
    "negative_space": {
        "partial_specification": {
            "resource_language": r"(finite|limited|resource|priorit|allocat|budget)",
            "allocation_markers": r"(\d+%|percent|proportion|ratio|allocated to|dedicated to)",
            "threshold": 2,
            "confidence": 0.7,
        },
        "generous_specification": {
            "pattern": r"(\d+%|percent of|ratio of|\$\d+|allocated \d+|specific amount)",
            "threshold": 3,
            "confidence": 0.85,
        },
    },
    "rhythm": {
        "polyrhythmic": {
            "time_horizons": r"(quarter|annual|year|month|week|biannual|five-year|strategic plan)",
            "threshold": 3,
            "confidence": 0.75,
        },
        "arrhythmic": {
            "pattern": r"(as needed|when necessary|ad hoc|ongoing|continuous)",
            "threshold": 3,
            "confidence": 0.7,
        },
    },
    "density": {
        "sparse_articulation": {
            "pattern": r"(goal|objective|priority|focus area|strategic initiative)",
            "max_count": 5,
            "confidence": 0.75,
        },
        "overflowing": {
            "pattern": r"(goal|objective|priority|focus area|strategic initiative)",
            "min_count": 12,
            "confidence": 0.8,
        },
    },
    "scale_hierarchy": {
        "inverted": {
            "goal_pattern": r"(goal\s+\d+|objective\s+\d+|priority\s+\d+|strategic goal\s+\d+)",
            "hierarchy_markers": r"(foundational|primary|first priority|core|essential|prerequisite)",
            "foundational_terms": r"(organizational|operation|competence|capacity|infrastructure|foundation)",
            "min_goals": 4,
            "confidence": 0.85,
        },
        "clear_precedence": {
            "pattern": r"(first|primary|foundational goal|core priority|must precede|prerequisite)",
            "threshold": 3,
            "confidence": 0.8,
        },
    },
    "diagonal_dominance": {
        "diagonal": {
            "pattern": r"(pivot|shift|transition|transform|change direction|evolve)",
            "threshold": 4,
            "confidence": 0.75,
        },
        "orthogonal": {
            "pattern": r"(maintain|sustain|continue|preserve|stability|consistent)",
            "threshold": 4,
            "confidence": 0.7,
        },
    },
    "containment": {
        "bleeding_edges": {
            "pattern": r"(partner|external|coalition|collaborate|joint|alliance|stakeholder engagement)",
            "threshold": 5,
            "confidence": 0.8,
        },
        "well_contained": {
            "pattern": r"(internal|within the|member-only|exclusive to|limited to members)",
            "threshold": 3,
            "confidence": 0.75,
        },
    },
}


def detect_asymmetric_balance(text: str) -> Tuple[Optional[str], float, List[str]]:
    """Detect authority distribution patterns."""
    
    # Check for radial dispersed pattern (multiple authority entities)
    authority_pattern = STRATEGIC_PATTERNS["asymmetric_balance"]["radial_dispersed"]["pattern"]
    authorities = re.findall(authority_pattern, text)
    unique_authorities = set(authorities)
    
    # Check for referral patterns (can refer to X or Y)
    referral_pattern = STRATEGIC_PATTERNS["asymmetric_balance"]["radial_dispersed"]["referral_pattern"]
    referral_matches = re.findall(referral_pattern, text)
    
    if len(unique_authorities) >= STRATEGIC_PATTERNS["asymmetric_balance"]["radial_dispersed"]["threshold"]:
        evidence = [
            f"Found {len(unique_authorities)} distinct authority entities: {', '.join(sorted(unique_authorities)[:5])}"
        ]
        if referral_matches:
            evidence.append(f"Referral pattern detected: '{referral_matches[0]}'")
        return "radial_dispersed", STRATEGIC_PATTERNS["asymmetric_balance"]["radial_dispersed"]["confidence"], evidence
    
    # Check for bilateral symmetric
    bilateral_pattern = STRATEGIC_PATTERNS["asymmetric_balance"]["bilateral_symmetric"]["pattern"]
    bilateral_matches = re.findall(bilateral_pattern, text)
    if len(bilateral_matches) >= STRATEGIC_PATTERNS["asymmetric_balance"]["bilateral_symmetric"]["threshold"]:
        return "bilateral_symmetric", STRATEGIC_PATTERNS["asymmetric_balance"]["bilateral_symmetric"]["confidence"], [
            f"Bilateral leadership pattern: {bilateral_matches[0]}"
        ]
    
    return None, 0.0, []


def detect_negative_space(text: str) -> Tuple[Optional[str], float, List[str]]:
    """Detect resource allocation specification levels."""
    
    # Check for partial specification (resource language without allocation details)
    resource_lang = re.findall(
        STRATEGIC_PATTERNS["negative_space"]["partial_specification"]["resource_language"],
        text
    )
    allocation_markers = re.findall(
        STRATEGIC_PATTERNS["negative_space"]["partial_specification"]["allocation_markers"],
        text
    )
    
    if len(resource_lang) >= STRATEGIC_PATTERNS["negative_space"]["partial_specification"]["threshold"]:
        if len(allocation_markers) == 0:
            return "partial_specification", STRATEGIC_PATTERNS["negative_space"]["partial_specification"]["confidence"], [
                f"Resource language present ({len(resource_lang)} instances) but no allocation mechanisms specified"
            ]
    
    # Check for generous specification (explicit percentages/amounts)
    generous_matches = re.findall(
        STRATEGIC_PATTERNS["negative_space"]["generous_specification"]["pattern"],
        text
    )
    if len(generous_matches) >= STRATEGIC_PATTERNS["negative_space"]["generous_specification"]["threshold"]:
        return "generous_specification", STRATEGIC_PATTERNS["negative_space"]["generous_specification"]["confidence"], [
            f"Explicit allocation markers found: {generous_matches[:3]}"
        ]
    
    return None, 0.0, []


def detect_rhythm(text: str) -> Tuple[Optional[str], float, List[str]]:
    """Detect temporal rhythm patterns."""
    
    # Check for polyrhythmic (multiple time horizons)
    time_matches = re.findall(
        STRATEGIC_PATTERNS["rhythm"]["polyrhythmic"]["time_horizons"],
        text
    )
    unique_horizons = set(time_matches)
    
    if len(unique_horizons) >= STRATEGIC_PATTERNS["rhythm"]["polyrhythmic"]["threshold"]:
        return "polyrhythmic", STRATEGIC_PATTERNS["rhythm"]["polyrhythmic"]["confidence"], [
            f"Multiple time horizons detected: {', '.join(sorted(unique_horizons)[:5])}"
        ]
    
    # Check for arrhythmic (as-needed, ad-hoc)
    arrhythmic_matches = re.findall(
        STRATEGIC_PATTERNS["rhythm"]["arrhythmic"]["pattern"],
        text
    )
    if len(arrhythmic_matches) >= STRATEGIC_PATTERNS["rhythm"]["arrhythmic"]["threshold"]:
        return "arrhythmic", STRATEGIC_PATTERNS["rhythm"]["arrhythmic"]["confidence"], [
            f"Arrhythmic pattern: {arrhythmic_matches[:3]}"
        ]
    
    return None, 0.0, []


def detect_scale_hierarchy(text: str) -> Tuple[Optional[str], float, List[str]]:
    """Detect goal hierarchy patterns - especially inverted hierarchies."""
    
    # Find all goals
    goal_matches = re.findall(
        STRATEGIC_PATTERNS["scale_hierarchy"]["inverted"]["goal_pattern"],
        text
    )
    
    if len(goal_matches) < STRATEGIC_PATTERNS["scale_hierarchy"]["inverted"]["min_goals"]:
        return None, 0.0, []
    
    # Check for explicit hierarchy markers
    hierarchy_markers = re.findall(
        STRATEGIC_PATTERNS["scale_hierarchy"]["inverted"]["hierarchy_markers"],
        text
    )
    
    # Check if foundational language appears late in goal sequence
    foundational_matches = re.findall(
        STRATEGIC_PATTERNS["scale_hierarchy"]["inverted"]["foundational_terms"],
        text
    )
    
    # Inverted hierarchy: foundational elements appear last WITHOUT precedence markers
    if foundational_matches and not hierarchy_markers:
        # Check if foundational language appears in later goals
        # Simple heuristic: split text into thirds, check if foundational terms in last third
        text_thirds = len(text) // 3
        last_third = text[text_thirds * 2:]
        foundational_in_last_third = any(
            re.search(STRATEGIC_PATTERNS["scale_hierarchy"]["inverted"]["foundational_terms"], last_third)
        )
        
        if foundational_in_last_third:
            return "inverted", STRATEGIC_PATTERNS["scale_hierarchy"]["inverted"]["confidence"], [
                f"Found {len(goal_matches)} goals without hierarchy markers",
                f"Foundational language ({foundational_matches[0]}) appears late in document",
                "Foundation positioned last without precedence specification"
            ]
    
    # Check for clear precedence
    if len(hierarchy_markers) >= STRATEGIC_PATTERNS["scale_hierarchy"]["clear_precedence"]["threshold"]:
        return "clear_precedence", STRATEGIC_PATTERNS["scale_hierarchy"]["clear_precedence"]["confidence"], [
            f"Clear hierarchy markers: {hierarchy_markers[:3]}"
        ]
    
    return None, 0.0, []


def analyze_strategy_document(strategy_text: str) -> dict:
    """
    Project strategy document through Grid Dynamics structural dimensions.
    
    Pure Layer 2 deterministic pattern matching - zero LLM cost.
    Returns findings with confidence scores and evidence.
    
    Args:
        strategy_text: Full text of strategy document
    
    Returns:
        Dictionary with structural findings, each containing:
        - dimension: structural dimension name
        - pattern: detected pattern type
        - confidence: 0.0-1.0 confidence score
        - evidence: list of supporting text patterns
        - categorical_family: "morphisms" for structural relationships
    """
    findings = []
    
    # Preprocess text once (optimization: avoid redundant lowercasing)
    text_lower = strategy_text.lower()
    
    # Run all pattern detectors
    detectors = [
        ("asymmetric_balance", detect_asymmetric_balance),
        ("negative_space", detect_negative_space),
        ("rhythm", detect_rhythm),
        ("scale_hierarchy", detect_scale_hierarchy),
    ]
    
    for dimension, detector in detectors:
        pattern, confidence, evidence = detector(text_lower)
        if pattern and confidence > 0.6:  # Threshold for reporting
            findings.append({
                "dimension": dimension,
                "pattern": pattern,
                "confidence": confidence,
                "evidence": evidence,
                "categorical_family": "morphisms",  # Structural relationships
            })
    
    return {
        "domain": "grid_dynamics",
        "findings": findings,
        "total_findings": len(findings),
        "methodology": "deterministic_pattern_matching",
        "llm_cost_tokens": 0,
    }


# ============================================================================
# PHASE 2.6: Rhythmic Presets & Trajectory Generation
# ============================================================================
#
# Grid Dynamics operates in a 15D morphospace (8 structural + 7 mood).
# Canonical states represent compositional archetypes — recognizable
# configurations that image-makers intuitively navigate between.
# Rhythmic presets define oscillation paths through this space.
#
# Parameter ordering (consistent across all states/presets):
#   Structural [0-7]: edge_tension, asymmetric_balance, negative_space,
#                     rhythm, density, scale_hierarchy, diagonal_dominance,
#                     containment
#   Mood [8-14]:      melancholy, intimacy, tension, stillness, energy,
#                     warmth, unease
# ============================================================================

GRID_STRUCTURAL_PARAMS = [
    "edge_tension",
    "asymmetric_balance",
    "negative_space",
    "rhythm",
    "density",
    "scale_hierarchy",
    "diagonal_dominance",
    "containment",
]

GRID_MOOD_PARAMS = [
    "melancholy",
    "intimacy",
    "tension",
    "stillness",
    "energy",
    "warmth",
    "unease",
]

GRID_PARAMETER_NAMES = GRID_STRUCTURAL_PARAMS + GRID_MOOD_PARAMS

# ---------------------------------------------------------------------------
# Canonical compositional states (15D coordinates)
# ---------------------------------------------------------------------------

GRID_CANONICAL_STATES: Dict[str, Dict[str, float]] = {
    "claustrophobic_tension": {
        # Compressed frame, subjects pressed against edges, high visual stress
        "edge_tension": 0.90, "asymmetric_balance": 0.65, "negative_space": 0.15,
        "rhythm": 0.70, "density": 0.85, "scale_hierarchy": 0.40,
        "diagonal_dominance": 0.60, "containment": 0.30,
        # Mood: anxious, close, tense
        "melancholy": 0.35, "intimacy": 0.75, "tension": 0.90,
        "stillness": 0.15, "energy": 0.65, "warmth": 0.25, "unease": 0.85,
    },
    "open_contemplation": {
        # Generous breathing room, centered subject, quiet
        "edge_tension": 0.10, "asymmetric_balance": 0.20, "negative_space": 0.90,
        "rhythm": 0.15, "density": 0.15, "scale_hierarchy": 0.30,
        "diagonal_dominance": 0.10, "containment": 0.10,
        # Mood: meditative, distanced, still
        "melancholy": 0.40, "intimacy": 0.20, "tension": 0.10,
        "stillness": 0.90, "energy": 0.10, "warmth": 0.45, "unease": 0.05,
    },
    "dynamic_chaos": {
        # Strong diagonals, irregular placement, kinetic energy
        "edge_tension": 0.55, "asymmetric_balance": 0.80, "negative_space": 0.35,
        "rhythm": 0.90, "density": 0.70, "scale_hierarchy": 0.55,
        "diagonal_dominance": 0.95, "containment": 0.65,
        # Mood: electric, urgent, volatile
        "melancholy": 0.10, "intimacy": 0.30, "tension": 0.75,
        "stillness": 0.05, "energy": 0.95, "warmth": 0.40, "unease": 0.55,
    },
    "intimate_stillness": {
        # Close framing, warm tones, quiet presence
        "edge_tension": 0.25, "asymmetric_balance": 0.35, "negative_space": 0.45,
        "rhythm": 0.20, "density": 0.40, "scale_hierarchy": 0.25,
        "diagonal_dominance": 0.15, "containment": 0.20,
        # Mood: tender, close, warm, hushed
        "melancholy": 0.30, "intimacy": 0.95, "tension": 0.10,
        "stillness": 0.80, "energy": 0.10, "warmth": 0.90, "unease": 0.05,
    },
    "monumental_order": {
        # Strong hierarchy, regular rhythm, contained and formal
        "edge_tension": 0.20, "asymmetric_balance": 0.15, "negative_space": 0.50,
        "rhythm": 0.05, "density": 0.55, "scale_hierarchy": 0.95,
        "diagonal_dominance": 0.05, "containment": 0.05,
        # Mood: imposing, distanced, stable
        "melancholy": 0.15, "intimacy": 0.10, "tension": 0.20,
        "stillness": 0.70, "energy": 0.25, "warmth": 0.20, "unease": 0.10,
    },
    "bleeding_urgency": {
        # Subjects escape frame, high diagonals, palpable tension
        "edge_tension": 0.80, "asymmetric_balance": 0.70, "negative_space": 0.20,
        "rhythm": 0.75, "density": 0.80, "scale_hierarchy": 0.60,
        "diagonal_dominance": 0.85, "containment": 0.95,
        # Mood: raw, exposed, urgent
        "melancholy": 0.20, "intimacy": 0.50, "tension": 0.85,
        "stillness": 0.05, "energy": 0.85, "warmth": 0.30, "unease": 0.75,
    },
    "melancholic_drift": {
        # Off-balance weight, generous but heavy space, wistful
        "edge_tension": 0.30, "asymmetric_balance": 0.75, "negative_space": 0.65,
        "rhythm": 0.35, "density": 0.30, "scale_hierarchy": 0.45,
        "diagonal_dominance": 0.40, "containment": 0.25,
        # Mood: sad, distanced, cool, nostalgic
        "melancholy": 0.90, "intimacy": 0.35, "tension": 0.25,
        "stillness": 0.60, "energy": 0.20, "warmth": 0.30, "unease": 0.30,
    },
}

# ---------------------------------------------------------------------------
# Phase 2.6 Rhythmic Presets
# ---------------------------------------------------------------------------
# Each preset oscillates between two canonical states at a defined period.
# Periods chosen for productive emergent behavior when composed with other
# domains (microscopy [10,16,20,24,30], nuclear [15,18], etc.).
# ---------------------------------------------------------------------------

GRID_RHYTHMIC_PRESETS: Dict[str, dict] = {
    "tension_release": {
        "state_a": "claustrophobic_tension",
        "state_b": "open_contemplation",
        "pattern": "sinusoidal",
        "num_cycles": 3,
        "steps_per_cycle": 20,
        "description": "Pressure builds in compressed framing then releases into open space",
    },
    "energy_cycle": {
        "state_a": "dynamic_chaos",
        "state_b": "intimate_stillness",
        "pattern": "sinusoidal",
        "num_cycles": 3,
        "steps_per_cycle": 24,
        "description": "Kinetic explosion collapses into tender quiet, then rebuilds",
    },
    "scale_breathing": {
        "state_a": "monumental_order",
        "state_b": "open_contemplation",
        "pattern": "triangular",
        "num_cycles": 4,
        "steps_per_cycle": 16,
        "description": "Formal hierarchy dissolves into meditative emptiness and reforms",
    },
    "mood_tide": {
        "state_a": "melancholic_drift",
        "state_b": "intimate_stillness",
        "pattern": "sinusoidal",
        "num_cycles": 4,
        "steps_per_cycle": 18,
        "description": "Wistful distance alternates with tender closeness",
    },
    "containment_pulse": {
        "state_a": "bleeding_urgency",
        "state_b": "monumental_order",
        "pattern": "sinusoidal",
        "num_cycles": 3,
        "steps_per_cycle": 22,
        "description": "Raw urgency snaps into rigid containment then fractures again",
    },
}


def _generate_oscillation(num_steps: int, num_cycles: float, pattern: str) -> np.ndarray:
    """Generate oscillation envelope α ∈ [0, 1]."""
    t = np.linspace(0, 2 * np.pi * num_cycles, num_steps, endpoint=False)

    if pattern == "sinusoidal":
        return 0.5 * (1.0 + np.sin(t))
    elif pattern == "triangular":
        t_norm = (t / (2 * np.pi)) % 1.0
        return np.where(t_norm < 0.5, 2.0 * t_norm, 2.0 * (1.0 - t_norm))
    elif pattern == "square":
        t_norm = (t / (2 * np.pi)) % 1.0
        return np.where(t_norm < 0.5, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown oscillation pattern: {pattern}")


def generate_grid_preset_trajectory(preset_name: str) -> List[Dict[str, float]]:
    """
    Generate a complete Phase 2.6 preset trajectory in 15D grid-dynamics space.

    Returns list of state dicts (one per timestep), each containing all 15
    parameter values interpolated between state_a and state_b.
    """
    if preset_name not in GRID_RHYTHMIC_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available: {list(GRID_RHYTHMIC_PRESETS.keys())}"
        )

    cfg = GRID_RHYTHMIC_PRESETS[preset_name]
    state_a = GRID_CANONICAL_STATES[cfg["state_a"]]
    state_b = GRID_CANONICAL_STATES[cfg["state_b"]]

    total_steps = cfg["num_cycles"] * cfg["steps_per_cycle"]
    alpha = _generate_oscillation(total_steps, cfg["num_cycles"], cfg["pattern"])

    trajectory: List[Dict[str, float]] = []
    for step_idx in range(total_steps):
        a = float(alpha[step_idx])
        state = {
            p: round(state_a[p] * (1.0 - a) + state_b[p] * a, 6)
            for p in GRID_PARAMETER_NAMES
        }
        trajectory.append(state)
    return trajectory


def generate_all_grid_preset_trajectories() -> Dict[str, List[Dict[str, float]]]:
    """Generate trajectories for every registered preset."""
    return {
        name: generate_grid_preset_trajectory(name)
        for name in GRID_RHYTHMIC_PRESETS
    }


# ============================================================================
# PHASE 2.7: Attractor Visualization — Vocabulary & Prompt Generation
# ============================================================================
#
# Translates 15D grid-dynamics coordinates into image-generation vocabulary.
# Each visual type is a recognizable compositional archetype with associated
# keywords that image generators understand natively.
# ============================================================================

GRID_VISUAL_TYPES: Dict[str, dict] = {
    "compressed_intimate": {
        "coords": {
            "edge_tension": 0.85, "asymmetric_balance": 0.55, "negative_space": 0.15,
            "rhythm": 0.55, "density": 0.80, "scale_hierarchy": 0.35,
            "diagonal_dominance": 0.45, "containment": 0.30,
            "melancholy": 0.30, "intimacy": 0.85, "tension": 0.70,
            "stillness": 0.20, "energy": 0.50, "warmth": 0.55, "unease": 0.60,
        },
        "keywords": [
            "tight close-up framing",
            "subjects pressed against frame edges",
            "shallow depth of field with bokeh",
            "warm directional side-lighting",
            "compressed negative space",
            "claustrophobic tenderness",
            "visible skin texture and pores",
        ],
    },
    "expansive_contemplative": {
        "coords": {
            "edge_tension": 0.10, "asymmetric_balance": 0.20, "negative_space": 0.90,
            "rhythm": 0.10, "density": 0.10, "scale_hierarchy": 0.25,
            "diagonal_dominance": 0.10, "containment": 0.10,
            "melancholy": 0.45, "intimacy": 0.15, "tension": 0.10,
            "stillness": 0.90, "energy": 0.10, "warmth": 0.40, "unease": 0.05,
        },
        "keywords": [
            "vast negative space surrounding lone subject",
            "centered minimalist composition",
            "even diffused ambient light",
            "muted desaturated palette",
            "meditative stillness",
            "deep depth of field",
            "architectural emptiness",
        ],
    },
    "dynamic_kinetic": {
        "coords": {
            "edge_tension": 0.60, "asymmetric_balance": 0.80, "negative_space": 0.30,
            "rhythm": 0.85, "density": 0.70, "scale_hierarchy": 0.55,
            "diagonal_dominance": 0.95, "containment": 0.70,
            "melancholy": 0.10, "intimacy": 0.30, "tension": 0.70,
            "stillness": 0.05, "energy": 0.95, "warmth": 0.45, "unease": 0.50,
        },
        "keywords": [
            "strong diagonal leading lines at 30-45° angles",
            "motion blur on secondary elements",
            "asymmetric weight distribution",
            "high contrast directional lighting",
            "irregular rhythm of repeating forms",
            "Dutch angle tilt",
            "kinetic frozen-moment energy",
        ],
    },
    "monumental_formal": {
        "coords": {
            "edge_tension": 0.15, "asymmetric_balance": 0.10, "negative_space": 0.50,
            "rhythm": 0.05, "density": 0.55, "scale_hierarchy": 0.95,
            "diagonal_dominance": 0.05, "containment": 0.05,
            "melancholy": 0.15, "intimacy": 0.10, "tension": 0.15,
            "stillness": 0.75, "energy": 0.20, "warmth": 0.20, "unease": 0.10,
        },
        "keywords": [
            "symmetric bilateral composition",
            "dominant central subject at large scale",
            "regular repeating columnar rhythm",
            "cool even lighting from above",
            "deep perspective vanishing point",
            "architectural formality",
            "imposing vertical proportions",
        ],
    },
    "melancholic_atmospheric": {
        "coords": {
            "edge_tension": 0.30, "asymmetric_balance": 0.70, "negative_space": 0.60,
            "rhythm": 0.30, "density": 0.30, "scale_hierarchy": 0.40,
            "diagonal_dominance": 0.35, "containment": 0.25,
            "melancholy": 0.90, "intimacy": 0.35, "tension": 0.25,
            "stillness": 0.65, "energy": 0.15, "warmth": 0.25, "unease": 0.30,
        },
        "keywords": [
            "off-center subject with weighted negative space",
            "overcast diffused grey light",
            "desaturated cool blue-grey palette",
            "soft atmospheric haze",
            "subject's gaze directed away from camera",
            "nostalgic film grain texture",
            "solitary figure in open landscape",
        ],
    },
}


def _extract_grid_visual_vocabulary(
    state: Dict[str, float],
    strength: float = 1.0,
) -> dict:
    """
    Map a 15D grid-dynamics state to nearest visual type via Euclidean distance.

    Returns the matched type name, distance, and strength-weighted keyword list.
    """
    state_vec = np.array([state.get(p, 0.5) for p in GRID_PARAMETER_NAMES])

    best_type = None
    best_dist = float("inf")

    for vtype_name, vtype in GRID_VISUAL_TYPES.items():
        ref_vec = np.array([vtype["coords"].get(p, 0.5) for p in GRID_PARAMETER_NAMES])
        dist = float(np.linalg.norm(state_vec - ref_vec))
        if dist < best_dist:
            best_dist = dist
            best_type = vtype_name

    keywords = GRID_VISUAL_TYPES[best_type]["keywords"]

    # At low strength, take fewer keywords
    if strength < 0.3:
        keywords = keywords[:2]
    elif strength < 0.6:
        keywords = keywords[:4]

    return {
        "domain": "grid_dynamics",
        "nearest_type": best_type,
        "distance": round(best_dist, 4),
        "strength": strength,
        "keywords": keywords,
    }


def generate_attractor_prompt(
    state: Dict[str, float],
    mode: str = "composite",
    seed_text: Optional[str] = None,
) -> dict:
    """
    Generate an image-generation prompt from a grid-dynamics attractor state.

    Modes:
        composite  — single blended prompt with all relevant keywords
        split_view — structural prompt + mood prompt kept separate

    Args:
        state: 15D parameter dict (structural + mood values in [0, 1])
        mode: "composite" or "split_view"
        seed_text: optional subject/context to weave in

    Returns:
        Dict with prompt string(s) and metadata
    """
    vocab = _extract_grid_visual_vocabulary(state, strength=1.0)

    # --- Composite mode --------------------------------------------------------
    if mode == "composite":
        parts = list(vocab["keywords"])

        # Derive a register phrase from structural-mood alignment
        structural_vals = {p: state.get(p, 0.5) for p in GRID_STRUCTURAL_PARAMS}
        mood_vals = {p: state.get(p, 0.5) for p in GRID_MOOD_PARAMS}
        alignment = propose_alignment(
            StructuralAnalysis(**structural_vals, summary=""),
            MoodAnalysis(dimensions=mood_vals, summary=""),
            None,
        )
        register = alignment.register_description

        prompt = f"{register} composition. " + ", ".join(parts)
        if seed_text:
            prompt += f". Subject: {seed_text}"

        return {
            "mode": "composite",
            "prompt": prompt,
            "nearest_visual_type": vocab["nearest_type"],
            "distance_to_type": vocab["distance"],
            "register": register,
            "alignment_type": alignment.relationship_type.value,
        }

    # --- Split-view mode -------------------------------------------------------
    elif mode == "split_view":
        # Structural keywords (first 4 tend toward geometry/framing)
        struct_keywords = [k for k in vocab["keywords"] if any(
            w in k.lower() for w in [
                "frame", "space", "angle", "line", "symmetr", "rhythm",
                "composition", "scale", "diagonal", "perspective", "proportion",
                "depth", "blur", "tilt", "repeat",
            ]
        )]
        # Mood keywords (remainder)
        mood_keywords = [k for k in vocab["keywords"] if k not in struct_keywords]
        # Ensure at least one in each bucket
        if not struct_keywords:
            struct_keywords = vocab["keywords"][:3]
        if not mood_keywords:
            mood_keywords = vocab["keywords"][3:]

        structural_prompt = "Compositional structure: " + ", ".join(struct_keywords)
        mood_prompt = "Emotional register: " + ", ".join(mood_keywords)

        if seed_text:
            structural_prompt += f". Subject: {seed_text}"

        return {
            "mode": "split_view",
            "structural_prompt": structural_prompt,
            "mood_prompt": mood_prompt,
            "nearest_visual_type": vocab["nearest_type"],
            "distance_to_type": vocab["distance"],
        }
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'composite' or 'split_view'.")


def generate_sequence_prompts(
    preset_name: str,
    num_keyframes: int = 4,
    seed_text: Optional[str] = None,
) -> dict:
    """
    Generate a sequence of prompts along a rhythmic preset trajectory.

    Samples *num_keyframes* evenly-spaced states from the preset and
    returns one prompt per keyframe — useful for animation storyboards
    or timelapse-style generation.

    Args:
        preset_name: Phase 2.6 preset name
        num_keyframes: how many frames to sample (default 4)
        seed_text: optional subject woven into every frame

    Returns:
        Dict with preset metadata and list of keyframe prompts
    """
    trajectory = generate_grid_preset_trajectory(preset_name)
    total = len(trajectory)
    indices = [int(i * total / num_keyframes) for i in range(num_keyframes)]

    keyframes = []
    for idx in indices:
        state = trajectory[idx]
        prompt_result = generate_attractor_prompt(state, mode="composite", seed_text=seed_text)
        keyframes.append({
            "step": idx,
            "phase": round(idx / total, 3),
            **prompt_result,
        })

    cfg = GRID_RHYTHMIC_PRESETS[preset_name]
    return {
        "preset": preset_name,
        "description": cfg["description"],
        "period": cfg["steps_per_cycle"],
        "total_steps": total,
        "num_keyframes": num_keyframes,
        "keyframes": keyframes,
    }


# ============================================================================
# MCP Tool Endpoints (wrapping testable core)
# ============================================================================

@mcp.tool()
def analyze_and_propose(
    structural_values: dict,
    mood_values: dict,
    seed_text: Optional[str] = None,
) -> dict:
    """
    Analyze pre-extracted compositional parameters and propose alignment.
    
    This is the core Grid Dynamics workflow: Claude extracts image parameters,
    Grid Dynamics proposes how structure and mood interact expressively.
    
    Args:
        structural_values: Dict with 8 structural parameters (0-1 scale)
          - edge_tension, asymmetric_balance, negative_space, rhythm,
          - density, scale_hierarchy, diagonal_dominance, containment
        mood_values: Dict with 7 mood dimensions (0-1 scale)
          - melancholy, intimacy, tension, stillness, energy, warmth, unease
        seed_text: Optional seed text describing subject/intent
    
    Returns:
        Dictionary with proposed_alignment, conflicts, and editable structural/mood values
    """
    return extract_impl(structural_values, mood_values, seed_text)


@mcp.tool()
def synthesize_with_modifications(
    structural_values: dict,
    mood_values: dict,
    alignment_override: Optional[str] = None,
    seed_text: Optional[str] = None,
    target_length: str = "medium",
) -> dict:
    """
    Synthesize enhanced prompt from modified structural and mood parameters.
    
    After reviewing proposed alignment from analyze_and_propose, Claude can
    modify parameters and use this tool to generate the final compositional prompt.
    
    Args:
        structural_values: Modified structural parameters
        mood_values: Modified mood parameters
        alignment_override: Optional override for register description
        seed_text: Optional seed text context
        target_length: "short", "medium", or "detailed"
    
    Returns:
        Dictionary with enhanced_prompt string
    """
    return synthesize_impl(
        structural_values, mood_values, alignment_override, seed_text, target_length
    )


@mcp.tool()
def analyze_strategy_document_tool(strategy_text: str) -> dict:
    """
    Analyze a strategy document through Grid Dynamics structural dimensions.
    
    This is the tomographic domain projection tool - it projects strategic
    text through compositional structure vocabulary to detect patterns.
    
    Zero LLM cost - pure deterministic pattern matching.
    
    Args:
        strategy_text: Full text of the strategy document to analyze
    
    Returns:
        Dictionary with structural findings including:
        - domain: "grid_dynamics"
        - findings: List of detected patterns with confidence and evidence
        - total_findings: Count of findings
        - methodology: "deterministic_pattern_matching"
        - llm_cost_tokens: 0
    
    Example:
        result = analyze_strategy_document_tool(strategy_pdf_text)
        Returns findings like:
        {
          "domain": "grid_dynamics",
          "findings": [
            {
              "dimension": "asymmetric_balance",
              "pattern": "radial_dispersed",
              "confidence": 0.75,
              "evidence": ["Found 6 authority entities..."],
              "categorical_family": "morphisms"
            }
          ],
          "total_findings": 4,
          "llm_cost_tokens": 0
        }
    """
    return analyze_strategy_document(strategy_text)


# ============================================================================
# Phase 2.6 MCP Tools — Rhythmic Presets & Trajectories
# ============================================================================

@mcp.tool()
def get_grid_canonical_states() -> dict:
    """
    Return all canonical compositional states in the 15D grid-dynamics morphospace.

    Each state is a recognizable compositional archetype with coordinates for
    8 structural parameters and 7 mood parameters (all in [0, 1]).

    Returns:
        Dictionary mapping state names to their 15D coordinate dicts,
        plus the ordered parameter name list.
    """
    return {
        "parameter_names": GRID_PARAMETER_NAMES,
        "structural_params": GRID_STRUCTURAL_PARAMS,
        "mood_params": GRID_MOOD_PARAMS,
        "states": GRID_CANONICAL_STATES,
        "count": len(GRID_CANONICAL_STATES),
    }


@mcp.tool()
def get_grid_rhythmic_presets() -> dict:
    """
    Return all Phase 2.6 rhythmic preset configurations.

    Each preset defines an oscillation between two canonical states at a
    specific period and waveform pattern. Periods are chosen to produce
    productive emergent attractor behavior when composed with other domains.

    Returns:
        Dictionary with preset configs, periods, and descriptions.
    """
    presets_out = {}
    for name, cfg in GRID_RHYTHMIC_PRESETS.items():
        presets_out[name] = {
            "state_a": cfg["state_a"],
            "state_b": cfg["state_b"],
            "pattern": cfg["pattern"],
            "steps_per_cycle": cfg["steps_per_cycle"],
            "num_cycles": cfg["num_cycles"],
            "total_steps": cfg["num_cycles"] * cfg["steps_per_cycle"],
            "description": cfg["description"],
        }
    return {
        "domain": "grid_dynamics",
        "presets": presets_out,
        "periods": sorted(set(
            cfg["steps_per_cycle"] for cfg in GRID_RHYTHMIC_PRESETS.values()
        )),
        "count": len(GRID_RHYTHMIC_PRESETS),
    }


@mcp.tool()
def generate_grid_trajectory(
    preset_name: str,
    include_trajectory: bool = True,
) -> dict:
    """
    Generate a Phase 2.6 rhythmic trajectory for a grid-dynamics preset.

    The trajectory oscillates through 15D compositional parameter space
    between two canonical states. Suitable for forced-orbit integration
    in the Aesthetic Dynamics framework.

    Args:
        preset_name: One of the registered presets (e.g. "tension_release")
        include_trajectory: If True, return full step-by-step trajectory.
            Set False to get metadata only.

    Returns:
        Dictionary with preset metadata and (optionally) the trajectory
        as a list of 15D state dicts.
    """
    if preset_name not in GRID_RHYTHMIC_PRESETS:
        return {
            "error": f"Unknown preset '{preset_name}'",
            "available_presets": list(GRID_RHYTHMIC_PRESETS.keys()),
        }

    cfg = GRID_RHYTHMIC_PRESETS[preset_name]
    result = {
        "preset": preset_name,
        "state_a": cfg["state_a"],
        "state_b": cfg["state_b"],
        "pattern": cfg["pattern"],
        "period": cfg["steps_per_cycle"],
        "total_steps": cfg["num_cycles"] * cfg["steps_per_cycle"],
        "description": cfg["description"],
        "parameter_names": GRID_PARAMETER_NAMES,
    }

    if include_trajectory:
        result["trajectory"] = generate_grid_preset_trajectory(preset_name)

    return result


# ============================================================================
# Phase 2.7 MCP Tools — Attractor Visualization & Prompt Generation
# ============================================================================

@mcp.tool()
def extract_grid_visual_vocabulary(
    state: dict,
    strength: float = 1.0,
) -> dict:
    """
    Map a 15D grid-dynamics state to the nearest visual vocabulary type.

    Uses Euclidean distance across all 15 parameters to find the closest
    compositional archetype, then returns image-generation keywords.

    Args:
        state: Dict with parameter values (0-1 scale). Missing params
            default to 0.5.
        strength: How strongly to weight this domain (0-1). Lower values
            return fewer keywords.

    Returns:
        Dictionary with nearest_type, distance, and keyword list.
    """
    return _extract_grid_visual_vocabulary(state, strength)


@mcp.tool()
def generate_grid_attractor_prompt(
    state: dict,
    mode: str = "composite",
    seed_text: Optional[str] = None,
) -> dict:
    """
    Generate an image-generation prompt from a grid-dynamics attractor state.

    Translates 15D compositional coordinates into concrete visual language
    suitable for text-to-image models (Stable Diffusion, DALL-E, Midjourney).

    Modes:
        composite  — Single blended prompt incorporating structure + mood
        split_view — Separate structural-framing and emotional-register prompts

    Args:
        state: 15D parameter dict (structural + mood, values in [0, 1])
        mode: "composite" or "split_view"
        seed_text: Optional subject/context (e.g. "two figures on a bridge")

    Returns:
        Dict with prompt string(s), nearest visual type, and metadata.

    Example:
        generate_grid_attractor_prompt(
            state={"edge_tension": 0.9, "negative_space": 0.1, ...},
            mode="composite",
            seed_text="woman reading by window light"
        )
    """
    return generate_attractor_prompt(state, mode, seed_text)


@mcp.tool()
def generate_grid_sequence_prompts(
    preset_name: str,
    num_keyframes: int = 4,
    seed_text: Optional[str] = None,
) -> dict:
    """
    Generate a sequence of prompts sampled along a rhythmic preset trajectory.

    Useful for storyboards, animation keyframes, or timelapse-style image
    sequences that track compositional evolution through time.

    Args:
        preset_name: Phase 2.6 preset (e.g. "tension_release")
        num_keyframes: Number of evenly-spaced frames to sample (default 4)
        seed_text: Optional subject woven into every frame prompt

    Returns:
        Dict with preset metadata and list of keyframe prompts, each
        containing the phase position, visual type, and prompt text.

    Example:
        generate_grid_sequence_prompts(
            preset_name="energy_cycle",
            num_keyframes=6,
            seed_text="crowded Tokyo street at dusk"
        )
    """
    if preset_name not in GRID_RHYTHMIC_PRESETS:
        return {
            "error": f"Unknown preset '{preset_name}'",
            "available_presets": list(GRID_RHYTHMIC_PRESETS.keys()),
        }
    return generate_sequence_prompts(preset_name, num_keyframes, seed_text)


@mcp.tool()
def get_grid_server_info() -> dict:
    """
    Return comprehensive server information including Phase 2.6/2.7 capabilities.

    Covers domain parameters, canonical states, rhythmic presets, visual
    vocabulary types, and available MCP tools.
    """
    return {
        "server": "grid-dynamics",
        "version": "2.7.0",
        "description": (
            "Compositional structure and mood analysis with rhythmic presets "
            "and attractor visualization prompt generation."
        ),
        "morphospace": {
            "total_dimensions": len(GRID_PARAMETER_NAMES),
            "structural_dimensions": len(GRID_STRUCTURAL_PARAMS),
            "mood_dimensions": len(GRID_MOOD_PARAMS),
            "parameter_names": GRID_PARAMETER_NAMES,
        },
        "phase_2_6": {
            "canonical_states": list(GRID_CANONICAL_STATES.keys()),
            "rhythmic_presets": {
                name: {
                    "period": cfg["steps_per_cycle"],
                    "pattern": cfg["pattern"],
                    "states": f"{cfg['state_a']} ↔ {cfg['state_b']}",
                }
                for name, cfg in GRID_RHYTHMIC_PRESETS.items()
            },
            "periods": sorted(set(
                cfg["steps_per_cycle"] for cfg in GRID_RHYTHMIC_PRESETS.values()
            )),
        },
        "phase_2_7": {
            "attractor_visualization": True,
            "visual_types": list(GRID_VISUAL_TYPES.keys()),
            "prompt_modes": ["composite", "split_view", "sequence"],
        },
        "tools": [
            "analyze_and_propose",
            "synthesize_with_modifications",
            "analyze_strategy_document_tool",
            "get_grid_canonical_states",
            "get_grid_rhythmic_presets",
            "generate_grid_trajectory",
            "extract_grid_visual_vocabulary",
            "generate_grid_attractor_prompt",
            "generate_grid_sequence_prompts",
            "get_grid_server_info",
        ],
    }


# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    mcp.run()
