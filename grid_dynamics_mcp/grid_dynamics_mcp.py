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
from typing import Optional
from dataclasses import dataclass, asdict
from enum import Enum

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



# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    mcp.run()
