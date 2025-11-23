"""
Grid Dynamics MCP Server
Analyzes image compositional structure and mood, proposes alignments, and synthesizes enhanced prompts.
Three-layer olog architecture: Domain → Interpretation → Synthesis
"""

import base64
import json
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, asdict
from enum import Enum

from fastmcp import FastMCP
from anthropic import Anthropic

# Initialize FastMCP server
mcp = FastMCP("grid-dynamics")
client = Anthropic()


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


# ============================================================================
# LAYER 2: Interpretation (Deterministic Extraction via Vision)
# ============================================================================

def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as img_file:
        return base64.standard_b64encode(img_file.read()).decode("utf-8")


def extract_structural_analysis(image_base64: str) -> StructuralAnalysis:
    """
    Structural Extractor functor: Vision model analyzes compositional elements
    Returns deterministic structural parameters
    """
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": """Analyze this image's compositional structure. Return JSON with these exact parameters (0-1 scale):

{
  "edge_tension": <float>,
  "asymmetric_balance": <float>,
  "negative_space": <float>,
  "rhythm": <float>,
  "density": <float>,
  "scale_hierarchy": <float>,
  "diagonal_dominance": <float>,
  "containment": <float>,
  "summary": "<natural language description of overall structure>"
}

Definitions:
- edge_tension: How much subjects crowd frame boundaries (0=centered, 1=cramped edges)
- asymmetric_balance: Weight distribution evenness (0=balanced, 1=heavily weighted)
- negative_space: Breathing room (0=generous whitespace, 1=compressed/crowded)
- rhythm: Repetition pattern regularity (0=regular grid, 1=chaotic variation)
- density: Element count and saturation (0=sparse/minimal, 1=overflowing)
- scale_hierarchy: Size variation dominance (0=uniform sizes, 1=clear dominant element)
- diagonal_dominance: Diagonal line prevalence (0=horizontal/vertical, 1=strong diagonals)
- containment: Elements within frame (0=contained, 1=bleeding/cropped)

Return ONLY valid JSON.""",
                    },
                ],
            }
        ],
    )

    # Parse Claude's JSON response
    json_str = response.content[0].text
    data = json.loads(json_str)

    return StructuralAnalysis(
        edge_tension=data["edge_tension"],
        asymmetric_balance=data["asymmetric_balance"],
        negative_space=data["negative_space"],
        rhythm=data["rhythm"],
        density=data["density"],
        scale_hierarchy=data["scale_hierarchy"],
        diagonal_dominance=data["diagonal_dominance"],
        containment=data["containment"],
        summary=data["summary"],
    )


def extract_mood_analysis(image_base64: str) -> MoodAnalysis:
    """
    Mood Extractor functor: Vision model analyzes emotional/sensory qualities
    Returns deterministic mood dimensions
    """
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": """Analyze this image's emotional and sensory mood. Return JSON with sentiment dimensions (0-1 scale):

{
  "dimensions": {
    "melancholy": <float>,
    "intimacy": <float>,
    "tension": <float>,
    "stillness": <float>,
    "energy": <float>,
    "warmth": <float>,
    "unease": <float>
  },
  "summary": "<natural language description of overall mood>"
}

Scale: 0 = absent/minimal, 1 = dominant/intense

Return ONLY valid JSON.""",
                    },
                ],
            }
        ],
    )

    json_str = response.content[0].text
    data = json.loads(json_str)

    return MoodAnalysis(
        dimensions=data["dimensions"],
        summary=data["summary"],
    )


def parse_seed_text(seed_text: str) -> SeedInterpretation:
    """
    Seed Parser: Extract subject matter and intent from seed text
    """
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": f"""Parse this seed text into subject and intent:

Seed: "{seed_text}"

Return JSON:
{{
  "subject": "<what is being depicted>",
  "intent": "<creative or emotional intent>"
}}

Return ONLY valid JSON.""",
            }
        ],
    )

    json_str = response.content[0].text
    data = json.loads(json_str)

    return SeedInterpretation(
        subject=data["subject"],
        intent=data["intent"],
    )


def propose_alignment(
    structural: StructuralAnalysis,
    mood: MoodAnalysis,
    seed: Optional[SeedInterpretation],
) -> AlignmentProposal:
    """
    Alignment Proposal: Rule-based + lightweight inference to suggest how
    structure, mood, and seed interact expressively
    """
    # Build context for Claude
    context = f"""
Structural Analysis:
{structural.summary}
Values: edge_tension={structural.edge_tension:.2f}, negative_space={structural.negative_space:.2f}, 
density={structural.density:.2f}, diagonal_dominance={structural.diagonal_dominance:.2f}

Mood Analysis:
{mood.summary}
Dimensions: {json.dumps(mood.dimensions, indent=2)}
"""

    if seed:
        context += f"""
Seed Intent:
Subject: {seed.subject}
Intent: {seed.intent}
"""

    prompt = f"""{context}

Propose how these elements interact expressively. Return JSON:

{{
  "relationship_type": "<reinforcing|contrasting|productive_tension>",
  "register_description": "<evocative combined mood, e.g. 'claustrophobic tenderness'>",
  "conflicts": [<list of detected tensions between elements, e.g. "melancholic mood contradicts celebratory intent">]
}}

Be specific about compositional-emotional resonance. Return ONLY valid JSON."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    json_str = response.content[0].text
    data = json.loads(json_str)

    return AlignmentProposal(
        relationship_type=AlignmentType(data["relationship_type"]),
        register_description=data["register_description"],
        conflicts=data["conflicts"],
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
    Synthesis: Single LLM call to produce compositionally-aware enhanced prompt
    Receives user-approved interpretation layer objects
    """
    length_guide = {
        PromptLength.SHORT: "1-2 sentences, concise",
        PromptLength.MEDIUM: "3-4 sentences, balanced detail",
        PromptLength.DETAILED: "5-7 sentences, comprehensive",
    }

    prompt = f"""Generate an enhanced image generation prompt based on these compositional intentions:

Structural Parameters:
{json.dumps(structural_values, indent=2)}

Mood Parameters:
{json.dumps(mood_values, indent=2)}

Target Register: {alignment_override or "natural composition of mood and structure"}
{"Seed Context: " + seed_text if seed_text else ""}

Write compositional direction that:
1. Translates numerical values to natural language aesthetic guidance
2. Shows how structure and mood work together
3. Instructs an image generator on compositional choices
4. Reconciles any tensions toward expressive coherence

Length: {length_guide[target_length]}

Return ONLY the enhanced prompt text, no JSON or explanation."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


# ============================================================================
# Testable Core Functions
# ============================================================================

def extract_impl(
    image_path: str,
    seed_text: Optional[str] = None,
) -> dict:
    """
    Core extraction implementation (testable).
    Extract compositional structure and mood from image.
    
    Args:
        image_path: Path to image file (<10MB)
        seed_text: Optional seed text describing subject/intent
    
    Returns:
        Dictionary with structural_analysis, mood_analysis, proposed_alignment,
        and editable flag set to true for user modification
    """
    # Convert image to base64
    image_b64 = image_to_base64(image_path)

    # Layer 2: Extract structure and mood in parallel
    structural = extract_structural_analysis(image_b64)
    mood = extract_mood_analysis(image_b64)

    # Parse seed if provided
    seed_interp = None
    if seed_text:
        seed_interp = parse_seed_text(seed_text)

    # Propose alignment
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


def analyze_and_synthesize_impl(
    image_path: str,
    seed_text: Optional[str] = None,
    auto_approve: bool = False,
    target_length: str = "medium",
) -> dict:
    """
    Core implementation (testable).
    Convenience endpoint combining extraction and synthesis.
    
    Args:
        image_path: Path to image file
        seed_text: Optional seed text
        auto_approve: If true, skips to synthesis; if false, returns interpretation layer
        target_length: "short", "medium", or "detailed"
    
    Returns:
        If auto_approve=false: extraction result for user review
        If auto_approve=true: final enhanced_prompt
    """
    # Extract everything
    extraction = extract_impl(image_path, seed_text)

    if not auto_approve:
        # Return interpretation layer for user review
        return extraction

    # Auto-approve: synthesize immediately
    structural_values = extraction["structural_analysis"]["values"]
    mood_values = extraction["mood_analysis"]["dimensions"]
    alignment_override = extraction["proposed_alignment"]["register_description"]

    synthesis_result = synthesize_impl(
        structural_values=structural_values,
        mood_values=mood_values,
        alignment_override=alignment_override,
        seed_text=seed_text,
        target_length=target_length,
    )

    return synthesis_result


# ============================================================================
# MCP Tool Endpoints (wrapping testable core)
# ============================================================================

@mcp.tool()
def extract(
    image_path: str,
    seed_text: Optional[str] = None,
) -> dict:
    """
    Extract compositional structure and mood from image.
    
    Args:
        image_path: Path to image file (<10MB)
        seed_text: Optional seed text describing subject/intent
    
    Returns:
        Dictionary with structural_analysis, mood_analysis, proposed_alignment,
        and editable flag set to true for user modification
    """
    return extract_impl(image_path, seed_text)


@mcp.tool()
def synthesize(
    structural_values: dict,
    mood_values: dict,
    alignment_override: Optional[str] = None,
    seed_text: Optional[str] = None,
    target_length: str = "medium",
) -> dict:
    """
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
    return synthesize_impl(
        structural_values, mood_values, alignment_override, seed_text, target_length
    )


@mcp.tool()
def analyze_and_synthesize(
    image_path: str,
    seed_text: Optional[str] = None,
    auto_approve: bool = False,
    target_length: str = "medium",
) -> dict:
    """
    Convenience endpoint combining extraction and synthesis.
    
    Args:
        image_path: Path to image file
        seed_text: Optional seed text
        auto_approve: If true, skips to synthesis; if false, returns interpretation layer
        target_length: "short", "medium", or "detailed"
    
    Returns:
        If auto_approve=false: extraction result for user review
        If auto_approve=true: final enhanced_prompt
    """
    return analyze_and_synthesize_impl(
        image_path, seed_text, auto_approve, target_length
    )


# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    mcp.run()
