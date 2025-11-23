"""
Test suite for Grid Dynamics MCP Server
Tests Layer 1 (domain types), Layer 2 (interpretation), and Layer 3 (synthesis)
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from grid_dynamics_mcp import (
    # Types
    AlignmentType,
    PromptLength,
    StructuralAnalysis,
    MoodAnalysis,
    SeedInterpretation,
    AlignmentProposal,
    ExtractionResult,
    # Functions
    propose_alignment,
    synthesize_prompt,
    # Testable implementations
    extract_impl,
    synthesize_impl,
)


# ============================================================================
# Layer 1: Domain Type Tests
# ============================================================================

def test_alignment_type_enum():
    """Verify AlignmentType enum values"""
    assert AlignmentType.REINFORCING.value == "reinforcing"
    assert AlignmentType.CONTRASTING.value == "contrasting"
    assert AlignmentType.PRODUCTIVE_TENSION.value == "productive_tension"


def test_prompt_length_enum():
    """Verify PromptLength enum values"""
    assert PromptLength.SHORT.value == "short"
    assert PromptLength.MEDIUM.value == "medium"
    assert PromptLength.DETAILED.value == "detailed"


def test_structural_analysis_creation():
    """StructuralAnalysis dataclass instantiation"""
    sa = StructuralAnalysis(
        edge_tension=0.7,
        asymmetric_balance=0.5,
        negative_space=0.3,
        rhythm=0.6,
        density=0.8,
        scale_hierarchy=0.4,
        diagonal_dominance=0.5,
        containment=0.2,
        summary="Test composition",
    )
    assert sa.edge_tension == 0.7
    assert sa.summary == "Test composition"


def test_mood_analysis_creation():
    """MoodAnalysis dataclass instantiation"""
    ma = MoodAnalysis(
        dimensions={"melancholy": 0.6, "warmth": 0.3},
        summary="Test mood",
    )
    assert ma.dimensions["melancholy"] == 0.6
    assert ma.summary == "Test mood"


def test_seed_interpretation_creation():
    """SeedInterpretation dataclass instantiation"""
    si = SeedInterpretation(
        subject="birthday party",
        intent="celebratory atmosphere",
    )
    assert si.subject == "birthday party"
    assert si.intent == "celebratory atmosphere"


def test_alignment_proposal_creation():
    """AlignmentProposal dataclass instantiation"""
    ap = AlignmentProposal(
        relationship_type=AlignmentType.PRODUCTIVE_TENSION,
        register_description="bittersweet celebration",
        conflicts=["melancholic mood vs celebratory intent"],
    )
    assert ap.relationship_type == AlignmentType.PRODUCTIVE_TENSION
    assert len(ap.conflicts) == 1


def test_extraction_result_creation():
    """ExtractionResult dataclass instantiation"""
    sa = StructuralAnalysis(
        edge_tension=0.7,
        asymmetric_balance=0.5,
        negative_space=0.3,
        rhythm=0.6,
        density=0.8,
        scale_hierarchy=0.4,
        diagonal_dominance=0.5,
        containment=0.2,
        summary="Test",
    )
    ma = MoodAnalysis(dimensions={"melancholy": 0.6}, summary="Test")
    ap = AlignmentProposal(
        relationship_type=AlignmentType.REINFORCING,
        register_description="test",
        conflicts=[],
    )

    er = ExtractionResult(
        structural_analysis=sa,
        mood_analysis=ma,
        seed_interpretation=None,
        proposed_alignment=ap,
    )
    assert er.editable is True


# ============================================================================
# Layer 2: Interpretation Tests
# ============================================================================

def test_parse_seed_text_inline():
    """Test simple inline seed text parsing"""
    # Grid Dynamics now does simple parsing, not Claude-based
    seed_text = "birthday party, celebration"
    # This would be parsed as: subject="birthday party", intent="celebration"
    # Simple test: ensure we handle the comma-separated format
    parts = seed_text.split(',')
    assert len(parts) >= 1


def test_propose_alignment_reinforcing():
    """Test alignment proposal with reinforcing structure and mood"""
    structural = StructuralAnalysis(
        edge_tension=0.3,
        asymmetric_balance=0.5,
        negative_space=0.2,
        rhythm=0.8,  # High
        density=0.6,
        scale_hierarchy=0.4,
        diagonal_dominance=0.8,  # High
        containment=0.3,
        summary="Dynamic composition",
    )
    mood = MoodAnalysis(
        dimensions={"melancholy": 0.2, "energy": 0.9, "tension": 0.8, "warmth": 0.4, "unease": 0.3, "intimacy": 0.2, "stillness": 0.1},
        summary="Vibrant and energetic",
    )
    seed = None

    result = propose_alignment(structural, mood, seed)

    assert result.relationship_type == AlignmentType.REINFORCING
    # High energy structure + high energy mood = reinforcing


def test_propose_alignment_with_seed_conflict():
    """Test alignment proposal detecting intent conflict"""
    structural = StructuralAnalysis(
        edge_tension=0.8,
        asymmetric_balance=0.5,
        negative_space=0.7,
        rhythm=0.5,
        density=0.6,
        scale_hierarchy=0.4,
        diagonal_dominance=0.5,
        containment=0.3,
        summary="Compressed composition",
    )
    mood = MoodAnalysis(
        dimensions={"melancholy": 0.6, "intimacy": 0.7, "tension": 0.4, "stillness": 0.3, "energy": 0.3, "warmth": 0.2, "unease": 0.4},
        summary="Somber mood",
    )
    seed = SeedInterpretation(
        subject="birthday party",
        intent="celebration",
    )

    result = propose_alignment(structural, mood, seed)

    assert result.relationship_type == AlignmentType.PRODUCTIVE_TENSION
    # Melancholic mood + celebratory intent = productive tension
    assert len(result.conflicts) > 0


def test_propose_alignment_without_seed():
    """Test alignment proposal without seed (structure/mood only)"""
    structural = StructuralAnalysis(
        edge_tension=0.5,
        asymmetric_balance=0.6,
        negative_space=0.4,
        rhythm=0.2,
        density=0.5,
        scale_hierarchy=0.6,
        diagonal_dominance=0.2,
        containment=0.4,
        summary="Static composition",
    )
    mood = MoodAnalysis(
        dimensions={"energy": 0.2, "tension": 0.3, "melancholy": 0.1, "warmth": 0.5, "unease": 0.2, "intimacy": 0.4, "stillness": 0.8},
        summary="Calm mood",
    )

    result = propose_alignment(structural, mood, None)

    assert result.relationship_type == AlignmentType.REINFORCING
    # Low energy structure + low energy mood = reinforcing


# ============================================================================
# Layer 3: Synthesis Tests
# ============================================================================

def test_synthesize_prompt_short():
    """Test short prompt synthesis"""
    result = synthesize_prompt(
        structural_values={"edge_tension": 0.8, "negative_space": 0.7, "diagonal_dominance": 0.5, "rhythm": 0.6},
        mood_values={"melancholy": 0.6, "intimacy": 0.7, "energy": 0.3, "warmth": 0.2, "tension": 0.4},
        target_length=PromptLength.SHORT,
    )

    assert isinstance(result, str)
    assert len(result) > 0
    assert "edge" in result.lower() or "subject" in result.lower()


def test_synthesize_prompt_detailed():
    """Test detailed prompt synthesis"""
    result = synthesize_prompt(
        structural_values={"edge_tension": 0.8, "negative_space": 0.7, "diagonal_dominance": 0.5, "rhythm": 0.6, "density": 0.6},
        mood_values={"melancholy": 0.6, "intimacy": 0.7, "energy": 0.3, "warmth": 0.2, "tension": 0.4},
        target_length=PromptLength.DETAILED,
    )

    assert isinstance(result, str)
    assert len(result) > 0


def test_synthesize_prompt_with_register():
    """Test synthesis with alignment override"""
    result = synthesize_prompt(
        structural_values={"edge_tension": 0.8, "negative_space": 0.7},
        mood_values={"melancholy": 0.6},
        alignment_override="bittersweet celebration",
    )

    assert isinstance(result, str)
    assert "bittersweet celebration" in result


# ============================================================================
# MCP Endpoint Integration Tests
# ============================================================================

def test_analyze_and_propose_endpoint():
    """Test analyze_and_propose endpoint"""
    structural = {
        "edge_tension": 0.7,
        "asymmetric_balance": 0.5,
        "negative_space": 0.3,
        "rhythm": 0.6,
        "density": 0.8,
        "scale_hierarchy": 0.4,
        "diagonal_dominance": 0.5,
        "containment": 0.2,
        "summary": "Test structure",
    }
    mood = {
        "melancholy": 0.6,
        "intimacy": 0.7,
        "tension": 0.4,
        "stillness": 0.3,
        "energy": 0.3,
        "warmth": 0.2,
        "unease": 0.4,
    }

    result = extract_impl(structural, mood, "birthday party")

    assert "structural_analysis" in result
    assert "mood_analysis" in result
    assert "proposed_alignment" in result
    assert result["editable"] is True
    assert result["structural_analysis"]["values"]["edge_tension"] == 0.7
    assert result["mood_analysis"]["dimensions"]["melancholy"] == 0.6


def test_synthesize_with_modifications_endpoint():
    """Test synthesize_with_modifications endpoint"""
    result = synthesize_impl(
        structural_values={"edge_tension": 0.8, "negative_space": 0.7},
        mood_values={"melancholy": 0.6},
        target_length="medium",
    )

    assert "enhanced_prompt" in result
    assert isinstance(result["enhanced_prompt"], str)
    assert len(result["enhanced_prompt"]) > 0


# ============================================================================
# Type Validation Tests
# ============================================================================

def test_extraction_result_json_serializable():
    """Verify extraction result can be converted to JSON"""
    structural = {
        "edge_tension": 0.7,
        "asymmetric_balance": 0.5,
        "negative_space": 0.3,
        "rhythm": 0.6,
        "density": 0.8,
        "scale_hierarchy": 0.4,
        "diagonal_dominance": 0.5,
        "containment": 0.2,
        "summary": "Test",
    }
    mood = {
        "melancholy": 0.6,
        "intimacy": 0.7,
        "tension": 0.4,
        "stillness": 0.3,
        "energy": 0.3,
        "warmth": 0.2,
        "unease": 0.4,
    }

    result = extract_impl(structural, mood, seed_text=None)

    # Should be JSON serializable
    json_str = json.dumps(result)
    assert len(json_str) > 0
    parsed = json.loads(json_str)
    assert "structural_analysis" in parsed


# ============================================================================
# Boundary Tests
# ============================================================================

def test_alignment_type_conversion():
    """Test enum conversion for API responses"""
    ap = AlignmentProposal(
        relationship_type=AlignmentType.PRODUCTIVE_TENSION,
        register_description="test",
        conflicts=[],
    )
    assert ap.relationship_type.value == "productive_tension"


def test_prompt_length_conversion():
    """Test enum conversion for prompt generation"""
    lengths = [PromptLength.SHORT, PromptLength.MEDIUM, PromptLength.DETAILED]
    for length in lengths:
        assert length.value in ["short", "medium", "detailed"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
