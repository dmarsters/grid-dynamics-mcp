# Grid Dynamics MCP Server

An MCP tool that analyzes image compositional structure and mood, proposes alignment strategies, and synthesizes enhanced image generation prompts. Implements a three-layer olog (ontology log) architecture for deterministic, semantically coherent analysis.

## Architecture

The system follows a three-layer olog pattern:

### Layer 1: Domain Types (Categorical Structure)

- **StructuralAnalysis**: 8 compositional parameters (edge tension, asymmetric balance, negative space, rhythm, density, scale hierarchy, diagonal dominance, containment)
- **MoodAnalysis**: 7 sentiment dimensions (melancholy, intimacy, tension, stillness, energy, warmth, unease)
- **SeedInterpretation**: Subject and intent extraction from optional seed text
- **AlignmentProposal**: Relationship type, register description, and detected conflicts

All types are dataclasses with deterministic structure.

### Layer 2: Interpretation (Deterministic Extraction)

Vision models analyze images through two parallel functors:

1. **Structural Extractor**: Analyzes compositional elements → produces 8 numerical parameters (0-1 scale) + natural language summary
2. **Mood Extractor**: Analyzes emotional/sensory qualities → produces 7 dimensional values + natural language summary
3. **Seed Parser** (if provided): Extracts subject and intent from text
4. **Alignment Proposer**: Rule-based inference to suggest how structure, mood, and seed interact expressively

Outputs are deterministic JSON with human-readable summaries.

### Layer 3: Synthesis (Enhanced Prompt Generation)

Single LLM call receives all Layer 2 outputs and user modifications, producing compositionally-aware prompt language that reconciles any tensions.

## MCP Endpoints

### `extract`

Extract compositional structure and mood from image.

**Args:**
- `image_path` (str): Path to image file (<10MB)
- `seed_text` (str, optional): Seed text describing subject/intent

**Returns:**
```json
{
  "structural_analysis": {
    "values": { "edge_tension": 0.7, ... },
    "summary": "Tight composition with strong edge presence"
  },
  "mood_analysis": {
    "dimensions": { "melancholy": 0.6, "intimacy": 0.7, ... },
    "summary": "Deeply intimate but melancholic atmosphere"
  },
  "seed_interpretation": {
    "subject": "children's birthday party",
    "intent": "celebration"
  },
  "proposed_alignment": {
    "relationship_type": "productive_tension",
    "register_description": "bittersweet celebration",
    "conflicts": ["melancholic mood contradicts celebratory intent"]
  },
  "editable": true
}
```

### `synthesize`

Synthesize enhanced prompt from modified parameters.

**Args:**
- `structural_values` (dict): Modified structural analysis (user can adjust values)
- `mood_values` (dict): Modified mood analysis (user can adjust dimensions)
- `alignment_override` (str, optional): Override register description
- `seed_text` (str, optional): Seed context
- `target_length` (str): "short", "medium", or "detailed"

**Returns:**
```json
{
  "enhanced_prompt": "Frame the birthday scene with tight, edge-crowding composition..."
}
```

### `analyze_and_synthesize`

Convenience endpoint combining extraction and synthesis.

**Args:**
- `image_path` (str): Path to image file
- `seed_text` (str, optional): Optional seed text
- `auto_approve` (bool): If true, skips to synthesis; if false, returns interpretation layer
- `target_length` (str): "short", "medium", or "detailed"

**Returns:**
- If `auto_approve=false`: Full extraction result for user review
- If `auto_approve=true`: Final enhanced_prompt

## Example Flow

```python
# Step 1: Extract
result = extract("birthday_photo.jpg", seed_text="children's birthday party")

# User reviews interpretation layer:
# - Structural: high edge_tension (0.8), compressed negative_space (0.7)
# - Mood: melancholy (0.6), intimacy (0.7), unease (0.4)
# - Alignment: "Conflict detected—melancholic mood tensions against celebratory intent"

# Step 2: User modifies (optional)
# Reduce melancholy to 0.3, approve "bittersweet celebration" register

# Step 3: Synthesize
prompt = synthesize(
    structural_values={
        "edge_tension": 0.8,
        "negative_space": 0.7,
        ...
    },
    mood_values={
        "melancholy": 0.3,  # User modified
        "intimacy": 0.7,
        ...
    },
    alignment_override="bittersweet celebration",
    seed_text="children's birthday party",
    target_length="detailed"
)

# Output: "Frame the birthday scene with tight, edge-crowding composition that 
# leaves no breathing room. Let the celebratory subject matter sit within 
# dramatic, intimate framing—the closeness should feel warm but the compression 
# hints at transience..."
```

## Installation

### Using FastMCP Cloud

1. Create a GitHub repository from this template
2. Configure FastMCP Cloud with:
   - **Runtime**: Python
   - **Entrypoint**: `grid_dynamics_mcp.grid_dynamics_mcp:mcp`
   - **Package Manager**: uv
3. Deploy

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest test_grid_dynamics.py -v

# Run server locally
python grid_dynamics_mcp.py
```

### Claude Desktop Integration

Add to `claude_desktop_config.json`:

```json
{
  "tools": {
    "grid-dynamics": {
      "command": "python",
      "args": ["-m", "fastmcp.cli", "grid_dynamics_mcp:mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "your-key-here"
      }
    }
  }
}
```

## Testing

All 22 tests pass:

```bash
pytest test_grid_dynamics.py -v
```

### Test Coverage

- **Layer 1 (Domain Types)**: 7 tests validating dataclass instantiation and serialization
- **Layer 2 (Interpretation)**: 8 tests verifying vision model integration and JSON parsing
- **Layer 3 (Synthesis)**: 3 tests for prompt generation with various intensities
- **Integration**: 4 tests for full endpoint workflows

## Composability

The output of `extract` feeds naturally into `synthesize`. The interpretation layer (structural/mood values) can be:

1. **User-modified via UI**: Sliders adjust numerical intensity
2. **Programmatically modified**: Upstream processes can adjust dimensions
3. **Injected from other tools**: Alignment preferences passed as parameters

This enables downstream composability in multi-olog systems (e.g., combining Grid Dynamics with game show aesthetics or perfume domains).

## Key Design Decisions

### Why Three Layers?

1. **Layer 1 (Domain)**: Mathematical rigor—categorical structure prevents circular dependencies
2. **Layer 2 (Interpretation)**: Deterministic extraction—vision models analyze, not synthesize
3. **Layer 3 (Synthesis)**: Creative LLM integration—single call reconciles all tensions

This separation allows:
- Independent testing of each layer
- User review/modification at Layer 2 before synthesis
- Reuse of interpretation layer in other contexts
- Mathematical proof of compositional coherence via commutative diagrams

### Why These 8 Structural Parameters?

Chosen to capture:
- **Spatial properties**: edge_tension, containment, negative_space
- **Distribution**: asymmetric_balance, scale_hierarchy
- **Temporal/kinetic properties**: rhythm, density, diagonal_dominance

Together they describe the "visual pressure" of a composition.

### Why These 7 Mood Dimensions?

Cover the sensory spectrum:
- **Emotional valence**: melancholy, warmth
- **Relational**: intimacy
- **Dynamic**: energy, tension, rhythm
- **Stillness**: stasis and presence

## Performance Considerations

- **Image analysis**: ~2-3 seconds (Claude vision model)
- **JSON parsing**: <100ms
- **Alignment proposal**: ~1 second
- **Prompt synthesis**: ~1 second

Total extraction: ~4-5 seconds
Total synthesis: ~1 second

## Limitations

- Image size: <10MB
- Format: JPEG recommended (PNG/WebP supported)
- Seed text: Optimized for 2-20 words
- Mood extraction: Deterministic but Claude-dependent (no fine-tuning)

## Future Extensions

1. **Layered composition**: Combine Grid Dynamics with other domains (game show, perfume, jazz)
2. **Iterative refinement**: Feed output back through extract for progressive enhancement
3. **Batch processing**: Analyze image series for consistent compositional direction
4. **Visual parameter export**: Generate ComfyUI/other workflow integrations from analysis

## Architecture References

- **Ologs**: David Spivak's ontology logs framework
- **Category Theory**: Morphisms as compositional structure preservation
- **Three-Layer Pattern**: Domain → Interpretation → Synthesis

See `grid_dynamics_mcp.py` for implementation details and type definitions.

## License

MIT
