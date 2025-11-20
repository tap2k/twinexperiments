# Twin-2K Experiments

Simple experiments for validating Twin-2K simulation against ground truth.

## Quick Start

### Setup

```bash
cd twinexperiments
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

Copy API keys to `.env` file.

### Run Minimal Test

Test with default settings (5 personas, full summary format):

```bash
python minimal_test.py
```

## Persona Formats

The system supports flexible persona formatting. Change `PERSONA_FORMAT` in `minimal_test.py`:

### Minimal Formats
- `empty` - No persona information (baseline)
- `demographics_only` - Just demographics (~441 chars)

### Small Formats
- `demographics_big5` - Demographics + Big 5 personality (~1.1KB)
- `demographics_qualitative` - Demographics + self-descriptions (~2.4KB)

### Medium Formats
- `demographics_personality` - Demographics + Big 5 + personality scores
- `demographics_cognitive` - Demographics + cognitive/intelligence scores
- `demographics_economic` - Demographics + economic game scores

### Full Formats
- `summary` - Full persona summary (~13.6KB) **[DEFAULT]**
- `full_text` - Complete question/answer history (~129KB)

### Custom Combinations
- `demographics_big5_qualitative`
- `demographics_cognitive_economic`
- `all_scores_no_demographics`

## Compare Persona Formats and Models

Test multiple formats and/or models to see their impact on accuracy:

```bash
# Compare formats only
python compare_formats.py

# Compare models only
python compare_formats.py --models gemini-2.5-flash-lite,haiku-4.5,sonnet-4.5 --formats summary

# Compare both (matrix comparison)
python compare_formats.py --models gemini-2.5-flash-lite,haiku-4.5 --formats empty,demographics_big5,summary
```

This will run tests with different combinations and output a comparison table.

## Configuration

Edit constants at the top of `minimal_test.py`:

```python
NUM_PERSONAS = 5                      # Number of personas to test
MODEL = "gemini-2.5-flash-lite"       # LLM model to use
PERSONA_FORMAT = "summary"            # Persona format (see above)
DATA_DIR = Path(__file__).parent.parent / "Twin-2K-500"  # Data location
```

To test different numbers of questions per persona, change line 247:
```python
for q_idx, q_data in enumerate(questions[:3]):  # Change :3 to :10, etc.
```

## Available Models

See `llm_client.py` MODEL_MAP for all available models. Examples:
- `gemini-2.5-flash-lite` ($0.1/$0.4 per 1M tokens)
- `gpt-5-nano` ($0.05/$0.40 per 1M tokens)
- `haiku-4.5` ($1.0/$5.0 per 1M tokens)
- `sonnet-4.5` ($3.0/$15.0 per 1M tokens)
- `llama-3.3-70b` (via DeepInfra)

## Custom Persona Formats

Create your own format in Python:

```python
from persona_formatter import create_custom_format

# Create format with specific components
my_format = create_custom_format(['demographics', 'big5', 'cognitive'])
persona_text = my_format(persona_row)
```

Available components:
- `demographics` - Age, gender, location, education, etc.
- `big5` - Big 5 personality scores
- `qualitative` - Self-description paragraphs
- `cognitive` - Intelligence and reasoning scores
- `economic` - Economic game behavior
- `personality` - Personality traits and values
- `wellbeing` - Anxiety and depression scores

## Output

Results are saved to `data/` directory:
- `minimal_test_results.csv` - Detailed predictions from minimal_test.py
- `format_comparison_results.csv` - Format/model comparison details
- `format_comparison_summary.csv` - Format/model comparison summary
- `comparison_results_{timestamp}.csv` - Timestamped results when comparing multiple models
- `comparison_summary_{timestamp}.csv` - Timestamped summary when comparing multiple models
