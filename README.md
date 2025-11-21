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

### Run Tests

Test with default settings (5 personas, 3 questions, summary format, gemini-2.5-flash-lite):

```bash
python compare_formats.py
```

## Persona Formats

The system supports flexible persona formatting via the `--formats` parameter:

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

## CLI Options

All experiments are run through `compare_formats.py` with flexible CLI options:

```bash
# Basic usage with defaults
python compare_formats.py

# Test specific model and format
python compare_formats.py --models sonnet-4.5 --formats demographics_big5

# Compare multiple formats
python compare_formats.py --formats empty,demographics_big5,summary

# Compare multiple models
python compare_formats.py --models gemini-2.5-flash-lite,haiku-4.5,sonnet-4.5

# Matrix comparison (models × formats)
python compare_formats.py --models gemini-2.5-flash-lite,haiku-4.5 --formats empty,summary

# More personas and questions
python compare_formats.py --personas 20 --questions 10

# Filter by question block
python compare_formats.py --block "False consensus"

# All options combined
python compare_formats.py \
  --models gemini-2.5-flash-lite,sonnet-4.5 \
  --formats demographics_big5,summary \
  --personas 10 \
  --questions 5 \
  --block "anchoring"
```

### Available Options

- `--models` - Comma-separated list of models (default: gemini-2.5-flash-lite)
- `--formats` - Comma-separated list of persona formats (default: summary)
- `--personas` - Number of personas to test (default: 5)
- `--questions` - Questions per persona (default: 3)
- `--block` - Filter questions by block name (case-insensitive partial match)
- `--data-dir` - Data directory path (default: ../Twin-2K-500)

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

## Additional Tools

### Inspect Persona Formats

Preview what each format contains before running experiments:

```bash
# Show all formats for default persona
python inspect_formats.py

# Show specific format
python inspect_formats.py demographics_big5

# Show different persona
python inspect_formats.py --persona-id 1348 summary

# Show more characters
python inspect_formats.py --max-chars 2000 summary
```

### Test Answer Matching

Debug answer matching with real LLM responses:

```bash
# Test with defaults
python test_llm_matching.py

# Test with specific model and format
python test_llm_matching.py --model sonnet-4.5 --format summary

# Test more extensively
python test_llm_matching.py --personas 5 --questions 10
```

### Dump Questions

Extract all questions from the dataset to CSV for analysis:

```bash
# List all available blocks
python dump_questions.py --list-blocks

# Dump all questions
python dump_questions.py

# Dump questions from a specific block
python dump_questions.py --block "False consensus"

# Dump from limited personas
python dump_questions.py --personas 10 --output data/sample_questions.csv
```

## Output

Results are saved to `data/` directory:
- `format_comparison_results.csv` - Detailed predictions (single model)
- `format_comparison_summary.csv` - Summary table (single model)
- `comparison_results_{timestamp}.csv` - Detailed results (multiple models)
- `comparison_summary_{timestamp}.csv` - Summary table (multiple models)
