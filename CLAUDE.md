# Twin Experiments - Developer Context for Claude Code

This document provides comprehensive context for Claude Code when working on the Twin-2K experiments codebase.

## Project Purpose

This is a **lightweight experimentation framework** for validating Twin-2K digital twin simulations against ground truth data. Unlike the full `Twin-2K-500-Mega-Study` pipeline (which uses Snakemake, complex workflows, and multiple processing stages), this directory focuses on **rapid iteration and hypothesis testing** with minimal dependencies.

### Why This Exists

The full Twin-2K-500-Mega-Study codebase is production-grade for running comprehensive evaluations, but it's heavyweight for quick experiments. This `twinexperiments/` directory enables:

1. **Fast persona format testing** - Test 12+ persona variants (empty, demographics-only, full summary, etc.)
2. **Model comparison** - Easily swap between 50+ LLM models via `llm_client.py`
3. **Category-level analysis** - Break down accuracy by Wave 4 study blocks (false consensus, base-rate fallacy, etc.)
4. **Cost-benefit analysis** - Compare persona length vs. accuracy vs. API cost
5. **Flexible experimentation** - All scripts are independent, fully CLI-configurable, and generate CSV outputs

## Repository Structure Context

```
Twinnning/          # Parent directory
├── Twin-2K-500/                 # Dataset repository (sibling)
│   ├── full_persona/            # Complete persona profiles (parquet chunks)
│   │   └── chunks/              # *.parquet files with pid, persona_text, persona_summary, persona_json
│   ├── wave_split/              # Train/test split for model evaluation
│   └── raw_data/                # Original Qualtrics QSF/CSV files
│
├── Twin-2K-500-Mega-Study/      # Full production pipeline (sibling)
│   ├── mega_study_evaluation/   # Meta-analysis across all studies
│   ├── processing_qualtrics_*/  # QSF/CSV processing
│   ├── text_simulation/         # LLM simulation pipeline
│   └── Snakefile                # Complex workflow orchestration
│
└── twinexperiments/             # THIS DIRECTORY - Simple experiments
    ├── minimal_test.py          # Main validation script (CLI enabled)
    ├── compare_formats.py       # Format comparison (CLI enabled)
    ├── inspect_formats.py       # Format preview tool (CLI enabled)
    ├── persona_formatter.py     # Flexible persona formatting system
    ├── llm_client.py            # Unified LLM client (~50 models)
    ├── data/                    # Output CSVs
    ├── README.md                # Quick start guide
    ├── USAGE.md                 # Comprehensive CLI documentation
    └── CLAUDE.md                # This file
```

## Dataset Details (Twin-2K-500)

### Core Dataset
- **2,058 personas** from representative US sample
- **Four waves** of data collection (Waves 1-3 training, Wave 4 validation)
- **Wave 4**: 63 questions × 2,058 participants = 129,654 total predictions to validate

### Available Persona Data Columns

When you read parquet files from `../Twin-2K-500/full_persona/chunks/*.parquet`:

```python
df.columns = ['pid', 'persona_text', 'persona_summary', 'persona_json']
```

**Key fields**:
- `pid` (int): Participant ID (e.g., 951, 1348)
- `persona_text` (~129KB): Full Q&A history with all survey responses
- `persona_summary` (~13.6KB): Structured summary with:
  - Demographics (age, gender, region, education, income, politics, religion)
  - Big 5 personality scores (openness, conscientiousness, extraversion, agreeableness, neuroticism)
  - Qualitative self-descriptions (3 open-ended paragraphs)
  - Cognitive/intelligence scores (~10 measures)
  - Economic game scores (~8 measures)
  - Personality trait scores (~15 measures)
  - Wellbeing scores (anxiety, depression)
- `persona_json` (JSON): Structured format of persona_text for selective question extraction

### Wave 4 Ground Truth Structure

Wave 4 data is stored in parquet files at `../Twin-2K-500/wave_split/chunks/*.parquet`:

```python
df.columns = ['pid', 'wave1_3_persona_text', 'wave1_3_persona_json',
              'wave4_Q_wave1_3_A', 'wave4_Q_wave4_A']
```

The `wave4_Q_wave4_A` column contains 18 study blocks:

**Wave 4 Study Blocks** (18 behavioral economics experiments):
1. `False Consensus Effect` - Overestimating agreement with one's opinions
2. `Base-rate Fallacy` - Ignoring prior probabilities
3. `Preference Redistribution` - Changing preferences when framing changes
4. `Ambiguity Aversion` - Preferring known vs. unknown probabilities
5. `Reflection Effect` - Risk preferences reversing for gains vs. losses
6. `Certainty Effect` - Overweighting certain outcomes
7. `Disposition Effect` - Selling winners too early, holding losers too long
8. `Ellsberg Paradox` - Ambiguity aversion in probability judgments
9. `Decoy Effect` - Adding irrelevant options changes preferences
10. `Isolation Effect` - Focusing on differences, ignoring similarities
11. `Present Bias` - Preferring immediate rewards
12. `Risky-choice Framing` - Gain/loss framing affects risk preferences
13. `Anchoring` - Over-relying on first piece of information
14. `Attribute Framing` - Positive/negative framing of same attribute
15. `IKEA Effect` - Valuing self-assembled items more
16. `Sunk-cost Fallacy` - Continuing due to past investment
17. `Endowment Effect` - Valuing owned items more than identical non-owned
18. `Status-quo Bias` - Preferring current state

**Question Types**:
- `MC` (Multiple Choice): Single selection from options
- `Matrix` (Likert Scale): Rating questions (e.g., 1-7 scale)
- `Slider`: Numeric range input
- `TE` (Text Entry): Free-form text responses

## Core Scripts Architecture

### Design Principles
1. **Full decoupling**: Each script has independent DEFAULT_* constants
2. **CLI-first**: All parameters configurable via argparse
3. **Backward compatible**: Scripts work without arguments (using defaults)
4. **CSV outputs**: All results saved to `data/` for analysis
5. **No global state**: Functions accept parameters, return values

### Script Descriptions

#### 1. minimal_test.py
**Purpose**: Single persona format validation against Wave 4 ground truth

**Key Functions**:
```python
def load_persona_data(data_dir, num_personas):
    """Load persona summaries from parquet chunks."""

def load_wave4_ground_truth(data_dir):
    """Load Wave 4 questions/answers from parquet."""

def extract_wave4_questions(wave4_json):
    """Parse JSON to extract question list with block_name tracking."""

def create_prompt(persona_text, question_data):
    """Create LLM prompt from persona + question."""

def check_answer(model_answer, correct_answer, question_type, options):
    """Validate model prediction against ground truth."""

def run_minimal_test(model, persona_format, num_personas, max_questions, data_dir):
    """Main test execution with category/type breakdown reporting."""
```

**Output Structure**:
```
ACCURACY BY CATEGORY
====================
                    count  sum   mean
block_name
Anchoring            15    12   0.800
Base-rate Fallacy    15     9   0.600
...

ACCURACY BY QUESTION TYPE
=========================
               count  sum   mean
question_type
MC              45    35   0.778
Matrix          10     8   0.800
...

Overall Accuracy: 72.3%
```

**CSV Output**: `data/minimal_test_results.csv`
```csv
pid,question_id,block_name,question_type,correct,model_answer,true_answer
951,QID12,Anchoring,MC,True,Option 2,Option 2
```

#### 2. compare_formats.py
**Purpose**: Compare multiple persona formats in single run

**Key Differences from minimal_test.py**:
- Duplicates data loading functions (for decoupling)
- Only imports `extract_wave4_questions` and `create_prompt` from minimal_test.py
- Loops over multiple formats and outputs comparison table

**Output Structure**:
```
FORMAT COMPARISON
=================
Format                          Accuracy  Avg Chars  Est. Cost
empty                           45.2%          0     $0.10
demographics_only              58.1%        441     $0.15
demographics_big5              64.3%       1149     $0.18
summary                        72.3%      13583     $0.45
```

**CSV Outputs**:
- `data/format_comparison_results.csv` - Detailed per-question results
- `data/format_comparison_summary.csv` - Summary table

#### 3. inspect_formats.py
**Purpose**: Preview persona format contents before testing

**Key Functions**:
```python
# Uses persona_formatter.py functions
from persona_formatter import PERSONA_FORMATS, format_persona

# Shows formatted output with character counts
# Supports inspecting different personas by PID
# Configurable character preview limits
```

**Output** (console only, no files):
```
PERSONA FORMAT INSPECTOR (PID: 951)
====================================

FORMAT: demographics_big5
Length: 1,149 characters

Content (first 2000 chars):
The person's demographics are the following...
Geographic region: West (WA, OR, ID, MT, WY, CA, NV, UT, CO, AZ, NM)
Gender: Female
...
```

#### 4. persona_formatter.py
**Purpose**: Flexible persona component extraction and combination

**Architecture**:
```python
# Component extractors (regex-based)
def extract_demographics(persona_summary: str) -> str
def extract_big5(persona_summary: str) -> str
def extract_qualitative(persona_summary: str) -> str
def extract_cognitive(persona_summary: str) -> str
def extract_economic(persona_summary: str) -> str
def extract_personality(persona_summary: str) -> str
def extract_wellbeing(persona_summary: str) -> str

# 12 predefined formats
PERSONA_FORMATS = {
    'empty': lambda row: "",
    'demographics_only': ...,
    'demographics_big5': ...,
    'demographics_qualitative': ...,
    'demographics_big5_qualitative': ...,
    'demographics_personality': ...,
    'demographics_cognitive': ...,
    'demographics_economic': ...,
    'demographics_cognitive_economic': ...,
    'all_scores_no_demographics': ...,
    'summary': lambda row: row['persona_summary'],
    'full_text': lambda row: row['persona_text'],
}

# Custom format builder
def create_custom_format(components: List[str]):
    """Build custom format from component list."""
```

**Design Rationale**:
- Regex extraction allows mixing/matching persona components
- Lambda-based PERSONA_FORMATS enables easy format definition
- Falls back to full persona_summary if extraction fails (defensive)

#### 5. llm_client.py
**Purpose**: Unified interface to ~50 LLM providers

**Supported Providers**:
- OpenAI (GPT-4, GPT-5, etc.)
- Anthropic (Claude models)
- Google (Gemini models)
- DeepInfra (Llama, Mixtral, etc.)
- Together AI
- AWS Bedrock
- Azure OpenAI

**Usage**:
```python
from llm_client import get_llm_response

response = get_llm_response(
    prompt="...",
    model="gemini-2.5-flash-lite",  # or sonnet-4.5, gpt-5, etc.
    system_prompt="You are a digital twin...",
    temperature=0.0
)
```

**Model Selection Guidance**:
- **Cheap/Fast**: `gemini-2.5-flash-lite` ($0.1/$0.4 per 1M tokens)
- **Medium**: `haiku-4.5` ($1/$5 per 1M tokens)
- **High Quality**: `sonnet-4.5` ($3/$15 per 1M tokens), `gpt-5`

## Configuration and CLI Arguments

All three executable scripts now support comprehensive CLI configuration:

### minimal_test.py
```bash
python minimal_test.py \
  --model sonnet-4.5 \
  --personas 20 \
  --questions 10 \
  --format demographics_big5 \
  --data-dir ../Twin-2K-500
```

### compare_formats.py
```bash
python compare_formats.py \
  --model gemini-2.5-flash-lite \
  --personas 10 \
  --questions 5 \
  --formats empty,demographics_only,demographics_big5,summary \
  --data-dir ../Twin-2K-500
```

### inspect_formats.py
```bash
python inspect_formats.py demographics_big5 \
  --persona-id 1348 \
  --max-chars 5000 \
  --data-dir ../Twin-2K-500
```

**See USAGE.md** for complete CLI documentation and workflow examples.

## Key Design Decisions

### 1. Why Decouple Scripts?
**Problem**: Original design had `compare_formats.py` importing constants from `minimal_test.py`, creating tight coupling.

**Solution**:
- Each script has independent `DEFAULT_*` constants
- `compare_formats.py` only imports functions, not config
- Enables testing different settings without affecting other scripts

### 2. Why Duplicate Data Loading?
**Trade-off**: Code duplication vs. configuration coupling

**Choice**: Duplicate `load_persona_data()` and `load_wave4_ground_truth()` in `compare_formats.py` to accept parameters rather than using shared functions with global config.

**Rationale**: Simpler mental model, easier to modify independently, no risk of unintended side effects.

### 3. Why CLI Arguments for All Scripts?
**User Workflow**: Experimentation requires rapid parameter changes without editing code.

**Solution**: `argparse` with sensible defaults matching original constants.

**Benefits**:
- Quick iteration: `python minimal_test.py --personas 3 --questions 2`
- Scriptable: Can be called from notebooks or other scripts
- Documented: `--help` shows all options

### 4. Why Category Breakdown in Results?
**Research Value**: Understanding which types of experiments are easier/harder for LLMs to simulate.

**Implementation**:
```python
# Added block_name tracking to extract_wave4_questions()
questions.append({
    'question_id': question.get('QuestionID', ''),
    'block_name': block_name,  # Track which study block
    ...
})

# Report breakdown using pandas groupby
results_df.groupby('block_name').agg({'correct': ['count', 'sum', 'mean']})
```

**Output**: Identifies specific cognitive biases where LLMs fail (e.g., "Base-rate Fallacy: 45% accuracy").

## Common Development Patterns

### Adding a New Persona Format

1. **Define extraction function** (if new component):
```python
# In persona_formatter.py
def extract_new_component(persona_summary: str) -> str:
    """Extract new component from persona summary."""
    pattern = r'SECTION HEADER(.*?)(?=\n\n[A-Z]|$)'
    match = re.search(pattern, persona_summary, re.DOTALL)
    return match.group(1).strip() if match else ""
```

2. **Add to PERSONA_FORMATS**:
```python
PERSONA_FORMATS['new_format'] = lambda row: '\n\n'.join(filter(None, [
    extract_demographics(row['persona_summary']),
    extract_new_component(row['persona_summary']),
]))
```

3. **Test with inspect_formats.py**:
```bash
python inspect_formats.py new_format
```

### Adding a New Model

Edit `llm_client.py` MODEL_MAP:
```python
MODEL_MAP = {
    'new-model-name': {
        'provider': 'openai',  # or anthropic, google, etc.
        'model_id': 'actual-api-model-name',
        'input_price': 1.0,   # per 1M tokens
        'output_price': 5.0,  # per 1M tokens
    },
    ...
}
```

No changes needed in test scripts - just use `--model new-model-name`.

### Running Full Evaluation (All 2,058 Personas, All 63 Questions)

**Warning**: This is expensive! Estimate cost first:

```python
# For summary format (~13.6KB per persona):
# Input tokens: 2,058 personas × 63 questions × ~3,400 tokens ≈ 441M tokens
# With gemini-2.5-flash-lite ($0.1 per 1M input): ~$44

# With sonnet-4.5 ($3 per 1M input): ~$1,323
```

**Run**:
```bash
python minimal_test.py --personas 2058 --questions 63 --model gemini-2.5-flash-lite
```

**Expected Runtime**: 2-4 hours depending on API rate limits and parallel processing.

## Integration with Full Pipeline

### When to Use twinexperiments/ vs. Twin-2K-500-Mega-Study/

**Use `twinexperiments/` for**:
- Quick hypothesis testing ("Does demographics alone predict better than empty baseline?")
- Persona format exploration ("What's the minimal info needed for 60% accuracy?")
- Model comparison ("Is Gemini good enough or do we need Claude?")
- Cost-benefit analysis ("Is the 10x token cost of full summary worth 5% accuracy gain?")

**Use `Twin-2K-500-Mega-Study/` for**:
- Production evaluation runs with Snakemake orchestration
- Processing new Qualtrics surveys (QSF/CSV → JSON)
- Meta-analysis across multiple studies
- XGBoost baseline comparisons
- Publishing results for papers

### Data Flow

```
Twin-2K-500/ (Dataset)
  └─> full_persona/chunks/*.parquet  ──┐
  └─> wave_split/chunks/*.parquet    ──┼──> twinexperiments/ (Quick experiments)
                                        │    ├─> minimal_test.py
                                        │    ├─> compare_formats.py
                                        │    └─> inspect_formats.py
                                        │
                                        └──> Twin-2K-500-Mega-Study/ (Production)
                                             └─> Snakemake pipeline
```

### Exporting Findings

After experiments in `twinexperiments/`, production runs in full pipeline:

1. **Identify best format** in twinexperiments:
```bash
python compare_formats.py --personas 50 --questions 20
# Result: demographics_big5 is 95% as accurate as summary, 10x cheaper
```

2. **Update production config** in Twin-2K-500-Mega-Study:
```yaml
# configs/my_study.yaml
text_simulation:
  personas_to_texts:
    persona_variant: demographics_big5  # Changed from 'full'
```

3. **Run production pipeline**:
```bash
poetry run snakemake --configfile configs/my_study.yaml --cores 4
```

## Troubleshooting Guide

### Issue: "ModuleNotFoundError: No module named 'pandas'"
**Solution**: Activate virtual environment:
```bash
source venv/bin/activate
python minimal_test.py
```

### Issue: Results seem random/low accuracy
**Check**:
1. **Temperature**: Use 0.0 for factual questions, 1.0 for creative tasks
2. **Model**: Cheap models may underperform (try `sonnet-4.5` baseline)
3. **Format**: `empty` baseline should be ~45%, full `summary` should be ~70%+

### Issue: "Persona ID not found"
**Cause**: Not all PIDs are in every parquet chunk.

**Solution**:
```python
# In inspect_formats.py, already handled:
persona_row = persona_df[persona_df['pid'] == args.persona_id]
if len(persona_row) == 0:
    print(f"Error: Persona ID {args.persona_id} not found")
    return
```

### Issue: API rate limits
**Solutions**:
1. Reduce `num_personas` or `max_questions`
2. Add delays between requests (edit `llm_client.py`)
3. Use cheaper model with higher rate limits (e.g., `gemini-2.5-flash-lite`)

## Testing Checklist for New Features

When adding new functionality:

- [ ] Update `DEFAULT_*` constants if adding new parameters
- [ ] Add argparse argument with default matching constant
- [ ] Update function signatures to accept new parameters
- [ ] Test with `--help` to verify documentation
- [ ] Test with custom arguments to verify override works
- [ ] Test without arguments to verify backward compatibility
- [ ] Update USAGE.md with new examples
- [ ] Update this CLAUDE.md if architectural changes

## Code Style and Conventions

### File Organization
```python
# 1. Imports (standard library, third-party, local)
import json
from pathlib import Path
import pandas as pd
from llm_client import get_llm_response

# 2. Constants (DEFAULT_* for configurable values)
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "Twin-2K-500"
DEFAULT_MODEL = "gemini-2.5-flash-lite"

# 3. Helper functions (pure functions, no side effects)
def extract_wave4_questions(wave4_json):
    """Parse JSON to extract question list."""
    ...

# 4. Main logic functions (accept parameters, return values)
def run_minimal_test(model, persona_format, num_personas, max_questions, data_dir):
    """Run minimal test with configurable parameters."""
    ...

# 5. CLI entry point
def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(...)
    args = parser.parse_args()
    run_minimal_test(...)

if __name__ == "__main__":
    main()
```

### Variable Naming
- `persona_df`: DataFrame of persona data
- `wave4_df`: DataFrame of Wave 4 ground truth
- `question_data`: Single question dict from extract_wave4_questions()
- `model_answer`: LLM prediction
- `correct_answer` or `true_answer`: Ground truth from data

### Error Handling Philosophy
- **Fail loudly for missing data**: Don't silently skip personas or questions
- **Defensive extraction**: Regex extractions should handle missing sections gracefully
- **User-facing errors**: Print clear error messages, not stack traces

## Performance Considerations

### Bottlenecks
1. **LLM API calls**: ~1-2 seconds per question (dominant cost)
2. **Parquet reading**: ~100ms to load chunks (negligible)
3. **Prompt construction**: <1ms (negligible)

### Optimization Strategies
- **Batch API calls**: Not yet implemented, could add ThreadPoolExecutor
- **Cache responses**: Not yet implemented, could add disk cache by (pid, question_id, model, format)
- **Stream processing**: Currently loads all data into memory (fine for 2K personas)

### Current Performance (Single-threaded)
- **5 personas × 3 questions = 15 predictions**: ~30 seconds
- **100 personas × 20 questions = 2,000 predictions**: ~1 hour
- **2,058 personas × 63 questions = 129,654 predictions**: ~72 hours

## Future Enhancements (Potential)

Based on current architecture, easy additions:

1. **Caching Layer**:
```python
# Could add to llm_client.py
def get_llm_response(prompt, model, system_prompt, temperature, cache_dir=None):
    if cache_dir:
        cache_key = hashlib.md5(f"{prompt}{model}{system_prompt}{temperature}".encode()).hexdigest()
        cache_file = cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            return json.load(open(cache_file))
    response = _call_api(...)
    if cache_dir:
        json.dump(response, open(cache_file, 'w'))
    return response
```

2. **Parallel Execution**:
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(get_llm_response, prompt, model) for prompt in prompts]
    results = [f.result() for f in futures]
```

3. **Incremental Results**:
```python
# Save after each persona instead of at end
for persona_idx, persona_row in persona_df.iterrows():
    # ... run questions ...
    pd.DataFrame(results).to_csv('data/minimal_test_results.csv', mode='a', header=False)
```

4. **More Persona Formats**:
- `demographics_big5_economic` (combine economic games with big5)
- `qualitative_only` (just the 3 self-description paragraphs)
- `scores_only` (all numeric scores, no text)

## References and Links

- **Dataset**: [Twin-2K-500 on HuggingFace](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500)
- **Paper**: [Digital Twin Simulation Paper](https://arxiv.org/abs/2505.17479)
- **Mega Study Paper**: [Twin-2K-500 Mega Study](https://arxiv.org/abs/2509.19088)
- **Documentation**: [ReadTheDocs](https://digital-twin-simulation-version2.readthedocs.io/en/latest/)
- **Parent Repo**: [Digital-Twin-Simulation GitHub](https://github.com/tianyipeng-lab/Digital-Twin-Simulation)

---

**Last Updated**: 2025-11-20
**Maintained By**: Research team
**Claude Code Version**: Compatible with Claude Sonnet 4.5
