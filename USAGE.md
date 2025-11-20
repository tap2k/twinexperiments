# Twin Experiments - Command Line Usage

## Scripts Overview

Three main scripts for experimentation:
1. **compare_formats.py** - Main testing script (models × formats)
2. **inspect_formats.py** - Preview persona formats
3. **test_llm_matching.py** - Debug answer matching

---

## compare_formats.py

Main testing script. Compare multiple persona formats and/or models in one run.

### Basic Usage
```bash
python compare_formats.py
```

### All Options
```
--models MODEL1,MODEL2,...  Comma-separated list of models (default: gemini-2.5-flash-lite)
--formats FMT1,FMT2,...     Comma-separated list of formats (default: summary)
--personas N                Number of personas to test (default: 5)
--questions N               Questions per persona (default: 3)
--block BLOCK               Filter by block name (case-insensitive partial match)
--data-dir PATH             Data directory (default: ../Twin-2K-500)
```

### Examples

**Single model, single format (default):**
```bash
python compare_formats.py
# Uses: gemini-2.5-flash-lite, summary format, 5 personas, 3 questions
```

**Compare formats:**
```bash
python compare_formats.py --formats empty,demographics_big5,summary
```

**Compare models:**
```bash
python compare_formats.py --models gemini-2.5-flash-lite,haiku-4.5,sonnet-4.5
```

**Matrix comparison (models × formats):**
```bash
python compare_formats.py --models gemini-2.5-flash-lite,haiku-4.5 --formats empty,summary
```

**Test specific question blocks:**
```bash
python compare_formats.py --block "False consensus" --questions 10
```

**Large-scale test:**
```bash
python compare_formats.py --personas 100 --questions 20
```

---

## inspect_formats.py

Preview and inspect persona format contents.

### Basic Usage
```bash
python inspect_formats.py
```

Shows all available formats with previews (first 300 characters of each).

### Inspect Specific Format
```bash
python inspect_formats.py demographics_big5
```

Shows the full content of a specific format (up to 2000 characters by default).

### All Options
```
format                  Format name to inspect (optional positional argument)
--persona-id PID        Persona ID to inspect (default: first persona)
--max-chars N           Max characters to show (default: 300 for all, 2000 for single)
--data-dir PATH         Data directory (default: ../Twin-2K-500)
```

### Examples

**Show all formats with default preview:**
```bash
python inspect_formats.py
```

**Inspect specific format with more detail:**
```bash
python inspect_formats.py summary --max-chars 5000
```

**Inspect different persona:**
```bash
python inspect_formats.py --persona-id 1348 demographics_big5
```

**Show more of all formats:**
```bash
python inspect_formats.py --max-chars 1000
```

---

## test_llm_matching.py

Debug answer matching logic with real LLM responses. Shows detailed output for each question including predicted vs. ground truth answers.

### Basic Usage
```bash
python test_llm_matching.py
```

### All Options
```
--model MODEL          LLM model to use (default: gemini-2.5-flash-lite)
--personas N           Number of personas to test (default: 2)
--questions N          Questions per persona (default: 5)
--format FORMAT        Persona format to use (default: demographics_big5)
--data-dir PATH        Data directory (default: ../Twin-2K-500)
```

### Examples

**Test with better model:**
```bash
python test_llm_matching.py --model sonnet-4.5
```

**Test with full summary format:**
```bash
python test_llm_matching.py --format summary --personas 3 --questions 10
```

**Quick debug test:**
```bash
python test_llm_matching.py --personas 1 --questions 3
```

---

## Available Models

Cheap/Fast:
- `gemini-2.5-flash-lite` (default)
- `gpt-5-nano`

Medium:
- `haiku-4.5`
- `llama-3.3-70b`

Expensive/High Quality:
- `sonnet-4.5`
- `gpt-5`

See `llm_client.py` MODEL_MAP for full list (~50+ models).

---

## Available Persona Formats

### Minimal
- `empty` - No persona information (baseline)
- `demographics_only` - Just demographics (~441 chars)

### Small
- `demographics_big5` - Demographics + Big 5 personality (~1.1KB)
- `demographics_qualitative` - Demographics + self-descriptions (~2.4KB)

### Medium
- `demographics_personality` - Demographics + personality scores
- `demographics_cognitive` - Demographics + cognitive/intelligence
- `demographics_economic` - Demographics + economic games

### Full
- `summary` - Full persona summary (~13.6KB) [default]
- `full_text` - Complete question/answer history (~129KB)

### Custom
- `demographics_big5_qualitative`
- `demographics_cognitive_economic`
- `all_scores_no_demographics`

---

## Outputs

### minimal_test.py outputs:
- Console: Accuracy by category and question type
- `data/minimal_test_results.csv` - Detailed results

### compare_formats.py outputs:
- Console: Comparison table (accuracy & cost per format)
- `data/format_comparison_results.csv` - Detailed results
- `data/format_comparison_summary.csv` - Summary table

### inspect_formats.py outputs:
- Console only: Format contents and character counts
- No files created (read-only inspection tool)

---

## Common Workflows

### Explore available formats first
```bash
python inspect_formats.py
```

### Inspect a specific format before testing
```bash
python inspect_formats.py demographics_big5
```

### Quick validation with cheap model
```bash
python minimal_test.py --model gemini-2.5-flash-lite --personas 5 --questions 3
```

### Full accuracy test with best model
```bash
python minimal_test.py --model sonnet-4.5 --personas 100 --questions 20
```

### Compare persona formats cost-effectively
```bash
python compare_formats.py --personas 10 --questions 5 --formats empty,demographics_only,demographics_big5,summary
```

### Test specific format across many personas
```bash
python minimal_test.py --format demographics_big5 --personas 50 --questions 10
```

---

## Tips

1. **Start small**: Use default settings (5 personas, 3 questions) for quick tests
2. **Cost estimation**: Cheaper models like `gemini-2.5-flash-lite` are good for experimentation
3. **Full validation**: Use 100+ personas and 20+ questions for robust results
4. **Format comparison**: Start with 3-5 formats, not all 12 at once
5. **Question limit**: Wave 4 has up to 63 questions per persona - use `--questions 63` for full coverage

