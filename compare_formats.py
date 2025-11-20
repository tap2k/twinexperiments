"""
Compare different persona formats to see their impact on accuracy.

Usage:
    python compare_formats.py [options]

Options:
    --model MODEL           LLM model to use (default: gemini-2.5-flash-lite)
    --personas N            Number of personas to test (default: 5)
    --questions N           Number of questions per persona (default: 3)
    --formats FORMAT1,FORMAT2,...  Comma-separated list of formats to test
"""

import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from minimal_test import (
    extract_wave4_questions,
    create_prompt,
)
from llm_client import call_llm, calculate_cost
from persona_formatter import format_persona


# Default configuration (independent from minimal_test.py)
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "Twin-2K-500"
DEFAULT_NUM_PERSONAS = 5
DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_MAX_QUESTIONS = 3

# Default formats to compare
DEFAULT_FORMATS = [
    'empty',
    'demographics_only',
    'demographics_big5',
    'demographics_big5_qualitative',
    'summary',
]


def load_persona_data(data_dir, num_personas):
    """Load persona summaries from parquet files."""
    persona_chunks = list((data_dir / "full_persona/chunks").glob("*.parquet"))

    if not persona_chunks:
        raise FileNotFoundError(f"No parquet files found in {data_dir / 'full_persona/chunks'}")

    # Load first chunk and get first N personas
    df = pd.read_parquet(persona_chunks[0])
    return df.head(num_personas)


def load_wave4_ground_truth(data_dir):
    """Load Wave 4 ground truth from parquet files."""
    wave4_chunks = list((data_dir / "wave_split/chunks").glob("*.parquet"))

    if not wave4_chunks:
        raise FileNotFoundError(f"No parquet files found in {data_dir / 'wave_split/chunks'}")

    # Load all chunks and concatenate
    dfs = [pd.read_parquet(chunk) for chunk in wave4_chunks]
    return pd.concat(dfs, ignore_index=True)


def test_format(format_name, persona_df, wave4_df, model, max_questions=3):
    """Test a single persona format and return results."""
    print(f"\nTesting format: {format_name}")
    print("-" * 80)

    results = []
    total_questions = 0
    correct_predictions = 0
    total_cost = 0.0

    for idx, persona_row in persona_df.iterrows():
        pid = persona_row['pid']
        persona_text = format_persona(persona_row, format_name)

        # Get Wave 4 questions for this persona
        pid_int = int(pid) if isinstance(pid, str) else pid
        persona_wave4 = wave4_df[wave4_df['pid'] == pid_int]

        if len(persona_wave4) == 0:
            continue

        wave4_json = persona_wave4.iloc[0]['wave4_Q_wave4_A']
        if not wave4_json:
            continue

        questions = extract_wave4_questions(wave4_json)
        if not questions:
            continue

        # Test first N questions
        for q_data in questions[:max_questions]:
            question_text = q_data['question_text']
            question_type = q_data['question_type']
            ground_truth = q_data['answer']
            options = q_data.get('options', [])

            prompt = create_prompt(persona_text, question_text, question_type, options)

            try:
                llm_response = call_llm(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )

                predicted_answer = llm_response.strip()

                # Estimate cost
                input_tokens = len(prompt) // 4
                output_tokens = len(predicted_answer) // 4
                cost_info = calculate_cost(input_tokens, output_tokens, model)
                total_cost += cost_info['total_cost']

                if not predicted_answer:
                    continue

                # Compare with ground truth
                if question_type in ["TE", "Slider"]:
                    try:
                        is_correct = abs(float(predicted_answer) - float(ground_truth)) < 0.01
                    except ValueError:
                        is_correct = predicted_answer.lower() == ground_truth.lower()
                else:
                    is_correct = predicted_answer.lower().strip() == ground_truth.lower().strip()

                total_questions += 1
                if is_correct:
                    correct_predictions += 1

                results.append({
                    'format': format_name,
                    'pid': pid,
                    'question_id': q_data['question_id'],
                    'question_type': question_type,
                    'block_name': q_data.get('block_name', 'Unknown'),
                    'predicted': predicted_answer,
                    'actual': ground_truth,
                    'correct': is_correct
                })

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                continue

    accuracy = (correct_predictions / total_questions * 100) if total_questions > 0 else 0
    print(f"  Questions: {total_questions}, Correct: {correct_predictions}, Accuracy: {accuracy:.1f}%, Cost: ${total_cost:.4f}")

    return {
        'format': format_name,
        'total_questions': total_questions,
        'correct': correct_predictions,
        'accuracy': accuracy,
        'cost': total_cost,
        'results': results
    }


def main():
    """Compare all persona formats."""
    parser = argparse.ArgumentParser(description='Compare persona formats')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f'LLM model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--personas', type=int, default=DEFAULT_NUM_PERSONAS,
                        help=f'Number of personas to test (default: {DEFAULT_NUM_PERSONAS})')
    parser.add_argument('--questions', type=int, default=DEFAULT_MAX_QUESTIONS,
                        help=f'Number of questions per persona (default: {DEFAULT_MAX_QUESTIONS})')
    parser.add_argument('--formats', type=str, default=None,
                        help=f'Comma-separated list of formats (default: {",".join(DEFAULT_FORMATS)})')
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help=f'Data directory (default: {DEFAULT_DATA_DIR})')

    args = parser.parse_args()

    # Parse formats
    if args.formats:
        formats_to_test = [f.strip() for f in args.formats.split(',')]
    else:
        formats_to_test = DEFAULT_FORMATS

    data_dir = Path(args.data_dir)

    print("=" * 80)
    print("PERSONA FORMAT COMPARISON")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Personas: {args.personas}")
    print(f"Questions per persona: {args.questions}")
    print(f"Formats to test: {', '.join(formats_to_test)}")
    print(f"Data directory: {data_dir}")
    print()

    # Load data once
    print("Loading data...")
    persona_df = load_persona_data(data_dir, args.personas)
    wave4_df = load_wave4_ground_truth(data_dir)
    print(f"  Loaded {len(persona_df)} personas, {len(wave4_df)} Wave 4 responses")

    # Test each format
    all_results = []
    summary_stats = []

    for format_name in formats_to_test:
        result = test_format(format_name, persona_df, wave4_df, args.model, args.questions)
        all_results.extend(result['results'])
        summary_stats.append({
            'format': format_name,
            'accuracy': result['accuracy'],
            'cost': result['cost'],
            'questions': result['total_questions']
        })

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Format':<40} {'Accuracy':<12} {'Cost':<12} {'Questions':<10}")
    print("-" * 80)
    for stat in summary_stats:
        print(f"{stat['format']:<40} {stat['accuracy']:>6.1f}%     ${stat['cost']:>8.4f}    {stat['questions']:>6}")

    # Save detailed results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_dir = Path(__file__).parent / "data"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "format_comparison_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")

    # Save summary
    summary_df = pd.DataFrame(summary_stats)
    summary_file = output_dir / "format_comparison_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
