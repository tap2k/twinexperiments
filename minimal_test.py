"""
Minimal test script for Twin-2K simulation validation.

Tests personas against Wave 4 ground truth using a single LLM model.

Usage:
    python minimal_test.py [options]

Options:
    --model MODEL           LLM model to use (default: gemini-2.5-flash-lite)
    --personas N            Number of personas to test (default: 5)
    --questions N           Number of questions per persona (default: 3)
    --format FORMAT         Persona format to use (default: summary)
    --data-dir PATH         Data directory (default: ../Twin-2K-500)
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from llm_client import call_llm, calculate_cost
from persona_formatter import format_persona, list_formats


# Default configuration
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "Twin-2K-500"
DEFAULT_NUM_PERSONAS = 5
DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_PERSONA_FORMAT = "summary"
DEFAULT_MAX_QUESTIONS = 3

# Persona format options:
# - 'empty': No persona information
# - 'demographics_only': Just demographics
# - 'demographics_big5': Demographics + Big 5 personality
# - 'demographics_qualitative': Demographics + self-descriptions
# - 'demographics_personality': Demographics + Big 5 + personality scores
# - 'demographics_cognitive': Demographics + cognitive/intelligence scores
# - 'demographics_economic': Demographics + economic game scores
# - 'summary': Full persona summary (~13.6KB) [DEFAULT]
# - 'full_text': Complete question/answer history (~129KB)
# - 'demographics_big5_qualitative': Custom combination
# - 'demographics_cognitive_economic': Custom combination
# - 'all_scores_no_demographics': All scores without demographics


def load_persona_data(data_dir=DEFAULT_DATA_DIR, num_personas=DEFAULT_NUM_PERSONAS):
    """Load persona summaries from parquet files."""
    persona_chunks = list((data_dir / "full_persona/chunks").glob("*.parquet"))

    if not persona_chunks:
        raise FileNotFoundError(f"No parquet files found in {data_dir / 'full_persona/chunks'}")

    # Load first chunk and get first N personas
    df = pd.read_parquet(persona_chunks[0])
    return df.head(num_personas)


def load_wave4_ground_truth(data_dir=DEFAULT_DATA_DIR):
    """Load Wave 4 ground truth from parquet files."""
    wave4_chunks = list((data_dir / "wave_split/chunks").glob("*.parquet"))

    if not wave4_chunks:
        raise FileNotFoundError(f"No parquet files found in {data_dir / 'wave_split/chunks'}")

    # Load all chunks and concatenate
    dfs = [pd.read_parquet(chunk) for chunk in wave4_chunks]
    return pd.concat(dfs, ignore_index=True)


def create_prompt(persona_summary: str, question: str, question_type: str, options: list = None) -> str:
    """Create prompt for LLM simulation with answer options."""
    # Format options if provided
    options_text = ""
    if options and len(options) > 0:
        if question_type == "Slider":
            options_text = f"\n{options[0]}"
        else:
            options_text = "\n\nOPTIONS:\n" + "\n".join([f"- {opt}" for opt in options])

    # Create instruction based on question type
    if question_type == "MC":
        instruction = "Respond with ONLY the exact text of your chosen option from the list above. Do not include any explanation or preamble."
    elif question_type == "Matrix":
        instruction = "Respond with ONLY the exact text of your chosen option from the list above. Do not include any explanation or preamble."
    elif question_type == "TE":
        instruction = "Respond with only your answer (a number or short text). Do not include any explanation or preamble."
    elif question_type == "Slider":
        instruction = "Respond with ONLY a number within the range shown above. Do not include any explanation or preamble."
    else:
        instruction = "Respond with only your answer. Do not include any explanation or preamble."

    return f"""You are simulating a person based on their survey responses. Answer the following question as this person would.

PERSONA INFORMATION:
{persona_summary}

QUESTION:
{question}{options_text}

INSTRUCTIONS:
{instruction}"""


def extract_answer(response: str) -> str:
    """Extract letter answer from LLM response."""
    # Remove whitespace and get first character
    cleaned = response.strip()

    # If response is just a letter, return it
    if len(cleaned) == 1 and cleaned.isalpha():
        return cleaned.upper()

    # Try to find first letter
    for char in cleaned:
        if char.isalpha():
            return char.upper()

    return None


def extract_wave4_questions(wave4_json_str):
    """Extract questions from Wave 4 JSON data with answer options and block categories."""
    import json

    if not wave4_json_str:
        return []

    # Parse JSON string
    data = json.loads(wave4_json_str)

    questions = []
    # Iterate through blocks
    for block in data:
        if block.get('ElementType') == 'Block':
            block_name = block.get('BlockName', 'Unknown')

            # Iterate through questions in the block
            for question in block.get('Questions', []):
                # Skip descriptive questions
                if question.get('is_descriptive'):
                    continue

                q_text = question.get('QuestionText', '')
                q_type = question.get('QuestionType', '')
                answers = question.get('Answers', {})

                # Extract answer options and ground truth answer
                answer = None
                options = []

                if q_type == 'MC':
                    answer = answers.get('SelectedText', '')
                    options = question.get('Options', [])
                elif q_type == 'Matrix':
                    # For matrix, use first selected text and first row
                    selected_texts = answers.get('SelectedText', [])
                    if selected_texts:
                        answer = selected_texts[0] if isinstance(selected_texts, list) else selected_texts
                    options = question.get('Columns', [])
                    # Include first row for context
                    rows = question.get('Rows', [])
                    if rows:
                        q_text = f"{q_text}\n\n{rows[0]}"
                elif q_type == 'TE':
                    answer = answers.get('Text', '')
                    # No options for text entry
                elif q_type == 'Slider':
                    values = answers.get('Values', [])
                    if values:
                        answer = values[0] if isinstance(values, list) else values
                    # Include slider range
                    range_info = question.get('Range', {})
                    if range_info:
                        min_val = range_info.get('Min', 0)
                        max_val = range_info.get('Max', 100)
                        options = [f"Range: {min_val} to {max_val}"]

                if q_text and answer:
                    questions.append({
                        'question_id': question.get('QuestionID', ''),
                        'question_text': q_text,
                        'question_type': q_type,
                        'answer': str(answer),
                        'options': options,
                        'block_name': block_name  # Add block category
                    })

    return questions


def run_minimal_test(model=DEFAULT_MODEL, persona_format=DEFAULT_PERSONA_FORMAT,
                     num_personas=DEFAULT_NUM_PERSONAS, max_questions=DEFAULT_MAX_QUESTIONS,
                     data_dir=DEFAULT_DATA_DIR):
    """Run minimal test with configurable parameters."""
    print("=" * 80)
    print("TWIN-2K MINIMAL VALIDATION TEST")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {model}")
    print(f"  Persona format: {persona_format}")
    print(f"  Number of personas: {num_personas}")
    print(f"  Questions per persona: {max_questions}")
    print(f"  Data directory: {data_dir}")
    print()

    # Load data
    print("Loading persona data...")
    persona_df = load_persona_data(data_dir, num_personas)
    print(f"  Loaded {len(persona_df)} personas")

    print("\nLoading Wave 4 ground truth...")
    wave4_df = load_wave4_ground_truth(data_dir)
    print(f"  Loaded {len(wave4_df)} Wave 4 responses")

    # Run simulations
    results = []
    total_questions = 0
    correct_predictions = 0
    total_cost = 0.0

    print("\n" + "=" * 80)
    print("RUNNING SIMULATIONS")
    print("=" * 80)

    for idx, persona_row in persona_df.iterrows():
        pid = persona_row['pid']
        # Format persona using the configured format
        persona_text = format_persona(persona_row, persona_format)

        print(f"\nPersona {idx + 1}/{num_personas} (PID: {pid})")
        print("-" * 80)

        # Get Wave 4 questions for this persona (convert pid to int for matching)
        pid_int = int(pid) if isinstance(pid, str) else pid
        persona_wave4 = wave4_df[wave4_df['pid'] == pid_int]

        if len(persona_wave4) == 0:
            print(f"  WARNING: No Wave 4 data found for PID {pid}")
            continue

        # Get the questions from wave4_Q_wave4_A column
        wave4_json = persona_wave4.iloc[0]['wave4_Q_wave4_A']

        if not wave4_json:
            print(f"  WARNING: No Wave 4 questions for PID {pid}")
            continue

        # Extract all questions and answers
        questions = extract_wave4_questions(wave4_json)

        if not questions:
            print(f"  WARNING: Could not parse Wave 4 questions for PID {pid}")
            continue

        print(f"  Found {len(questions)} Wave 4 questions")

        # Process each question
        for q_idx, q_data in enumerate(questions[:max_questions]):
            question_text = q_data['question_text']
            question_type = q_data['question_type']
            ground_truth = q_data['answer']
            options = q_data.get('options', [])

            # Create prompt and call LLM
            prompt = create_prompt(persona_text, question_text, question_type, options)

            try:
                llm_response = call_llm(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )

                predicted_answer = llm_response.strip()

                # Estimate cost (rough: 4 chars per token)
                input_tokens = len(prompt) // 4
                output_tokens = len(predicted_answer) // 4
                cost_info = calculate_cost(input_tokens, output_tokens, model)
                total_cost += cost_info['total_cost']

                if not predicted_answer:
                    print(f"  Q{q_idx + 1}: ERROR - Empty response from LLM")
                    continue

                # Compare with ground truth (case-insensitive for text, exact for numbers)
                if question_type in ["TE", "Slider"]:
                    # For numeric answers, compare values
                    try:
                        is_correct = abs(float(predicted_answer) - float(ground_truth)) < 0.01
                    except ValueError:
                        is_correct = predicted_answer.lower() == ground_truth.lower()
                else:
                    # For text answers, case-insensitive comparison
                    is_correct = predicted_answer.lower().strip() == ground_truth.lower().strip()

                total_questions += 1
                if is_correct:
                    correct_predictions += 1

                status = "✓" if is_correct else "✗"
                # Truncate long answers for display
                pred_display = predicted_answer[:50] + "..." if len(predicted_answer) > 50 else predicted_answer
                truth_display = ground_truth[:50] + "..." if len(ground_truth) > 50 else ground_truth
                print(f"  Q{q_idx + 1} ({question_type}): {status} Predicted: {pred_display}, Actual: {truth_display}")

                results.append({
                    'pid': pid,
                    'question_id': q_data['question_id'],
                    'question_type': question_type,
                    'block_name': q_data.get('block_name', 'Unknown'),
                    'predicted': predicted_answer,
                    'actual': ground_truth,
                    'correct': is_correct
                })

            except Exception as e:
                print(f"  Q{q_idx + 1}: ERROR - {str(e)}")
                continue

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTotal questions answered: {total_questions}")
    print(f"Correct predictions: {correct_predictions}")

    if total_questions > 0:
        accuracy = correct_predictions / total_questions * 100
        print(f"Accuracy: {accuracy:.1f}%")
    else:
        print("Accuracy: N/A (no questions answered)")

    print(f"\nEstimated total cost: ${total_cost:.2f}")

    # Print breakdown by category/block
    if results:
        results_df = pd.DataFrame(results)

        print("\n" + "=" * 80)
        print("ACCURACY BY CATEGORY")
        print("=" * 80)

        category_stats = results_df.groupby('block_name').agg({
            'correct': ['count', 'sum', 'mean']
        }).round(3)

        category_stats.columns = ['Total', 'Correct', 'Accuracy']
        category_stats = category_stats.sort_values('Accuracy', ascending=False)

        print(f"\n{'Category':<40} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
        print("-" * 80)
        for category, row in category_stats.iterrows():
            total = int(row['Total'])
            correct = int(row['Correct'])
            acc = row['Accuracy'] * 100
            print(f"{category:<40} {total:<8} {correct:<8} {acc:>6.1f}%")

        # Print breakdown by question type
        print("\n" + "=" * 80)
        print("ACCURACY BY QUESTION TYPE")
        print("=" * 80)

        type_stats = results_df.groupby('question_type').agg({
            'correct': ['count', 'sum', 'mean']
        }).round(3)

        type_stats.columns = ['Total', 'Correct', 'Accuracy']
        type_stats = type_stats.sort_values('Accuracy', ascending=False)

        print(f"\n{'Type':<10} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
        print("-" * 80)
        for qtype, row in type_stats.iterrows():
            total = int(row['Total'])
            correct = int(row['Correct'])
            acc = row['Accuracy'] * 100
            print(f"{qtype:<10} {total:<8} {correct:<8} {acc:>6.1f}%")

        # Save results
        output_dir = Path(__file__).parent / "data"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "minimal_test_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n{'=' * 80}")
        print(f"Results saved to: {output_file}")

    print("\n" + "=" * 80)
    return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Twin-2K minimal validation test')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f'LLM model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--personas', type=int, default=DEFAULT_NUM_PERSONAS,
                        help=f'Number of personas to test (default: {DEFAULT_NUM_PERSONAS})')
    parser.add_argument('--questions', type=int, default=DEFAULT_MAX_QUESTIONS,
                        help=f'Number of questions per persona (default: {DEFAULT_MAX_QUESTIONS})')
    parser.add_argument('--format', type=str, default=DEFAULT_PERSONA_FORMAT,
                        help=f'Persona format to use (default: {DEFAULT_PERSONA_FORMAT})')
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help=f'Data directory (default: {DEFAULT_DATA_DIR})')

    args = parser.parse_args()

    run_minimal_test(
        model=args.model,
        persona_format=args.format,
        num_personas=args.personas,
        max_questions=args.questions,
        data_dir=Path(args.data_dir)
    )


if __name__ == "__main__":
    main()
