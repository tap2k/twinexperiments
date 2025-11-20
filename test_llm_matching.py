"""
Test answer matching with real LLM responses.

This script:
1. Loads a few personas and questions from the dataset
2. Gets real LLM responses
3. Checks if the answer matching logic is working correctly
4. Shows examples of matched/mismatched answers
"""

import argparse
from pathlib import Path
import pandas as pd
import json
from dotenv import load_dotenv

load_dotenv()

from utils import extract_wave4_questions, create_prompt
from llm_client import call_llm
from persona_formatter import format_persona


DEFAULT_DATA_DIR = Path(__file__).parent.parent / "Twin-2K-500"


def check_answer_match(predicted, ground_truth, question_type):
    """Answer matching logic from compare_formats.py"""
    if question_type in ["TE", "Slider"]:
        try:
            is_correct = abs(float(predicted) - float(ground_truth)) < 0.01
        except ValueError:
            is_correct = predicted.lower() == ground_truth.lower()
    else:
        is_correct = predicted.lower().strip() == ground_truth.lower().strip()
    return is_correct


def test_real_llm_responses(model='gemini-2.5-flash-lite', num_personas=2, questions_per_persona=5,
                            persona_format='summary', data_dir=None):
    """Test answer matching with real LLM responses."""

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    else:
        data_dir = Path(data_dir)

    print("=" * 80)
    print(f"TESTING REAL LLM ANSWER MATCHING")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Personas: {num_personas}")
    print(f"Questions per persona: {questions_per_persona}")
    print(f"Format: {persona_format}")
    print()

    # Load data
    persona_chunks = list((data_dir / "full_persona/chunks").glob("*.parquet"))
    persona_df = pd.read_parquet(persona_chunks[0]).head(num_personas)

    wave4_chunks = list((data_dir / "wave_split/chunks").glob("*.parquet"))
    wave4_dfs = [pd.read_parquet(chunk) for chunk in wave4_chunks]
    wave4_df = pd.concat(wave4_dfs, ignore_index=True)

    results = []
    correct_matches = []
    incorrect_matches = []

    for idx, persona_row in persona_df.iterrows():
        pid = persona_row['pid']
        persona_text = format_persona(persona_row, persona_format)

        # Get Wave 4 questions
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

        print(f"\n{'=' * 80}")
        print(f"PERSONA {pid}")
        print(f"{'=' * 80}")

        # Test questions
        for i, q_data in enumerate(questions[:questions_per_persona]):
            question_text = q_data['question_text']
            question_type = q_data['question_type']
            ground_truth = q_data['answer']
            options = q_data.get('options', [])
            block_name = q_data.get('block_name', 'Unknown')

            print(f"\n{'-' * 80}")
            print(f"Question {i+1}/{questions_per_persona}")
            print(f"Block: {block_name}")
            print(f"Type: {question_type}")
            print(f"Question: {question_text[:100]}{'...' if len(question_text) > 100 else ''}")
            if options:
                print(f"Options: {options}")
            print(f"Ground Truth: '{ground_truth}'")

            # Create prompt and get LLM response
            prompt = create_prompt(persona_text, question_text, question_type, options)

            try:
                llm_response = call_llm(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                predicted_answer = llm_response.strip()

                print(f"LLM Response: '{predicted_answer}'")

                # Check match
                is_correct = check_answer_match(predicted_answer, ground_truth, question_type)

                print(f"Match Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

                result = {
                    'pid': pid,
                    'block_name': block_name,
                    'question_type': question_type,
                    'question': question_text[:50] + '...',
                    'ground_truth': ground_truth,
                    'predicted': predicted_answer,
                    'correct': is_correct
                }
                results.append(result)

                if is_correct:
                    correct_matches.append(result)
                else:
                    incorrect_matches.append(result)

            except Exception as e:
                print(f"ERROR: {str(e)}")
                continue

    # Summary
    print("\n\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(results)
    correct = len(correct_matches)
    incorrect = len(incorrect_matches)
    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"\nTotal Questions: {total}")
    print(f"Correct: {correct} ({accuracy:.1f}%)")
    print(f"Incorrect: {incorrect} ({100-accuracy:.1f}%)")

    # Breakdown by question type
    print(f"\n{'BREAKDOWN BY QUESTION TYPE'}")
    print("-" * 80)
    df_results = pd.DataFrame(results)
    if len(df_results) > 0:
        type_breakdown = df_results.groupby('question_type').agg({
            'correct': ['count', 'sum', 'mean']
        })
        print(type_breakdown)

    # Show some examples
    if incorrect_matches:
        print(f"\n{'EXAMPLES OF INCORRECT MATCHES'}")
        print("-" * 80)
        for i, result in enumerate(incorrect_matches[:5], 1):
            print(f"\n{i}. {result['block_name']} ({result['question_type']})")
            print(f"   Question: {result['question']}")
            print(f"   Ground Truth: '{result['ground_truth']}'")
            print(f"   Predicted: '{result['predicted']}'")
            print(f"   Issue: ", end="")

            # Analyze the mismatch
            gt = result['ground_truth']
            pred = result['predicted']

            if result['question_type'] in ['TE', 'Slider']:
                try:
                    diff = abs(float(pred) - float(gt))
                    print(f"Numeric difference: {diff:.4f} (threshold: 0.01)")
                except ValueError:
                    print(f"String mismatch: '{pred.lower()}' != '{gt.lower()}'")
            else:
                print(f"String mismatch: '{pred.lower().strip()}' != '{gt.lower().strip()}'")

    if correct_matches:
        print(f"\n{'EXAMPLES OF CORRECT MATCHES'}")
        print("-" * 80)
        for i, result in enumerate(correct_matches[:5], 1):
            print(f"\n{i}. {result['block_name']} ({result['question_type']})")
            print(f"   Ground Truth: '{result['ground_truth']}'")
            print(f"   Predicted: '{result['predicted']}'")

    return results


def main():
    parser = argparse.ArgumentParser(description='Test answer matching with real LLM responses')
    parser.add_argument('--model', type=str, default='gemini-2.5-flash-lite',
                        help='LLM model to use')
    parser.add_argument('--personas', type=int, default=2,
                        help='Number of personas to test')
    parser.add_argument('--questions', type=int, default=5,
                        help='Questions per persona')
    parser.add_argument('--format', type=str, default='demographics_big5',
                        help='Persona format to use')
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help='Data directory')

    args = parser.parse_args()

    test_real_llm_responses(
        model=args.model,
        num_personas=args.personas,
        questions_per_persona=args.questions,
        persona_format=args.format,
        data_dir=args.data_dir
    )


if __name__ == "__main__":
    main()
